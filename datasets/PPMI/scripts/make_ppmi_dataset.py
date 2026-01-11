import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import datasets as hfds
import numpy as np
import pandas as pd

import fmri_fm_eval.nisc as nisc
import fmri_fm_eval.readers as readers

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.INFO,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)

_logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]

SPLITS = ["train", "validation", "test"]

# Resample all time series to fixed TR
# PPMI TR is 1.0 (for AP/PA) or 2.5 (for LR/RL).
TARGET_TR = 2.5
# Keep first 120 TRs (5 mins) from each run
# All PPMI runs are 10 min long. We could also consider keeping more data.
MAX_NUM_TRS = 120


def main(args):
    outdir = ROOT / f"data/processed/ppmi.{args.space}.arrow"
    _logger.info("Generating dataset: %s", outdir)
    if outdir.exists():
        _logger.warning("Output %s exists; exiting.", outdir)
        return 1

    # load curated subjects
    curated_df = pd.read_csv(ROOT / "metadata/PPMI_curated.csv", dtype={"Subject": str})

    # load curated paths and get first run per subject
    all_curated_paths = open(ROOT / "metadata/PPMI_curated_paths.txt").read().splitlines()
    curated_paths = {}
    for path in all_curated_paths:
        sub_with_prefix = path.split("/")[0]
        sub = sub_with_prefix.split("-")[1]
        if sub not in curated_paths:
            curated_paths[sub] = path
    curated_paths = list(curated_paths.values())

    if args.space in {"a424", "mni", "mni_cortex"}:
        suffix = "_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    else:
        suffix = "_space-fsLR_den-91k_bold.dtseries.nii"

    # preprocessed data paths
    # note, preprocessed fmriprep data should go in the dataset folder under data/
    fmriprep_root = ROOT / "data/fmriprep"
    curated_paths = [fmriprep_root / p.replace("_bold.nii.gz", suffix) for p in curated_paths]

    # mapping of subs to assigned splits
    sub_split_map = {sub: split for sub, split in zip(curated_df["Subject"], curated_df["split"])}

    missing_paths = [p for p in curated_paths if not p.exists()]
    if missing_paths:
        _logger.error(f"Missing {len(missing_paths)} files. First few:")
        for p in missing_paths[:5]:
            _logger.error(f"  {p}")
        return 1

    # data paths for each split
    path_splits = {split: [] for split in SPLITS}
    for path in curated_paths:
        sub_with_prefix = str(path.relative_to(fmriprep_root)).split("/")[0]
        sub = sub_with_prefix.split("-")[1]
        split = sub_split_map[sub]
        path_splits[split].append(path)

    # load the data reader for the target space and look up the data dimension
    reader = readers.READER_DICT[args.space]()
    dim = readers.DATA_DIMS[args.space]

    # define features
    features = hfds.Features(
        {
            "sub": hfds.Value("string"),
            "ses": hfds.Value("string"),
            "dir": hfds.Value("string"),
            "sex": hfds.Value("string"),
            "age": hfds.Value("float32"),
            "age_bin": hfds.Value("string"),
            "dx": hfds.Value("string"),
            "path": hfds.Value("string"),
            "n_frames": hfds.Value("int32"),
            "tr": hfds.Value("float32"),
            "bold": hfds.Array2D(shape=(None, dim), dtype="float16"),
            "mean": hfds.Array2D(shape=(1, dim), dtype="float32"),
            "std": hfds.Array2D(shape=(1, dim), dtype="float32"),
        }
    )

    # generate the datasets with huggingface. cache to a temp dir to save space.
    with tempfile.TemporaryDirectory(prefix="huggingface-") as tmpdir:
        dataset_dict = {}
        for split, paths in path_splits.items():
            dataset_dict[split] = hfds.Dataset.from_generator(
                generate_samples,
                features=features,
                gen_kwargs={
                    "paths": paths,
                    "curated_df": curated_df,
                    "root": fmriprep_root,
                    "reader": reader,
                    "dim": dim,
                },
                num_proc=args.num_proc,
                split=hfds.NamedSplit(split),
                cache_dir=tmpdir,
            )
        dataset = hfds.DatasetDict(dataset_dict)
        outdir.parent.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(outdir, max_shard_size="300MB")
        _logger.info(f"Dataset saved to {outdir}")


def generate_samples(paths, *, curated_df, root, reader, dim):
    for path in paths:
        sidecar_path = path.parent / (path.name.split(".")[0] + ".json")
        with sidecar_path.open() as f:
            sidecar_data = json.load(f)
        tr = float(sidecar_data["RepetitionTime"])

        # Extract subject and session from path
        stem = path.name.split(".")[0]
        meta = dict(item.split("-") for item in stem.split("_") if "-" in item)
        sub = meta["sub"]

        # Get demographics from curated_df
        row = curated_df[curated_df["Subject"] == sub].iloc[0]
        meta = {
            "sub": sub,
            "ses": meta["ses"],
            "dir": meta["dir"],
            "sex": row["Sex"],
            "age": float(row["Age"]),
            "age_bin": row["age_bin"],
            "dx": row["Group"],
        }

        series = reader(path)
        series = nisc.resample_timeseries(series, tr=tr, new_tr=TARGET_TR, kind="pchip")

        T, D = series.shape
        assert T >= MAX_NUM_TRS, f"Path {path} has too few TRs ({T}<{MAX_NUM_TRS})"

        series = series[:MAX_NUM_TRS]
        series, mean, std = nisc.scale(series)

        sample = {
            **meta,
            "path": str(path.relative_to(root)),
            "n_frames": MAX_NUM_TRS,
            "tr": TARGET_TR,
            "bold": series.astype(np.float16),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }
        yield sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", type=str, default="fslr64k", choices=list(readers.READER_DICT))
    parser.add_argument("--num_proc", "-j", type=int, default=32)
    args = parser.parse_args()
    sys.exit(main(args))
