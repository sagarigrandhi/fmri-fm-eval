# PPMI 

This workflow processes the PPMI fMRI dataset from the original raw DICOM ZIP files through all the major steps:

1. Extract raw DICOMs from ZIP files.  
2. Convert DICOMs to BIDS structure using `dcm2bids`.  
3. Preprocess fMRI data with `fMRIprep` in both CIFTI 91k and MNI 152 (NLin6Asym)spaces.  
4. Curate a subset of complete data, selecting subjects with PD or prodromal diagnosis, and splitting into train/validation/test sets.  
5. Generate Arrow datasets for downstream modeling and analysis.  


The detailed instructions for each step are listed below. 

This folder contains all scripts needed to convert the PPMI dataset
from the original DICOM ZIP files into a BIDS structure using `dcm2bids`. These instructions assume the PPMI zip files (`PPMI_01.zip.*`, `PPMI_02.zip.*`, …) are already available in a local directory such as:

```
datasets/PPMI/dicom/
```


## Prerequisites
* Ensure `dcm2niix` and `dcm2bids` are installed.
* Ensure `bash` is available


## Reconstruct and Extract the DICOM data
Run the `unzip_ppmi.sh` script in the `datasets/PPMI/scripts/` folder:

```bash
./datasets/PPMI/scripts/unzip_ppmi.sh [base_dir]
```
`[base_dir]` is optional, default: `datasets/PPMI`.

After extraction, the directory structure will look like:
``` 
/datasets/PPMI/unzipped/PPMI/<subject_id>/...
```
`<subject_id>` folders come from the zip contents.


## DICOM to BIDS Conversion

### Create the BIDS directory structure
```bash
mkdir -p datasets/PPMI/bids
```

### Generate BIDS metadata files
```bash
dcm2bids_scaffold -o datasets/PPMI/bids
```

### Copy the configuration file 
The conversion requires a configuration file, provided at: 

`datasets/PPMI/scripts/dcm2bids_config.json`. 

Copy it into the BIDS `code` folder: 
```bash 
cp datasets/PPMI/scripts/dcm2bids_config.json datasets/PPMI/bids/code/dcm2bids_config.json
```

### Run the conversion script
```bash
./datasets/PPMI/scripts/run_dcm2bids.sh [base_dir] [bids_dir] [config_file]
```

* All arguments are optional. 
* Defaults:
* base_dir: `./datasets/PPMI/unzipped/PPMI`
* bids_dir: `./datasets/PPMI/bids`
* config_file: `./datasets/PPMI/bids/code/dcm2bids_config.json`

### Expected output and logging
After conversion, the BIDS dataset will look like:

```
datasets/PPMI/bids/sub-<ID>/ses-<ID>/
```
with:
* ```anat/sub-<ID>_ses-<ID>_T1w.nii.gz```
* ```func/sub-<ID>_ses-<ID>_task-rest_dir-<phase>_bold.nii.gz```

The script creates a log file at:
```
datasets/PPMI/bids/dcm2bids_conversion.log
```

## Index BIDS data

Generate an index of all the BIDS dataset metadata to use for filtering the image data

```bash
uv run scripts/index_ppmi_bids.py
```

* Output: [`metadata/PPMI_BIDS_index.csv`](metadata/PPMI_BIDS_index.csv)

## Generate Filtered Subject List

Run the BIDS filter notebook to generate a filtered list of valid subjects, sessions, runs:

```bash
datasets/PPMI/notebooks/ppmi_bids_filter.ipynb
```

* This notebook excludes invalid T1w and bold runs (e.g. short acquisition, invalid shape), excludes incomplete sessions, and excludes subjects without at least one complete session.
* Outputs:
    * [`metadata/PPMI_BIDS_complete.csv`](metadata/PPMI_BIDS_complete.csv)
    * [`metadata/PPMI_BIDS_complete_paths.txt`](metadata/PPMI_BIDS_complete_paths.txt)
    * [`metadata/PPMI_BIDS_complete_subs.txt`](metadata/PPMI_BIDS_complete_subs.txt)
* These files will be used as input for the next curation step.

## Generate filtered BIDS dataset

Create a filtered BIDS dataset `bids_complete/` by symlinking all included runs from the full `bids/` directory.

```bash
uv run python scripts/link_ppmi_bids_complete.py
```

The complete filtered BIDS data are backed up to `s3://medarc/fmri-fm-eval/PPMI/bids_complete/`

## Data preprocessing

Run fMRIprep preprocessing using the [`scripts/preprocess.sh`](scripts/preprocess.sh) script.

Preprocessing jobs were run in batches on [lightning.ai](https://lightning.ai). See [`notebooks/submit_batch.ipynb`](notebooks/submit_batch.ipynb) for example batch job submission.


## Curate Complete PPMI Dataset

Construct a curated subset of PPMI fMRI data. The inclusion criteria are:
* Subjects with diagnosis PD or Prodromal.
* Runs with complete preprocessing outputs in CIFTI 91k and MNI 152 (NLin6Asym).
* Split into train, validation, and test sets (70:15:15), stratifying by sex, age (three bins), and diagnosis.
See [`notebooks/ppmi_curation.ipynb`](notebooks/ppmi_curation.ipynb) for the curation script.

The output table of curated subjects is in `metadata/PPMI_curated.csv`.


## Generate Arrow datasets

Finally, generate output Arrow datasets for all target standard spaces

```bash
uv run python scripts/make_ppmi_dataset.py --space mni
```