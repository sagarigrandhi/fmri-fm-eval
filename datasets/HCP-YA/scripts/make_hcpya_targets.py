from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
HCP_UNRESTRICTED_CSV_PATH = ROOT / "metadata/hcpya_unrestricted.csv"

# mapping of age ranges to discrete targets
AGE_MAP = {
    "22-25": 0,
    "26-30": 1,
    "31-35": 2,
    "36+": 2,  # merge with previous due to only 14 instances
}

# Phenotypic/Cognitive targets
#
# - NEOFAC_N: Neuroticism Scale Score NEO-FFI.
# - Flanker_Unadj: Flanker task score.
# - PMAT24_A_CR: Penn Matrix Test (PMAT): Number of Correct Responses.
#
# Reference:
#   https://wiki.humanconnectome.org/docs/HCP-YA%20Data%20Dictionary-%20Updated%20for%20the%201200%20Subject%20Release.html
TARGETS = ["NEOFAC_N", "Flanker_Unadj", "PMAT24_A_CR"]

# number of discrete quantile bins
# 3 bins is consistent with age, and with brain-semantoks
# (though it is a bit odd to use 3 bins and not 4).
NUM_BINS = 3


def main():
    raw_df = pd.read_csv(HCP_UNRESTRICTED_CSV_PATH, dtype={"Subject": str})

    target_df = raw_df.loc[:, ["Subject", "Gender", "Age"]]
    target_df["Age_Q"] = target_df["Age"].map(AGE_MAP)

    for col in TARGETS:
        target_df[col] = raw_df[col]
        target_df[f"{col}_Q"] = pd.qcut(raw_df[col], q=NUM_BINS, labels=False)

    target_df.to_csv(ROOT / "metadata/hcpya_pheno_targets.csv", index=False)


if __name__ == "__main__":
    main()
