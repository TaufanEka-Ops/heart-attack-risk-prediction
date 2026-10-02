"""Single source of truth for paths, feature groups, and the random seed.

Everything that the rest of the code needs to agree on lives here, so a reviewer
can change one constant and have the change propagate everywhere.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TRANSCRIPTS_DIR = RESULTS_DIR / "transcripts"
RAW_CSV = DATA_DIR / "framingham.csv"

# --------------------------------------------------------------------------- #
# Experiment protocol                                                          #
# --------------------------------------------------------------------------- #
TARGET = "TenYearCHD"
RANDOM_STATE = 42
TEST_SIZE = 0.2          # matches the original notebook's 80/20 split
N_SPLITS = 5             # stratified folds for cross-validation

# The original notebook used a single 80/20 split with no cross-validation at
# all.  We keep an 80/20 holdout so the comparison is like-for-like, but every
# headline number comes from the 5-fold CV mean, not from that one split.

# --------------------------------------------------------------------------- #
# Feature groups                                                               #
# --------------------------------------------------------------------------- #
# Each group gets a *different* imputation rule inside the pipeline.  The
# original notebook applied one rule (the column mean) to all three groups,
# which is why `education` ended up with fractional values like 2.94.
CONTINUOUS_FEATURES = [
    "age",
    "cigsPerDay",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

# 0/1 indicators.  `BPMeds` is the only one of these with missing values.
BINARY_FEATURES = [
    "male",
    "currentSmoker",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
]

# Ordered 1..4 but genuinely categorical; a mean of 2.94 is not a category.
ORDINAL_FEATURES = ["education"]

FEATURE_COLUMNS = CONTINUOUS_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES

# Column order exactly as it appears in the raw CSV.  Tests assert on this.
EXPECTED_COLUMNS = [
    "male",
    "age",
    "education",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
    "TenYearCHD",
]

# --------------------------------------------------------------------------- #
# Data contract — values verified against the raw file                         #
# --------------------------------------------------------------------------- #
EXPECTED_N_ROWS = 4238

# Missing-value counts as they actually are in data/raw/framingham.csv.  If a
# future edit to the CSV changes any of these, the tests fail loudly.
EXPECTED_MISSING = {
    "education": 105,
    "cigsPerDay": 29,
    "BPMeds": 53,
    "totChol": 50,
    "BMI": 19,
    "heartRate": 1,
    "glucose": 388,
}

# Domain ranges used by the data-contract check.  Deliberately generous: this is
# a sanity check against a mangled file, not a clinical validity claim.
PLAUSIBLE_RANGES = {
    "age": (30, 80),
    "education": (1, 4),
    "cigsPerDay": (0, 70),
    "totChol": (100, 700),
    "sysBP": (80, 300),
    "diaBP": (40, 150),
    "BMI": (14, 60),
    "heartRate": (40, 150),
    "glucose": (40, 400),
}
