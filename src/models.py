"""Model zoo.

Every model is returned as a *complete* pipeline (preprocessing + estimator) so
that cross-validation cannot accidentally leak.  Nothing here is fitted.

Design notes
------------
* ``class_weight="balanced"`` is used wherever the estimator supports it.  With
  ~15% positives, an unweighted logistic regression optimises overall accuracy
  and mostly predicts "no CHD" — which is exactly the trap the original
  notebook fell into.
* The dummy classifier is included on purpose: on an imbalanced target, a
  baseline that always answers "no" is the number a model must beat, and the
  original analysis never reported it.
* The MLP keeps the original's architecture ``(100, 100, 100)`` but fixes the
  optimiser (``sgd`` at lr 0.01 on unscaled-ish inputs is why the original's MLP
  scored *worse* than logistic regression).
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .config import RANDOM_STATE
from .preprocess import build_pipeline

__all__ = ["get_models", "MODEL_ORDER"]


# Display order: baselines first, then models from most to least interpretable.
MODEL_ORDER = [
    "Dummy (majority class)",
    "Logistic Regression",
    "Logistic Regression (balanced)",
    "Linear SVM (balanced)",
    "Random Forest (balanced)",
    "Hist Gradient Boosting",
    "MLP (100,100,100)",
]


def get_models() -> dict[str, object]:
    """Return ``{display name: unfitted pipeline}``.

    A fresh dict of fresh objects is built on every call, so callers can mutate
    or fit without cross-talk between scripts.
    """
    models: dict[str, object] = {
        # ---- baseline: always predicts the majority class ------------------ #
        "Dummy (majority class)": build_pipeline(
            DummyClassifier(strategy="prior", random_state=RANDOM_STATE)
        ),
        # ---- the original's model, correctly preprocessed ------------------ #
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        # ---- same model, imbalance-aware ----------------------------------- #
        "Logistic Regression (balanced)": build_pipeline(
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            )
        ),
        # ---- the original's second model, imbalance-aware ------------------- #
        # SVC(probability=True) is deprecated in scikit-learn 1.9; wrapping a
        # deterministic linear SVC in Platt scaling is the supported equivalent
        # and produces the same kind of out-of-fold calibrated probabilities.
        "Linear SVM (balanced)": build_pipeline(
            CalibratedClassifierCV(
                SVC(kernel="linear", class_weight="balanced"),
                method="sigmoid",
                cv=3,
                ensemble=False,
            )
        ),
        # ---- a non-linear tree ensemble ------------------------------------ #
        "Random Forest (balanced)": build_pipeline(
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        # ---- a strong tabular baseline ------------------------------------- #
        "Hist Gradient Boosting": build_pipeline(
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=RANDOM_STATE,
            )
        ),
        # ---- the original's third model, architecture kept, solver fixed --- #
        "MLP (100,100,100)": build_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(100, 100, 100),
                activation="relu",
                solver="adam",
                learning_rate_init=1e-3,
                alpha=1e-4,
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=RANDOM_STATE,
            )
        ),
    }
    return models
