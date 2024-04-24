# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Author: Dillon Lohr (djl70@txstate.edu)
# Property of Texas State University.

from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def build_roc(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    return fpr, tpr, thresholds


def estimate_fnr_at_fpr(
    fpr: np.ndarray, tpr: np.ndarray, fixed_fpr: float
) -> float:
    fnr = 1.0 - interp1d(fpr, tpr)(fixed_fpr)
    return float(fnr)


def estimate_eer(fpr: np.ndarray, tpr: np.ndarray) -> float:
    eer = brentq(lambda x: estimate_fnr_at_fpr(fpr, tpr, x) - x, 0.0, 1.0)
    return float(eer)


def estimate_threshold_at_fpr(
    fpr: np.ndarray, thresholds: np.ndarray, fixed_fpr: float
) -> float:
    fpr_threshold = interp1d(fpr, thresholds)(fixed_fpr)
    return float(fpr_threshold)
