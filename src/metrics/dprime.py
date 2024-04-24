# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Author: Dillon Lohr (djl70@txstate.edu)
# Property of Texas State University.

import numpy as np


def decidability_index(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positive_scores = y_score[y_true == 1]
    negative_scores = y_score[y_true == 0]

    x1_mu, x2_mu = np.mean(positive_scores), np.mean(negative_scores)
    x1_sigma, x2_sigma = np.std(positive_scores), np.std(negative_scores)
    x1_variance, x2_variance = x1_sigma**2, x2_sigma**2
    dprime = np.abs(x1_mu - x2_mu) / np.sqrt(0.5 * (x1_variance + x2_variance))
    return float(dprime)
