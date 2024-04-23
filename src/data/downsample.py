# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Author: Dillon Lohr (djl70@txstate.edu)
# Property of Texas State University.

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import decimate


def downsample_recording(
    df: pd.DataFrame,
    downsample_factors: Sequence[int],
    initial_sampling_rate_hz: int,
) -> Tuple[np.ndarray, int]:
    """
    Decimate gaze positions while preserving NaNs.

    First, NaNs are interpolated over using pchip.  Then, the data is
    decimated by each downsample factor in sequence.  Lastly, NaNs are
    put back into the downsampled data, using linear interpolation to
    propagate them as needed.

    If `downsample_factors` is empty, `df` is simply converted to an
    np.array; no interpolation or decimation is performed.

    Parameters
    ----------
    df : DataFrame
        A data frame containing at least 2 columns named `x` and `y`,
        representing horizontal and vertical gaze position,
        respectively.
    downsample_factors : list of ints
        A (potentially empty) list of factors by which the data will be
        decimated.  The factors are applied in sequence.
    initial_sampling_rate_hz : int
        The initial sampling rate (Hz) before downsampling.

    Returns
    -------
    gaze : np.array
        The downsampled data.  If `downsample_factors` is empty, this is
        simply `df` converted to an np.array.
    ideal_sampling_rate : int
        The ideal sampling rate (Hz) after downsampling.
    """
    data_cols = ["x", "y"]
    if len(downsample_factors) == 0:
        return df[data_cols].to_numpy(), initial_sampling_rate_hz

    # We can't decimate with NaNs in the data (it propagates them too
    # far), so we'll first interpolate with pchip to fill in any NaNs
    nan_free_df = df.interpolate(method="pchip", limit_direction="both")
    gaze = nan_free_df[data_cols].to_numpy()

    # Now we can downsample with an anti-aliasing filter
    ideal_sampling_rate = initial_sampling_rate_hz
    for factor in downsample_factors:
        gaze = decimate(gaze, factor, axis=0)
        ideal_sampling_rate /= factor

    # Since we had to remove NaNs ealier, we'll now put them back in.
    # But, since the signals were decimated, we'll also resample the
    # NaNs using np.interp to preserve and propagate them.
    new_nans = np.stack(
        [
            np.interp(
                np.arange(gaze.shape[0]) * ideal_sampling_rate,
                np.arange(df.shape[0]),
                df[col].to_numpy(),
            )
            for col in data_cols
        ],
        axis=-1,
    )
    gaze[np.isnan(new_nans)] = np.nan

    return gaze, ideal_sampling_rate
