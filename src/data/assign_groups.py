# This work is licensed under a "Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License"
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# Author: Dillon Lohr (djl70@txstate.edu)
# Property of Texas State University.

import heapq
from typing import Dict, Sequence, Tuple

import numpy as np


def assign_groups(
    n_groups: int, values: Sequence[int], weights: Sequence[int]
) -> Tuple[Dict[int, int], np.ndarray]:
    """
    Assign values to groups with equal sizes and weights.

    The first priority is that each group is the same size.  The second
    priority is that each group has equal weight.  The largest group
    will have no more than one extra element than the smallest group.

    Parameters
    ----------
    n_groups : int
        The number of groups to which the values will be assigned.  Does
        not have to evenly divide `len(values)`.
    values : list of ints
        The values to assign to groups.  Must have the same length as
        `weights`.
    weights : list of ints
        The weight of each value, where a higher weight indicates a
        higher priority for that value.  Must have the same length as
        `values`.

    Returns
    -------
    dict int->int
        A dictionary mapping each group to its values.
    np.array
        `n_groups` x 3 array.  Each row contains the number of values
        assigned to the group, the total weight of the group, and the
        group index.  The group index is a number in `range(n_groups)`.
        Intended for debugging purposes.

    References
    ----------
    https://stackoverflow.com/a/49416480
    """
    groups = [[0, 0, i, []] for i in range(n_groups)]
    heapq.heapify(groups)

    items = [(w, v) for w, v in zip(weights, values)]
    heapq.heapify(items)

    while items:
        group = heapq.heappop(groups)
        item = heapq.heappop(items)
        # Our primary concern is equal-sized groups
        group[0] += 1
        # Our secondary concern is equal-weighted groups.  We negate the
        # weight added to the group so that groups with higher weights
        # are prioritized less.
        group[1] -= item[0]
        # Keep track of the values assigned to each group
        group[3].append(item[1])
        # Put the group back onto the heap
        heapq.heappush(groups, group)

    group_to_values = {g[2]: g[3] for g in groups}
    return group_to_values, np.array([g[:3] for g in groups])
