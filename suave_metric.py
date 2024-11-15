#!/usr/bin/env python3
import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=False, nogil=True, cache=True)
def getMetric(dE, dP):
    """
    This function computes the ordering metric for a given set of finite differences.
    :param dE: the finite differences of the energies at the current point (shape: (numStates))
    :param dP: the finite differences of the properties at the current point (shape: (numStates, numFeatures))
    :return the ordering metric for the given finite differences
    """

    # reduce dP to a single vector
    dP = dP @ np.ones(dP.shape[1])

    return (dE * dP).sum()
