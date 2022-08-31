"""
   Copyright 2022 Marcus D. Liebenthal

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       https://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import copy as cp
import sys

import numba
import numpy as np
from numba import njit, prange, set_num_threads
from numpy import zeros, stack, insert, savetxt, inf, genfromtxt

from nstencil import makeStencil


def generateDerivatives(N, center, F, allPnts, minh, orders, width, cutoff, maxPan):
    """
    This function approximates the n-th order derivatives of a function, F, at a point, center.
    The derivatives are computed using a stencil of width, width.
    The cutoff sets how many points to consider from the right of the center
    the maxPan sets how far the window of size width can shift from the center point
    The stencil is centered at center and the derivatives are computed using the points in the stencil.
    The stencil is computed using the points in allPnts.
    """

    if orders is None:
        orders = [1]
    if width <= 1 or width > N:
        width = N
    if maxPan is None:
        maxPan = N
    if cutoff is None:
        cutoff = N

    combinedStencils = {}
    setDiff = False

    for order in orders:
        offCount = 0

        for off in range(-width, 1):
            if offCount >= maxPan:
                continue
            # get size of stencil
            s = []
            for i in range(width):
                idx = (i + off)
                if idx > cutoff:
                    break
                if 0 <= center + idx < N:
                    s.append(idx)
            sN = len(s)

            # ensure stencil is large enough for finite difference and not larger than data set
            if sN <= order or sN > N or sN <= 1:
                continue

            # ensure center point is included in stencil
            s = np.asarray(s)  # convert to np array
            if 0 not in s:
                continue

            # scale stencil to potentially non-uniform mesh based off smallest point spacing
            sh = zeros(sN)
            for idx in range(sN):
                sh[idx] = (allPnts[center + s[idx]] - allPnts[center]) / minh
            sh.flags.writeable = False

            # get finite difference coefficients from stencil points
            alpha = makeStencil(sN, sh, order)

            # collect finite difference coefficients from this stencil
            h_n = minh ** order
            for idx in range(sN):
                try:
                    combinedStencils[s[idx]] += alpha[idx] / h_n
                except KeyError:
                    combinedStencils[s[idx]] = alpha[idx] / h_n

            # mark if any finite difference coefficients have yet been computed
            if not setDiff:
                setDiff = True
            offCount += 1

    if not setDiff:
        raise "No finite differences were computed!"

    stencils = np.asarray(list(combinedStencils.items()), dtype=np.float32)
    stencil = stencils[:, 0].astype(int)
    alphas = stencils[:, 1]
    sN = len(combinedStencils)
    if len(F.shape) == 3:
        diff = zeros((F.shape[0], sN, F.shape[2]))
    elif len(F.shape) == 2:
        diff = zeros((F.shape[0], sN))
    else:
        raise "energies and/or features have incompatible dimensions"

    # compute combined finite differences from all stencils considered in panning window
    return approxDeriv(F, diff, center, stencil, alphas, sN)


# The function takes as input the sequence of points, the difference between the points, the center point,
# the stencil, the coefficients and the size of the stencil.
# The function returns the absolute value of the difference between the points
@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def approxDeriv(F, diff, center, stencil, alphas, sN):
    for s in prange(sN):
        pnt = center + stencil[s]
        diff[:, s] = alphas[s] * F[:, pnt]
    return np.absolute(diff.sum(axis=1))


# enforce ordering metric
@njit(parallel=True, fastmath=True)
def getMetric(diffE, diffN):
    # mean approximate differences summed for each state
    Esum = diffE.sum()
    Nsum = diffN.mean()
    return abs(Nsum * Esum)


# sort the state such that the first point is in ascending energy order
def sortEnergies(Evals, Nvals):
    idx = Evals[:, 0].argsort()
    Evals[:] = Evals[idx]
    Nvals[:] = Nvals[idx]


def arrangeStates(Evals, Nvals, allPnts, configPath=None, maxiter=-1, repeatMax=2, numStateRepeat=10):
    # This function will take a sequence of points for multiple states with energies and properties and reorder them such that the energies and properties are continuous
    #
    # Inputs:
    #   Evals: A 2D array of energies for each state at each point
    #   Nvals: A 2D array of properties for each state at each point
    #   allPnts: A 1D array of points
    #   bounds: A 2D array of the start and end points to consider
    #   maxiter: The maximum number of iterations to try to converge
    #   repeatMax: The maximum number of times to repeat the same point
    #   numStateRepeat: The maximum number of times to repeat the same state
    #
    # Outputs:
    #   Evals: A reordered 2D array of energies for each state at each point
    #   Nvals: A reordered 2D array of properties for each state at each point

    numStates = Evals.shape[0]
    numPoints = Evals.shape[1]
    if maxiter is None:
        maxiter = numStates + 1

    # find smallest step size
    minh = np.inf
    for idx in range(numPoints - 1):
        h = allPnts[idx + 1] - allPnts[idx]
        if h < minh:
            minh = h

    # copy initial curve info
    lastEvals = cp.deepcopy(Evals)
    lastNvals = cp.deepcopy(Nvals)

    # containers to count unmodified states
    stateRepeatList = np.zeros(numStates)

    # set parameters from configuration file
    orders, width, cutoff, maxPan, stateBounds, pntBounds, nthreads, makePos, doShuffle = applyConfig(configPath)

    if pntBounds is None:
        pntBounds = [0, numPoints]

    # arrange states such that energy and amplitude norms are continuous
    numSweeps = numPoints ** 2
    for sweep in range(numSweeps):
        for state in range(numStates):
            stateEvals = cp.deepcopy(Evals)
            stateNvals = cp.deepcopy(Nvals)
            saveOrder(Evals, Nvals, allPnts)
            for pnt in range(pntBounds[0], pntBounds[1]):
                if stateRepeatList[state] >= numStateRepeat:  # Modify
                    print("\n%%%%%%%%%%", "SWEEP ", sweep, "%%%%%%%%%%")
                    print("@@@@@@@", "STATE", state, "@@@@@@@@@")
                    stateRepeatList[state] += 1
                    if stateRepeatList[state] >= numStateRepeat + 5:
                        stateRepeatList[state] = 0
                        print("###", "POINT", pnt, "###  -- retest")
                    else:
                        print("###", "POINT ", pnt, "###  -- Converged")
                        continue
                else:
                    print("\n%%%%%%%%%%", "SWEEP ", sweep, "%%%%%%%%%%")
                    print("@@@@@@@", "STATE", state, "@@@@@@@@@")
                    print("###", "POINT", pnt, "###")
                repeat = 0
                itr = 0
                lastDif = (inf, state)

                # selection Sort algorithm to arrange states
                while repeat <= repeatMax and itr < maxiter:

                    # nth order finite difference
                    diffE = generateDerivatives(numPoints, pnt, Evals[state:], allPnts, minh, orders, width, cutoff,
                                                maxPan)
                    diffN = generateDerivatives(numPoints, pnt, Nvals[state:], allPnts, minh, orders, width, cutoff,
                                                maxPan)

                    if diffE.size == 0 or diffN.size == 0:
                        continue

                    diff = getMetric(diffE, diffN)
                    minDif = (diff, state)

                    # compare continuity differences from this state swapped with all other states
                    for i in range(state + 1, numStates):
                        # get energy and norm for each state at this point
                        Evals[[state, i], pnt] = Evals[[i, state], pnt]
                        Nvals[[state, i], pnt] = Nvals[[i, state], pnt]

                        # nth order finite difference
                        diffE = generateDerivatives(numPoints, pnt, Evals[state:], allPnts, minh, orders, width, cutoff,
                                                    maxPan)
                        diffN = generateDerivatives(numPoints, pnt, Nvals[state:], allPnts, minh, orders, width, cutoff,
                                                    maxPan)

                        if diffE.size == 0 or diffN.size == 0:
                            continue

                        diff = getMetric(diffE, diffN)

                        if diff < minDif[0]:
                            minDif = (diff, i)

                        Evals[[state, i], pnt] = Evals[[i, state], pnt]
                        Nvals[[state, i], pnt] = Nvals[[i, state], pnt]

                    if lastDif[1] == minDif[1]:
                        repeat += 1
                    else:
                        repeat = 0
                        print(state, "<---", minDif[1], flush=True)

                    # swap state in point with new state that has the most continuous change in amplitude norms
                    newState = minDif[1]
                    if state != newState:
                        Evals[[state, newState], pnt] = Evals[[newState, state], pnt]
                        Nvals[[state, newState], pnt] = Nvals[[newState, state], pnt]

                    lastDif = minDif
                    itr += 1
                if itr >= maxiter:
                    print("WARNING: state could not converge. Increase maxiter or maxRepeat")
                else:
                    print(lastDif)
                sys.stdout.flush()
            delMax = stateDifference(Evals, Nvals, stateEvals, stateNvals, state)

            if delMax < 1e-12:
                stateRepeatList[state] += 1
            else:
                stateRepeatList[state] = 0
            stateEvals = cp.deepcopy(Evals)
            stateNvals = cp.deepcopy(Nvals)

        delEval = Evals - lastEvals
        delNval = Nvals - lastNvals

        delMax = delEval.max() + delNval.max()
        print("CONVERGENCE PROGRESS: {:e}".format(delMax))
        if delMax < 1e-12:
            print("%%%%%%%%%%%%%%%%%%%% CONVERGED {:e} %%%%%%%%%%%%%%%%%%%%%%".format(delMax))
            break
        lastEvals = cp.deepcopy(Evals)
        lastNvals = cp.deepcopy(Nvals)
        sortEnergies(Evals, Nvals)


@njit(parallel=True)
def stateDifference(Evals, Nvals, stateEvals, stateNvals, state):
    # This function will calculate the difference in the energy from a previous reordering to the current order
    # This is used to determine if the current order is better than the previous order
    delEval = Evals[state] - stateEvals[state]
    delNval = Nvals[state] - stateNvals[state]
    delMax = delEval.max() + delNval.max()
    return delMax


def shuffle_energy(curves):
    # This function will randomize the state ordering for each point
    for pnt in range(curves.shape[1]):
        np.random.shuffle(curves[:, pnt])


# this function loads the state information of a reorder scan from a previous run of this script
def loadPrevRun(numStates, numPoints, numColumns):
    Ecurves = genfromtxt('Evals.csv')
    Ncurves = genfromtxt('Nvals.csv')
    allPnts = genfromtxt('allPnts.csv')

    Evals = Ecurves.reshape((numStates, numPoints))
    Nvals = Ncurves.reshape((numStates, numPoints, numColumns))

    return Evals, Nvals, allPnts


# this function reformats the state information for saving
@njit(parallel=True)
def combineVals(Evals, Nvals, allPnts, tempInput):
    numPoints = allPnts.shape[0]
    numStates = Evals.shape[0]
    numFeat = Nvals.shape[2]
    for pnt in prange(numPoints):
        for state in prange(numStates):
            tempInput[pnt * numStates + state, 0] = allPnts[pnt]
            tempInput[pnt * numStates + state, 1] = Evals[state, pnt]
            for feat in prange(numFeat):
                tempInput[pnt * numStates + state, feat + 2] = Nvals[state, pnt, feat]


# this function saves the state information of a reorder scan for a future run of this script
def saveOrder(Evals, Nvals, allPnts):
    tempInput = zeros((Nvals.shape[0] * Nvals.shape[1], Nvals.shape[2] + 2))
    combineVals(Evals, Nvals, allPnts, tempInput)

    savetxt("tempInput.csv", tempInput, fmt='%20.12f')

    newCurvesList = []
    for pnt in range(Evals.shape[1]):
        newCurvesList.append(Evals[:, pnt])
    newCurves = stack(newCurvesList, axis=1)
    newCurves = insert(newCurves, 0, allPnts, axis=0)
    savetxt('tempOutput.csv', newCurves, fmt='%20.12f')


def stringToList(string):
    return string.replace(" ", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "").split(",")


def applyConfig(configPath=None):
    orders = [1]
    width = 8
    cutoff = 1
    maxPan = None
    stateBounds = None
    pntBounds = None
    nthreads = 1
    makePos = False
    doShuffle = False
    if configPath is None:
        return orders, width, cutoff, maxPan, stateBounds, pntBounds, nthreads, makePos, doShuffle

    configer = open(configPath, 'r')
    for line in configer.readlines():
        line = line.replace("\n", "").replace(" ", "").strip()
        if line[0] == "#":
            continue
        splitLine = line.split("=")
        if "order" in line:
            try:
                orders = stringToList(splitLine[1])
                orders = [int(order) for order in orders]
                if len(orders) == 0:
                    print("The orders of the derivatives desired for computation are required. Defaulting to '[1]'")
                    orders = [1]
            except ValueError:
                print("The orders of the derivatives desired for computation are required. Defaulting to '[1]'")
                orders = [1]
        if "width" in line:
            try:
                width = int(splitLine[1])
            except ValueError:
                print("invalid type for width. Defaulting to '8'")
                width = 8
        if "cutoff" in line:
            try:
                cutoff = int(splitLine[1])
            except ValueError:
                if "None" not in splitLine[1]:
                    print("invalid type for cutoff. Defaulting to '1'")
                cutoff = 1
        if "maxPan" in line:
            try:
                maxPan = int(splitLine[1])
            except ValueError:
                if "None" not in splitLine[1]:
                    print("invalid type for maxPan. Defaulting to 'None'")
                maxPan = None
        if "pntBounds" in line:
            try:
                pntBounds = stringToList(splitLine[1])
                pntBounds = [int(pntBound) for pntBound in pntBounds]
                if len(pntBounds) == 0:
                    print("The pntBounds provided is invalid. Defaulting to 'None'")
                    pntBounds = None
            except ValueError:
                if "None" not in splitLine[1]:
                    print("The pntBounds provided is invalid. Defaulting to 'None'")
                pntBounds = None
        if "stateBounds" in line:
            try:
                stateBounds = stringToList(splitLine[1])
                stateBounds = [int(stateBound) for stateBound in stateBounds]
                if len(stateBounds) == 0:
                    print("The stateBounds provided is invalid. Defaulting to 'None'")
                    stateBounds = None
            except ValueError:
                if "None" not in splitLine[1]:
                    print("The stateBounds provided is invalid. Defaulting to 'None'")
                stateBounds = None
        if "nthreads" in line:
            try:
                nthreads = int(splitLine[1])
            except ValueError:
                print("Invalid nthread size. Defaulting to 1.")
                nthreads = 1
        if "makePos" in line:
            try:
                makePos = bool(splitLine[1])
            except ValueError:
                print("Invalid makePos. Defaulting to False.")
                makePos = False
        if "doShuffle" in line:
            try:
                doShuffle = bool(splitLine[1])
            except ValueError:
                print("Invalid doShuffle. Defaulting to False.")
                doShuffle = False
    if width <= 0 or width <= max(orders):
        print(
            "invalid size for width. width must be positive integer greater than max order. Defaulting to 'max(orders)+3'")
        width = max(orders) + 3
    configer.close()
    return orders, width, cutoff, maxPan, stateBounds, pntBounds, nthreads, makePos, doShuffle


def parseInputFile(infile, numStates, stateBounds, makePos, doShuffle):
    # This function extracts state information from a file
    #
    # infile: the name of the file containing the data
    # numStates: the number of states in the file
    # stateBounds: the range of states to extract from the file
    # makePos: if True, the properties are made positive
    # doShuffle: if True, the energies are shuffled
    #
    # The file is assumed to contain a sequence of points for multiple states with energies and properties
    # The function reorders the data such that the energies and properties are continuous.
    # The file will be filled with rows corresponding to the reaction coordinate and then by state,
    # with the energy and features of each state printed along the columns
    #
    #     rc1 energy1 feature1.1 feature1.2 --->
    #     rc1 energy2 feature2.1 feature2.2 --->
    #                     |
    #                     V
    #     rc2 energy1 feature1.1 feature1.2 --->
    #     rc2 energy2 feature2.1 feature2.2 --->
    #
    # The function returns the energies, properties, and points

    if stateBounds is None:
        stateBounds = [0, numStates]
    inputMatrix = genfromtxt(infile)

    numPoints = int(inputMatrix.shape[0] / numStates)
    numColumns = inputMatrix.shape[1]
    print("Number of States:", numStates)
    print("Number of Points:", numPoints)
    print("Number of Features (w. E_h and r.c.):", numColumns)
    sys.stdout.flush()

    curves = inputMatrix.reshape((numPoints, numStates, numColumns))
    curves = np.swapaxes(curves, 0, 1)
    if doShuffle:
        shuffle_energy(curves)

    allPnts = curves[0, :, 0]
    Evals = curves[stateBounds[0]:stateBounds[1], :, 1]
    Nvals = curves[stateBounds[0]:stateBounds[1], :, 2:]
    if makePos:
        Nvals = abs(Nvals)

    return Evals, Nvals, allPnts


def main(infile="input.csv", outfile="output.csv", numStates=10, configPath=None):
    """
    This function takes a sequence of points for multiple states with energies and properties
    and reorders them such that the energies and properties are continuous

    Parameters
    ----------
    :param infile : str
        The name of the input file
    :param outfile : str
        The name of the output file
    :param numStates : int
        The number of states
    :param configPath : The path of the configuration file for setting stencil properties
    :param bounds : list of tuples, optional
        The bounds of the points
    :param stateBounds : list of tuples, optional
        The bounds of the states
    :param nthreads : int, optional
        The number of threads to use
    :param makePos : bool, optional
        Whether to make the properties positive
    :param doShuffle : bool, optional
        Whether to shuffle the points

    Returns None
    """

    orders, width, cutoff, maxPan, stateBounds, pntBounds, nthreads, makePos, doShuffle = applyConfig(configPath)
    #os.environ["NUMBA_NUM_THREADS"] = str(nthreads)
    numba.set_num_threads(1 if nthreads is None else nthreads)
    Evals, Nvals, allPnts = parseInputFile(infile, numStates, stateBounds, makePos, doShuffle)
    sortEnergies(Evals, Nvals)
    arrangeStates(Evals, Nvals, allPnts, configPath, maxiter=200, repeatMax=2, numStateRepeat=50)
    newCurvesList = []
    for pnt in range(Evals.shape[1]):
        newCurvesList.append(Evals[:, pnt])
    newCurves = stack(newCurvesList, axis=1)
    newCurves = insert(newCurves, 0, allPnts, axis=0)
    savetxt(outfile, newCurves, fmt='%20.12f')


if __name__ == "__main__":
    try:
        infile = sys.argv[1]
        outfile = sys.argv[2]
    except (ValueError, IndexError):
        raise ValueError("First two arguments must be the path for the input file with the data and the output file")

    try:
        numStates = int(sys.argv[3])
    except (ValueError, IndexError):
        raise ValueError("Third argument must specify the number of states in the input data")

    if len(sys.argv) > 4:
        configPath = sys.argv[4]
    else:
        configPath = None

    main(infile, outfile, numStates, configPath)
