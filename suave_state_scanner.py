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

import copy
import sys
import time

import numba
import numpy as np
from numba import njit, prange, set_num_threads
from numpy import zeros, stack, insert, savetxt, inf, genfromtxt
import scipy.interpolate as interpolate

from nstencil import makeStencil


def generateDerivatives(N, center, F, allPnts, minh, orders, width, futurePnts, maxPan, backwards=False):
    """
    @brief This function approximates the n-th order derivatives of a the energies and properties at a point.
    @param N: the number of points in the state
    @param center: the index of the point to approximate the derivatives at
    @param F: the energies and properties of the state to be reordered and each state above it across each point
    @param allPnts: the reaction coordinate values of each point
    @param minh: the minimum step size to use in the finite difference approximation
    @param orders: the orders of the derivatives to approximate
    @param width: the maximum width of the stencil to use in the finite difference approximation
    @param futurePnts: the number of points to the right of the center to use in the finite difference approximation
    @param maxPan: the maximum number of pivots of the sliding windows for the stencil to use in the 
                   finite difference approximation
    
    @return: the n-th order finite differences of the energies and properties at the center point    
    """

    if orders is None:
        orders = [1]
    if width <= 1 or width > N:
        width = N
    if maxPan is None:
        maxPan = N
    if futurePnts is None:
        futurePnts = N

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
                if idx > futurePnts:
                    break
                if backwards:
                    idx = -idx
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
            alpha = makeStencil(sh, order)

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
    diffX = approxDeriv(F, diff, center, stencil, alphas, sN)
    return diffX


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def approxDeriv(F, diff, center, stencil, alphas, sN):
    """
    @brief This function approximates the n-th order derivatives of a the energies and properties
           at a point for a given stencil size.
    @param F: the energies and properties of the state to be reordered and each state above it across each point
    @param diff: the finite differences of the energies and properties at the center point
    @param center: the index of the point to approximate the derivatives at
    @param stencil: the stencil to use in the finite difference approximation
    @param alphas: the coefficients of the stencil to use in the finite difference approximation
    @param sN: the size of the stencil to use in the finite difference approximation

    @return: the n-th order finite differences of the energies and properties at the center point with the current stencil
    """

    # evaluate finite difference terms for each state
    for s in prange(sN):
        pnt = center + stencil[s]
        diff[:, s] = alphas[s] * F[:, pnt]

    # sum finite difference terms for each state for current stencil.
    # the absolute value of the finite differences of the energies and properties
    # should be minimized for the best ordering metric
    return np.absolute(diff.sum(axis=1))


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def getMetric(diffE, diffP):
    """
    @brief This function computes the ordering metric for a given set of finite differences.
    @param diffE: the finite differences of the energies at the center point
    @param diffP: the finite differences of the properties at the center point
    
    @return: the ordering metric for the given finite differences
    """

    # sum finite differences of all properties for each state
    diffP = np.sum(diffP, axis=1)

    # metric uses the sum of the product of the properties and energies for each state
    # to enforce the product rule of differentiation for the change in the ordering metric
    return (np.log(1 + diffE * diffP)).sum()


def sortEnergies(Evals, Pvals):
    """
    @brief This function sorts the energies and properties of the state such that the first point is in ascending energy order.
    @param Evals: the energies of the states to be reordered across each point
    @param Pvals: the properties of the states to be reordered across each point

    @return: the energies and properties of the states in ascending energy order of the first point
    """
    idx = Evals[:, 0].argsort()
    Evals[:] = Evals[idx]
    Pvals[:] = Pvals[idx]


def arrangeStates(Evals, Pvals, allPnts, configVars):
    """
    @brief: This function will take a sequence of points for multiple states with energies and properties and reorder them such that the energies and properties are continuous
    @param Evals: The energies for each state at each point
    @param Pvals: The properties for each state at each point
    @param allPnts: The sequence of points
    @param configVars: The configuration variables for the algorithm
    
    @return: The reordered energies and properties for each state at each point
    """

    # get dimensions of data
    numStates = Evals.shape[0]
    numPoints = Evals.shape[1]

    # find the smallest step size
    minh = np.inf
    for idx in range(numPoints - 1):
        h = allPnts[idx + 1] - allPnts[idx]
        if h < minh:
            minh = h

    # containers to count unmodified states
    stateRepeatList = np.zeros(numStates)

    # set parameters from configuration file
    printVar, orders, width, futurePnts, maxPan, \
    stateBounds, maxStateRepeat, pntBounds, sweepBack, eBounds, keepInterp, nthreads, \
    makePos, doShuffle = configVars

    print("\n\n\tConfiguration Parameters:\n", flush=True)
    print("printVar", printVar, flush=True)
    print("orders", orders, flush=True)
    print("width", width, flush=True)
    print("maxPan", maxPan, flush=True)
    print("futurePnts", futurePnts, flush=True)
    print("pntBounds", pntBounds, flush=True)
    print("sweepBack", sweepBack, flush=True)
    print("stateBounds", stateBounds, flush=True)
    print("maxStateRepeat", maxStateRepeat, flush=True)
    print("eBounds", eBounds, flush=True)
    print("keepInterp", keepInterp, flush=True)
    print("nthreads", nthreads, flush=True)
    print("makePos", makePos, flush=True)
    print("doShuffle", doShuffle, flush=True)

    # check if any points are nan or inf
    hasMissing = np.isnan(Evals).any() or np.isinf(Evals).any() or np.isnan(Pvals).any() or np.isinf(Pvals).any()
    if hasMissing:
        print("\nWARNING: Energies or properties contain nan or inf values.\n"
              "These will be ignored by interpolating over them during the optimization.\n"
              "Final results will not include these points.", flush=True)

    print("\n\n", flush=True)
    time.sleep(1)

    # arrange states such that energy and amplitude norms are continuous
    numSweeps = numPoints ** 2
    for sweep in range(numSweeps):
        startSweeptime = time.time()

        # copy initial state info
        lastEvals = copy.deepcopy(Evals)
        lastPvals = copy.deepcopy(Pvals)
        saveOrder(Evals, Pvals, allPnts, printVar)

        # create map of current state to last state for each point
        stateMap = np.zeros((numStates, numPoints), dtype=np.int32)
        for pnt in range(numPoints):
            stateMap[:, pnt] = np.arange(numStates)

        # create copy of Evals and Pvals with interpolated missing values (if any)
        if hasMissing:
            interpMissing(Evals, Pvals, allPnts, numStates)
            if keepInterp:
                saveOrder(Evals, Pvals, allPnts, printVar)

        for state in range(numStates):
            # skip state if out of bounds
            if not (stateBounds[0] <= state <= stateBounds[1]):
                continue

            # save initial state info
            stateEvals = copy.deepcopy(Evals)
            statePvals = copy.deepcopy(Pvals)



            # check if state has been modified too many times without improvement
            if stateRepeatList[state] >= maxStateRepeat > 0:
                stateRepeatList[state] += 1
                if stateRepeatList[state] < maxStateRepeat + 10:
                    # if so, ignore it for 10 sweeps
                    print("###", "Skipping", "###", flush=True)
                    continue
                else:
                    # if state has been ignored for too long, test it again twice
                    stateRepeatList[state] = abs(maxStateRepeat - 2)

            # reorder states across points for current state moving forwards
            modifiedStates = sweepPoints(Evals, Pvals, stateMap, allPnts, minh, state, sweep, numPoints, numStates, configVars)

            if sweepBack:
                # reorder states across points for current state moving backwards
                backModifiedStates = sweepPoints(Evals, Pvals, stateMap, allPnts, minh, state, sweep, numPoints, numStates, configVars, backwards=True)

                # merge modified states from forward and backward sweeps
                for modstates in backModifiedStates:
                    if modstates not in modifiedStates:
                        modifiedStates.append(modstates)

            # check if state has been modified
            delMax = stateDifference(Evals, Pvals, stateEvals, statePvals, state)
            if delMax < 1e-12:
                stateRepeatList[state] += 1
            else:
                stateRepeatList[state] = 0 # reset counter if state has been modified

            # check if any other states have been modified
            for modstate in modifiedStates:
                stateRepeatList[modstate] = 0 # reset counter for modified states

        endSweeptime = time.time()

        # reset states to original order with missing values
        Evals = copy.deepcopy(lastEvals)
        Pvals = copy.deepcopy(lastPvals)

        # use stateMap to reorder states
        for pnt in range(numPoints):
            Evals[:, pnt] = Evals[stateMap[:, pnt], pnt]
            Pvals[:, pnt] = Pvals[stateMap[:, pnt], pnt]

        # check if states have converged
        delEval = Evals - lastEvals
        delPval = Pvals - lastPvals

        delEval = delEval[np.isfinite(delEval)]
        delPval = delPval[np.isfinite(delPval)]

        delMax = delEval.max() + delPval.max()
        print("CONVERGENCE PROGRESS: {:e}".format(delMax), flush=True)
        print("SWEEP TIME: {:e}".format(endSweeptime - startSweeptime), flush=True)
        if delMax < 1e-12:
            print("%%%%%%%%%%%%%%%%%%%% CONVERGED {:e} %%%%%%%%%%%%%%%%%%%%%%".format(delMax), flush=True)
            if keepInterp:
                if hasMissing:
                    interpMissing(Evals, Pvals, allPnts, numStates, accurate=True)
            sortEnergies(Evals, Pvals)
            return Evals, Pvals

        sortEnergies(Evals, Pvals)

    print("!!!!!!!!!!!!!!!!!!!! FAILED TO CONVERRGE !!!!!!!!!!!!!!!!!!!!", flush=True)
    if keepInterp:
        if hasMissing:
            interpMissing(Evals, Pvals, allPnts, numStates, accurate=True)
    sortEnergies(Evals, Pvals)
    return Evals, Pvals


def sweepPoints(Evals, Pvals, stateMap, allPnts, minh, state, sweep, numPoints, numStates, configVars, backwards=False):
    """
    Sweep through points and sort states based on continuity of energy and amplitude norms
    :param Evals: array of energies
    :param Pvals: array of amplitude norms
    :param stateMap: map of current state to last state for each point
    :param allPnts: list of all points
    :param minh: minimum energy difference
    :param state: current state
    :param sweep: current sweep
    :param numPoints: number of points
    :param numStates: number of states
    :param configVars: list of configuration variables
    :param backwards: if True, sweep backwards
    :return: List of states that were modified
    """

    # set parameters from configuration file
    printVar, orders, width, futurePnts, maxPan, \
    stateBounds, maxStateRepeat, pntBounds, sweepBack, eBounds, keepInterp, nthreads, \
    makePos, doShuffle = configVars

    if pntBounds is None:
        pntBounds = [0, numPoints]

    # reorder states across points for current state moving forwards or backwards in 'time'
    start = pntBounds[0]
    end = pntBounds[1]
    delta = 1

    if backwards:
        start = pntBounds[1] - 1
        end = pntBounds[0] - 1
        delta = -1

    # ensure that bounds include enough points to make a valid finite difference
    maxorder = max(orders)
    if not backwards:
        start += maxorder
    else:
        start -= maxorder


    modifiedStates = []
    for pnt in range(start, end, delta):
        sys.stdout.flush()
        direction = "FORWARDS"
        if backwards:
            direction = "BACKWARDS"
        print("\n%%%%%%%%%%", "SWEEP " + direction + ":", sweep, "%%%%%%%%%%", flush=True)
        print("@@@@@@@", "STATE", state, "@@@@@@@@@", flush=True)
        print("###", "POINT", pnt, "###", flush=True)

        repeat = 0
        repeatMax = 1
        maxiter = 500
        itr = 0
        lastDif = (inf, state)

        # set bounds for states to be reordered
        lobound = state  # default is to only reorder states after the current state
        upbound = numStates
        if stateBounds is not None:
            # if bounds are specified, use them
            lobound = stateBounds[0]
            upbound = stateBounds[1]

        if eBounds is not None:
            # if energy bounds are specified, specify upper bound as the last state with energy below the upper energy bound
            for idx in range(state, numStates):
                if Evals[idx, pnt] > eBounds[1]:
                    if upbound > idx > lobound:
                        upbound = idx
                    break
            # if energy bounds are specified, specify lower bound as the first state with energy above the lower energy bound
            for idx in range(state, -1, -1):
                if Evals[idx, pnt] < eBounds[0]:
                    if lobound < idx < upbound:
                        lobound = idx
                    break

        # selection Sort algorithm to rearrange states
        while repeat <= repeatMax and itr < maxiter:

            # nth order finite difference
            diffE = generateDerivatives(numPoints, pnt, Evals[lobound:upbound], allPnts, minh, orders, width,
                                        futurePnts,
                                        maxPan, backwards=backwards)
            diffP = generateDerivatives(numPoints, pnt, Pvals[lobound:upbound], allPnts, minh, orders, width,
                                        futurePnts,
                                        maxPan, backwards=backwards)

            if diffE.size == 0 or diffP.size == 0:
                continue

            diff = getMetric(diffE, diffP)
            minDif = (diff, state)

            # compare continuity differences from this state swapped with all other states
            for i in range(state + 1, numStates):
                # point is allowed to swap with states outside of bounds

                Evals[[state, i], pnt] = Evals[[i, state], pnt]
                Pvals[[state, i], pnt] = Pvals[[i, state], pnt]

                # nth order finite difference
                diffE = generateDerivatives(numPoints, pnt, Evals[lobound:upbound], allPnts, minh, orders, width,
                                            futurePnts,
                                            maxPan, backwards=backwards)
                diffP = generateDerivatives(numPoints, pnt, Pvals[lobound:upbound], allPnts, minh, orders, width,
                                            futurePnts,
                                            maxPan, backwards=backwards)

                if diffE.size == 0 or diffP.size == 0:
                    continue

                diff = getMetric(diffE, diffP)

                if diff < minDif[0]:
                    minDif = (diff, i)

                Evals[[state, i], pnt] = Evals[[i, state], pnt]
                Pvals[[state, i], pnt] = Pvals[[i, state], pnt]

            if lastDif[1] == minDif[1]:
                repeat += 1
            else:
                repeat = 0
                print(state, "<---", minDif[1], flush=True)

            # swap state in point with new state that has the most continuous change in amplitude norms
            newState = minDif[1]
            if state != newState:
                Evals[[state, newState], pnt] = Evals[[newState, state], pnt]
                Pvals[[state, newState], pnt] = Pvals[[newState, state], pnt]

                # update stateMap
                stateMap[[state, newState], pnt] = stateMap[[newState, state], pnt]

                # update modifiedStates
                if newState not in modifiedStates:
                    modifiedStates.append(newState)

            lastDif = minDif
            itr += 1
        if itr >= maxiter:
            print("WARNING: state could not converge. Increase maxiter or repeatMax", flush=True)
        else:
            print(lastDif, flush=True)
        sys.stdout.flush()
    return modifiedStates


def interpMissing(Evals, Pvals, allPnts, numStates, accurate=False):
    """Interpolate missing values in Evals and Pvals"""
    print("Interpolating missing values", flush=True)

    interpKind = 'zero' # zero order interpolation (cause abrupt changes in amplitude, which makes it easier to find discontinuities)
    if accurate:
        interpKind = 'cubic' # cubic interpolation for more accurate results at print
    # interpolate all missing values
    for i in range(numStates):
        for j in range(Pvals.shape[2]): # loop over properties
            # get the indices of non-missing values
            idx = np.isfinite(Pvals[i, :, j])
            # interpolate the missing values over points
            Pvals[i, :, j] = interpolate.interp1d(allPnts[idx], Pvals[i, idx, j], kind=interpKind, fill_value='extrapolate')(allPnts)

        # get the indices of non-missing values
        idx = np.isfinite(Evals[i, :])
        # interpolate the missing values over points
        Evals[i, :] = interpolate.interp1d(allPnts[idx], Evals[i, idx], kind=interpKind, fill_value='extrapolate')(allPnts)


@njit(parallel=True, cache=True)
def stateDifference(Evals, Pvals, stateEvals, statePvals, state):
    """
    @brief This function will calculate the difference in the energy from a previous reordering to the current order
    @param Evals: The energy values for the current order
    @param Pvals: The properties for the current order
    @param stateEvals: The energy values for the previous order
    @param statePvals: The properties for the previous order
    @param state: The state that is being compared

    @return delMax: The maximum difference between the current and previous order
    """

    delEval = Evals[state] - stateEvals[state]
    delNval = Pvals[state] - statePvals[state]
    delMax = delEval.max() + delNval.max()
    return delMax


# This function will randomize the state ordering for each point
def shuffle_energy(curves):
    """
    @brief This function will randomize the state ordering for each point, except the first point
    @param curves: The energies and properties of each state at each point

    @return: The states with randomized energy ordering
    """

    for pnt in range(1, curves.shape[1]):
        np.random.shuffle(curves[:, pnt])


# this function loads the state information of a reorder scan from a previous run of this script
def loadPrevRun(numStates, numPoints, numColumns):
    """
    @brief This function will load the state information of a reorder scan from a previous run of this script
    @param numStates: The number of states
    @param numPoints: The number of points
    @param numColumns: The number of columns in the file
    """

    Ecurves = genfromtxt('Evals.csv')
    Ncurves = genfromtxt('Pvals.csv')
    allPnts = genfromtxt('allPnts.csv')

    Evals = Ecurves.reshape((numStates, numPoints))
    Pvals = Ncurves.reshape((numStates, numPoints, numColumns))

    return Evals, Pvals, allPnts


@njit(parallel=True)
def combineVals(Evals, Pvals, allPnts, tempInput):
    """
    @brief This function will reformat the state information for saving
    @param Evals: The energy values for the current order
    @param Pvals: The properties for the current order
    @param allPnts: The points that are being evaluated
    @param tempInput: The input file for the current run

    @return: The energy and properties for each state at each point
    """

    numPoints = allPnts.shape[0]
    numStates = Evals.shape[0]
    numFeat = Pvals.shape[2]
    for pnt in prange(numPoints):
        for state in prange(numStates):
            tempInput[pnt * numStates + state, 0] = allPnts[pnt]
            tempInput[pnt * numStates + state, 1] = Evals[state, pnt]
            for feat in prange(numFeat):
                tempInput[pnt * numStates + state, feat + 2] = Pvals[state, pnt, feat]


def saveOrder(Evals, Pvals, allPnts, printVar=0):
    """
    @brief This function will save the state information of a reorder scan for a future run of this script
    @param Evals: The energy values for the current order
    @param Pvals: The properties for the current order
    @param allPnts: The points that are being evaluated
    @param printVar: The index that determines which state information is saved

    @return: The energy and properties for each state at each point written to a file "tempInput.csv"
    """

    tempInput = zeros((Pvals.shape[0] * Pvals.shape[1], Pvals.shape[2] + 2))
    combineVals(Evals, Pvals, allPnts, tempInput)

    savetxt("tempInput.csv", tempInput, fmt='%20.12f')

    newCurvesList = []
    for pnt in range(allPnts.shape[0]):
        if printVar == 0:
            newCurvesList.append(Evals[:, pnt])
        elif printVar < 0:
            newCurvesList.append(Pvals[:, pnt, printVar])
        else:
            newCurvesList.append(Pvals[:, pnt, printVar - 1])

    newCurves = stack(newCurvesList, axis=1)
    newCurves = insert(newCurves, 0, allPnts, axis=0)
    savetxt('tempOutput.csv', newCurves, fmt='%20.12f')


# this function will convert a string to a list of integers
def stringToList(string):
    return string.replace(" ", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "").split(",")


def applyConfig(configPath=None):
    """
    @brief This function will set parameters from the configuration file
    @param configPath: The path to the configuration file

    @return: The parameters from the configuration file and default values
    """

    printVar = 0
    orders = [1]
    width = 5
    futurePnts = 0
    maxPan = None
    stateBounds = None
    pntBounds = None
    sweepBack = True
    eBounds = None
    keepInterp = False
    nthreads = 1
    makePos = False
    doShuffle = False
    maxStateRepeat = -1
    if configPath is None:
        return printVar, orders, width, futurePnts, maxPan, stateBounds, maxStateRepeat, pntBounds, sweepBack, eBounds, keepInterp, nthreads, makePos, doShuffle

    configer = open(configPath, 'r')
    for line in configer.readlines():
        line = line.replace("\n", "").replace(" ", "").strip()
        if line[0] == "#":
            continue
        splitLine = line.split("=")
        if "printVar" in line:
            try:
                printVar = int(splitLine[1])
            except ValueError:
                print("invalid index for variable to print. Defaulting to '0' for energy", flush=True)
                printVar = 0
        if "order" in line:
            try:
                orders = stringToList(splitLine[1])
                orders = [int(order) for order in orders]
                if len(orders) == 0:
                    print("The orders of the derivatives desired for computation are required. Defaulting to '[1]'",
                          flush=True)
                    orders = [1]
            except ValueError:
                print("The orders of the derivatives desired for computation are required. Defaulting to '[1]'",
                      flush=True)
                orders = [1]
        if "width" in line:
            try:
                width = int(splitLine[1])
            except ValueError:
                print("invalid type for width. Defaulting to '8'", flush=True)
                width = 8
        if "futurePnts" in line:
            try:
                futurePnts = int(splitLine[1])
            except ValueError:
                if "None" not in splitLine[1] and "none" not in splitLine[1]:
                    print("invalid type for futurePnts. Defaulting to '0'", flush=True)
                futurePnts = 0
        if "maxPan" in line:
            try:
                maxPan = int(splitLine[1])
            except ValueError:
                if "None" not in splitLine[1] and "none" not in splitLine[1]:
                    print("invalid type for maxPan. Defaulting to 'None'", flush=True)
                maxPan = None
        if "pntBounds" in line:
            try:
                pntBounds = stringToList(splitLine[1])
                pntBounds = [int(pntBound) for pntBound in pntBounds]
                if len(pntBounds) == 0:
                    print("The pntBounds provided is invalid. Defaulting to 'None'", flush=True)
                    pntBounds = None
            except ValueError:
                if "None" not in splitLine[1] and "none" not in splitLine[1]:
                    print("The pntBounds provided is invalid. Defaulting to 'None'", flush=True)
                pntBounds = None
        if "sweepBack" in line:
            if "False" in splitLine[1] or "false" in splitLine[1]:
                sweepBack = False
            else:
                sweepBack = True
        if "stateBounds" in line:
            try:
                stateBounds = stringToList(splitLine[1])
                stateBounds = [int(stateBound) for stateBound in stateBounds]
                if len(stateBounds) == 0:
                    print("The stateBounds provided is invalid. Defaulting to 'None'", flush=True)
                    stateBounds = None
            except ValueError:
                if "None" not in splitLine[1] and "none" not in splitLine[1]:
                    print("The stateBounds provided is invalid. Defaulting to 'None'", flush=True)
                stateBounds = None
        if "eBounds" in line:
            try:
                eBounds = stringToList(splitLine[1])
                eBounds = [float(eBound) for eBound in eBounds]
                if len(eBounds) == 0:
                    print("The eBounds provided is invalid. Defaulting to 'None'", flush=True)
                    eBounds = None
            except ValueError:
                if "None" not in splitLine[1] and "none" not in splitLine[1]:
                    print("The eBounds provided is invalid. Defaulting to 'None'", flush=True)
                eBounds = None
        if "keepInterp" in line:
            if "True" in splitLine[1] or "true" in splitLine[1]:
                keepInterp = True
            else:
                keepInterp = False
        if "maxStateRepeat" in line:
            try:
                maxStateRepeat = int(splitLine[1])
            except ValueError:
                if "None" not in splitLine[1] and "none" not in splitLine[1]:
                    print("invalid type for maxStateRepeat. Defaulting to 'None'", flush=True)
                maxStateRepeat = -1
        if "nthreads" in line:
            try:
                nthreads = int(splitLine[1])
            except ValueError:
                print("Invalid nthread size. Defaulting to 1.", flush=True)
                nthreads = 1
        if "makePos" in line:
            if "True" in splitLine[1] or "true" in splitLine[1]:
                makePos = True
            else:
                makePos = False
        if "doShuffle" in line:
            if "True" in splitLine[1] or "true" in splitLine[1]:
                doShuffle = True
            else:
                doShuffle = False
    if width <= 0 or width <= max(orders):
        print(
            "invalid size for width. width must be positive integer greater than max order. Defaulting to 'max(orders)+3'")
        width = max(orders) + 3
    configer.close()
    
    return printVar, orders, width, futurePnts, maxPan, stateBounds, maxStateRepeat, pntBounds, sweepBack, eBounds, keepInterp, nthreads, makePos, doShuffle


def parseInputFile(infile, numStates, stateBounds, makePos, doShuffle, printVar=0, eBounds=None):
    """
    This function extracts state information from a file

    infile: the name of the file containing the data
    numStates: the number of states in the file
    stateBounds: the range of states to extract from the file
    makePos: if True, the properties are made positive
    doShuffle: if True, the energies are shuffled

    The file is assumed to contain a sequence of points for multiple states with energies and properties
    The function reorders the data such that the energies and properties are continuous.
    The file will be filled with rows corresponding to the reaction coordinate and then by state,
    with the energy and features of each state printed along the columns

        rc1 energy1 feature1.1 feature1.2 --->
        rc1 energy2 feature2.1 feature2.2 --->
                        |
                        V
        rc2 energy1 feature1.1 feature1.2 --->
        rc2 energy2 feature2.1 feature2.2 --->

    The function returns the energies, properties, and points
    """

    # if stateBounds is None:
    #     stateBounds = [0, numStates]
    inputMatrix = genfromtxt(infile)

    numPoints = int(inputMatrix.shape[0] / numStates)
    numColumns = inputMatrix.shape[1]
    print("Number of States:", numStates, flush=True)
    print("Number of Points:", numPoints, flush=True)
    print("Number of Features (w. E_h and r.c.):", numColumns, flush=True)
    sys.stdout.flush()

    curves = inputMatrix.reshape((numPoints, numStates, numColumns))
    curves = np.swapaxes(curves, 0, 1)
    if doShuffle:
        shuffle_energy(curves)

    allPnts = curves[0, :, 0]
    Evals = curves[:, :, 1]
    Pvals = curves[:, :, 2:]

    if makePos:
        Pvals = abs(Pvals)
    if printVar > numColumns - 1:
        raise ValueError("Invalid printVar index. Must be less than the number of columns "
                         "in the input file (excluding the reaction coordinate).")

    return Evals, Pvals, allPnts


def main(infile, outfile, numStates, configPath=None):
    """
    @brief: This function takes a sequence of points for multiple states with energies and properties
    and reorders them such that the energies and properties are continuous

    Parameters
    ----------
    @param infile : str
        The name of the input file
    @param outfile : str
        The name of the output file
    @param numStates : int
        The number of states
    @param configPath : The path of the configuration file for setting stencil properties
    @param bounds : list of tuples, optional
        The bounds of the points
    @param stateBounds : list of tuples, optional
        The bounds of the states
    @param nthreads : int, optional
        The number of threads to use
    @param makePos : bool, optional
        Whether to make the properties positive
    @param doShuffle : bool, optional
        Whether to shuffle the points

    Returns None
    """

    print("\nInput File:", infile, flush=True)
    print("Output File:", outfile, flush=True)
    print()
    sys.stdout.flush()

    # Parse the configuration file
    configVars = applyConfig(configPath)

    printVar, orders, width, futurePnts, maxPan, \
    stateBounds, maxStateRepeat, pntBounds, sweepBack, eBounds, keepInterp, nthreads, \
    makePos, doShuffle = configVars


    # set the number of threads
    numba.set_num_threads(1 if nthreads is None else nthreads)

    # Parse the input file
    Evals, Pvals, allPnts = parseInputFile(infile, numStates, stateBounds, makePos, doShuffle, printVar, eBounds)
    sortEnergies(Evals, Pvals)

    # Calculate the stencils and reorder the data
    startArrange = time.time()
    Evals, Pvals = arrangeStates(Evals, Pvals, allPnts, configVars)

    endArrange = time.time()
    print("\n\nTotal time to arrange states:", endArrange - startArrange, flush=True)

    # Create the output file
    newCurvesList = []
    for pnt in range(allPnts.shape[0]):
        if printVar == 0:
            newCurvesList.append(Evals[:, pnt])
        elif printVar < 0:
            newCurvesList.append(Pvals[:, pnt, printVar])
        else:
            newCurvesList.append(Pvals[:, pnt, printVar - 1])
    newCurves = stack(newCurvesList, axis=1)
    newCurves = insert(newCurves, 0, allPnts, axis=0)
    savetxt(outfile, newCurves, fmt='%20.12f')
    print("Output file created:", outfile, flush=True)


if __name__ == "__main__":
    try:
        in_file = sys.argv[1]
    except (ValueError, IndexError):
        raise ValueError("First argument must be the input file")

    try:
        out_file = sys.argv[2]
    except (ValueError, IndexError):
        raise ValueError("Second argument must be the output file")
    try:
        num_states = int(sys.argv[3])
    except (ValueError, IndexError):
        raise ValueError("Third argument must specify the number of states in the input data")

    if len(sys.argv) > 4:
        config_Path = sys.argv[4]
    else:
        config_Path = None
        print("No configuration file specified. Using default values.", flush=True)

    main(in_file, out_file, num_states, config_Path)
