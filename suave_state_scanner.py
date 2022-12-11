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

@njit(parallel=True, fastmath=True, nogil=True, cache=True)
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

@njit(parallel=True, fastmath=True, nogil=True, cache=True)
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

@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def mergediff(diff):
    """
    @brief This function will merge the finite differences for the energy and properties
    @param diff: The finite differences for the energy and properties
    @return: The merged finite differences
    """
    return np.absolute(diff.sum(axis=1))


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def buildValidArray(validArray, Evals, lobound, pnt, ref, upbound, eBounds, eWidth, hasMissing):
    """
    @brief This function will build the valid array for the current point
    @param validArray: The array of valid states
    @param Evals: The energy values for the current order
    @param lobound: The lower bound for the current point
    @param pnt: The current point
    @param ref: The reference point
    @param upbound: The upper bound for the current point
    @param eBounds: The energy bounds for the current point
    @param eWidth: The energy width for the current point
    @param hasMissing: The array of missing states
    @return: The valid array for the current point
    """
    # set all states at points that are not valid to False
    for state in prange(lobound, upbound):
        if eBounds is not None:
            if not (eBounds[0] <= Evals[state, pnt] <= eBounds[1]):
                validArray[state] = False
        if eWidth is not None:
            if eWidth < abs(Evals[ref, pnt] - Evals[state, pnt]):
                validArray[state] = False
        if hasMissing:  # only check for missing energies
            Eval_missing = not np.isfinite(Evals[state, pnt])
            if Eval_missing:
                validArray[state] = False
    return validArray

def stringToList(string):
    """
    this function will convert a string to a list of integers
    @return:
    """
    return string.replace(" ", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "").split(",")


class SuaveStateScanner:
    """
        @brief: This class takes a sequence of points for multiple states with their energies and properties
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
        """
    def __init__(self, infile, outfile, numStates, configPath=None):
        print("\nInput File:", infile, flush=True)
        print("Output File:", outfile, flush=True)
        print()
        sys.stdout.flush()

        # save constructor parameters
        self.infile = infile
        self.outfile = outfile
        self.numStates = numStates
        self.configPath = configPath

        # assign the metric function to a class variable (can be changed by the user)
        self.getMetric = getMetric

        self.hasMissing = False # flag for presence of missing data

        ### Default configuration values
        self.printVar = 0 # index for energy or property to print
        self.maxiter = 1000 # maximum number of iterations
        self.orders = [1] # orders of derivatives to use
        self.width = 5 # stencil width
        self.futurePnts = 0 # number of future points to use
        self.maxPan = None # maximum pivots of stencil width
        self.stateBounds = None # bounds for states to use
        self.pntBounds = None # bounds for points to use
        self.sweepBack = True # whether to sweep backwards across points after reordering forwards
        self.eBounds = None # bounds for energies to use
        self.eWidth = None # width for valid energies to swap with current state at a point
        self.interpolate = False # whether to keep interpolated energies and properties
        self.keepInterp = False # whether to keep interpolated energies and properties
        self.nthreads = 1 # number of threads to use
        self.makePos = False # whether to make the properties positive
        self.doShuffle = False # whether to shuffle the points before reordering
        self.maxStateRepeat = -1 # maximum number of times to repeat swapping an unmodified state
        self.redundantSwaps = False # whether to allow redundant swaps of lower-lying states


        # Parse the configuration file
        self.applyConfig()

        # set the number of threads
        numba.set_num_threads(1 if self.nthreads is None else self.nthreads)

        # parse the input file to get the energies and properties for each state at each point
        self.Evals, self.Pvals, self.allPnts = self.parseInputFile()

        if self.doShuffle: # shuffle the points before reordering
            self.shuffle_energy()
        self.sortEnergies() # sort the states by energy of the first point

        self.numPoints = len(self.allPnts) # number of points

        if self.pntBounds is None:
            self.pntBounds = [0, len(self.allPnts)]
        if self.stateBounds is None:
            self.stateBounds = [0, self.numStates]

        # print out all the configuration values
        print("\n\n\tConfiguration Parameters:\n", flush=True)
        print("printVar", self.printVar, flush=True)
        print("maxiter", self.maxiter, flush=True)
        print("orders", self.orders, flush=True)
        print("width", self.width, flush=True)
        print("maxPan", self.maxPan, flush=True)
        print("futurePnts", self.futurePnts, flush=True)
        print("pntBounds", self.pntBounds, flush=True)
        print("sweepBack", self.sweepBack, flush=True)
        print("stateBounds", self.stateBounds, flush=True)
        print("maxStateRepeat", self.maxStateRepeat, flush=True)
        print("eBounds", self.eBounds, flush=True)
        print("eWidth", self.eWidth, flush=True)
        print("interpolate", self.interpolate, flush=True)
        print("keepInterp", self.keepInterp, flush=True)
        print("nthreads", self.nthreads, flush=True)
        print("makePos", self.makePos, flush=True)
        print("doShuffle", self.doShuffle, flush=True)
        print("redundantSwaps", self.redundantSwaps, flush=True)
        print("", flush=True)

    def generateDerivatives(self, center, F, minh, backwards=False):
        """
        @brief This function approximates the n-th order derivatives of the energies and properties at a point.
        @param center: The index of the center point
        @param F: the energies and properties of the state to be reordered and each state above it across each point
        @param minh: the minimum step size to use in the finite difference approximationtion
        @param backwards: whether to approximate the derivatives backwards or forwards from the center
        @return: the n-th order finite differences of the energies and properties at the center point
        """
        bounds = self.pntBounds
        if self.orders is None:
            self.orders = [1]
        if self.width <= 1 or self.width > bounds[1]:
            self.width = bounds[1]
        if self.maxPan is None:
            self.maxPan = bounds[1]
        if self.futurePnts is None:
            self.futurePnts = 0

        combinedStencils = {}
        setDiff = False

        for order in self.orders:
            offCount = 0

            for off in range(-self.width, 1):
                if offCount >= self.maxPan:
                    continue
                # get size of stencil
                s = []
                for i in range(self.width):
                    idx = (i + off)
                    if idx > self.futurePnts:
                        break
                    if backwards:
                        idx = -idx
                    if bounds[0] <= center + idx < bounds[1]:
                        s.append(idx)
                sN = len(s)

                # ensure stencil is large enough for finite difference and not larger than data set
                if sN <= order or sN > bounds[1] or sN <= 1:
                    continue

                # ensure center point is included in stencil
                s = np.asarray(s)  # convert to np array
                if 0 not in s:
                    continue

                # scale stencil to potentially non-uniform mesh based off smallest point spacing
                sh = zeros(sN)
                for idx in range(sN):
                    sh[idx] = (self.allPnts[center + s[idx]] - self.allPnts[center]) / minh
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
            raise ValueError("No finite difference coefficients were computed. Try increasing the width of the stencil.")

        stencils = np.asarray(list(combinedStencils.items()), dtype=np.float32)
        stencil = stencils[:, 0].astype(int)
        alphas = stencils[:, 1]
        sN = len(combinedStencils)
        nStates = F.shape[0]
        if len(F.shape) == 3:
            num_props = F.shape[2]
            diff = zeros((nStates, sN, num_props))
        elif len(F.shape) == 2:
            diff = zeros((nStates, sN))
        else:
            raise "energies and/or features have incompatible dimensions"

        # compute combined finite differences from all stencils considered in panning window
        approxDeriv(F, diff, center, stencil, alphas, sN)

        if self.hasMissing and not self.interpolate:
            # set nan/inf values to mean value (not sure if this is the best way to handle this)
            maxVal = np.nanmax(diff)
            diff[np.isnan(diff) | np.isinf(diff)] = maxVal
        return mergediff(diff)


    def sortEnergies(self):
        """
        @brief This function sorts the energies and properties of the state such that the first point is in ascending energy order.
        """
        idx = self.Evals[:, 0].argsort()
        self.Evals[:] = self.Evals[idx]
        self.Pvals[:] = self.Pvals[idx]


    def run(self):
        """
        @brief: This function will take a sequence of points for multiple states with energies and properties and reorder them such that the energies and properties are continuous
        """
        startArrange = time.time() # start timer

        # find the smallest step size
        minh = np.inf
        for idx in range(self.numPoints - 1):
            h = self.allPnts[idx + 1] - self.allPnts[idx]
            if h < minh:
                minh = h

        # containers to count unmodified states
        stateRepeatList = np.zeros(self.numStates)

        # check if any points are nan or inf
        self.hasMissing = np.isnan(self.Evals).any() or np.isinf(self.Evals).any() or np.isnan(self.Pvals).any() or np.isinf(self.Pvals).any()
        if self.hasMissing:
            print("\nWARNING: Energies or properties contain nan or inf values.")
            if self.interpolate:
                print("These will be ignored by interpolating over them during the optimization.\n")
            else:
                print("These will be ignored explicitly in the finite difference calculations.\n")
            if self.keepInterp:
                print("WARNING: final energies and properties will contain interpolated values.\n")
            else:
                print("Final results will not include these points.", flush=True)

        print("\n\n", flush=True)
        time.sleep(1)

        # sort energies and properties
        self.sortEnergies()

        # arrange states such that energy and amplitude norms are continuous
        delMax = np.inf # initialize delMax to infinity
        converged = False # initialize convergence flag to false
        sweep = 0 # initialize sweep counter
        while not converged and sweep < self.maxiter: # iterate until convergence or max iterations reached
            sweep += 1

            startSweeptime = time.time()

            # copy initial state info
            lastEvals = copy.deepcopy(self.Evals)
            lastPvals = copy.deepcopy(self.Pvals)

            # create map of current state to last state for each point
            stateMap = np.zeros((self.numStates, self.numPoints), dtype=np.int32)
            for pnt in range(self.numPoints):
                stateMap[:, pnt] = np.arange(self.numStates)

            if self.interpolate:
                # save copy of Evals and Pvals with interpolated missing values (if any)
                if self.hasMissing and self.keepInterp:
                    self.interpMissing(interpKind="cubic") # interpolate missing values with cubic spline
                self.sortEnergies()
                self.saveOrder()
            else:
                self.sortEnergies() # only sort energies and properties when saving order
                self.saveOrder()

            # reset Evals and Pvals to original order
            self.Evals = copy.deepcopy(lastEvals)
            self.Pvals = copy.deepcopy(lastPvals)


            for state in range(self.numStates):
                # skip state if out of bounds
                if not (self.stateBounds[0] <= state <= self.stateBounds[1]):
                    continue

                # interpolate missing values (if any); only for reordering
                if self.interpolate:
                    if self.hasMissing:
                        self.interpMissing() # use linear interpolation

                # save initial state info
                stateEvals = copy.deepcopy(self.Evals)
                statePvals = copy.deepcopy(self.Pvals)

                # check if state has been modified too many times without improvement
                if stateRepeatList[state] >= self.maxStateRepeat > 0:
                    stateRepeatList[state] += 1
                    if stateRepeatList[state] < self.maxStateRepeat + 10:
                        # if so, ignore it for 10 sweeps
                        print("###", "Skipping", "###", flush=True)
                        continue
                    else:
                        # if state has been ignored for too long, test it again twice
                        stateRepeatList[state] = abs(self.maxStateRepeat - 2)

                # reorder states across points for current state moving forwards
                modifiedStates = self.sweepPoints(stateMap, minh, state, sweep)

                if self.sweepBack:
                    # reorder states across points for current state moving backwards
                    backModifiedStates = self.sweepPoints(stateMap, minh, state, sweep, backwards=True)

                    # merge modified states from forward and backward sweeps
                    for modstates in backModifiedStates:
                        if modstates not in modifiedStates:
                            modifiedStates.append(modstates)

                # check if state has been modified
                delMax = stateDifference(self.Evals, self.Pvals, stateEvals, statePvals, state)
                if delMax < 1e-12:
                    stateRepeatList[state] += 1
                else:
                    stateRepeatList[state] = 0 # reset counter if state has been modified

                # check if any other states have been modified
                for modstate in modifiedStates:
                    stateRepeatList[modstate] = 0 # reset counter for modified states

            endSweeptime = time.time()

            # reset states to original order with missing values
            self.Evals = copy.deepcopy(lastEvals)
            self.Pvals = copy.deepcopy(lastPvals)

            # use stateMap to reorder states
            for pnt in range(self.numPoints):
                statesToSwap = stateMap[:, pnt].tolist()
                self.Evals[:, pnt] = self.Evals[statesToSwap, pnt]
                self.Pvals[:, pnt] = self.Pvals[statesToSwap, pnt]

            # check if states have converged
            delEval = self.Evals - lastEvals
            delPval = self.Pvals - lastPvals

            delEval = delEval[np.isfinite(delEval)]
            delPval = delPval[np.isfinite(delPval)]

            delMax = delEval.max() + delPval.max()
            print("CONVERGENCE PROGRESS: {:e}".format(delMax), flush=True)
            print("SWEEP TIME: {:e}".format(endSweeptime - startSweeptime), flush=True)
            if delMax < 1e-12:
                converged = True

        if converged:
            print("%%%%%%%%%%%%%%%%%%%% CONVERGED {:e} %%%%%%%%%%%%%%%%%%%%%%".format(delMax), flush=True)
        else:
            print("!!!!!!!!!!!!!!!!!!!! FAILED TO CONVERRGE !!!!!!!!!!!!!!!!!!!!", flush=True)

        # create copy of Evals and Pvals with interpolated missing values (if any)
        if self.interpolate:
            # save copy of Evals and Pvals with interpolated missing values (if any)
            if self.hasMissing and self.keepInterp:
                self.interpMissing(interpKind="cubic") # interpolate missing values with cubic spline

        endArrange = time.time()
        print("\n\nTotal time to arrange states:", endArrange - startArrange, flush=True)

        self.sortEnergies()
        self.saveOrder(isFinalResults=True)
        print("Output file created:", self.outfile, flush=True)


    def sweepPoints(self, stateMap, minh, state, sweep, backwards=False):
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

        # reorder states across points for current state moving forwards or backwards in 'time'
        start = self.pntBounds[0]
        end = self.pntBounds[1]
        delta = 1

        if backwards:
            start = self.pntBounds[1] - 1
            end = self.pntBounds[0] - 1
            delta = -1

        # ensure that bounds include enough points to make a valid finite difference
        maxorder = max(self.orders)
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
            print("\n%%%%%%%%%%", "SWEEP " + direction + ":", str(sweep) + " / " + str(self.maxiter), "%%%%%%%%%%", flush=True)
            print("@@@@@@@", "STATE", str(state) + " / " + str(self.numStates), "@@@@@@@@@", flush=True)
            print("###", "POINT", str(pnt) + " / " + str(self.numPoints), "###", flush=True)

            repeat = 0
            repeatMax = 1
            maxiter = 500
            itr = 0
            lastDif = (inf, state)

            # set bounds for states to be reordered
            lobound = 0 # lower state bound
            upbound = self.numStates # upper state bound
            if self.stateBounds is not None:
                # if bounds are specified, use them
                lobound = self.stateBounds[0]
                upbound = self.stateBounds[1]

            swapStart = state + 1 # start swapping states at the next state
            if self.redundantSwaps: # if redundant swaps are allowed, start swapping states at the first state
                swapStart = 0

            validStates = self.findValidStates(lobound, pnt, state, upbound)

            if state not in validStates: # if current state is not valid, skip it
                continue

            if not np.isfinite(self.Evals[state, pnt]): # if current state is missing, skip it
                continue

            # Bubble Sort algorithm to rearrange states
            while repeat <= repeatMax and itr < maxiter:

                # nth order finite difference
                # compare continuity differences from this state swapped with all other states
                diffE = self.generateDerivatives(pnt, self.Evals[validStates], minh, backwards=backwards)
                diffP = self.generateDerivatives(pnt, self.Pvals[validStates], minh, backwards=backwards)

                if diffE.size == 0 or diffP.size == 0:
                    continue

                diff = self.getMetric(diffE, diffP)
                minDif = (diff, state)

                # loop through all states to find the best swap
                for i in range(swapStart, self.numStates): # point is allowed to swap with states outside of bounds

                    if i not in validStates: # skip states that are not valid
                        continue

                    if not np.isfinite(self.Evals[i, pnt]):  # if state is missing, skip it
                        # this shouldn't happen since validStates should not include missing states
                        continue

                    # swap states
                    self.Evals[[state, i], pnt] = self.Evals[[i, state], pnt]
                    self.Pvals[[state, i], pnt] = self.Pvals[[i, state], pnt]

                    # nth order finite difference from swapped states
                    diffE = self.generateDerivatives(pnt, self.Evals[validStates], minh, backwards=backwards)
                    diffP = self.generateDerivatives(pnt, self.Pvals[validStates], minh, backwards=backwards)

                    # swap back
                    self.Evals[[state, i], pnt] = self.Evals[[i, state], pnt]
                    self.Pvals[[state, i], pnt] = self.Pvals[[i, state], pnt]

                    # if derivatives are missing, skip this state
                    if diffE.size == 0 or diffP.size == 0:
                        continue

                    # get metric of continuity difference
                    diff = self.getMetric(diffE, diffP)

                    if diff < minDif[0]: # if this state is better than the current best, update the best
                        minDif = (diff, i)

                if lastDif[1] == minDif[1]:
                    repeat += 1
                else:
                    repeat = 0
                    print(state, "<---", minDif[1], flush=True)

                # swap state in point with new state that has the most continuous change in amplitude norms
                newState = minDif[1]
                if state != newState:
                    self.Evals[[state, newState], pnt] = self.Evals[[newState, state], pnt]
                    self.Pvals[[state, newState], pnt] = self.Pvals[[newState, state], pnt]

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

    def findValidStates(self, lobound, pnt, ref, upbound):
        """
        Find states that are valid for the current point
        @param lobound: lower bound for states
        @param pnt: current point
        @param ref: current state
        @param upbound: upper bound for states
        @return validStates : list of valid states
        """
        validArray = np.zeros(self.numStates, dtype=bool)
        validArray.fill(True)  # set all states to be valid
        validArray = buildValidArray(validArray, self.Evals, lobound, pnt, ref, upbound, self.eBounds, self.eWidth, self.hasMissing)

        # convert validArray to list of valid states
        validStates = np.where(validArray)[0].tolist()
        validStates.sort()

        return validStates # return sorted list of valid states at each point

    def interpMissing(self, interpKind = 'linear'):
        """Interpolate missing values in Evals and Pvals"""
        print("\nInterpolating missing values...", end=" ", flush=True)

        # interpolate all missing values
        for i in range(self.numStates):
            for j in range(self.Pvals.shape[2]): # loop over properties
                # get the indices of non-missing values
                idx = np.isfinite(self.Pvals[i, :, j])
                # interpolate the missing values over points
                self.Pvals[i, :, j] = interpolate.interp1d(self.allPnts[idx], self.Pvals[i, idx, j], kind=interpKind, fill_value='extrapolate')(self.allPnts)

            # get the indices of non-missing values
            idx = np.isfinite(self.Evals[i, :])
            # interpolate the missing values over points
            self.Evals[i, :] = interpolate.interp1d(self.allPnts[idx], self.Evals[i, idx], kind=interpKind, fill_value='extrapolate')(self.allPnts)
        print("Done", flush=True)


    # This function will randomize the state ordering for each point
    def shuffle_energy(self):
        """
        @brief This function will randomize the state ordering for each point, except the first point
        @return: The states with randomized energy ordering
        """

        for pnt in range(self.pntBounds[0], self.pntBounds[1]):
            # shuffle indices of state for each point
            idx = np.arange(self.numStates) # get indices of states
            np.random.shuffle(idx) # shuffle indices
            self.Evals[:, pnt] = self.Evals[idx, pnt] # shuffle energies
            self.Pvals[:, pnt, :] = self.Pvals[idx, pnt, :] # shuffle properties


    # this function loads the state information of a reorder scan from a previous run of this script
    def loadPrevRun(self, numStates, numPoints, numColumns):
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


    def saveOrder(self, isFinalResults = False):
        """
        @brief This function will save the state information of a reorder scan for a future run of this script
        @param Evals: The energy values for the current order
        @param Pvals: The properties for the current order
        @param allPnts: The points that are being evaluated
        @param printVar: The index that determines which state information is saved
        @param isFinalResults: A boolean that determines if the final results are being saved

        @return: The energy and properties for each state at each point written to a file "tempInput.csv"
        """

        tempInput = zeros((self.Pvals.shape[0] * self.Pvals.shape[1], self.Pvals.shape[2] + 2))
        combineVals(self.Evals, self.Pvals, self.allPnts, tempInput)

        savetxt("tempInput.csv", tempInput, fmt='%20.12f')

        newCurvesList = []
        for pnt in range(self.allPnts.shape[0]):
            if self.printVar == 0:
                newCurvesList.append(self.Evals[:, pnt])
            elif self.printVar < 0:
                newCurvesList.append(self.Pvals[:, pnt, self.printVar])
            else:
                newCurvesList.append(self.Pvals[:, pnt, self.printVar - 1])

        newCurves = stack(newCurvesList, axis=1)
        newCurves = insert(newCurves, 0, self.allPnts, axis=0)
        savetxt('tempOutput.csv', newCurves, fmt='%20.12f')


        if isFinalResults: # save the final results
            # Create the output file
            newCurvesList = []
            for pnt in range(self.allPnts.shape[0]):
                if self.printVar == 0:
                    newCurvesList.append(self.Evals[:, pnt])
                elif self.printVar < 0:
                    newCurvesList.append(self.Pvals[:, pnt, self.printVar])
                else:
                    newCurvesList.append(self.Pvals[:, pnt, self.printVar - 1])
            newCurves = stack(newCurvesList, axis=1)
            newCurves = insert(newCurves, 0, self.allPnts, axis=0)
            savetxt(self.outfile, newCurves, fmt='%20.12f')




    def applyConfig(self):
        """
        @brief This function will set parameters from the configuration file
        """
        if self.configPath is None: # no config file
            return # use default values

        configer = open(self.configPath, 'r')
        for line in configer.readlines():
            line = line.replace("\n", "").replace(" ", "").strip()
            if line == "": # skip empty lines
                continue
            if line[0] == "#": # skip comments
                continue

            splitLine = line.split("=") # split the line into the parameter and the value


            if "printVar" in splitLine[0]:
                try:
                    self.printVar = int(splitLine[1])
                except ValueError:
                    print("invalid index for variable to print. Defaulting to '0' for energy", flush=True)
                    self.printVar = 0
            if "maxiter" in splitLine[0]:
                try:
                    self.maxiter = int(splitLine[1])
                except ValueError:
                    print("invalid maxiter. Defaulting to '1000'", flush=True)
                    self.maxiter = 1000
            if "order" in splitLine[0]:
                try:
                    self.orders = stringToList(splitLine[1])
                    self.orders = [int(order) for order in self.orders]
                    if len(self.orders) == 0:
                        print("The orders of the derivatives desired for computation are required. Defaulting to '[1]'",
                              flush=True)
                        self.orders = [1]
                except ValueError:
                    print("The orders of the derivatives desired for computation are required. Defaulting to '[1]'",
                          flush=True)
                    self.orders = [1]
            if "width" in splitLine[0]:
                try:
                    self.width = int(splitLine[1])
                except ValueError:
                    print("invalid type for width. Defaulting to '8'", flush=True)
                    self.width = 8
            if "futurePnts" in splitLine[0]:
                try:
                    self.futurePnts = int(splitLine[1])
                except ValueError:
                    if "None" not in splitLine[1] and "none" not in splitLine[1]:
                        print("invalid type for futurePnts. Defaulting to '0'", flush=True)
                    self.futurePnts = 0
            if "maxPan" in splitLine[0]:
                try:
                    self.maxPan = int(splitLine[1])
                except ValueError:
                    if "None" not in splitLine[1] and "none" not in splitLine[1]:
                        print("invalid type for maxPan. Defaulting to 'None'", flush=True)
                    self.maxPan = None
            if "pntBounds" in splitLine[0]:
                try:
                    self.pntBounds = stringToList(splitLine[1])
                    self.pntBounds = [int(pntBound) for pntBound in self.pntBounds]
                    if len(self.pntBounds) == 0:
                        print("The pntBounds provided is invalid. Defaulting to 'None'", flush=True)
                        self.pntBounds = None
                except ValueError:
                    if "None" not in splitLine[1] and "none" not in splitLine[1]:
                        print("The pntBounds provided is invalid. Defaulting to 'None'", flush=True)
                    self.pntBounds = None
            if "sweepBack" in splitLine[0]:
                if "False" in splitLine[1] or "false" in splitLine[1]:
                    self.sweepBack = False
                else:
                    self.sweepBack = True
            if "stateBounds" in splitLine[0]:
                try:
                    self.stateBounds = stringToList(splitLine[1])
                    self.stateBounds = [int(stateBound) for stateBound in self.stateBounds]
                    if len(self.stateBounds) == 0:
                        print("The stateBounds provided is invalid. Defaulting to 'None'", flush=True)
                        self.stateBounds = None
                except ValueError:
                    if "None" not in splitLine[1] and "none" not in splitLine[1]:
                        print("The stateBounds provided is invalid. Defaulting to 'None'", flush=True)
                    self.stateBounds = None
            if "eBounds" in splitLine[0]:
                try:
                    self.eBounds = stringToList(splitLine[1])
                    self.eBounds = [float(eBound) for eBound in self.eBounds]
                    if len(self.eBounds) == 0:
                        print("The eBounds provided is invalid. Defaulting to 'None'", flush=True)
                        self.eBounds = None
                except ValueError:
                    if "None" not in splitLine[1] and "none" not in splitLine[1]:
                        print("The eBounds provided is invalid. Defaulting to 'None'", flush=True)
                    self.eBounds = None
            if "eWidth" in splitLine[0]:
                try:
                    self.eWidth = float(splitLine[1])
                    if self.eWidth <= 0:
                        print("The eWidth provided is invalid. Defaulting to 'None'", flush=True)
                        self.eWidth = None
                except ValueError:
                    if "None" not in splitLine[1] and "none" not in splitLine[1]:
                        print("invalid type for eWidth. Defaulting to 'None", flush=True)
                    self.eWidth = None
            if "interpolate" in splitLine[0]:
                if "True" in splitLine[1] or "true" in splitLine[1]:
                    self.interpolate = True
                else:
                    self.interpolate = False
            if "keepInterp" in splitLine[0]:
                if "True" in splitLine[1] or "true" in splitLine[1]:
                    self.keepInterp = True
                else:
                    self.keepInterp = False
            if "maxStateRepeat" in splitLine[0]:
                try:
                    self.maxStateRepeat = int(splitLine[1])
                except ValueError:
                    if "None" not in splitLine[1] and "none" not in splitLine[1]:
                        print("invalid type for maxStateRepeat. Defaulting to 'None'", flush=True)
                    self.maxStateRepeat = -1
            if "nthreads" in splitLine[0]:
                try:
                    self.nthreads = int(splitLine[1])
                except ValueError:
                    print("Invalid nthread size. Defaulting to 1.", flush=True)
                    self.nthreads = 1
            if "makePos" in splitLine[0]:
                if "True" in splitLine[1] or "true" in splitLine[1]:
                    self.makePos = True
                else:
                    self.makePos = False
            if "doShuffle" in splitLine[0]:
                if "True" in splitLine[1] or "true" in splitLine[1]:
                    self.doShuffle = True
                else:
                    self.doShuffle = False
            if "redundantSwaps" in splitLine[0]:
                if "True" in splitLine[1] or "true" in splitLine[1]:
                    self.redundantSwaps = True
                else:
                    self.redundantSwaps = False
        if self.width <= 0 or self.width <= max(self.orders):
            print(
                "invalid size for width. width must be positive integer greater than max order. Defaulting to 'max(orders)+3'")
            self.width = max(self.orders) + 3
        configer.close()


    def parseInputFile(self):
        """
        This function extracts state information from a file

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

        inputMatrix = genfromtxt(self.infile)

        numPoints = int(inputMatrix.shape[0] / self.numStates)
        numColumns = inputMatrix.shape[1]
        print("Number of States:", self.numStates, flush=True)
        print("Number of Points:", numPoints, flush=True)
        print("Number of Features (w. E_h and r.c.):", numColumns, flush=True)
        sys.stdout.flush()

        curves = inputMatrix.reshape((numPoints, self.numStates, numColumns))
        curves = np.swapaxes(curves, 0, 1)

        allPnts = curves[0, :, 0]
        Evals = curves[:, :, 1]
        Pvals = curves[:, :, 2:]

        if self.makePos:
            Pvals = abs(Pvals)
        if self.printVar > numColumns - 1:
            raise ValueError("Invalid printVar index. Must be less than the number of columns "
                             "in the input file (excluding the reaction coordinate).")

        return Evals, Pvals, allPnts


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

    suave = SuaveStateScanner(in_file, out_file, num_states, config_Path)
    suave.run()
