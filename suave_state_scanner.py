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

@njit(parallel=True, fastmath=False, nogil=True, cache=True)
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

@njit(parallel=True, fastmath=False, nogil=True, cache=True)
def getMetric(dE, dP):
    """
    @brief This function computes the ordering metric for a given set of finite differences.
    @param dE: the finite differences of the energies at the center point (shape: (numStates))
    @param dP: the finite differences of the properties at the center point (shape: (numStates, numFeatures))
    @return: the ordering metric for the given finite differences
    """

    return (dE @ dP).sum()

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

@njit(parallel=True, fastmath=False, nogil=True, cache=True)
def mergediff(diff):
    """
    @brief This function will merge the finite differences for the energy and properties
    @param diff: The finite differences for the energy and properties
    @return: The merged finite differences
    """
    return np.absolute(diff.sum(axis=1))


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def buildValidArray(validArray, Evals, lobound, pnt, ref, upbound, eLow, eHigh, eWidth, hasEBounds, hasEWidth):
    """
    @brief This function will build the valid array for the current point
    @param validArray: The array of valid states
    @param Evals: The energy values for the current order
    @param lobound: The lower bound for the current point
    @param pnt: The current point
    @param ref: The reference point
    @param upbound: The upper bound for the current point
    @param eLow: The lower bound for the energy
    @param eHigh: The upper bound for the energy
    @param eWidth: The energy width for the current point
    @param hasEBounds: Whether the energy bounds are being used
    @param hasEWidth: Whether the energy width is being used
    @return: The valid array for the current point
    """

    # set all states at points that are not valid to False
    for state in prange(0, Evals.shape[0]):
        # set all states less than lower bound and greater than upper bound to False
        if state < lobound or state >= upbound:
            validArray[state] = False
        if hasEBounds:
            if not (eLow <= Evals[state, pnt] <= eHigh):
                validArray[state] = False
        if hasEWidth:
            if eWidth < abs(Evals[ref, pnt] - Evals[state, pnt]):
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
        self.stateMap = None # map of states to their indices

        ### Default configuration values
        self.printVar = 0 # index for energy or property to print
        self.maxiter = 1000 # maximum number of iterations
        self.tol = 1e-12 # tolerance for convergence of energy/properties
        self.orders = [1] # orders of derivatives to use
        self.width = 5 # stencil width
        self.futurePnts = 0 # number of future points to use
        self.maxPan = None # maximum pivots of stencil width
        self.stateBounds = None # bounds for states to use
        self.pntBounds = None # bounds for points to use
        self.propList = None # list of properties to use
        self.ignoreProps = False # flag for ignoring properties
        self.sweepBack = False # whether to sweep backwards across points after reordering forwards
        self.eBounds = None # bounds for energies to use
        self.eWidth = None # width for valid energies to swap with current state at a point
        self.interpolate = False # whether to interpolated energies during reordering
        self.keepInterp = False # whether to keep interpolated energies and properties
        self.nthreads = 1 # number of threads to use
        self.makePos = False # whether to make the properties positive
        self.doShuffle = False # whether to shuffle the points before reordering
        self.maxStateRepeat = -1 # maximum number of times to repeat swapping an unmodified state
        self.redundantSwaps = True # whether to allow redundant swaps of lower-lying states

        self.interactive = False # whether to interactive the reordering
        self.halt = False # whether to halt the reordering

        # Parse the configuration file
        self.applyConfig()

        # set the number of threads
        numba.set_num_threads(1 if self.nthreads is None else self.nthreads)

        # parse the input file to get the energies and properties for each state at each point
        self.E, self.P, self.allPnts, self.minh = self.parseInputFile()

        self.numPoints = len(self.allPnts)  # number of points
        self.numProps = self.P.shape[2]  # number of properties

        # set the bounds to use
        if self.pntBounds is None:
            self.pntBounds = [0, self.numPoints]
        if self.stateBounds is None:
            self.stateBounds = [0, self.numStates]
        if self.propList is None:
            self.propList = [i for i in range(self.numProps)]

        # check input for bounds. Throw error if bounds are invalid
        if self.pntBounds[0] >= self.pntBounds[1] - 1:
            raise ValueError("Invalid point bounds")
        if self.stateBounds[0] >= self.stateBounds[1] - 1:
            raise ValueError("Invalid state bounds")
        if self.eBounds is not None and self.eBounds[0] >= self.eBounds[1] - 1:
            raise ValueError("Invalid energy bounds")
        if self.eWidth is not None and self.eWidth <= 0:
            raise ValueError("Invalid energy width")

        # check if prop list is valid
        if len(self.propList) == 0:
            print("No properties to use; ignoring properties", flush=True)
        for prop in self.propList:
            if not (0 <= prop < self.numProps):
                raise ValueError("Invalid property index in propList:", prop)

        if self.doShuffle: # shuffle the points before reordering
            self.shuffle_energy()
        self.sortEnergies() # sort the states by energy of the first point

        # print out all the configuration values
        print("\n\n\tConfiguration Parameters:\n", flush=True)
        print("printVar", self.printVar, flush=True)
        print("interactive", self.interactive, flush=True)
        print("maxiter", self.maxiter, flush=True)
        print("orders", self.orders, flush=True)
        print("width", self.width, flush=True)
        print("maxPan", self.maxPan, flush=True)
        print("interpolate", self.interpolate, flush=True)
        print("futurePnts", self.futurePnts, flush=True)
        print("pntBounds", self.pntBounds, flush=True)
        print("stateBounds", self.stateBounds, flush=True)
        print("propList", self.propList, flush=True)
        print("eBounds", self.eBounds, flush=True)
        print("eWidth", self.eWidth, flush=True)
        print("sweepBack", self.sweepBack, flush=True)
        print("redundantSwaps", self.redundantSwaps, flush=True)
        print("maxStateRepeat", self.maxStateRepeat, flush=True)
        print("keepInterp", self.keepInterp, flush=True)
        print("makePos", self.makePos, flush=True)
        print("doShuffle", self.doShuffle, flush=True)
        print("\n ==> Using {} threads <==\n".format(self.nthreads), flush=True)
        print("", flush=True)

        # container to count unmodified states
        self.stateRepeatList = np.zeros(self.numStates)

        # check if any points are nan or inf
        self.hasMissing = np.isnan(self.E).any() or np.isinf(self.E).any() or np.isnan(
            self.P).any() or np.isinf(self.P).any()
        if self.hasMissing:
            print("\nWARNING: Energies or properties contain nan or inf values.")
            if self.interpolate:
                print("These will be ignored by interpolating over them during the optimization.")
            else:
                print("These will be ignored explicitly in the finite difference calculations.")
            if self.keepInterp:
                print("WARNING: final energies and properties will contain interpolated values.")
            else:
                print("Final results will not include these points.", flush=True)

        print("\n", flush=True)
        time.sleep(1)

    def generateDerivatives(self, center, F, backwards=False):
        """
        @brief This function approximates the n-th order derivatives of the energies and properties at a point.
        @param center: The index of the center point
        @param F: the energies and properties of the state to be reordered and each state above it across each point
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
                    sh[idx] = (self.allPnts[center + s[idx]] - self.allPnts[center]) / self.minh
                sh.flags.writeable = False

                # get finite difference coefficients from stencil points
                alpha = makeStencil(sh, order)

                # collect finite difference coefficients from this stencil
                h_n = self.minh ** order
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
            # interpolate derivatives at missing points (not interpolating energies or properties)
            diff = self.interpolateDerivatives(diff, sN)
        return mergediff(diff)


    def sortEnergies(self):
        """
        @brief This function sorts the energies and properties of the state such that the first point is in ascending energy order.
        """
        idx = self.E[:, 0].argsort()
        self.E[:] = self.E[idx]
        self.P[:] = self.P[idx]

    def prepareSweep(self):
        """
        @brief: This function prepares the states for a sweep by saving the current state of the energies and properties
                and creating a map of the states to their original order
        @return: lastEvals, lastPvals: the energies and properties of the states in their original order
        """

        # copy initial state info
        lastEvals = copy.deepcopy(self.E)
        lastPvals = copy.deepcopy(self.P)
        if self.interpolate:
            # save copy of E and P with interpolated missing values (if any)
            if self.hasMissing and self.keepInterp:
                self.interpMissing(interpKind="cubic")  # interpolate missing values with cubic spline
            self.sortEnergies()
            self.saveOrder()
        else:
            self.sortEnergies()  # only sort energies and properties when saving order
            self.saveOrder()
        # reset E and P to original order
        self.E = copy.deepcopy(lastEvals)
        self.P = copy.deepcopy(lastPvals)
        self.sortEnergies()  # sort energies and properties
        # create map of current state to last state for each point
        self.stateMap = np.zeros((self.numStates, self.numPoints), dtype=np.int32)
        for pnt in range(self.numPoints):
            self.stateMap[:, pnt] = np.arange(self.numStates)
        return lastEvals, lastPvals

    def analyzeSweep(self, lastEvals, lastPvals):
        """
        @brief: This function analyzes the results of a sweep to determine if the states have converged
        @param lastEvals:  the energies and properties of the states in their original order
        @param lastPvals:  the energies and properties of the states in their original order

        @return: delMax: the maximum change in the energies and properties of the states
        @return: E, P: the energies and properties of the states in their new order
        """
        # reset states to original order with missing values
        self.E = copy.deepcopy(lastEvals)
        self.P = copy.deepcopy(lastPvals)
        # use self.stateMap to reorder states
        for pnt in range(self.numPoints):
            statesToSwap = self.stateMap[:, pnt].tolist()
            self.E[:, pnt] = self.E[statesToSwap, pnt]
            self.P[:, pnt] = self.P[statesToSwap, pnt]
        # check if states have converged
        delEval = self.E - lastEvals
        delPval = self.P - lastPvals
        delEval = delEval[np.isfinite(delEval)]
        delPval = delPval[np.isfinite(delPval)]
        delMax = delEval.max() + delPval.max()
        return delMax

    def sweepState(self, state, sweep, backward=False):
        """
        This function sweeps through the states, performing the reordering.
        @param state: the state to be reordered
        @param sweep: current sweep number (used for printing)
        """

        if self.halt:
            print("Halting state sweeps")
            return
        # skip state if out of bounds
        if not (self.stateBounds[0] <= state <= self.stateBounds[1]):
            return

        # interpolate missing values (if any); only for reordering
        if self.hasMissing:
            if self.interpolate:
                self.interpMissing()  # use linear interpolation

        # save initial state info
        stateEvals = copy.deepcopy(self.E)
        statePvals = copy.deepcopy(self.P)

        # check if state has been modified too many times without improvement
        if self.stateRepeatList[state] >= self.maxStateRepeat > 0:
            self.stateRepeatList[state] += 1
            if self.stateRepeatList[state] < self.maxStateRepeat + 10:
                # if so, ignore it for 10 sweeps
                print("###", "Skipping", "###", flush=True)
                return
            else:
                # if state has been ignored for too long, test it again twice
                self.stateRepeatList[state] = abs(self.maxStateRepeat - 2)

        # reorder states across points for current state moving forwards
        modifiedStates = self.sweepPoints(state, sweep, backwards=backward)

        # check if state has been modified
        delMax = stateDifference(self.E, self.P, stateEvals, statePvals, state)
        if delMax < 1e-12:
            self.stateRepeatList[state] += 1
        else:
            self.stateRepeatList[state] = 0  # reset counter if state has been modified

        # check if any other states have been modified
        for modstate in modifiedStates:
            self.stateRepeatList[modstate] = 0  # reset counter for modified states


    def sweepPoints(self, state, sweep, backwards=False):
        """
        Sweep through points and sort states based on continuity of energy and amplitude norms
        :param state: current state
        :param sweep: current sweep
        :param backwards: if True, sweep backwards
        :return: List of states that were modified
        """

        if self.halt:
            print("Halting point sweeps")
            return
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

        # create bounds for states to be reordered
        lobound = self.stateBounds[0]  # lower bound for state
        upbound = self.stateBounds[1]  # upper bound for state

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

            swapStart = state + 1 # start swapping states at the next state
            if self.redundantSwaps: # if redundant swaps are allowed, start swapping states at the first state
                swapStart = 0

            validStates = self.findValidStates(lobound, pnt, state, upbound)

            if state not in validStates: # if current state is not valid, skip it
                continue

            if ~np.isfinite(self.E[state, pnt]): # if current state is missing, skip it
                continue

            # Bubble Sort algorithm to rearrange states
            while repeat <= repeatMax and itr < maxiter:

                # nth order finite difference
                # compare continuity differences from this state swapped with all other states
                dE = self.generateDerivatives(pnt, self.E[validStates], backwards=backwards)
                if not self.ignoreProps:
                    dP = self.generateDerivatives(pnt, self.P[:,:,self.propList][validStates], backwards=backwards)
                else:
                    dP = np.ones_like(self.P[validStates, :, 0])

                if dE.size == 0 or dP.size == 0:
                    continue

                diff = self.getMetric(dE, dP)
                minDif = (diff, state)

                # loop through all states to find the best swap
                for i in range(swapStart, self.numStates): # point is allowed to swap with states outside of bounds

                    if i not in validStates: # skip states that are not valid
                        continue
                    if i == state: # skip current state
                        continue
                    if ~np.isfinite(self.E[i, pnt]): # skip missing states
                        continue

                    # swap states
                    self.E[[state, i], pnt] = self.E[[i, state], pnt]
                    self.P[[state, i], pnt] = self.P[[i, state], pnt]

                    # nth order finite difference from swapped states
                    dE = self.generateDerivatives(pnt, self.E[validStates], backwards=backwards)
                    if not self.ignoreProps:
                        dP = self.generateDerivatives(pnt, self.P[:,:,self.propList][validStates],
                                                      backwards=backwards)
                    else:
                        dP = np.ones_like(self.P[validStates, :, 0])

                    # swap back
                    self.E[[state, i], pnt] = self.E[[i, state], pnt]
                    self.P[[state, i], pnt] = self.P[[i, state], pnt]

                    # if derivatives are missing, skip this state
                    if dE.size == 0 or dP.size == 0:
                        continue

                    # get metric of continuity difference
                    diff = self.getMetric(dE, dP)

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
                    self.E[[state, newState], pnt] = self.E[[newState, state], pnt]
                    self.P[[state, newState], pnt] = self.P[[newState, state], pnt]

                    # update self.stateMap
                    self.stateMap[[state, newState], pnt] = self.stateMap[[newState, state], pnt]

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


        hasEBounds = self.eBounds is not None # check if energy bounds are specified
        if hasEBounds:
            eLow = self.eBounds[0] # lower bound for energy
            eHigh = self.eBounds[1] # upper bound for energy
        else:
            eLow = -np.inf
            eHigh = np.inf
        hasEWidth = self.eWidth is not None  # check if energy width is provided
        if hasEWidth:
            eWidth = self.eWidth
        else:
            eWidth = np.inf

        fullRange = lobound == 0 and upbound == self.E.shape[0]  # check if the full range is being used

        if hasEBounds or hasEWidth or not fullRange:
            # set states outside of bounds or missing to be invalid
            validArray = buildValidArray(validArray, self.E, lobound, pnt, ref, upbound,
                                         eLow, eHigh, eWidth, hasEBounds, hasEWidth)
        else:
            pass # if no bounds or width are specified, all states are valid


        # convert validArray to list of valid states
        validStates = np.where(validArray)[0].tolist()
        validStates.sort() # sort states in ascending order

        return validStates # return sorted list of valid states at each point

    def interpMissing(self, interpKind = 'linear'):
        """Interpolate missing values in E and P"""
        print("\nInterpolating missing values...", end=" ", flush=True)

        # interpolate all missing values
        for i in range(self.numStates):
            for j in range(self.P.shape[2]): # loop over properties
                # get the indices of non-missing values
                idx = np.isfinite(self.P[i, :, j])
                # interpolate the missing values over points
                self.P[i, :, j] = interpolate.interp1d(self.allPnts[idx], self.P[i, idx, j], kind=interpKind, fill_value='extrapolate')(self.allPnts)

            # get the indices of non-missing values
            idx = np.isfinite(self.E[i, :])
            # interpolate the missing values over points
            self.E[i, :] = interpolate.interp1d(self.allPnts[idx], self.E[i, idx], kind=interpKind, fill_value='extrapolate')(self.allPnts)
        print("Done\n", flush=True)

    def interpolateDerivatives(self, diff, sN):
        """
        Interpolate the derivatives using the given stencil
        @param diff: derivatives
        @param sN: stencil size
        @return: interpolated derivatives
        """

        # interpolate the derivatives
        diff_shape = diff.shape
        flat_diff = diff.flatten()

        # get the indices of non-missing values
        idx = np.where(np.isfinite(flat_diff))[0]

        # interpolate the missing values over points
        flat_diff = interpolate.interp1d(idx, flat_diff[idx], kind='previous', fill_value='extrapolate')(np.arange(flat_diff.size))

        return flat_diff.reshape(diff_shape)
    
    def moveMissing(self):
        """Move missing values at each point to the last states"""
        print("\nMoving missing values to the end of the array...", end=" ", flush=True)
        for pnt in range(self.numPoints):
            # get the indices of missing values
            idx = np.where(~np.isfinite(self.E[:, pnt]))[0]
            
            count = 0 
            for i in idx: # loop over missing states
                # move missing values to the end of the array at each point and update stateMap
                end = self.numStates - 1 - count
                self.E[[i, end], pnt] = self.E[[end, i], pnt]
                self.P[[i, end], pnt] = self.P[[end, i], pnt]
                self.stateMap[[i, end], pnt] = self.stateMap[[end, i], pnt]
                count += 1

        print("Done\n", flush=True)
            
            
    

    # This function will randomize the state ordering for each point
    def shuffle_energy(self):
        """
        @brief This function will randomize the state ordering for each point, except the first point
        @return: The states with randomized energy ordering
        """

        for pnt in range(self.pntBounds[0], self.pntBounds[1]):
            # shuffle indices of state for each point
            idx = np.arange(self.numStates) # get indices of states

            #shuffle subset of indices from stateBounds[0] to stateBounds[1]
            np.random.shuffle(idx[self.stateBounds[0]:self.stateBounds[1]]) # shuffle subset of indices

            self.E[:, pnt] = self.E[idx, pnt] # shuffle energies
            self.P[:, pnt, :] = self.P[idx, pnt, :] # shuffle properties
            if self.stateMap is not None:
                self.stateMap[:, pnt] = self.stateMap[idx, pnt] # shuffle stateMap


    # this function loads the state information of a reorder scan from a previous run of this script
    def loadPrevRun(self, numStates, numPoints, numColumns):
        """
        @brief This function will load the state information of a reorder scan from a previous run of this script
        @param numStates: The number of states
        @param numPoints: The number of points
        @param numColumns: The number of columns in the file
        """

        Ecurves = genfromtxt('E.csv')
        Ncurves = genfromtxt('P.csv')

        self.allPnts = genfromtxt('allPnts.csv')
        self.E = Ecurves.reshape((numStates, numPoints))
        self.P = Ncurves.reshape((numStates, numPoints, numColumns))


    def saveOrder(self, isFinalResults = False):
        """
        @brief This function will save the state information of a reorder scan for a future run of this script
        @param isFinalResults: A boolean that determines if the final results are being saved

        @return: The energy and properties for each state at each point written to a file "tempInput.csv"
        """

        tempInput = zeros((self.P.shape[0] * self.P.shape[1], self.P.shape[2] + 2))
        combineVals(self.E, self.P, self.allPnts, tempInput)

        savetxt("tempInput.csv", tempInput, fmt='%20.12f')

        newCurvesList = []
        for pnt in range(self.allPnts.shape[0]):
            if self.printVar == 0:
                newCurvesList.append(self.E[:, pnt])
            elif self.printVar < 0:
                newCurvesList.append(self.P[:, pnt, self.printVar])
            else:
                newCurvesList.append(self.P[:, pnt, self.printVar - 1])

        newCurves = stack(newCurvesList, axis=1)
        newCurves = insert(newCurves, 0, self.allPnts, axis=0)
        savetxt('tempOutput.csv', newCurves, fmt='%20.12f')


        if isFinalResults: # save the final results
            # Create the output file
            newCurvesList = []
            for pnt in range(self.allPnts.shape[0]):
                if self.printVar == 0:
                    newCurvesList.append(self.E[:, pnt])
                elif self.printVar < 0:
                    newCurvesList.append(self.P[:, pnt, self.printVar])
                else:
                    newCurvesList.append(self.P[:, pnt, self.printVar - 1])
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
            if len(splitLine) != 2: # skip lines that don't have a parameter and a value
                continue
            if "#" in splitLine[1]: # remove comments from the value
                splitLine[1] = splitLine[1].split("#")[0]

            if "printVar" in splitLine[0]:
                try:
                    self.printVar = int(splitLine[1])
                except ValueError:
                    print("invalid index for variable to print. Defaulting to '0' for energy", flush=True)
                    self.printVar = 0
            if "interactive" in splitLine[0]:
                if splitLine[1] == "True" or splitLine[1] == "true":
                    self.interactive = True
                else:
                    self.interactive = False
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
                    self.pntBounds = [int(self.pntBounds[0]), int(self.pntBounds[1])]
                    if len(self.pntBounds) == 0:
                        print("The pntBounds provided is invalid. Defaulting to 'None'", flush=True)
                        self.pntBounds = None
                except ValueError:
                    if "None" not in splitLine[1] and "none" not in splitLine[1]:
                        print("The pntBounds provided is invalid. Defaulting to 'None'", flush=True)
                    self.pntBounds = None
            if "propList" in splitLine[0]:
                try:
                    self.propList = stringToList(splitLine[1])
                    self.propList = [int(prop) for prop in self.propList]
                except ValueError:
                    if "None" not in splitLine[1] and "none" not in splitLine[1]:
                        print("The propList provided is invalid. Defaulting to 'None'", flush=True)
                    self.propList = None
            if "sweepBack" in splitLine[0]:
                if "False" in splitLine[1] or "false" in splitLine[1]:
                    self.sweepBack = False
                else:
                    self.sweepBack = True
            if "stateBounds" in splitLine[0]:
                try:
                    self.stateBounds = stringToList(splitLine[1])
                    self.stateBounds = [int(self.stateBounds[0]), int(self.stateBounds[1])]
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
                    self.eBounds = [float(self.eBounds[0]), float(self.eBounds[1])]
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

        # find the smallest step size
        minh = np.inf
        for idx in range(numPoints - 1):
            h = allPnts[idx + 1] - allPnts[idx]
            if h < minh:
                minh = h

        return Evals, Pvals, allPnts, minh

    def run(self):
        """
        @brief: This function will take a sequence of points for multiple states with energies and properties and reorder them such that the energies and properties are continuous
        """
        startArrange = time.time() # start timer

        # arrange states such that energy and amplitude norms are continuous
        delMax = np.inf # initialize delMax to infinity
        converged = False # initialize convergence flag to false
        sweep = 0 # initialize sweep counter
        while not converged and sweep < self.maxiter: # iterate until convergence or max iterations reached
            sweep += 1

            startSweeptime = time.time() # start sweep timer

            lastEvals, lastPvals = self.prepareSweep() # save last energies and properties

            for state in range(self.numStates):
                if sweep % 2 == 1:
                    self.sweepState(state, sweep) # sweep state forwards
                else:
                    self.sweepState(state, sweep, self.sweepBack) # sweep state backwards

            endSweeptime = time.time() # end sweep timer

            delMax = self.analyzeSweep(lastEvals, lastPvals)

            print("CONVERGENCE PROGRESS: {:e}".format(delMax), flush=True)
            print("SWEEP TIME: {:e}".format(endSweeptime - startSweeptime), flush=True)
            if delMax < self.tol:
                converged = True

        if converged:
            print("%%%%%%%%%%%%%%%%%%%% CONVERGED {:e} %%%%%%%%%%%%%%%%%%%%%%".format(delMax), flush=True)
        else:
            print("!!!!!!!!!!!!!!!!!!!! FAILED TO CONVERRGE !!!!!!!!!!!!!!!!!!!!", flush=True)

        # create copy of E and P with interpolated missing values (if any)
        if self.interpolate:
            # save copy of E and P with interpolated missing values (if any)
            if self.hasMissing and self.keepInterp:
                self.interpMissing(interpKind="cubic") # interpolate missing values with cubic spline

        endArrange = time.time()
        print("\n\nTotal time to arrange states:", endArrange - startArrange, flush=True)

        self.sortEnergies()
        self.saveOrder(isFinalResults=True)
        print("Output file created:", self.outfile, flush=True)

    def run_interactive(self):
        """
        This creates an interactive session for the user to manipulate the states
        @requires: Plotly, dash
        """

        from dash import Dash, dcc, html, Input, Output, no_update
        from dash.exceptions import PreventUpdate
        import plotly.graph_objects as go

        app = Dash(__name__) # create dash app

        # get label of y-axis
        if self.printVar == 0:
            data0 = self.E
        elif self.printVar < 0:
            data0 = self.P[:, :, self.printVar]
        else:
            data0 = self.P[:, :, self.printVar - 1]

        minE = np.nanmin(self.E)
        maxE = np.nanmax(self.E)

        app.layout = html.Div([
            html.Div([ # create a div for the plot
            dcc.Graph(id='graph',
                      figure={
                          'data': [go.Scatter(x=self.allPnts, y=data0[state, :], mode='lines+markers',
                                              name="State {}".format(state)) for state in range(self.numStates)],
                          'layout': go.Layout(title="SuaveStateScanner", xaxis={'title': 'Reaction Coordinate'},
                                              hovermode='closest',
                                              uirevision='constant',
                                              # set dark theme
                                              paper_bgcolor='rgba(42,42,42,0.42)',
                                              plot_bgcolor='rgba(42,42,42,0.80)',
                                              font={'color': 'white'})

                      },
                      style={'height': 800},
                      config={'displayModeBar': False}
                      ),
            ]),
            html.Div([ # create a div for the buttons

                html.H2("Controls"),

                html.Div([  # Print Selection
                    html.Label("Print Selection"),
                    dcc.Dropdown(id="print-var", value=0,
                                 options=[{'label': "Energy" if i == 0 else "Property " + str(i), 'value': i}
                                          for i
                                          in range(self.numProps + 1)], clearable=False),
                    html.Div([  # Redraw
                        html.Button('Redraw', id='redraw', n_clicks=0, style={'padding': '2px'}),
                        html.Button('Sweep and Reorder', id='button', n_clicks=0, style={'padding': '2px'}),
                        html.Button('Abort', id='stop-button', n_clicks=0, style={'padding': '2px'}),
                    ], style={'display': 'inline-block'}),
                ], style={'display': 'inline-block', 'padding': '10px'}),
                html.Div([  # Save
                    html.Button('Save Output', id='save-button', n_clicks=0),
                ], style={'display': 'inline-block', 'width': '100%'}),
                html.Div([  # Undo Redo
                    html.Button('Undo', id='undo', n_clicks=0, style={'padding': '2px'}),
                    html.Button('Redo', id='redo', n_clicks=0, style={'padding': '2px'}),
                ], style={'display': 'inline-block', 'width': '100%'}),
            ], style={'width': '100%', 'display': 'inline-block', 'padding': '10px'}),

            html.Div([  # create a div for swapping two states by index with button
                html.Div([
                    html.Button('Swap States', id='swap-button', n_clicks=0),  # make a button to start the animation
                ], style={'display': 'inline-block', 'padding': '2px 10px 2px 2px', 'margin': 'auto'}),
                html.Div([
                    html.Label('State 1:'),  # make an input for state 1
                    dcc.Input(id='swap-input1', type='number', value=0, min=0, max=self.numStates - 1, step=1),
                ], style={'display': 'inline-block', 'padding': '2px 2px 2px 2px', 'margin': 'auto'}),
                html.Div([
                    html.Label('State 2:'),  # make an input for state 2
                    dcc.Input(id='swap-input2', type='number', value=1, min=0, max=self.numStates - 1, step=1)
                ], style={'display': 'inline-block', 'padding': '2px 2px 2px 2px', 'margin': 'auto'}),
                html.Div([  # create a div for shuffle
                    html.Button('Shuffle Values', id='shuffle-button', n_clicks=0),
                    # make a button to start the animation
                ], style={'display': 'inline-block', 'padding': '2px 2px 2px 50px', 'margin': 'auto'}),
            ], style={'display': 'inline-block', 'width': '100%'}),

            dcc.Loading(id="loading-save", children=[html.Div(id="loading-save-out")], type="default"),
            dcc.Loading(id="loading-reorder", children=[html.Div(id="loading-reorder-out")], type="default"),

            html.H2("Settings"),

            html.Div([  # Property List
                html.Label("Property List"),
                dcc.Checklist(id="prop-list",
                              options=[{'label': str(i + 1), 'value': i} for i in range(self.numProps)],
                              value=self.propList, labelStyle={'display': 'inline-block', 'padding': '10px'})
            ], style={'display': 'inline-block', 'padding': '10px'}),

            html.Div([ # make a slider to control the points to be reordered
                html.Div("Point Range", style={'display': 'inline-block', 'width': '30%', 'padding': '10px 10px 10px 10px'}),
                dcc.RangeSlider(id="point-slider", min=0, max=self.numPoints, step=1, value=self.pntBounds,
                                marks={i: "{}".format(self.allPnts[i]) for i in range(self.numPoints)},
                                allowCross=False, tooltip={'always_visible': True, 'placement': 'left'}),
                    ], style={'width': '95%', 'display': 'inline-block', 'padding': '5px 0px 5px 50px', 'margin': 'auto'}),

            html.Div([ # make a slider to control the states to be reordered
                html.Div("State Range", style={'display': 'inline-block', 'width': '30%', 'padding': '10px 10px 10px 10px'}),
                dcc.RangeSlider(id="state-slider", min=0, max=self.numStates, step=1, value=self.stateBounds,
                                marks={i: "{}".format(i) for i in range(self.numStates)},
                                allowCross=False, tooltip={'always_visible': True, 'placement': 'left'}),
            ], style={'width': '95%', 'display': 'inline-block', 'padding': '5px 0px 5px 50px', 'margin': 'auto'}),

            html.Div([  # make a slider to control the energy range
                html.Div("Energy Range", style={'display': 'inline-block', 'width': '30%', 'padding': '10px 10px 10px 10px'}),
                dcc.RangeSlider(id="energy-slider", min=minE - 1e-3, max=maxE + 1e-3, step=1e-6, value=[minE - 1e-3, maxE + 1e-3],
                                marks={minE: "Minimum Energy", maxE: "Maximum Energy"},
                                allowCross=False, tooltip={'always_visible': True, 'placement': 'left'}),
            ], style={'width': '95%', 'display': 'inline-block', 'padding': '5px 0px 5px 50px', 'margin': 'auto'}),

            html.Div([  # create a div for energy width
                html.Div("Energy Width", style={'display': 'inline-block', 'width': '30%', 'padding': '10px 10px 10px 10px'}),
                dcc.Slider(id="energy-width", value=abs(maxE - minE) + 1e-3, min=0, max=abs(maxE - minE) + 1e-3,
                            marks={1e-12: "Minimum Energy Width", abs(maxE - minE) + 1e-3: "Maximum Energy Width"},
                            step=1e-6, tooltip={'always_visible': True, 'placement': 'left'}),
            ], style={'width': '95%', 'display': 'inline-block', 'padding': '5px 0px 20px 50px', 'margin': 'auto'}),

            html.Div([ # make a div for all checklist options
                html.Div([  # create a div for checklist sweep backwards
                    dcc.Checklist(id='backSweep', options=[{'label': 'Sweep Backwards', 'value': 'sweepBack'}],
                                  value=False),
                ], style={'display': 'inline-block', 'width': '33%'}),

                html.Div([  # create a div for checklist interpolate missing values
                    dcc.Checklist(id='interpolate', options=[{'label': 'Interpolative Reorder', 'value': 'interpolate'}],
                                  value=False),
                ], style={'display': 'inline-block', 'width': '33%'}),

                html.Div([  # create a div for checklist redundant swaps
                    dcc.Checklist(id='redundant', options=[{'label': 'Redundant Swaps', 'value': 'redundant'}],
                                  value=True),
                ], style={'display': 'inline-block', 'width': '33%'}),


            ], style={'width': '30%', 'display': 'inline-block', 'padding': '0px 10px 0px 10px', 'margin': 'left'}),

            html.Div([  # make a div for all manual inputs
                html.Div([  # create a div for number of sweep to do
                    html.Label("Number of Sweeps:"),
                    dcc.Input(id='numSweeps', type='number', value=1, min=1, max=100),
                ], style={'display': 'inline-block', 'width': '25%'}),

                html.Div([  # create a div for stencil width
                    html.Label("Stencil Width:"),
                    dcc.Input(placeholder="Stencil Width", id="stencil-width", type="number", value=self.width, debounce=True),
                ], style={'display': 'inline-block', 'width': '25%'}),

                html.Div([  # create a div for maxPan
                    html.Label("Max Pan of Stencil:"),
                    dcc.Input(placeholder="Max Pan", id="max-pan", type="number", value=1000, debounce=True),
                ], style={'display': 'inline-block', 'width': '25%'}),

                html.Div([  # create a div for derivative order
                    html.Label("Derivative Order:"),
                    dcc.Input(placeholder="Derivative Order", id="order-value", type="number", value=self.orders[0],
                              debounce=True),
                ], style={'display': 'inline-block', 'width': '25%'}),
            ], style={'width': '65%', 'display': 'inline-block', 'padding': '0px 10px 0px 10px', 'margin': 'right'}),
        ], style={'width': '100%', 'padding': '10px 10px 10px 10px', 'margin': 'auto',
                    # set dark theme
                    'backgroundColor': '#111111', 'color': '#7FDBFF'})

        sweep = 0 # initialize sweep counter
        def make_figure():
            """
            This function creates the figure to be plotted
            @return: fig: the figure to be plotted
            """
            nonlocal sweep


            # update plot data
            lastEvals, lastPvals = self.prepareSweep()  # save last energies and properties
            if self.printVar == 0:
                data = self.E
            elif self.printVar < 0:
                data = self.P[:, :, self.printVar]
            else:
                data = self.P[:, :, self.printVar - 1]

            # create figure
            fig = go.Figure(
                data=[go.Scatter(x=self.allPnts, y=data[state, :], mode='lines+markers', name="State {}".format(state))
                      for state in range(self.numStates)],

                layout= go.Layout(title="SuaveStateScanner", xaxis={'title': 'Reaction Coordinate'},
                                    hovermode='closest',
                                    uirevision='constant',
                                    # set dark theme
                                    paper_bgcolor='rgba(42,42,42,0.42)',
                                    plot_bgcolor='rgba(42,42,42,0.80)',
                                    font={'color': 'white'})
            )


            self.E = copy.deepcopy(lastEvals)
            self.P = copy.deepcopy(lastPvals)
            return fig, f"Sweep {sweep}"

        eval_undo = [] # initialize history of energies for undo
        pval_undo = [] # initialize history of properties for undo
        eval_redo = [] # initialize memory of energies for redo
        pval_redo = [] # initialize memory of properties for redo

        def undo_callback():
            """
            This function undoes the last sweep
            @return: Boolean: True if successful, False otherwise
            """
            nonlocal eval_undo, pval_undo, eval_redo, pval_redo
            if len(eval_undo) < 1: # if there is nothing to undo
                return False

            if len(eval_redo) > 10: # if there are too many redos
                # remove the oldest redo
                eval_redo.pop(0)
                pval_redo.pop(0)

            # add the current state to the redo memory
            eval_redo.append(self.E)
            pval_redo.append(self.P)

            # set the current state to the previous state and remove the previous state from the undo memory
            self.E = copy.deepcopy(eval_undo.pop(-1))
            self.P = copy.deepcopy(pval_undo.pop(-1))

            return True

        def redo_callback():
            """
            This function redoes the last sweep
            @return: None
            """
            nonlocal eval_undo, pval_undo, eval_redo, pval_redo
            if len(eval_redo) < 1: # if there is nothing to redo
                return False # do nothing

            if len(eval_undo) > 10: # if there are too many undos
                # remove the oldest undo
                eval_undo.pop(0)
                pval_undo.pop(0)

            # add the current state to the undo memory
            eval_undo.append(self.E)
            pval_undo.append(self.P)

            # set the current state to the next state and remove the next state from the redo memory
            self.E = copy.deepcopy(eval_redo.pop(-1))
            self.P = copy.deepcopy(pval_redo.pop(-1))

            return True

        def store_update():
            """
            This function stores the current state of the plot
            @return: None
            """
            nonlocal eval_undo, pval_undo, eval_redo, pval_redo
            if len(eval_undo) > 10:
                eval_undo.pop(0)
                pval_undo.pop(0)
            eval_undo.append(copy.deepcopy(self.E))
            pval_undo.append(copy.deepcopy(self.P))
            eval_redo = []
            pval_redo = []


        last_sweep_click = 0 # initialize last sweep click
        last_shuffle_click = 0 # initialize last shuffle click
        last_redraw_click = 0 # initialize last redraw click
        last_swap_click = 0 # initialize last swap click
        last_stop_click = 0 # initialize last stop click
        last_undo_click = 0 # initialize last undo click
        last_redo_click = 0 # initialize last redo click
        callback_running = False # initialize callback running check
        @app.callback(
            [Output('graph', 'figure'), Output('loading-reorder-out', 'children')],
            [Input('button', 'n_clicks'), Input('point-slider', 'value'), Input('state-slider', 'value'),
             Input('print-var', 'value'), Input('stencil-width', 'value'),
             Input('order-value', 'value'), Input('shuffle-button', 'n_clicks'), Input('backSweep', 'value'),
             Input('interpolate', 'value'), Input('numSweeps', 'value'), Input('redraw', 'n_clicks'), Input('prop-list', 'value'),
             Input('max-pan', 'value'), Input('energy-width', 'value'), Input('redundant', 'value'), Input('energy-slider', 'value'),
             Input('swap-button', 'n_clicks'), Input('swap-input1', 'value'), Input('swap-input2', 'value'), Input('undo', 'n_clicks'),
             Input('stop-button', 'n_clicks'), Input('redo', 'n_clicks')])
        def update_graph(sweep_clicks, point_bounds, state_bounds, print_var, stencil_width, order, shuffle_clicks, backSweep,
                         interpolative, numSweeps, redraw_clicks, prop_list, maxPan, energyWidth, redundant, energy_bounds,
                         swap_clicks, swap1, swap2, undo_clicks, stop_clicks, redo_clicks):
            """
            This function updates the graph

            @param sweep_clicks:  the number of times the button has been clicked
            @param point_bounds:  the bounds of the points to be plotted
            @param state_bounds:  the bounds of the states to be plotted
            @param print_var:  the variable to be plotted
            @param stencil_width:  the width of the stencil to use
            @param order:  the order to use for each sweep
            @param shuffle_clicks:  the number of times the shuffle button has been clicked
            @param backSweep:  whether to sweep backwards
            @param interpolative:  whether to interpolate
            @param numSweeps:  the number of sweeps to do
            @param redraw_clicks:  the number of times the redraw button has been clicked
            @param prop_list: the properties to enforce continuity for
            @param maxPan:  the maximum number of points to pan in stencil
            @param energyWidth:  the width of the energy window
            @param redundant:  whether to use redundant swaps
            @param energy_bounds:  the bounds of the energies to be plotted
            @param swap_clicks:  the number of times the swap button has been clicked
            @param swap1:  the first state to swap
            @param swap2:  the second state to swap
            @param undo_clicks:  the number of times the undo button has been clicked
            @param stop_clicks:  the number of times the stop button has been clicked
            @param redo_clicks:  the number of times the redo button has been clicked
            @return: the figure to be plotted
            """
            nonlocal last_sweep_click, last_shuffle_click, last_redraw_click, last_swap_click
            nonlocal last_undo_click, last_stop_click, last_redo_click
            nonlocal eval_undo, pval_undo, eval_redo, pval_redo
            nonlocal sweep, callback_running

            if stop_clicks > last_stop_click: # if stop button has been clicked
                last_stop_click = stop_clicks # update last stop click
                self.halt = True # halt the scanner
                return no_update, no_update # return no update
            else:
                self.halt = False # otherwise, don't halt the scanner

            if callback_running: # if callback is running
                # update all click counters
                last_sweep_click = sweep_clicks
                last_shuffle_click = shuffle_clicks
                last_redraw_click = redraw_clicks
                last_swap_click = swap_clicks
                last_undo_click = undo_clicks
                last_stop_click = stop_clicks
                raise PreventUpdate # prevent update

            callback_running = True # set callback running to true

            # assign values to global variables
            self.pntBounds = point_bounds
            self.stateBounds = state_bounds
            self.propList = [int(prop) for prop in prop_list]
            self.ignoreProps = False
            self.energyBounds = energy_bounds
            self.printVar = int(print_var)
            self.width = int(stencil_width)
            self.orders = [int(order)] # only use one order for now
            self.interpolate = interpolative
            self.redundantSwaps = redundant
            self.maxPan = int(maxPan)
            self.energyWidth = float(energyWidth)

            # check input values
            if self.printVar > self.numProps: # if print variable is greater than number of properties
                self.printVar = 0 # set print variable to 0 (energy)
                print("Invalid print variable. Printing energy instead.", flush=True)

            if self.orders[0] >= self.numPoints: # if order is greater than number of points
                self.orders[0] = self.numPoints - 1 # set order to number of points - 1
                print("Order too large. Using minimum order instead.", flush=True)

            if self.width <= order: # if stencil width is less than or equal to order
                self.width = order + 1 # set stencil width to order + 1
                print("Stencil width too small. Using minimum stencil width instead.", flush=True)

            if self.width >= self.numPoints: # if stencil width is greater than or equal to number of points
                self.width = self.numPoints - 1 # set stencil width to number of points - 1
                print("Stencil width too large. Using minimum stencil width instead.", flush=True)

            if self.pntBounds[0] >= self.pntBounds[1] - 1: # if point bounds are invalid
                self.pntBounds[1] = self.pntBounds[0] + 1 # set point bounds to minimum
                print("Point bounds too small. Using minimum point bounds instead.", flush=True)
            if self.stateBounds[0] >= self.stateBounds[1] - 1: # if state bounds are invalid
                self.stateBounds[1] = self.stateBounds[0] + 1 # set state bounds to minimum
                print("State bounds too small. Using minimum state bounds instead.", flush=True)
            if abs(self.energyBounds[1] - self.energyBounds[0]) <= 1e-6: # if energy bounds are invalid
                self.energyBounds[1] = self.energyBounds[0] + 1e-6 # set energy bounds to minimum
                print("Energy bounds too small. Using minimum energy bounds instead.", flush=True)


            # check which button was clicked and update the graph accordingly
            if sweep_clicks > last_sweep_click and sweep_clicks > 0: # if sweep button was clicked
                # perform a sweep
                last_sweep_click = sweep_clicks  # update last sweep click
            elif redraw_clicks > last_redraw_click: # redraw button clicked
                last_redraw_click = redraw_clicks
                ret = make_figure()[0], "Redrawn"
                callback_running = False
                return ret
            elif undo_clicks > last_undo_click: # undo button clicked
                last_undo_click = undo_clicks

                if undo_callback(): # undo the last action
                    ret = make_figure()[0], "Undone" # update figure
                else:
                    ret = no_update, "Nothing to undo"
                callback_running = False
                return ret
            elif redo_clicks > last_redo_click:
                last_redo_click = redo_clicks # redo button clicked
                if redo_callback(): # redo the last action
                    ret = make_figure()[0], "Redone" # update figure
                else:
                    ret = no_update, "Nothing to redo"
                callback_running = False
                return ret
            elif shuffle_clicks > last_shuffle_click: # shuffle button clicked
                last_shuffle_click = shuffle_clicks # update last shuffle click

                store_update() # store current state
                self.shuffle_energy() # shuffle energy

                ret = make_figure()[0], "Shuffled" # update figure
                callback_running = False # set callback running to false
                return ret # return figure
            elif swap_clicks > last_swap_click: # swap button clicked
                last_swap_click = swap_clicks # update last swap click

                store_update() # store current state
                self.E[[swap1, swap2], point_bounds[0]:point_bounds[1]] = self.E[[swap2, swap1], point_bounds[0]:point_bounds[1]]
                self.P[[swap1, swap2], point_bounds[0]:point_bounds[1]] = self.P[[swap2, swap1], point_bounds[0]:point_bounds[1]]

                ret = make_figure()[0], "Swapped"
                callback_running = False
                return ret
            else: # otherwise, do nothing
                callback_running = False
                return no_update, no_update


            store_update() # store current state

            # perform a sweep
            lastEvals, lastPvals = self.prepareSweep()

            # skip first sweep
            for i in range(numSweeps):
                lastEvals, lastPvals = self.prepareSweep()
                sweep += 1
                for state in range(self.numStates):
                    self.sweepState(state, sweep, backward=backSweep)
                self.analyzeSweep(lastEvals, lastPvals)
            time.sleep(0.1)

            delMax = self.analyzeSweep(lastEvals, lastPvals)
            print("CONVERGENCE PROGRESS: {:e}".format(delMax), flush=True)

            if delMax < self.tol:
                print("%%%%%%%%%%%%%%%%%%%% CONVERGED {:e} %%%%%%%%%%%%%%%%%%%%%%".format(delMax), flush=True)
                self.sortEnergies()

            # update plot data
            last_sweep_click = sweep_clicks  # update last sweep click (should already be updated, but just in case)
            ret = make_figure()
            callback_running = False
            return ret

        last_save_clicks = 0
        @app.callback(
            Output('loading-save-out', 'children'),
            [Input('save-button', 'n_clicks')])
        def save_order(save_clicks):
            """
            This function saves the order
            @param save_clicks:  the number of times the button has been clicked
            @return: the bounds of the points to be plotted
            """
            nonlocal last_save_clicks, callback_running
            if callback_running:
                last_save_clicks = save_clicks
                raise PreventUpdate
            callback_running = True

            if save_clicks > last_save_clicks:
                lastEvals = copy.deepcopy(self.E)
                lastPvals = copy.deepcopy(self.P)

                if self.hasMissing and self.interpolate:
                    if self.keepInterp:
                        self.interpMissing(interpKind="cubic")
                self.saveOrder(isFinalResults=True)

                self.E = copy.deepcopy(lastEvals)
                self.P = copy.deepcopy(lastPvals)
                callback_running = False
                last_save_clicks = save_clicks
                return "Order saved"
            else:
                callback_running = False
                last_save_clicks = save_clicks
                return ""


        # run app without verbose output
        app.run_server(debug=False, use_reloader=False)



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

    if suave.interactive:
        suave.run_interactive()
    else:
        suave.run()
