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
import json
import argparse
import sys
import time

import numba
import numpy as np
from numba import njit, prange
from numpy import zeros, stack, insert, savetxt, inf, genfromtxt
import scipy.interpolate as interpolate

from nstencil import makeStencil
from suave_metric import getMetric
from gui_client import startClient

@njit(parallel=True, fastmath=False, nogil=True, cache=True)
def approxDeriv(F, diff, center, stencil, alphas, sN):
    """
    This function approximates the n-th order derivatives of the energies and properties
           at a point for a given stencil size.
    :param F: the energies and properties of the state to be reordered and each state above it across each point
    :param diff: the finite differences of the energies and properties at the center point
    :param center: the index of the point to approximate the derivatives at
    :param stencil: the stencil to use in the finite difference approximation
    :param alphas: the coefficients of the stencil to use in the finite difference approximation
    :param sN: the size of the stencil to use in the finite difference approximation
    :return the n-th order finite differences of the energies and properties at the center point with the current stencil
    """

    # evaluate finite difference terms for each state
    for s in prange(sN):
        pnt = center + stencil[s]
        diff[:, s] = alphas[s] * F[:, pnt]

@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def stateDifference(Evals, Pvals, stateEvals, statePvals, state):
    """
    This function will calculate the difference in the energy from a previous reordering to the current order
    :param Evals: The energy values for the current order
    :param Pvals: The properties for the current order
    :param stateEvals: The energy values for the previous order
    :param statePvals: The properties for the previous order
    :param state: The state that is being compared

    :return delMax: The maximum difference between the current and previous order
    """

    delEval = Evals[state] - stateEvals[state]
    delNval = Pvals[state] - statePvals[state]
    delMax = delEval.max() + delNval.max()
    return delMax

@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def combineVals(Evals, Pvals, allPnts, tempInput):
    """
    This function will reformat the state information for saving
    :param Evals: The energy values for the current order
    :param Pvals: The properties for the current order
    :param allPnts: The points that are being evaluated
    :param tempInput: The input file for the current run
    :return The energy and properties for each state at each point
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

def initialize_diff(F, nStates, sN):
    """
    Initialize the array to store the finite differences
    :param F: the energies or properties of the state to be reordered and each state above it across each point
    :param nStates: The number of states
    :param sN: The number of stencil points
    :return:
    """
    if len(F.shape) == 3:
        num_props = F.shape[2]
        diff = zeros((nStates, sN, num_props))
    elif len(F.shape) == 2:
        diff = zeros((nStates, sN))
    else:
        raise ValueError("energies and/or features have incompatible dimensions")
    return diff

def mergediff(diff):
    """
    This function will merge the finite differences for the energy and properties
    :param diff: The finite differences for the energy and properties
    :return The merged finite differences
    """
    return np.absolute(diff.sum(axis=1))


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def buildValidArray(validArray, Evals, lobound, pnt, ref, upbound, eLow, eHigh, eWidth, hasEBounds, hasEWidth):
    """
    This function will build the valid array for the current point
    :param validArray: The array of valid states
    :param Evals: The energy values for the current order
    :param lobound: The lower bound for the current point
    :param pnt: The current point
    :param ref: The reference point
    :param upbound: The upper bound for the current point
    :param eLow: The lower bound for the energy
    :param eHigh: The upper bound for the energy
    :param eWidth: The energy width for the current point
    :param hasEBounds: Whether the energy bounds are being used
    :param hasEWidth: Whether the energy width is being used
    :return The valid array for the current point
    """

    # set all states at points that are not valid to False
    for state in prange(lobound, upbound):
        validArray[state] = True
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
    :return
    """
    return string.replace(" ", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "").split(",")


def interpolateDerivatives(diff):
    """
    Interpolate the derivatives using the given stencil
    :param diff: derivatives
    :return interpolated derivatives
    """

    # interpolate the derivatives
    diff_shape = diff.shape
    flat_diff = diff.flatten()

    # get the indices of non-missing values
    idx = np.where(np.isfinite(flat_diff))[0]

    # interpolate the missing values over points
    flat_diff = interpolate.interp1d(idx, flat_diff[idx], kind='previous', fill_value='extrapolate')(np.arange(flat_diff.size))

    return flat_diff.reshape(diff_shape)


class SuaveStateScanner:
    """
        This class takes a sequence of points for multiple states with their energies and properties
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
        self.sortLast = False # whether to sort by the last point (default is to sort by the first point)
        self.keepInterp = False # whether to keep interpolated energies and properties
        self.nthreads = 1 # number of threads to use
        self.makePos = False # whether to make the properties positive
        self.normalize = False # whether to normalize the properties
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
        print("normalize", self.normalize, flush=True)
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

        # initialize parameters for the finite differences
        self.center = None
        self.validPnts = None
        self.backwards = None
        self.bounds = None
        self.setDiff = None
        self.combinedStencils = None


    def initialize_variables(self, center, validPnts, backwards):
        """
        Initialize variables for the finite difference calculation
        :param center: center point
        :param validPnts: valid points
        :param backwards: whether to approximate the derivatives backwards or forwards from the center
        """

        self.center = center
        self.validPnts = validPnts
        self.backwards = backwards
        self.bounds = self.pntBounds
        self.setDiff = False
        self.combinedStencils = {}

        if self.orders is None:
            self.orders = [1]
        if self.width <= 1 or self.width > self.bounds[1]:
            self.width = self.bounds[1]
        if self.maxPan is None:
            self.maxPan = self.bounds[1]
        if self.futurePnts is None:
            self.futurePnts = 0

    def generateDerivatives(self, center, validPnts, F, backwards=False):
        """
        This function approximates the n-th order derivatives of the energies and properties at a point.
        :param center: The index of the center point
        :param validPnts: The indices of the points that are valid for the stencil
        :param F: the energies and properties of the state to be reordered and each state above it across each point
        :param backwards: whether to approximate the derivatives backwards or forwards from the center
        :return the n-th order finite differences of the energies and properties at the center point
        """
        self.initialize_variables(center, validPnts, backwards)
        self.process_orders()

        if not self.setDiff:
            raise ValueError(
                "No finite difference coefficients were computed. Try increasing the width of the stencil.")

        stencils = np.asarray(list(self.combinedStencils.items()), dtype=np.float32) # convert to numpy array
        stencil = stencils[:, 0].astype(int) # stencil sizes
        alphas = stencils[:, 1] # finite difference coefficients
        sN = len(self.combinedStencils) # number of stencils
        nStates = F.shape[0] # number of states
        diff = initialize_diff(F, nStates, sN) # initialize the derivatives

        # compute combined finite differences from all stencils considered in panning window
        approxDeriv(F, diff, center, stencil, alphas, sN)

        if self.hasMissing and not self.interpolate:
            # interpolate derivatives at missing points (not interpolating energies or properties)
            diff = interpolateDerivatives(diff)

        return mergediff(diff) # return the sum of finite derivatives for each offset

    def calculate_stencil_size(self, off):
        """
        Calculate the stencil size for a given offset.
        :param off: offset
        :return: stencil size
        """
        s = [] # stencil
        for i in range(self.width):
            idx = (i + off) # index at offset
            if idx > self.futurePnts:
                break # stop if we are past points that will be swapped in the future
            if self.backwards:
                idx = -idx # if we are going backwards, make the index negative

            # index is valid if it is within the bounds and is a valid point
            if self.bounds[0] <= self.center + idx < self.bounds[1] \
                    and self.center + idx in self.validPnts:
                s.append(idx) # add the index to the stencil if it is valid
        return s # return the stencil

    def scale_stencil(self, s):
        """
        Scale the stencil points to match the minimum spacing between points
        :param s: stencil points
        :return: scaled stencil points
        """
        sN = len(s)
        sh = zeros(sN)
        for idx in range(sN): # scale stencil points to match the minimum spacing between points
            sh[idx] = (self.allPnts[self.center + s[idx]] - self.allPnts[self.center]) / self.minh
        sh.flags.writeable = False # make stencil points immutable
        return sh

    def compute_finite_difference_coefficients(self, s, order):
        """
        Compute the finite difference coefficients for a given stencil and order
        :param s: stencil points
        :param order: order of the finite difference
        """
        # get finite difference coefficients from stencil points
        sh = self.scale_stencil(s)
        alpha = makeStencil(sh, order)

        # collect finite difference coefficients from this stencil
        h_n = self.minh ** order # h^n
        for idx in range(len(s)):
            try:
                self.combinedStencils[s[idx]] += alpha[idx] / h_n # add to existing stencil if it exists
            except KeyError:
                self.combinedStencils[s[idx]] = alpha[idx] / h_n # create new stencil if it does not exist

    def process_orders(self):
        """
        Process the various orders to use as finite difference coefficients
        :return: True if any finite difference coefficients were computed, False otherwise
        """
        self.setDiff = False
        for order in self.orders:
            offCount = 0
            for off in range(-self.width, 1):
                if offCount >= self.maxPan:
                    continue

                s = self.calculate_stencil_size(off)
                sN = len(s)

                if sN <= order or sN > self.bounds[1] or sN <= 1:
                    continue # stencil is too small or too large

                s = np.asarray(s)  # convert to np array

                if 0 not in s:
                    continue # stencil does not contain the center point

                self.compute_finite_difference_coefficients(s, order) # compute finite difference coefficients

                if not self.setDiff:
                    self.setDiff = True # set flag to indicate that finite difference coefficients were computed

                offCount += 1

    def sortEnergies(self):
        """
        This function sorts the energies and properties of the state such that the first point is in ascending energy order.
        """
        sort_pos = self.pntBounds[0] if not self.sortLast else self.pntBounds[1]-1
        idx = self.E[:, sort_pos].argsort()
        self.E[:] = self.E[idx]
        self.P[:] = self.P[idx]

    def prepareSweep(self):
        """
        This function prepares the states for a sweep by saving the current state of the energies and properties
                and creating a map of the states to their original order
        :return lastEvals, lastPvals: the energies and properties of the states in their original order
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
        This function analyzes the results of a sweep to determine if the states have converged
        :param lastEvals:  the energies and properties of the states in their original order
        :param lastPvals:  the energies and properties of the states in their original order

        :return delMax: the maximum change in the energies and properties of the states
        :return E, P: the energies and properties of the states in their new order
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
        :param state: the state to be reordered
        :param sweep: current sweep number (used for printing)
        :param backward: whether to sweep backwards or forwards through the points (default: False)
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
        :return List of states that were modified
        """

        if self.halt:
            print("Halting point sweeps")
            return

        # create bounds for states to be reordered
        lobound = self.stateBounds[0]  # lower bound for state
        upbound = self.stateBounds[1]  # upper bound for state

        if not (lobound <= state < upbound):
            return []  # skip state if out of bounds

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
        validPnts = [i for i in range(start, end, delta) if 0 <= i < self.numPoints]

        modifiedStates = []
        futurePnts_copy = self.futurePnts
        for pnt in range(start, end, delta):

            set_off = abs(pnt - start) <= maxorder
            if set_off:
                new_off = maxorder - abs(pnt - start)
                self.futurePnts = new_off if self.futurePnts <= new_off else self.futurePnts

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
                swapStart = lobound

            # find valid states to compare with current state at current point
            validStates = self.getValidStates(pnt, lobound, upbound, state)

            if ~np.isfinite(self.E[state, pnt]): # if current state is missing, skip it
                continue

            # Bubble Sort algorithm to rearrange states
            while repeat <= repeatMax and itr < maxiter:

                # nth order finite difference
                # compare continuity differences from this state swapped with all other states
                dE = self.generateDerivatives(pnt, validPnts, self.E[validStates], backwards=backwards)

                pTest = self.P[validStates][:,:,self.propList]
                if self.makePos:
                    pTest = np.abs(pTest)
                if self.normalize:
                    pTest = pTest / np.sum(pTest)

                if not self.ignoreProps:
                    dP = self.generateDerivatives(pnt, validPnts, pTest, backwards=backwards)
                else:
                    dP = np.ones_like(self.P[validStates, :, 0])

                if dE.size == 0 or dP.size == 0:
                    continue

                diff = self.getMetric(dE, dP)
                minDif = (diff, state)

                # loop through all states to find the best swap
                for i in range(swapStart, upbound): # point is allowed to swap with states outside of bounds

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
                    dE = self.generateDerivatives(pnt, validPnts, self.E[validStates], backwards=backwards)
                    pTest = self.P[validStates][:, :, self.propList]
                    if self.makePos:
                        pTest = np.abs(pTest)
                    if self.normalize:
                        pTest = pTest / np.linalg.norm(pTest)

                    if not self.ignoreProps:
                        dP = self.generateDerivatives(pnt, validPnts, pTest, backwards=backwards)
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
            self.futurePnts = futurePnts_copy
        return modifiedStates

    def getValidStates(self, pnt, lobound, upbound, state):
        """
        Find states that are valid for the current point
        :param lobound: lower bound for states
        :param pnt: current point
        :param state: current state
        :param upbound: upper bound for states
        :return validStates : list of valid states
        """
        validArray = np.zeros(self.numStates, dtype=bool)


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
            validArray.fill(False)  # set all states to be invalid
            # set states outside of bounds or missing to be invalid
            validArray = buildValidArray(validArray, self.E, lobound, pnt, state, upbound,
                                         eLow, eHigh, eWidth, hasEBounds, hasEWidth)
        else:
            validArray.fill(True)  # set all states to be valid
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
        This function will randomize the state ordering for each point, except the first point
        :return The states with randomized energy ordering
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
        This function will load the state information of a reorder scan from a previous run of this script
        :param numStates: The number of states
        :param numPoints: The number of points
        :param numColumns: The number of columns in the file
        """

        Ecurves = genfromtxt('E.csv')
        Ncurves = genfromtxt('P.csv')

        self.allPnts = genfromtxt('allPnts.csv')
        self.E = Ecurves.reshape((numStates, numPoints))
        self.P = Ncurves.reshape((numStates, numPoints, numColumns))


    def saveOrder(self, isFinalResults = False):
        """
        This function will save the state information of a reorder scan for a future run of this script
        :param isFinalResults: A boolean that determines if the final results are being saved
        :return The energy and properties for each state at each point written to a file "checkpoint.csv"
        """

        tempInput = zeros((self.P.shape[0] * self.P.shape[1], self.P.shape[2] + 2))
        combineVals(self.E, self.P, self.allPnts, tempInput)

        savetxt("checkpoint.csv", tempInput, fmt='%20.12f')

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
        savetxt('temp_out.csv', newCurves, fmt='%20.12f')


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
        This function will set parameters from the configuration file
        """
        if self.configPath is None:
            return

        with open(self.configPath, 'r') as configer:
            data = json.load(configer)
            self.printVar = data.get("printVar", 0)
            self.interactive = data.get("interactive", True)
            self.maxiter = data.get("maxiter", 1000)
            self.orders = data.get("orders", [1])
            self.width = data.get("width", 8)
            self.futurePnts = data.get("futurePnts", 0)
            self.maxPan = data.get("maxPan", None)
            self.pntBounds = data.get("pntBounds", None)
            self.propList = data.get("propList", None)
            self.sweepBack = data.get("sweepBack", True)
            self.stateBounds = data.get("stateBounds", None)
            self.eBounds = data.get("eBounds", None)
            self.eWidth = data.get("eWidth", None)
            self.interpolate = data.get("interpolate", False)
            self.sortLast = data.get("sortLast", False)
            self.keepInterp = data.get("keepInterp", False)
            self.maxStateRepeat = data.get("maxStateRepeat", -1)
            self.nthreads = data.get("nthreads", 1)
            self.makePos = data.get("makePos", False)
            self.normalize = data.get("normalize", False)
            self.doShuffle = data.get("doShuffle", False)
            self.redundantSwaps = data.get("redundantSwaps", False)
        if self.width <= 0 or self.width <= max(self.orders):
            print(
                "invalid size for width. width must be positive integer greater than max order. Defaulting to 'max(orders)+3'")
            self.width = max(self.orders) + 3
        if self.futurePnts < 0:
            raise ValueError("Invalid number of future points")
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
        This function will take a sequence of points for multiple states with energies and properties and reorder them such that the energies and properties are continuous
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

        startClient(self)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Suave State Scanner", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("in_file", help="input file that contains the data in the format:\n"
                                        "\trc1 energy1 feature1.1 feature1.2 ---> \n"
                                        "\trc1 energy2 feature2.1 feature2.2 ---> \n"
                                        "\trc2 energy1 feature1.1 feature1.2 ---> \n"
                                        "\trc2 energy2 feature2.1 feature2.2 ---> \n\n")
    parser.add_argument("out_file", help="output file that will store the final results")
    parser.add_argument("num_states", type=int, help="number of states in the input data")

    # optional arguments
    parser.add_argument("-c", "--config", dest="config_path",
                        help="\n"
                            "configuration file path (optional). parameters are:\n\n"
                            "printVar:\n\tindex of property to print\n"
                            "interactive:\n\twhether to run in interactive mode\n"
                            "orders:\n\torders of derivatives to compute\n"
                            "width:\n\twidth of stencil\n"
                            "futurePnts:\n\tnumber of points from the right of the center that is included in the stencils\n"
                            "maxPan:\n\tmaximum number of times the stencil of size width can pivot around the center point\n"
                            "pntBounds:\n\tbounds of the points in the input file\n"
                            "propList:\n\tlist of the indices of the properties to enforce continuity for\n"
                            "sweepBack:\n\twhether to sweep backwards through the points after it has finished sweeping forwards\n"
                            "stateBounds:\n\tbounds of the states in the input file\n"
                            "eBounds:\n\tbounds of the energies in the input file\n"
                            "eWidth:\n\tenergy width for valid energies to swap with current state at a point\n"
                            "interpolate:\n\twhether to linearly interpolate over nan or inf values\n"
                            "keepInterp:\n\twhether to keep the interpolated missing points in the output file\n"
                            "maxStateRepeat:\n\tmaximum number of times a state can be re-swapped without it changing\n"
                            "nthreads:\n\tnumber of numba threads to use\n"
                            "makePos:\n\twhether to make all extracted properties positive when sorting\n"
                            "normalize:\n\twhether to normalize the extracted properties when sorting\n"
                            "doShuffle:\n\twhether to shuffle order or energy eigenvalues along each curve\n"
                            "redundantSwaps:\n\twhether to allow redundant swaps\n\n"
                            "See the documentation for more information on these parameters.\n\n"
                        )


    args = parser.parse_args()

    in_file = args.in_file
    out_file = args.out_file
    num_states = args.num_states
    config_path = args.config_path

    if config_path is None:
        print("No configuration file specified. Using default values.", flush=True)

    suave = SuaveStateScanner(in_file, out_file, num_states, config_path)

    if suave.interactive:
        suave.run_interactive()
    else:
        suave.run()