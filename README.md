![Alt text](suave.svg)

[SuaveStateScanner] - A tool for excited electronic state reordering
=======================================================================

Table of Contents
-----------------
* [Introduction](#introduction)
* [Usage](#usage)
* [Arguments](#arguments)
* [Input File Format](#input-file-format)
* [Configuration File Format](#configuration-file-format)
* [Troubleshooting](#troubleshooting)
* [How to cite](#how-to-cite)


Introduction
------------

This tool takes a sequence of points for multiple electronic states with energies and properties and reorders them so that each state is continuous for these quantities. This script reorders states by defining [n-point finite differences](https://en.wikipedia.org/wiki/Finite_difference_coefficient) along a
sliding window centered at each point. The point is swapped with all states, and the state that is most continuous at
that point is kept.

This script is useful, for example, in electronic structure calculations where excited state energies are often discontinuous
w.r.t their properties. The script can help identify the symmetry of electronic states without explicitly running a symmetry
calculation.

The script is a general mathematical tool to enforce the continuity of eigenvalues along a set of points with properties extracted from the eigenvectors
that need to be continuous along some coordinate. This will only work well for inherently continuous data with closely spaced points.


Usage
-----

To use this tool, run:

    python suave_state_scanner.py <infile> <outfile> <numStates> [<configFile>]

This will output the new energy ordering from \<infile\> to \<outfile\>, with the reaction coordinate as the first row. The files "tempInput.csv" and "tempOutput.csv" are generated during the reordering procedure. The "tempInput.csv" file stores all the state information for each point in a file that the user can use to restart this script. The "tempOutput.csv" file stores the output at any given iteration for the states, which is used to track the progress of the reordering.

Arguments
---------

* `infile` - The name of the input file. This file should contain a sequence of points for multiple states with energies
  and properties.


* `outfile` - The name of the output file. This file will be filled with rows corresponding to the reaction coordinate
  and then by state, with each state's energy printed along the columns.


* `numStates` - The number of states in the input file for each point in the reaction coordinate.


* `configPath` (optional) - This is the path to a configuration file that will set up parameters for the stencils used
  to enforce continuity of states. If not specified, default values will be set (default: None)

Input File Format
-----------------

The file is assumed to contain a sequence of points for multiple states with energies and properties. The file will be
filled with rows corresponding to the reaction coordinate and then by state, with the energy and properties of each state
printed along the columns. The first column must be the reaction coordinate. The second column must be the state's energy or some other target variable.
The remaining columns are the properties of the state. 

The number of states is specified by the user.


    rc1 energy1 property1.1 property1.2 --->
    rc1 energy2 property2.1 property2.2 --->
                    |
                    V
    rc2 energy1 property1.1 property1.2 --->
    rc2 energy2 property2.1 property2.2 --->

For data that is not perfectly square, the script will break. If it is not feasible to run further calculations,
the user can replace missing energies and/or properties with nan or inf. The script will ignore these points for 
calculating finite differences by interpolating over them (A warning message will indicate this). The final output will 
keep the nan or inf values and will not interpolate over them. However, the user should be aware that the script will 
not effectively reorder states if there are too many missing.

Configuration File Format
-------------------------
The configuration file is a text file with the following format:

```
# This is a comment line. All lines starting with '#' will be ignored.
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
maxStateRepeat = None
nthreads = 7
makePos = True
doShuffle = False
redundantSwaps = False
```

All configurations in the configuration file are optional and are defined as follows:

* `printVar` - The index for the target property to be printed to the output file. If this is not specified, the
  default is to print the energy of the state. Other values will print a specific property for each state. Negative
  values will index from the end of the list of properties. (default: 0)


* `orders` - The 'orders' parameter defines the orders of derivatives desired for computation. This parameter must be a list of integers.
   The default value '[1]' will be used if this parameter is not provided. (default: [1])


* `width` - The 'width' parameter defines the width of the stencil used for finite differences.
   This parameter is optional and must be a positive integer greater than the max order.
   If this parameter is not provided, the default value '8' or 'max(orders)+3' will be used. (default: 8)


* `futurePnts` - The 'futurePnts' parameter defines the number of points from the right of the center that is included in the stencils.
   This parameter is optional and must be a positive integer. If this parameter is not provided, 
   the default value '0' will be used. A zero/small value is ideal since the script reorders points from left to right; 
   points on the right will be unsorted making their derivatives mostly invalid (default: 0)


* `maxPan` - The 'maxPan' parameter defines the maximum number of times the stencil of size width can pivot 
   around the center point. This parameter is optional and must be a positive integer. If this parameter is not provided, 
   the default value 'None' will be used, meaning there is no limit to the pivoting of the stencil.  (default: None)


* `pntBounds` - The bounds of the points in the input file. If provided, this should be a list
  in the form [xmin, xmax] where xmin/xmax are the minimum/maximum index for the reaction coordinates (default: None)


* `sweepBack` - The 'sweepBack' parameter defines whether the script will sweep backwards through the points
   after it has finished sweeping forwards. This parameter is optional and must be a boolean. If this parameter is not provided, 
   the default value 'True' will be used. It is recommended to keep this parameter as 'True' since the script loses 
   accuracy at the boundaries due to the finite difference stencil. (default: True)


* `stateBounds` - The bounds of the states in the input file. If provided, this should be a
  list in the form [statemin, statemax] specifying inclusive lower and
  upper bounds on indices identifying individual electronic states to sort (e.g., if numStates=3 then
  stateBounds=[0, 1] would reorder only two out of three available electronic states). 
  Electronic states from outside the range can swap with interior states to further improve continuity. By default all available
  electronic states will be included in analysis/output unless stateBounds is specified otherwise. (default: None)


* `eBounds` - The bounds of the energies in the input file. If provided, this should be a list
  in the form [emin, emax] specifying inclusive lower and upper bounds on energies. By default all available
  energies will be included in analysis/output unless eBounds is specified otherwise. (default: None)

* `keepInterp` - The 'keepInterp' parameter defines whether to keep the interpolated missing points in the output file. 
   This parameter is optional and must be a boolean. If this parameter is not provided, 
   the default value 'False' will be used. Note: if this parameter is set to True, the missing points will not be identified
   in the output file. (default: False)


* `maxStateRepeat` - The maximum number of times a state can be repeated without changes in reordering procedure. If this parameter is not provided, the default value 'None' will be used, meaning there is no limit to the number of times a state can be repeated. (default: None)


* `nthreads` - The number of numba threads to use (default: 1)


* `makePos` - Whether to make all extracted properties positive before sorting according to increasing energy
  eigenvalue along each curve/trajectory/reaction path segment sampled during electronic structure calculations or other
  types of calculations from which data was obtained as input to the SuaveStateScanner program. Making properties positive
  can sometimes improve performance on data processed by SuaveStateScanner as input but isn't strictly necessary so
  default value is False. (default: False)


* `doShuffle` - Whether to shuffle order or energy eigenvalues along each curve sampled from electronic
  structure calculations or other types of calculations. Shuffling can sometimes improve performance on data processed
  by SuaveStateScanner as input but isn't strictly necessary so default value is False. (default: False)

* `redundantSwaps` - Whether to allow redundant swaps. The current alogrithm will only swap the current state with higher lying states. 
  This for performance reasons since the lower lying states should already be in the correct order after a few iterations. However,
  if the user wants to allow the redundant swaps for troubleshooting purposes, this parameter can be set to True. (default: False)

Troubleshooting
------------
If your data is not converging and/or the order of states is not accurate, there are a few things you can try:

- Make sure that your data is continuous at each point and that the points are closely spaced. This tool will not work well with discontinuous data or has widely spaced points.

- Try changing the configuration parameters in the config file. You can experiment with different values for the 'orders', 'width', 'cutoff', and 'maxPan' parameters to see if that helps convergence and accuracy. 

    - In almost all cases, the default value of [1] for 'orders' gives the correct behavior. This value should be the last to be experiemented with.
    - It's usually ideal to have a large width with a small spacing of points; this helps capture more local information of the state. However, this will increase the cost of reordering. 
    - The 'cutoff' should be small since points towards the right will often be sorted inaccurately, however sometimes a value greater than 1 may give better results.
    - The 'maxPan' value defaults to None which puts no limits on how much the sliding window for finite differences pivots around a central point. It can help to set a small value for this, so fewer combinations of points are considered to compute the change of energy w.r.t a state swapping.
    - The 'futurePnts' parameter should almost always be set to 0. This parameter is used to include points on the right of the center point in the stencil. However, these points are not sorted and will often give inaccurate derivatives.
    - Setting redundantSwaps to True will allow the algorithm to swap the current state with lower lying states. This should not be necessary in most cases, but can be useful for troubleshooting.

- Experiment with using different properties to describe each state. Some properties will contribute more to the calculation of the finite differences than others, or some properties may be less continuous than others at each point (i.e. x-, y- transition dipoles that can mix arbitrarily for c1-symmetry calculations). It may help to normalize all properties for each state so properties are treated on equal footing or enable the 'makePos' flag in the config file to eliminate dependence on sign.

- Try shuffling the order of energy eigenvalues for each state along each point sampled from electronic structure calculations or other types of calculations. This can sometimes improve performance on data processed by SuaveStateScanner as input, where parameters from the initial configuration prevents states from being swapped. To do this, simply set the 'doShuffle' parameter to True in the config file.

- Consider sorting the states in batches with different bounds for the points (pntBounds), states (stateBounds), and energies (eBounds). This can help to identify the correct order of states for each batch, which can then be used to sort the entire data set.

- For problematic points that cause the script to fail, you can try to manually reorder the states at that point. This can be done by editing the input file and changing the order of the states at that point. The script will then use this new order as the starting point for its reordering procedure.

- Additionally, you can set the energy and properties of the problematic points to 'nan' in the input file. The script will then ignore these points via interpolation and will not try to reorder the states at these points. Unless the 'keepInterp' parameter is set to True, the script will not output the interpolated points in the output file (those points will have 'nan' for the energy and properties).

- If you are still having trouble, please contact me at my [email](mailto:mliebenthal@fsu.edu) or submit a ticket and I will try to help you out.

Make sure to plot the guess ordering at different iterations from the 'tempOutput.csv' file. This can help you recognize if a set of parameters will work or not before waiting for the reordering procedure to terminate. The reordering procedure will repeat the scan over all points for the number of points squared, or until no more state swaps improve the continuity at each point. For degenerate states with identical properties, this can sometimes cause issues where the two states swap with each other arbitrarily for each scan. In this case, it is ideal to terminate the calculation and use tempOutput.csv as the final result.

How to cite
-----------

When using SuaveStateScanner for research projects, please cite:

```
@misc{SuaveStateScanner,
    author       = {Marcus Dante Liebenthal},
    title        = {{SuaveStateScanner}: https://github.com/Marclie/SuaveStateScanner},
    month        = {August},
    year         = {2022},
    url          = {https://github.com/Marclie/SuaveStateScanner} 
}
```
