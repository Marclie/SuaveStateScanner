
![Alt text](suave.svg)

[SuaveStateScanner] - A tool for excited electronic state reordering
=======================================================================

Introduction
------------

This tool takes a sequence of points for multiple electronic states with energies and properties and reorders them such
that the energies with properties are continuous for each state. This is done by defining n-point finite differences along a
sliding window centered at each point. The point is swapped with all states, and the state that is most continuous at
that point is kept.

This is useful, for example, in electronic structure calculations where excited state energies are often discontinuous
w.r.t their properties. This can help identify the symmetry of electronic states without explicitly running a symmetry
calculation.

This is a general mathematical tool to enforce continuity of any set of states along a set of points with quantities
that need the states to be continuous along their points. This will only work well for data that is inherently
continuous with closely spaced points.


Usage
-----

To use this tool, simply run:

    python suave_state_scanner.py <infile> <outfile> <numStates> [<configFile>]

This will output the new energy ordering from \<infile\> to \<outfile\>, with the reaction coordinate as the first row. During the reordering procedure, the files "tempInput.csv" and "tempOutput.csv" are generated. "tempOutput.csv" stores the output at any given iteration for the states, which is used to track the progress of the reordering. "tempInput.csv" stores all the state information for each point in a file that can be used to restart this script.

Arguments
---------

* `infile` - The name of the input file. This file should contain a sequence of points for multiple states with energies
  and properties.

* `outfile` - The name of the output file. This file will be filled with rows corresponding to the reaction coordinate
  and then by state, with the energy of each state printed along the columns.

* `numStates` - The number of states in the input file.

* `configPath` (optional) - This is the path to a configuration file that will set up parameters for the stencils used
  to enforce continuity of states. If not specified, default values will be set (default: None)

Input File Format
-----------------

The file is assumed to contain a sequence of points for multiple states with energies and properties. The file will be
filled with rows corresponding to the reaction coordinate and then by state, with the energy and features of each state
printed along the columns. The first column must be the reaction coordinate. The second column must be the energy of the state, or some other target variable.

    rc1 energy1 feature1.1 feature1.2 --->
    rc1 energy2 feature2.1 feature2.2 --->
                    |
                    V
    rc2 energy1 feature1.1 feature1.2 --->
    rc2 energy2 feature2.1 feature2.2 --->

Configuration File Format
-------------------------
The configuration file is a text file with the following format:

```
# This is a comment line. All lines starting with '#' will be ignored.
orders = [1]
width = 8
cutoff = 1
maxPan = None
stateBounds = None
pntBounds = None
nthreads = 1
makePos = False
doShuffle = False
```

All configurations in the configuration file are optional and are defined as follows:

* `orders` - The 'orders' parameter defines the orders of derivatives desired for computation.
   This parameter is required and must be a list of integers.
   If this parameter is not provided, the default value '[1]' will be used. (default: [1])

* `width` - The 'width' parameter defines the width of the stencil used for finite differences.
   This parameter is optional and must be a positive integer greater than max order.
   If this parameter is not provided, the default value '8' or 'max(orders)+3' will be used. (default: 8)

* `cutoff` - The 'cutoff' parameter defines the number of points from the right of the center that is included in the stencils.
   This parameter is optional and must be a positive integer. If this parameter is not provided, 
   the default value '1' will be used. A small value is ideal since the script reorders points from left to right; 
   points on the right will be unsorted making their derivatives mostly invalid (default: 1)

* `maxPan` - The 'maxPan' parameter defines the maximum number of times the stencil of size width can pivot 
   around the center point. This parameter is optional and must be a positive integer. If this parameter is not provided, 
   the default value 'None' will be used, meaning there is no limit to the pivoting of the stencil.  (default: False)

* `pntBounds` - The bounds of the points in the input file. If provided, this should be a list
  in the form [xmin, xmax] where xmin/xmax are the minimum/maximum index for the reaction coordinates (default: None)

* `stateBounds` - The bounds of the states in the input file. If provided, this should be a
  list in the form [statemin, statemax] specifying inclusive lower and
  upper bounds on indices identifying individual electronic states within an ensemble (e.g., if numStates=3 then
  stateBounds=[0, 1] would select only two out of three available electronic states). By default all available
  electronic states will be included in analysis/output unless stateBounds is specified otherwise. (default: None)

* `nthreads` - The number of numba threads to use (default: 1)

* `makePos` - Whether to make all extracted features positive before sorting according to increasing energy
  eigenvalue along each curve/trajectory/reaction path segment sampled during electronic structure calculations or other
  types of calculations from which data was obtained as input to the SuaveStateScanner program. Making features positive
  can sometimes improve performance on data processed by SuaveStateScanner as input but isn't strictly necessary so
  default value is False. (default: False)

* `doShuffle` - Whether to shuffle order or energy eigenvalues along each curve sampled from electronic
  structure calculations or other types of calculations. Shuffling can sometimes improve performance on data processed
  by SuaveStateScanner as input but isn't strictly necessary so default value is False. (default: False)


How to cite
------------------

When using SuaveStateScanner for research projects, please cite:

```
@misc{SuaveStateScanner,
    author       = {Marcus D. Liebenthal},
    title        = {{SuaveStateScanner}: https://github.com/Marclie/SuaveStateScanner},
    month        = {August},
    year         = {2022},
    url          = {https://github.com/Marclie/SuaveStateScanner} 
}
```
