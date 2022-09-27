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

import numpy as np
from numba import njit, prange

# This script generates n-point finite difference stencils to approximate derivatives for functions at any order.
# The stencils are generated using the method of undetermined coefficients.
# The stencils are generated using the numba library to speed up the calculations.


# This is a lookup table for the factorial function.
LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')


@njit(parallel=False, cache=True)
def numbaFactorial(n):
    # This function is a numba implementation of the factorial function.
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]


@njit(parallel=True, cache=True)
def makeStencil(N, s, d):
    S = np.zeros((N, N))
    for i in prange(N):
        S[i][:] = s[:] ** i

    delta = np.zeros(N)
    delta[d] = numbaFactorial(d)

    # The stencil coefficients is generated for a function with N points.
    alpha = np.linalg.solve(S, delta)
    return alpha


def mytest():
    # This function tests the makeStencil function.
    for k in range(7):
        N = k + 1
        s = np.arange(N)
        alpha = makeStencil(N, s, 2)

        print(s)
        print(alpha)
        print()
# mytest()
