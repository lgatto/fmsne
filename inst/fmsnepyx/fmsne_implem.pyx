# distutils: language = c++

#cython: binding=False
#cython: wraparound=False
#cython: boundscheck=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: profile=False
#cython: linetrace=False
#cython: infer_types=False
#cython: embedsignature=False
#cython: cdivision=True
#cython: cdivision_warnings=False
#cython: overflowcheck=False
#cython: overflowcheck.fold=False
#cython: language_level=3
#cython: always_allow_keywords=False
#cython: type_version_tag=True
#cython: iterable_coroutine=False
#cython: optimize.use_switch=True
#cython: optimize.unpack_method_calls=True
#cython: warn.undeclared=False
#cython: warn.unreachable=False
#cython: warn.maybe_uninitialized=False
#cython: warn.unused=False
#cython: warn.unused_arg=False
#cython: warn.unused_result=False
#cython: warn.multiple_declarators=False

#######################################################
####################################################### Imports
#######################################################

# Numpy is needed to define FLOAT64_EPS. 'cimport' is used to import compile-time information about the numpy module.
import numpy as np
cimport numpy as np
# Importing some functions from the C math library
from libc.math cimport sqrt, log, exp, fabs, round, log2, pow
# Import Python C-API memory allocation functions
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc
# Sorting function
from libcpp.algorithm cimport sort
# Random number generation and exit function
from libc.stdlib cimport rand, srand, exit, EXIT_FAILURE
# Print
from libc.stdio cimport printf
# Limits of the floating-point values
from libc.float cimport DBL_MIN, DBL_MAX
# Memory copy function
from libc.string cimport memcpy, memset

#######################################################
####################################################### Global variables
#######################################################

# If some double is smaller than EPSILON_DBL in magnitude, it is considered as close to zero.
cdef double EPSILON_DBL = 1e-8

# To avoid dividing by zeros in similarity-related quantities.
cdef double FLOAT64_EPS = np.finfo(dtype=np.float64).eps

#######################################################
####################################################### Minimum and maximum functions
#######################################################

cdef inline double min_arr_ptr(const double* x, Py_ssize_t m) nogil:
    """
    Return the minimum value in a one-dimensional array, assuming the latter has at least one element.
    m is the size of the array x.
    """
    cdef Py_ssize_t i
    cdef double v = x[0]
    for i in range(1, m, 1):
        if x[i] < v:
            v = x[i]
    return v

cdef inline double max_arr_ptr(const double* x, Py_ssize_t m) nogil:
    """
    Return the maximum value in a one-dimensional array, assuming the latter has at least one element.
    m is the size of the array x.
    """
    cdef Py_ssize_t i
    cdef double v = x[0]
    for i in range(1, m, 1):
        if x[i] > v:
            v = x[i]
    return v

cdef inline Py_ssize_t max_arr_ptr_Pysst(const Py_ssize_t* x, Py_ssize_t m) nogil:
    """
    Return the maximum value in a one-dimensional array, assuming the latter has at least one element.
    m is the size of the array x.
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t v = x[0]
    for i in range(1, m, 1):
        if x[i] > v:
            v = x[i]
    return v

cdef inline double max_arr2d_col(double** x, Py_ssize_t m, Py_ssize_t c) nogil:
    """
    Search the maximum of some column in a 2d array. m is the number of rows, c is the column to search.
    """
    cdef Py_ssize_t i
    cdef double v = x[0][c]
    for i in range(1, m, 1):
        if x[i][c] > v:
            v = x[i][c]
    return v

cdef inline double min_arr_ptr_step(const double* x, Py_ssize_t m, Py_ssize_t start, Py_ssize_t step) nogil:
    """
    Similar to min_arr_ptr, but with start and step parameters.
    """
    cdef double v = x[start]
    start += step
    while start < m:
        if x[start] < v:
            v = x[start]
        start += step
    return v

cdef inline double max_arr_ptr_step(const double* x, Py_ssize_t m, Py_ssize_t start, Py_ssize_t step) nogil:
    """
    Similar to max_arr_ptr, but with start and step parameters.
    """
    cdef double v = x[start]
    start += step
    while start < m:
        if x[start] > v:
            v = x[start]
        start += step
    return v

#######################################################
####################################################### Euclidean distance function
#######################################################

cdef inline double sqeucl_dist_ptr(const double* x, const double* y, Py_ssize_t m) nogil:
    """
    Computes the squared Euclidean distance between x and y, which are assumed to be one-dimensional and containing the same number m of elements.
    In:
    - x, y: two one-dimensional arrays with the same number of elements.
    - m: size of x and y.
    Out:
    The squared Euclidean distance between x and y.
    """
    cdef Py_ssize_t i
    cdef double d = 0.0
    cdef double v
    for i in range(m):
        v = x[i] - y[i]
        d += v*v
    return d

#######################################################
####################################################### Infinite distance function
#######################################################

cdef inline double inf_dist_ptr(const double* x, const double* y, Py_ssize_t m) nogil:
    """
    Computes the infinite distance (i.e. the distance based on the infinite norm) between x and y, which are assumed to be one-dimensional and with the same number of elements. x and y are assumed to have at least one element.
    In:
    - x, y: pointers to two one-dimensional arrays with the same number of elements. They are assumed to have at least one element.
    - m: size of x and y.
    Out:
    The infinite distance between x and y.
    """
    cdef Py_ssize_t i
    cdef double d = fabs(x[0] - y[0])
    cdef double v
    for i in range(1, m, 1):
        v = fabs(x[i] - y[i])
        if v > d:
            d = v
    return d

#######################################################
####################################################### Mean of an array
#######################################################

cdef inline double mean_arr_ptr_step(const double* x, Py_ssize_t m, Py_ssize_t start, Py_ssize_t step, double N) nogil:
    """
    Return the mean of the elements pointed by x, at the indexes start, start+step, start+2*step, ..., until start+m-1.
    m is the total size of x.
    N is the number of elements over which we compute the mean.
    """
    cdef double v = x[start]
    start += step
    while start < m:
        v += x[start]
        start += step
    return v/N

#######################################################
####################################################### Variance of an array
#######################################################

cdef inline double var_arr_ptr_step(const double* x, Py_ssize_t m, Py_ssize_t start, Py_ssize_t step, double N, double den_var) nogil:
    """
    Computes the variance of the elements pointed by x, at the indexes start, start + step, start + 2*step, ..., until start+m-1.
    m is the total size of x.
    m must be at least 2.
    den_var can be set to N-1.0
    """
    cdef double mu = mean_arr_ptr_step(x, m, start, step, N)
    cdef double diff = x[start] - mu
    cdef double v = diff * diff
    start += step
    while start < m:
        diff = x[start] - mu
        v += diff * diff
        start += step
    return v/den_var

#######################################################
####################################################### Allocation functions. The returned values must be freed.
#######################################################

cdef inline void free_int_2dmat(int** arr, Py_ssize_t M):
    """
    """
    cdef Py_ssize_t m
    for m in range(M):
        PyMem_Free(arr[m])
    PyMem_Free(arr)

cdef inline void free_int_3dmat(int*** arr, Py_ssize_t M, Py_ssize_t N):
    """
    """
    cdef Py_ssize_t m, n
    for m in range(M):
        for n in range(N):
            PyMem_Free(arr[m][n])
        PyMem_Free(arr[m])
    PyMem_Free(arr)

cdef inline void free_dble_2dmat(double** arr, Py_ssize_t M):
    """
    """
    cdef Py_ssize_t m
    for m in range(M):
        PyMem_Free(arr[m])
    PyMem_Free(arr)

cdef inline void free_dble_3dmat(double*** arr, Py_ssize_t M, Py_ssize_t N):
    """
    """
    cdef Py_ssize_t m, n
    for m in range(M):
        for n in range(N):
            PyMem_Free(arr[m][n])
        PyMem_Free(arr[m])
    PyMem_Free(arr)

cdef inline void free_Pysst_2dmat(Py_ssize_t** arr, Py_ssize_t M):
    """
    """
    cdef Py_ssize_t m
    for m in range(M):
        PyMem_Free(arr[m])
    PyMem_Free(arr)

cdef inline void free_Pysst_3dmat(Py_ssize_t*** arr, Py_ssize_t M, Py_ssize_t N):
    """
    """
    cdef Py_ssize_t m, n
    for m in range(M):
        for n in range(N):
            PyMem_Free(arr[m][n])
        PyMem_Free(arr[m])
    PyMem_Free(arr)

cdef inline int* seq_1step(Py_ssize_t N):
    """
    """
    cdef int* all_ind = <int*> PyMem_Malloc(N*sizeof(int))
    if all_ind is NULL:
        return NULL
    cdef int i
    for i in range(N):
        all_ind[i] = i
    return all_ind

cdef inline int** calloc_int_2dmat(Py_ssize_t M, Py_ssize_t N):
    """
    """
    cdef int** mat_ret = <int**> PyMem_Malloc(M*sizeof(int*))
    if mat_ret is NULL:
        return NULL
    cdef Py_ssize_t m
    cdef size_t shdp = N*sizeof(int)
    for m in range(M):
        mat_ret[m] = <int*> PyMem_Malloc(shdp)
        if mat_ret is NULL:
            free_int_2dmat(mat_ret, m)
            return NULL
        # Setting the elements of mat_ret[m] to zero.
        memset(mat_ret[m], 0, shdp)
    return mat_ret

cdef inline int*** alloc_int_3dmat(Py_ssize_t M, Py_ssize_t N, Py_ssize_t K):
    """
    """
    cdef int*** mat_ret = <int***> PyMem_Malloc(M*sizeof(int**))
    if mat_ret is NULL:
        return NULL
    cdef Py_ssize_t m, n
    for m in range(M):
        mat_ret[m] = <int**> PyMem_Malloc(N*sizeof(int*))
        if mat_ret[m] is NULL:
            free_int_3dmat(mat_ret, m, N)
            return NULL
        for n in range(N):
            mat_ret[m][n] = <int*> PyMem_Malloc(K*sizeof(int))
            if mat_ret[m][n] is NULL:
                free_int_2dmat(mat_ret[m], n)
                free_int_3dmat(mat_ret, m, N)
                return NULL
    return mat_ret

cdef inline double** alloc_dble_2dmat(Py_ssize_t M, Py_ssize_t N):
    """
    """
    cdef double** mat_ret = <double**> PyMem_Malloc(M*sizeof(double*))
    if mat_ret is NULL:
        return NULL
    cdef Py_ssize_t m
    for m in range(M):
        mat_ret[m] = <double*> PyMem_Malloc(N*sizeof(double))
        if mat_ret[m] is NULL:
            free_dble_2dmat(mat_ret, m)
            return NULL
    return mat_ret

cdef inline double** alloc_dble_2dmat_varKpysst(Py_ssize_t M, Py_ssize_t* N):
    """
    """
    cdef double** mat_ret = <double**> PyMem_Malloc(M*sizeof(double*))
    if mat_ret is NULL:
        return NULL
    cdef Py_ssize_t m
    for m in range(M):
        mat_ret[m] = <double*> PyMem_Malloc(N[m]*sizeof(double))
        if mat_ret[m] is NULL:
            free_dble_2dmat(mat_ret, m)
            return NULL
    return mat_ret

cdef inline double*** alloc_dble_3dmat(Py_ssize_t M, Py_ssize_t N, Py_ssize_t K):
    """
    """
    cdef double*** mat_ret = <double***> PyMem_Malloc(M*sizeof(double**))
    if mat_ret is NULL:
        return NULL
    cdef Py_ssize_t m, n
    for m in range(M):
        mat_ret[m] = <double**> PyMem_Malloc(N*sizeof(double*))
        if mat_ret[m] is NULL:
            free_dble_3dmat(mat_ret, m, N)
            return NULL
        for n in range(N):
            mat_ret[m][n] = <double*> PyMem_Malloc(K*sizeof(double))
            if mat_ret[m][n] is NULL:
                free_dble_2dmat(mat_ret[m], n)
                free_dble_3dmat(mat_ret, m, N)
                return NULL
    return mat_ret

cdef inline double*** alloc_dble_3dmat_varK(Py_ssize_t M, Py_ssize_t N, int** K):
    """
    Same as alloc_dble_3dmat, but the size of the third dimension may change.
    """
    cdef double*** mat_ret = <double***> PyMem_Malloc(M*sizeof(double**))
    if mat_ret is NULL:
        return NULL
    cdef Py_ssize_t m, n
    for m in range(M):
        mat_ret[m] = <double**> PyMem_Malloc(N*sizeof(double*))
        if mat_ret[m] is NULL:
            free_dble_3dmat(mat_ret, m, N)
            return NULL
        for n in range(N):
            mat_ret[m][n] = <double*> PyMem_Malloc(K[m][n]*sizeof(double))
            if mat_ret[m][n] is NULL:
                free_dble_2dmat(mat_ret[m], n)
                free_dble_3dmat(mat_ret, m, N)
                return NULL
    return mat_ret

cdef inline Py_ssize_t** alloc_Pysst_2dmat_varN(Py_ssize_t M, Py_ssize_t* N):
    cdef Py_ssize_t** mat_ret = <Py_ssize_t**> PyMem_Malloc(M*sizeof(Py_ssize_t*))
    if mat_ret is NULL:
        return NULL
    cdef Py_ssize_t m
    for m in range(M):
        mat_ret[m] = <Py_ssize_t*> PyMem_Malloc(N[m]*sizeof(Py_ssize_t))
        if mat_ret[m] is NULL:
            free_Pysst_2dmat(mat_ret, m)
            return NULL
    return mat_ret

cdef inline Py_ssize_t*** alloc_Pysst_3dmat_varK(Py_ssize_t M, Py_ssize_t N, int** K):
    """
    Same as alloc_dble_3dmat, but the size of the third dimension may change.
    """
    cdef Py_ssize_t*** mat_ret = <Py_ssize_t***> PyMem_Malloc(M*sizeof(Py_ssize_t**))
    if mat_ret is NULL:
        return NULL
    cdef Py_ssize_t m, n
    for m in range(M):
        mat_ret[m] = <Py_ssize_t**> PyMem_Malloc(N*sizeof(Py_ssize_t*))
        if mat_ret[m] is NULL:
            free_Pysst_3dmat(mat_ret, m, N)
            return NULL
        for n in range(N):
            mat_ret[m][n] = <Py_ssize_t*> PyMem_Malloc(K[m][n]*sizeof(Py_ssize_t))
            if mat_ret[m][n] is NULL:
                free_Pysst_2dmat(mat_ret[m], n)
                free_Pysst_3dmat(mat_ret, m, N)
                return NULL
    return mat_ret

cdef inline Py_ssize_t*** alloc_Pysst_3dmat_varK_3dK(Py_ssize_t M, Py_ssize_t N, Py_ssize_t*** K, Py_ssize_t idk):
    """
    Same as alloc_dble_3dmat, but the size of the third dimension may change.
    """
    cdef Py_ssize_t*** mat_ret = <Py_ssize_t***> PyMem_Malloc(M*sizeof(Py_ssize_t**))
    if mat_ret is NULL:
        return NULL
    cdef Py_ssize_t m, n
    for m in range(M):
        mat_ret[m] = <Py_ssize_t**> PyMem_Malloc(N*sizeof(Py_ssize_t*))
        if mat_ret[m] is NULL:
            free_Pysst_3dmat(mat_ret, m, N)
            return NULL
        for n in range(N):
            mat_ret[m][n] = <Py_ssize_t*> PyMem_Malloc(K[m][n][idk]*sizeof(Py_ssize_t))
            if mat_ret[m][n] is NULL:
                free_Pysst_2dmat(mat_ret[m], n)
                free_Pysst_3dmat(mat_ret, m, N)
                return NULL
    return mat_ret

#######################################################
####################################################### L-BFGS optimization (C library)
#######################################################

cdef extern from "lbfgs.h":
    ctypedef double lbfgsfloatval_t
    ctypedef lbfgsfloatval_t* lbfgsconst_p "const lbfgsfloatval_t *"

    ctypedef lbfgsfloatval_t (*lbfgs_evaluate_t)(void *, lbfgsconst_p, lbfgsfloatval_t *, int, lbfgsfloatval_t)
    ctypedef int (*lbfgs_progress_t)(void *, lbfgsconst_p, lbfgsconst_p, lbfgsfloatval_t, lbfgsfloatval_t, lbfgsfloatval_t, lbfgsfloatval_t, int, int, int)

    cdef enum LineSearchAlgo:
        LBFGS_LINESEARCH_DEFAULT,
        LBFGS_LINESEARCH_MORETHUENTE,
        LBFGS_LINESEARCH_BACKTRACKING_ARMIJO,
        LBFGS_LINESEARCH_BACKTRACKING_WOLFE,
        LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE

    cdef enum ReturnCode:
        LBFGS_SUCCESS,
        LBFGS_ALREADY_MINIMIZED,
        LBFGSERR_UNKNOWNERROR,
        LBFGSERR_LOGICERROR,
        LBFGSERR_OUTOFMEMORY,
        LBFGSERR_CANCELED,
        LBFGSERR_INVALID_N,
        LBFGSERR_INVALID_N_SSE,
        LBFGSERR_INVALID_X_SSE,
        LBFGSERR_INVALID_EPSILON,
        LBFGSERR_INVALID_TESTPERIOD,
        LBFGSERR_INVALID_DELTA,
        LBFGSERR_INVALID_LINESEARCH,
        LBFGSERR_INVALID_MINSTEP,
        LBFGSERR_INVALID_MAXSTEP,
        LBFGSERR_INVALID_FTOL,
        LBFGSERR_INVALID_WOLFE,
        LBFGSERR_INVALID_GTOL,
        LBFGSERR_INVALID_XTOL,
        LBFGSERR_INVALID_MAXLINESEARCH,
        LBFGSERR_INVALID_ORTHANTWISE,
        LBFGSERR_INVALID_ORTHANTWISE_START,
        LBFGSERR_INVALID_ORTHANTWISE_END,
        LBFGSERR_OUTOFINTERVAL,
        LBFGSERR_INCORRECT_TMINMAX,
        LBFGSERR_ROUNDING_ERROR,
        LBFGSERR_MINIMUMSTEP,
        LBFGSERR_MAXIMUMSTEP,
        LBFGSERR_MAXIMUMLINESEARCH,
        LBFGSERR_MAXIMUMITERATION,
        LBFGSERR_WIDTHTOOSMALL,
        LBFGSERR_INVALIDPARAMETERS,
        LBFGSERR_INCREASEGRADIENT

    ctypedef struct lbfgs_parameter_t:
        int m
        lbfgsfloatval_t epsilon
        int past
        lbfgsfloatval_t delta
        int max_iterations
        int linesearch
        int max_linesearch
        lbfgsfloatval_t min_step
        lbfgsfloatval_t max_step
        lbfgsfloatval_t ftol
        lbfgsfloatval_t wolfe
        lbfgsfloatval_t gtol
        lbfgsfloatval_t xtol
        lbfgsfloatval_t orthantwise_c
        int orthantwise_start
        int orthantwise_end

    int lbfgs(int, lbfgsfloatval_t *, lbfgsfloatval_t *, lbfgs_evaluate_t, lbfgs_progress_t, void *, lbfgs_parameter_t *)
    void lbfgs_parameter_init(lbfgs_parameter_t *)
    lbfgsfloatval_t *lbfgs_malloc(int)
    void lbfgs_free(lbfgsfloatval_t *)

#######################################################
####################################################### Multi-scale SNE
#######################################################

cdef inline int ms_def_n_scales(double Nd, int K_star, int L_min, bint isLmin1) nogil:
    """
    """
    if isLmin1:
        return <int> round(log2(Nd/(<double> K_star)))
    else:
        return (<int> round(log2(Nd/(<double> K_star)))) + 1 - L_min

cdef inline int ms_def_shift_Lmin(bint isnotLmin1, Py_ssize_t L_min) nogil:
    """
    """
    cdef int shift_L_min = 1
    cdef Py_ssize_t h
    if isnotLmin1:
        for h in range(L_min-1):
            shift_L_min *= 2
    return shift_L_min

cdef inline int* ms_def_Kh(int K_star, bint isnotLmin1, int shift_L_min, Py_ssize_t L):
    """
    The returned value must be freed.
    """
    cdef int* K_h = <int*> PyMem_Malloc(L*sizeof(int))
    if K_h is NULL:
        return NULL
    K_h[0] = K_star
    if isnotLmin1:
        K_h[0] *= shift_L_min
    cdef Py_ssize_t h
    for h in range(1, L, 1):
        K_h[h] = K_h[h-1]*2
    return K_h

cdef inline double** sne_ds_hd(double* xhds, Py_ssize_t N, Py_ssize_t d_hds, Py_ssize_t N_1):
    """
    The returned value must be freed.
    The HD distances with respect to each data point are substracted from their minimum, to avoid doing it during the computation of the similarities.
    xhds = pointer toward the start of the HDS, with dim = d_hds
    N_1 = N-1
    """
    cdef Py_ssize_t i, j, idx, idxj
    cdef double** ds_hd = alloc_dble_2dmat(N, N_1)
    if ds_hd is NULL:
        return NULL
    # Computing the pairwise squared Euclidean distances
    cdef double* x
    idx = 0
    for i in range(N_1):
        x = &xhds[idx]
        idx += d_hds
        idxj = idx
        for j in range(i, N_1, 1):
            ds_hd[i][j] = sqeucl_dist_ptr(x, &xhds[idxj], d_hds)
            ds_hd[j+1][i] = ds_hd[i][j]
            idxj += d_hds
    # Computing the minimum of the distances with respect to each data point and substracting the minimum from the distances
    cdef double min_ds
    for i in range(N):
        min_ds = min_arr_ptr(ds_hd[i], N_1)
        for j in range(N_1):
            ds_hd[i][j] = min_ds - ds_hd[i][j]
    return ds_hd

cdef inline void sne_hdpinn_nolog(const double* ds_nn, double tau, Py_ssize_t nnn, double* pinn) nogil:
    """
    Computes SNE sim, without their log.
    ds_nn is assumed to contain the minimum squared distance - the squared distance.
    tau is the denominator of the exponentials
    nnn is the number of neighbors
    pinn is the location at which the similarities will be stored.
    """
    cdef double den = 0.0
    cdef Py_ssize_t i
    for i in range(nnn):
        pinn[i] = exp(ds_nn[i]/tau)
        den += pinn[i]
    for i in range(nnn):
        pinn[i] /= den

cdef inline double sne_densim(const double* ds_nn, double tau, Py_ssize_t nnn) nogil:
    """
    Computes the denominator of the similarities
    ds_nn is assumed to contain the minimum squared distance - the squared distance.
    tau is the denominator of the exponentials
    nnn is the number of neighbors
    """
    cdef double den = 0.0
    cdef Py_ssize_t i
    for i in range(nnn):
        den += exp(ds_nn[i]/tau)
    return den

cdef inline double sne_binsearch_fct(const double* ds_nn, double tau, Py_ssize_t nnn, double log_perp) nogil:
    """
    Computes the entropry of the similarities minus the logarithm of the perplexity
    ds_nn is assumed to contain the minimum squared distance - the squared distance.
    tau is the denominator of the exponentials
    nnn is the number of neighbors
    """
    cdef double a, b, v, den
    v = 0.0
    den = 0.0
    cdef Py_ssize_t i
    for i in range(nnn):
        a = ds_nn[i]/tau
        b = exp(a)
        v -= a*b
        den += b
    return v/den + log(den) - log_perp

cdef inline double sne_binsearch_bandwidth_fit(const double* ds_nn, Py_ssize_t nnn, double log_perp, double tau) nogil:
    """
    Tune the bandwidths of HD SNE similarities.
    ds_nn is assumed to contain the minimum squared distance - the squared distance.
    nnn is the number of neighbors
    Returns the bandwidths.
    The 4th parameter, tau, is the starting point for the binary search. It can be set to 1.0 if no prior guess is known.
    """
    cdef double f_tau = sne_binsearch_fct(ds_nn, tau, nnn, log_perp)
    if fabs(f_tau) <= EPSILON_DBL:
        return tau
    cdef double tau_up, tau_low
    if f_tau > 0:
        tau_low = tau*0.5
        if (tau_low < DBL_MIN) or (fabs(sne_densim(ds_nn, tau_low, nnn)) < DBL_MIN):
            # Binary search failed. The root is too close from 0 for the numerical precision: the denominator of the similarities is almost 0.
            return tau
        f_tau = sne_binsearch_fct(ds_nn, tau_low, nnn, log_perp)
        if fabs(f_tau) <= EPSILON_DBL:
            return tau_low
        tau_up = tau
        while f_tau > 0:
            tau_up = tau_low
            tau_low *= 0.5
            if (tau_low < DBL_MIN) or (fabs(sne_densim(ds_nn, tau_low, nnn)) < DBL_MIN):
                # Binary search failed. The root is too close from 0 for the numerical precision.
                return tau_up
            f_tau = sne_binsearch_fct(ds_nn, tau_low, nnn, log_perp)
            if fabs(f_tau) <= EPSILON_DBL:
                return tau_low
    else:
        tau_up = 2.0*tau
        if fabs(sne_densim(ds_nn, tau_up, nnn)-nnn) <= EPSILON_DBL:
            # Binary search failed. The root is too big for the numerical precision of the exponentials of the similarities. All the exponentials at the denominator = 1 and hence, the denominator = nnn.
            return tau
        f_tau = sne_binsearch_fct(ds_nn, tau_up, nnn, log_perp)
        if fabs(f_tau) <= EPSILON_DBL:
            return tau_up
        tau_low = tau
        while f_tau < 0:
            tau_low = tau_up
            tau_up *= 2.0
            if fabs(sne_densim(ds_nn, tau_up, nnn)-nnn) <= EPSILON_DBL:
                # Binary search failed. The root is too big for the numerical precision of the exponentials of the similarities.
                return tau_low
            f_tau = sne_binsearch_fct(ds_nn, tau_up, nnn, log_perp)
            if fabs(f_tau) <= EPSILON_DBL:
                return tau_up
    cdef Py_ssize_t nit = 0
    cdef Py_ssize_t nit_max = 1000
    while nit < nit_max:
        tau = (tau_up+tau_low)*0.5
        f_tau = sne_binsearch_fct(ds_nn, tau, nnn, log_perp)
        if fabs(f_tau) <= EPSILON_DBL:
            return tau
        elif f_tau > 0:
            tau_up = tau
        else:
            tau_low = tau
        nit += 1
    # Binary search failed
    return tau

cdef inline double** ms_hdsim(double** ds_hd, Py_ssize_t N, Py_ssize_t L, int* K_h, Py_ssize_t N_1):
    """
    Return NULL if memory problem.
    """
    cdef double** tau_h = alloc_dble_2dmat(L, N)
    if tau_h is NULL:
        return NULL
    cdef Py_ssize_t i, h, L_1, L_2
    cdef double* log_perp = <double*> PyMem_Malloc(L*sizeof(double))
    if log_perp is NULL:
        free_dble_2dmat(tau_h, L)
        return NULL
    for h in range(L):
        log_perp[h] = log(<double> min(K_h[h], N_1))
    L_1 = L - 1
    L_2 = L_1 - 1
    # For each data point
    for i in range(N):
        # Computing the bandwidth for the last scale. The binary search is initialized with 1.
        tau_h[L_1][i] = sne_binsearch_bandwidth_fit(ds_hd[i], N_1, log_perp[L_1], 1.0)
        # For the other scales, the binary search is initialized with the bandwidth of the previous scale.
        for h in range(L_2, -1, -1):
            tau_h[h][i] = sne_binsearch_bandwidth_fit(ds_hd[i], N_1, log_perp[h], tau_h[h+1][i])
    PyMem_Free(log_perp)
    return tau_h

cdef inline double msld_def_div2N(bint isnc2, double Nd, double n_c_f) nogil:
    """
    """
    if isnc2:
        return 1.0/Nd
    else:
        return 2.0/(Nd*n_c_f)

cdef inline double eval_mean_var_X_lds(double Nd, Py_ssize_t n_components, double* xlds, Py_ssize_t prod_N_nc, double n_c_f, double Nd_1) nogil:
    """
    """
    cdef double mean_var_X_lds = 0.0
    cdef Py_ssize_t i
    for i in range(n_components):
        mean_var_X_lds += var_arr_ptr_step(xlds, prod_N_nc, i, n_components, Nd, Nd_1)
    return mean_var_X_lds/n_c_f

cdef inline void ms_ldprec_nofitU(double* p_h, double* t_h, bint isnc2, Py_ssize_t L, int* K_h, double ihncf, double ihncfexp, double mean_var_X_lds) nogil:
    """
    """
    cdef Py_ssize_t h
    if isnc2:
        for h in range(L):
            p_h[h] = <double> K_h[h]
    else:
        for h in range(L):
            p_h[h] = pow(<double> K_h[h], ihncf)
    cdef double mf = max_arr_ptr(p_h, L)*ihncfexp
    for h in range(L):
        p_h[h] *= mean_var_X_lds
        if p_h[h] < FLOAT64_EPS:
            p_h[h] = FLOAT64_EPS
        p_h[h] = mf/p_h[h]
        if p_h[h] < FLOAT64_EPS:
            t_h[h] = 2.0/FLOAT64_EPS
        else:
            t_h[h] = 2.0/p_h[h]
        if t_h[h] < FLOAT64_EPS:
            t_h[h] = FLOAT64_EPS

cdef inline void ms_ldprec(Py_ssize_t n_components, double Nd, double* xlds, Py_ssize_t prod_N_nc, bint fit_U, Py_ssize_t L, Py_ssize_t N, double** tau_h, int* K_h, double* p_h, double* t_h, int N_1) nogil:
    """
    """
    cdef bint isnc2 = n_components == 2
    cdef double Dhmax, td, mf, ihncf, ihncfexp, n_c_f, mean_var_X_lds
    n_c_f = <double> n_components
    if isnc2:
        ihncf = 1.0
        ihncfexp = 4.0
    else:
        ihncf = 2.0/n_c_f
        ihncfexp = pow(2.0, 1.0+ihncf)
    mean_var_X_lds = eval_mean_var_X_lds(Nd, n_components, xlds, prod_N_nc, n_c_f, <double> N_1)
    cdef Py_ssize_t i, k, h
    if fit_U:
        # Computing the U and storing it in mf
        Dhmax = -DBL_MAX
        for k in range(1, L, 1):
            mf = 0.0
            h = k-1
            for i in range(N):
                td = log2(tau_h[k][i]) - log2(tau_h[h][i])
                if td >= DBL_MIN:
                    mf += 1.0/td
            if mf > Dhmax:
                Dhmax = mf
        mf = Dhmax*msld_def_div2N(isnc2, Nd, n_c_f)
        if mf < 1.0:
            mf = 1.0
        elif mf > 2.0:
            mf = 2.0
        # Computing the LD precisions
        if isnc2:
            for h in range(L):
                p_h[h] = pow(<double> K_h[h], mf)
        else:
            for h in range(L):
                p_h[h] = pow(<double> K_h[h], mf*ihncf)
        mf = max_arr_ptr(p_h, L)*ihncfexp
        for h in range(L):
            p_h[h] *= mean_var_X_lds
            if p_h[h] < FLOAT64_EPS:
                p_h[h] = FLOAT64_EPS
            p_h[h] = mf/p_h[h]
            if p_h[h] < FLOAT64_EPS:
                t_h[h] = 2.0/FLOAT64_EPS
            else:
                t_h[h] = 2.0/p_h[h]
            if t_h[h] < FLOAT64_EPS:
                t_h[h] = FLOAT64_EPS
    else:
        ms_ldprec_nofitU(p_h, t_h, isnc2, L, K_h, ihncf, ihncfexp, mean_var_X_lds)

cdef inline lbfgsfloatval_t* init_lbfgs_var(size_t shdp, int prod_N_nc, double* xlds):
    """
    """
    # Variables for the optimization. We must use lbfgs_malloc to benefitt from SSE2 optimization.
    cdef lbfgsfloatval_t* xopt = lbfgs_malloc(prod_N_nc)
    if xopt is NULL:
        return NULL
    # Initializing the the variables to the current LDS. We can use memcpy as lbfgsfloatval_t is, in our case, strictly equivalent to a double.
    memcpy(xopt, xlds, shdp)
    # Returning
    return xopt

cdef inline double ms_update_mso_step(Py_ssize_t k, Py_ssize_t h, Py_ssize_t N, Py_ssize_t N_1, double** ds_hd, double** tau_h, double** simhd_ms, double** simhd_h) nogil:
    """
    k refers to the number of currently considered scales, between 1 and the number of scales.
    h is the index of the current scale.
    """
    cdef Py_ssize_t i, j
    cdef double kd, ikd
    # Computing the multi-scale similarities for the current multi-scale optimization step
    if k == 1:
        # Computing the similarities at the last scale and storing them in simhd_ms
        for i in range(N):
            sne_hdpinn_nolog(ds_hd[i], tau_h[h][i], N_1, simhd_ms[i])
        return 1.0
    else:
        # Storing the current value of k, in double
        kd = <double> k
        # Inverse of k
        ikd = 1.0/kd
        # Value of kd at the previous step
        kd -= 1.0
        # Computing the similarities at the current scale and updating simhd_ms
        for i in range(N):
            sne_hdpinn_nolog(ds_hd[i], tau_h[h][i], N_1, simhd_h[i])
            for j in range(N_1):
                simhd_ms[i][j] = (kd*simhd_ms[i][j] + simhd_h[i][j])*ikd
        return ikd

cdef struct OpMssne:
    Py_ssize_t ns           # Number of scales which are considered in the current multi-scale optimization step
    Py_ssize_t N            # Number of data points
    Py_ssize_t N_1          # N-1
    Py_ssize_t n_components # Dimension of the LDS
    size_t sstx             # Size, in bytes, of the vector of variables and hence, of the gradient
    double inv_ns           # Inverse of the number of scales
    double** simhd_ms       # Multi-scale HD similarities
    double* p_h             # LD precisions for all scales when fit_U is False
    double* t_h             # 2/p_h
    double* simld_ms        # Memory to store the multi-scale LD similarities with respect to a data point, hence with N-1 elements
    double** simld_h        # Memory to store the LD similarities with respect to a data point at each scale, hence with ns x (N-1) elements

cdef inline lbfgsfloatval_t mssne_evaluate(void* instance, const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int n, const lbfgsfloatval_t step) nogil:
    """
    Computes cost function and gradient for the current LD coordinates.
    See documentation on the web.
    n stores the number of variables
    """
    cdef OpMssne* popt = <OpMssne*> instance
    # Cost function value to return
    cdef lbfgsfloatval_t fx = 0.0
    # Initializing the gradient to 0
    memset(g, 0, popt.sstx)
    # Index variables
    cdef Py_ssize_t i, j, h, k, idx, idxj
    # Intermediate variables
    cdef double a, b, c
    cdef const double* xi
    # Stores the index of the currently considered data point in x
    idx = 0
    # For each data point
    for i in range(popt.N):
        # Currently considered data point
        xi = &x[idx]
        # Computing the LD distances with respect to the other data points and storing them in simld_ms
        idxj = 0
        for j in range(i):
            popt.simld_ms[j] = sqeucl_dist_ptr(xi, &x[idxj], popt.n_components)
            idxj += popt.n_components
        for j in range(i, popt.N_1, 1):
            idxj += popt.n_components
            popt.simld_ms[j] = sqeucl_dist_ptr(xi, &x[idxj], popt.n_components)
        # Minimum distance
        a = min_arr_ptr(popt.simld_ms, popt.N_1)
        # Removing the minimum from the distances and changing the sign
        for j in range(popt.N_1):
            popt.simld_ms[j] = a - popt.simld_ms[j]
        # Computing the LD similarities with respect to the other data points for each scale
        for h in range(popt.ns):
            a = 0.0
            for j in range(popt.N_1):
                popt.simld_h[h][j] = exp(popt.simld_ms[j]/popt.t_h[h])
                a += popt.simld_h[h][j]
            if a < FLOAT64_EPS:
                a = FLOAT64_EPS
            for j in range(popt.N_1):
                popt.simld_h[h][j] /= a
        # Computing the multi-scale LD similarities, updating the cost function evaluation and storing the multi-scale HD similarities divided by the LD ones in popt.simld_ms
        for j in range(popt.N_1):
            popt.simld_ms[j] = 0.0
            for h in range(popt.ns):
                popt.simld_ms[j] += popt.simld_h[h][j]
            popt.simld_ms[j] *= popt.inv_ns
            if popt.simld_ms[j] < FLOAT64_EPS:
                popt.simld_ms[j] = FLOAT64_EPS
            fx -= popt.simhd_ms[i][j] * log(popt.simld_ms[j])
            popt.simld_ms[j] = popt.simhd_ms[i][j]/popt.simld_ms[j]
        # Updating the gradient
        for h in range(popt.ns):
            a = 0.0
            for j in range(popt.N_1):
                a += popt.simld_ms[j]*popt.simld_h[h][j]
            idxj = 0
            for j in range(popt.N_1):
                if j == i:
                    idxj += popt.n_components
                b = popt.p_h[h]*popt.simld_h[h][j]*(popt.simld_ms[j] - a)
                for k in range(popt.n_components):
                    c = b * (xi[k] - x[idxj])
                    g[idx+k] += c
                    g[idxj] -= c
                    idxj += 1
        # Updating idx
        idx += popt.n_components
    # Normalizing the gradient
    for i in range(n):
        g[i] *= popt.inv_ns
    # Returning the cost function value
    return fx

cpdef inline void mssne_implem(double[::1] X_hds, double[::1] X_lds, int N, int d_hds, int n_components, bint fit_U, int nit_max, double gtol, double ftol, int maxls, int maxcor, int L_min):
    """
    Cython implementation of Ms SNE.
    L_min is provided in argument.
    X_hds and X_lds must both be in a 1d array
    """
    # Number of data points in double
    cdef double Nd = <double> N

    #####
    ##### Perplexity-related quantities
    #####

    cdef int K_star = 2
    cdef bint isnotLmin1 = L_min != 1
    # Number of scales
    cdef int L = ms_def_n_scales(Nd, K_star, L_min, not isnotLmin1)

    # Perplexity at each scale
    cdef int* K_h = ms_def_Kh(K_star, isnotLmin1, ms_def_shift_Lmin(isnotLmin1, L_min), L)
    if K_h is NULL:
        printf("Error in mssne_implem function of fmsne_implem.pyx: out of memory for K_h.")
        exit(EXIT_FAILURE)

    #####
    ##### Computing the pairwise HD distances between the data points. The HD distances with respect to each data point are substracted from their minimum, to avoid doing it during the computation of the similarities.
    #####

    # K_star now refers to N-1
    K_star = N-1

    cdef double** ds_hd = sne_ds_hd(&X_hds[0], N, d_hds, K_star)
    if ds_hd is NULL:
        PyMem_Free(K_h)
        printf("Error in mssne_implem function of fmsne_implem.pyx: out of memory for ds_hd.")
        exit(EXIT_FAILURE)

    #####
    ##### Computing the HD bandwidths for all data points and scales
    #####

    # HD bandwidths for each scale and data point. Only stored if fit_U is True.
    cdef double** tau_h = ms_hdsim(ds_hd, N, L, K_h, K_star)
    if tau_h is NULL:
        PyMem_Free(K_h)
        free_dble_2dmat(ds_hd, N)
        printf("Error in mssne_implem function of fmsne_implem.pyx: out of memory in function ms_hdsim.")
        exit(EXIT_FAILURE)

    #####
    ##### Computing the LD precisions
    #####

    # Array storing the LD precisions for each scale when fit_U is False.
    cdef double* p_h = <double*> PyMem_Malloc(L*sizeof(double))
    if p_h is NULL:
        PyMem_Free(K_h)
        free_dble_2dmat(ds_hd, N)
        free_dble_2dmat(tau_h, L)
        printf("Error in mssne_implem function of fmsne_implem.pyx: out of memory for p_h.")
        exit(EXIT_FAILURE)
    # Array storing 2/p_h
    cdef double* t_h = <double*> PyMem_Malloc(L*sizeof(double))
    if t_h is NULL:
        PyMem_Free(K_h)
        free_dble_2dmat(ds_hd, N)
        free_dble_2dmat(tau_h, L)
        PyMem_Free(p_h)
        printf("Error in mssne_implem function of fmsne_implem.pyx: out of memory for t_h.")
        exit(EXIT_FAILURE)
    # Pointer toward the start of the LDS
    cdef double* xlds = &X_lds[0]
    cdef int prod_N_nc = N*n_components
    # Computing the LD precisions
    ms_ldprec(n_components, Nd, xlds, prod_N_nc, fit_U, L, N, tau_h, K_h, p_h, t_h, K_star)

    # Free stuff which will not be used anymore
    PyMem_Free(K_h)

    #####
    ##### Allocating memory to store the HD similarities
    #####

    # Array storing the multi-scale HD similarities, as computed during the multi-scale optimization
    cdef double** simhd_ms = alloc_dble_2dmat(N, K_star)
    if simhd_ms is NULL:
        free_dble_2dmat(ds_hd, N)
        free_dble_2dmat(tau_h, L)
        PyMem_Free(p_h)
        PyMem_Free(t_h)
        printf("Error in mssne_implem function of fmsne_implem.pyx: out of memory for simhd_ms.")
        exit(EXIT_FAILURE)

    # Array storing the HD similarities at some scale h, during the multi-scale optimization
    cdef double** simhd_h = alloc_dble_2dmat(N, K_star)
    if simhd_h is NULL:
        free_dble_2dmat(ds_hd, N)
        free_dble_2dmat(tau_h, L)
        PyMem_Free(p_h)
        PyMem_Free(t_h)
        free_dble_2dmat(simhd_ms, N)
        printf("Error in mssne_implem function of fmsne_implem.pyx: out of memory for simhd_h.")
        exit(EXIT_FAILURE)

    #####
    ##### Multi-scale optimization
    #####

    # Number of bytes of the array for the optimization
    cdef size_t shdp = prod_N_nc*sizeof(double)
    # Variables for the optimization, initialized to the current LDS.
    cdef lbfgsfloatval_t* xopt = init_lbfgs_var(shdp, prod_N_nc, xlds)
    if xopt is NULL:
        free_dble_2dmat(ds_hd, N)
        free_dble_2dmat(tau_h, L)
        PyMem_Free(p_h)
        PyMem_Free(t_h)
        free_dble_2dmat(simhd_ms, N)
        free_dble_2dmat(simhd_h, N)
        printf('Error in function mssne_implem of fmsne_implem.pyx: out of memory for xopt')
        exit(EXIT_FAILURE)

    # Structure gathering the data which are necessary to evaluate the cost function and the gradient
    cdef OpMssne* popt = <OpMssne*> PyMem_Malloc(sizeof(OpMssne))
    if popt is NULL:
        free_dble_2dmat(ds_hd, N)
        free_dble_2dmat(tau_h, L)
        PyMem_Free(p_h)
        PyMem_Free(t_h)
        free_dble_2dmat(simhd_ms, N)
        free_dble_2dmat(simhd_h, N)
        lbfgs_free(xopt)
        printf("Error in function mssne_implem of module cyfastpyx: out of memory for popt")
        exit(EXIT_FAILURE)
    # Filling popt
    popt.N = N
    popt.N_1 = K_star
    popt.n_components = n_components
    popt.sstx = shdp
    popt.simhd_ms = simhd_ms
    popt.simld_ms = <double*> PyMem_Malloc(K_star*sizeof(double))
    if popt.simld_ms is NULL:
        free_dble_2dmat(ds_hd, N)
        free_dble_2dmat(tau_h, L)
        PyMem_Free(p_h)
        PyMem_Free(t_h)
        free_dble_2dmat(simhd_ms, N)
        free_dble_2dmat(simhd_h, N)
        lbfgs_free(xopt)
        PyMem_Free(popt)
        printf("Error in function mssne_implem of module cyfastpyx: out of memory for popt.simld_ms")
        exit(EXIT_FAILURE)
    popt.simld_h = alloc_dble_2dmat(L, K_star)
    if popt.simld_h is NULL:
        free_dble_2dmat(ds_hd, N)
        free_dble_2dmat(tau_h, L)
        PyMem_Free(p_h)
        PyMem_Free(t_h)
        free_dble_2dmat(simhd_ms, N)
        free_dble_2dmat(simhd_h, N)
        lbfgs_free(xopt)
        PyMem_Free(popt.simld_ms)
        PyMem_Free(popt)
        printf("Error in function mssne_implem of module cyfastpyx: out of memory for popt.simld_h")
        exit(EXIT_FAILURE)

    # Parameters of the L-BFGS optimization
    cdef lbfgs_parameter_t param
    cdef lbfgs_parameter_t* pparam = &param
    # Initializing param with default values
    lbfgs_parameter_init(pparam)
    # Updating some parameters
    param.m = maxcor
    param.epsilon = gtol
    param.delta = ftol
    param.max_iterations = nit_max
    param.max_linesearch = maxls
    param.past = 1
    # We modify the default values of the minimum and maximum step sizes of the line search because the problem is badly scaled
    param.max_step = DBL_MAX
    param.min_step = DBL_MIN

    # k refers to the number of currently considered scales and h to the index of the current scale. Nd will store the inverse of the number of currently considered scales.
    cdef Py_ssize_t k, h
    h = L-1
    for k in range(1, L+1, 1):
        # Updates related to the current multi-scale optimization step
        Nd = ms_update_mso_step(k, h, N, K_star, ds_hd, tau_h, simhd_ms, simhd_h)
        # Updating the data structure to evaluate the cost function and the gradient
        popt.ns = k
        popt.inv_ns = Nd
        popt.p_h = &p_h[h]
        popt.t_h = &t_h[h]
        # Performing the optimization
        lbfgs(prod_N_nc, xopt, NULL, mssne_evaluate, NULL, popt, pparam)
        h -= 1
    # Gathering the optimized LD coordinates
    memcpy(xlds, xopt, shdp)

    # Free the ressources
    free_dble_2dmat(ds_hd, N)
    free_dble_2dmat(tau_h, L)
    PyMem_Free(p_h)
    PyMem_Free(t_h)
    free_dble_2dmat(simhd_ms, N)
    free_dble_2dmat(simhd_h, N)
    lbfgs_free(xopt)
    PyMem_Free(popt.simld_ms)
    free_dble_2dmat(popt.simld_h, L)
    PyMem_Free(popt)

#######################################################
####################################################### Multi-scale t-SNE
#######################################################

cdef inline void mstsne_symmetrize(Py_ssize_t N_1, double** a, double** a_sym) nogil:
    """
    Stores a symmetric version of a in the top half of a_sym.
    """
    cdef double tot = 0.0
    cdef Py_ssize_t i, j
    for i in range(N_1):
        for j in range(i, N_1, 1):
            a_sym[i][j] = a[i][j] + a[j+1][i]
            tot += a_sym[i][j]
    tot = 1.0/(2.0*tot)
    for i in range(N_1):
        for j in range(i, N_1, 1):
            a_sym[i][j] *= tot

cdef struct OpMstsne:
    Py_ssize_t N_1          # Number of data points-1
    Py_ssize_t n_components # Dimension of the LDS
    size_t sstx             # Size, in bytes, of the vector of variables and hence, of the gradient
    double** simhd_ms       # Symmetrized multi-scale HD similarities
    double** simld          # Memory to store the LD similarities between all pairs of distinct data points, hence with N x N-1 elements

cdef inline lbfgsfloatval_t mstsne_evaluate(void* instance, const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int n, const lbfgsfloatval_t step) nogil:
    """
    Computes cost function and gradient for the current LD coordinates.
    See documentation on the web.
    n stores the number of variables
    Pay attention to the fact that only the top half of popt.simhd_ms can be used.
    """
    cdef OpMstsne* popt = <OpMstsne*> instance
    # Cost function value to return
    cdef lbfgsfloatval_t fx = 0.0
    # Initializing the gradient to 0
    memset(g, 0, popt.sstx)
    # Index variables
    cdef Py_ssize_t i, j, k, idx, idxj
    # d stores the denominator of the LD similarities, while a and b are an intermediate variables
    cdef double d, a, b
    # Pointer toward the coordinates of the considered data point
    cdef const double* xi
    # Computing the numerators of the LD similarities as well as the denominator
    idx = 0
    d = 0.0
    for i in range(popt.N_1):
        xi = &x[idx]
        idx += popt.n_components
        idxj = idx
        for j in range(i, popt.N_1, 1):
            popt.simld[i][j] = 1.0 + sqeucl_dist_ptr(xi, &x[idxj], popt.n_components)
            k = j+1
            popt.simld[k][i] = 1.0/popt.simld[i][j]
            d += popt.simld[k][i]
            idxj += popt.n_components
    d *= 2.0
    if d < FLOAT64_EPS:
        d = FLOAT64_EPS
    # Stores the index of the currently considered data point in x
    idx = 0
    # For each data point
    for i in range(popt.N_1):
        # Currently considered data point
        xi = &x[idx]
        idxj = idx + popt.n_components
        for j in range(i, popt.N_1, 1):
            # We use the fact that the similarities are symmetric to double fx afterwards.
            fx += log(popt.simld[i][j]) * popt.simhd_ms[i][j]
            k = j+1
            a = (popt.simhd_ms[i][j] - popt.simld[k][i]/d) * popt.simld[k][i]
            for k in range(popt.n_components):
                b = a * (xi[k] - x[idxj])
                g[idx+k] += b
                g[idxj] -= b
                idxj += 1
        # Updating idx
        idx += popt.n_components
    # Scaling the gradient
    for i in range(n):
        g[i] *= 4.0
    # Returning the cost function value
    return fx * 2.0 + log(d)

cpdef inline void mstsne_implem(double[::1] X_hds, double[::1] X_lds, int N, int d_hds, int n_components, int nit_max, double gtol, double ftol, int maxls, int maxcor, int L_min):
    """
    Cython implementation of Ms SNE.
    L_min is provided in argument.
    X_hds and X_lds must both be in a 1d array
    """
    #####
    ##### Perplexity-related quantities
    #####
    cdef int K_star = 2
    cdef bint isnotLmin1 = L_min != 1
    # Number of scales
    cdef int L = ms_def_n_scales(<double> N, K_star, L_min, not isnotLmin1)

    # Perplexity at each scale
    cdef int* K_h = ms_def_Kh(K_star, isnotLmin1, ms_def_shift_Lmin(isnotLmin1, L_min), L)
    if K_h is NULL:
        printf("Error in mstsne_implem function of fmsne_implem.pyx: out of memory for K_h.")
        exit(EXIT_FAILURE)

    #####
    ##### Computing the pairwise HD distances between the data points. The HD distances with respect to each data point are substracted from their minimum, to avoid doing it during the computation of the similarities.
    #####

    # K_star now refers to N-1
    K_star = N-1

    cdef double** ds_hd = sne_ds_hd(&X_hds[0], N, d_hds, K_star)
    if ds_hd is NULL:
        PyMem_Free(K_h)
        printf("Error in mstsne_implem function of fmsne_implem.pyx: out of memory for ds_hd.")
        exit(EXIT_FAILURE)

    #####
    ##### Computing the HD bandwidths for all data points and scales
    #####

    # HD bandwidths for each scale and data point. Only stored if fit_U is True.
    cdef double** tau_h = ms_hdsim(ds_hd, N, L, K_h, K_star)
    if tau_h is NULL:
        free_dble_2dmat(ds_hd, N)
        PyMem_Free(K_h)
        printf("Error in mstsne_implem function of fmsne_implem.pyx: out of memory in function ms_hdsim.")
        exit(EXIT_FAILURE)

    # Free stuff which will not be used anymore
    PyMem_Free(K_h)

    #####
    ##### Allocating memory to store the HD similarities
    #####

    # Array storing the multi-scale HD similarities, as computed during the multi-scale optimization
    cdef double** simhd_ms = alloc_dble_2dmat(N, K_star)
    if simhd_ms is NULL:
        free_dble_2dmat(ds_hd, N)
        free_dble_2dmat(tau_h, L)
        printf("Error in mstsne_implem function of fmsne_implem.pyx: out of memory for simhd_ms.")
        exit(EXIT_FAILURE)

    # Array storing the HD similarities at some scale h, during the multi-scale optimization
    cdef double** simhd_h = alloc_dble_2dmat(N, K_star)
    if simhd_h is NULL:
        free_dble_2dmat(ds_hd, N)
        free_dble_2dmat(tau_h, L)
        free_dble_2dmat(simhd_ms, N)
        printf("Error in mstsne_implem function of fmsne_implem.pyx: out of memory for simhd_h.")
        exit(EXIT_FAILURE)

    #####
    ##### Multi-scale optimization
    #####

    cdef int prod_N_nc = N*n_components
    # Number of bytes of the array for the optimization
    cdef size_t shdp = prod_N_nc*sizeof(double)
    # Pointer toward the start of the LDS
    cdef double* xlds = &X_lds[0]
    # Variables for the optimization, initialized to the current LDS.
    cdef lbfgsfloatval_t* xopt = init_lbfgs_var(shdp, prod_N_nc, xlds)
    if xopt is NULL:
        free_dble_2dmat(ds_hd, N)
        free_dble_2dmat(tau_h, L)
        free_dble_2dmat(simhd_ms, N)
        free_dble_2dmat(simhd_h, N)
        printf('Error in function mstsne_implem of fmsne_implem.pyx: out of memory for xopt')
        exit(EXIT_FAILURE)

    # Structure gathering the data which are necessary to evaluate the cost function and the gradient
    cdef OpMstsne* popt = <OpMstsne*> PyMem_Malloc(sizeof(OpMstsne))
    if popt is NULL:
        free_dble_2dmat(ds_hd, N)
        free_dble_2dmat(tau_h, L)
        free_dble_2dmat(simhd_ms, N)
        free_dble_2dmat(simhd_h, N)
        lbfgs_free(xopt)
        printf("Error in function mstsne_implem of module cyfastpyx: out of memory for popt")
        exit(EXIT_FAILURE)
    # Filling popt
    popt.N_1 = K_star
    popt.n_components = n_components
    popt.sstx = shdp
    # popt.simhd_ms is set to simhd_h since the multi-scale HD similarities are symmetrized
    popt.simhd_ms = simhd_h
    popt.simld = alloc_dble_2dmat(N, K_star)
    if popt.simld is NULL:
        free_dble_2dmat(ds_hd, N)
        free_dble_2dmat(tau_h, L)
        free_dble_2dmat(simhd_ms, N)
        free_dble_2dmat(simhd_h, N)
        lbfgs_free(xopt)
        PyMem_Free(popt)
        printf("Error in function mstsne_implem of module cyfastpyx: out of memory for popt.simld")
        exit(EXIT_FAILURE)

    # Parameters of the L-BFGS optimization
    cdef lbfgs_parameter_t param
    cdef lbfgs_parameter_t* pparam = &param
    # Initializing param with default values
    lbfgs_parameter_init(pparam)
    # Updating some parameters
    param.m = maxcor
    param.epsilon = gtol
    param.delta = ftol
    param.max_iterations = nit_max
    param.max_linesearch = maxls
    param.past = 1
    # We modify the default values of the minimum and maximum step sizes of the line search because the problem is badly scaled
    param.max_step = DBL_MAX
    param.min_step = DBL_MIN

    # k refers to the number of currently considered scales and h to the index of the current scale.
    cdef Py_ssize_t k, h
    h = L-1
    for k in range(1, L+1, 1):
        # Updates related to the current multi-scale optimization step
        ms_update_mso_step(k, h, N, K_star, ds_hd, tau_h, simhd_ms, simhd_h)
        # Symmetrizing the multi-scale HD similarities in simhd_h. Be careful that only the top half of simhd_h contains the symmetric HD similarities.
        mstsne_symmetrize(K_star, simhd_ms, simhd_h)
        # Performing the optimization
        lbfgs(prod_N_nc, xopt, NULL, mstsne_evaluate, NULL, popt, pparam)
        h -= 1
    # Gathering the optimized LD coordinates
    memcpy(xlds, xopt, shdp)

    # Free the ressources
    free_dble_2dmat(ds_hd, N)
    free_dble_2dmat(tau_h, L)
    free_dble_2dmat(simhd_ms, N)
    free_dble_2dmat(simhd_h, N)
    lbfgs_free(xopt)
    free_dble_2dmat(popt.simld, N)
    PyMem_Free(popt)

#######################################################
####################################################### Vantage-point trees
#######################################################

cdef extern from "vptree.h":
    cdef cppclass VpTree:
        VpTree(const double* X, int N, int D)
        void search(const double* x, int k, int* idx)

#######################################################
####################################################### Space-partitioning trees
#######################################################

cdef struct SpNode:
    # Number of points inside the cell
    int npt
    # dim is the dimension of the space-partitioning tree, n_sa is the number of supplementary attributes, n_childs is the number of childs
    Py_ssize_t dim, n_sa, n_childs
    # Pointers toward the center of mass of the cell, its minimum, middle value and maximum bounds along each dimension, and supplementary attributes
    double* cm
    const double* min_ax
    double* mid_ax
    const double* max_ax
    double* suppl_attr
    # Boolean indicating whether there are supplementary attributes, whether they must be copied or not and whether the current cell is a leaf or not
    bint has_sa, copy_up_sa, is_leaf
    # Radius of the current cell
    double radius
    # Pointers toward the childs
    SpNode** childs
    # Pointers enabling to determine which childs are initialized
    bint* has_child

cdef inline SpNode* cinit_SpNode(const double* x, Py_ssize_t dim, const double* min_ax, const double* max_ax, bint has_sa, const double* suppl_attr, Py_ssize_t n_sa):
    """
    Initialize a node of a space partitioning tree
    Exit the program if an error occured.
    """
    cdef SpNode* node = <SpNode*> PyMem_Malloc(sizeof(SpNode))
    if node is NULL:
        printf("Out of memory in cinit_SpNode")
        exit(EXIT_FAILURE)
    # Different notation than C: instead of ->, just use .
    node.npt = 1
    node.dim = dim
    node.cm = <double*> x
    node.min_ax = min_ax
    node.max_ax = max_ax
    node.mid_ax = <double*> PyMem_Malloc(dim*sizeof(double))
    if node.mid_ax is NULL:
        PyMem_Free(node)
        printf("Out of memory in cinit_SpNode")
        exit(EXIT_FAILURE)
    cdef Py_ssize_t i
    cdef double diff
    node.radius = 0.0
    node.n_childs = 1
    for i in range(dim):
        node.n_childs *= 2
        node.mid_ax[i] = 0.5*(min_ax[i] + max_ax[i])
        diff = max_ax[i] - min_ax[i]
        node.radius += diff*diff
    # Space-partitioning trees are working with squared Euclidean distance, squared radius and squared threshold theta. This is why node.radius = sqrt(node.radius) is not performed.
    node.has_sa = has_sa
    if has_sa:
        node.suppl_attr = <double*> suppl_attr
    node.n_sa = n_sa
    node.copy_up_sa = True
    node.is_leaf = True
    return node

cdef inline void addPnt_SpNode(SpNode* node, const double* x, const double* new_suppl_attr):
    """
    """
    # If the current node is a leaf, save the coordinates of its previous center of mass and supplementary attribute. Caution: it is not because the current node is a leaf that it contains only one datum, as several distinct data points may have the same coordinates.
    cdef double* pcm
    cdef double* psa
    cdef size_t sdimd = node.dim*sizeof(double)
    cdef size_t snsad
    if node.has_sa:
        snsad = node.n_sa*sizeof(double)
    if node.is_leaf:
        if node.npt == 1:
            # Keeping track of the original data
            pcm = node.cm
            # Avoiding to modify the original data
            node.cm = <double*> PyMem_Malloc(sdimd)
            if node.cm is NULL:
                printf("Out of memory in addPnt_SpNode")
                exit(EXIT_FAILURE)
            memcpy(node.cm, pcm, sdimd)
            if node.has_sa:
                psa = node.suppl_attr
                node.suppl_attr = <double*> PyMem_Malloc(snsad)
                if node.suppl_attr is NULL:
                    PyMem_Free(node.cm)
                    printf("Out of memory in addPnt_SpNode")
                    exit(EXIT_FAILURE)
                memcpy(node.suppl_attr, psa, snsad)
            else:
                psa = NULL
        else:
            pcm = <double*> PyMem_Malloc(sdimd)
            if pcm is NULL:
                printf("Out of memory in addPnt_SpNode")
                exit(EXIT_FAILURE)
            memcpy(pcm, node.cm, sdimd)
            if node.has_sa:
                psa = <double*> PyMem_Malloc(snsad)
                if psa is NULL:
                    PyMem_Free(pcm)
                    printf('Out of memory in addPnt_SpNode')
                    exit(EXIT_FAILURE)
                memcpy(psa, node.suppl_attr, snsad)
            else:
                psa = NULL
    # Updating the center of mass of the current cell
    cdef int new_npt = node.npt + 1
    cdef double nnpt_d = <double> new_npt
    cdef Py_ssize_t i
    for i in range(node.dim):
        node.cm[i] = (node.cm[i]*node.npt + x[i])/nnpt_d
    # Updating the supplementary attributes of the current cell
    if node.has_sa:
        for i in range(node.n_sa):
            node.suppl_attr[i] = (node.suppl_attr[i]*node.npt + new_suppl_attr[i])/nnpt_d
    # Updating the number of data points that the current cell contains
    node.npt = new_npt

    # A boolean value indicating whether or not we should further dig in the tree to insert x
    cdef bint dig = True

    #### Updating the child nodes
    # Index of the child to process and variable used to determine it
    cdef Py_ssize_t ic, nc
    # Data structures for child nodes
    cdef double* min_ax_c
    cdef double* max_ax_c
    if node.is_leaf:
        # As the current node is a leaf, the cell should be further divided, unless x and the previous center of mass of the cell (pcm) are too close from each other, as it would be nearly impossible to separate x from the other data points of the leaf, since they are all very close to pcm.
        if inf_dist_ptr(pcm, x, node.dim) >= EPSILON_DBL:
            # Mark the node as a non-leaf
            node.is_leaf = False
            # Allocating space for the childs
            node.childs = <SpNode**> PyMem_Malloc(node.n_childs*sizeof(SpNode*))
            if node.childs is NULL:
                printf("Out of memory in addPnt_SpNode")
                exit(EXIT_FAILURE)
            snsad = node.n_childs*sizeof(bint)
            node.has_child = <bint*> PyMem_Malloc(snsad)
            if node.has_child is NULL:
                PyMem_Free(node.childs)
                printf("Out of memory in addPnt_SpNode")
                exit(EXIT_FAILURE)
            memset(node.has_child, False, snsad)
            # Determining the index of the child to process and allocating memory for its underlying data structures
            ic = 0
            nc = 1
            min_ax_c = <double*> PyMem_Malloc(sdimd)
            if min_ax_c is NULL:
                PyMem_Free(node.childs)
                PyMem_Free(node.has_child)
                printf('Out of memory in addPnt_SpNode')
                exit(EXIT_FAILURE)
            max_ax_c = <double*> PyMem_Malloc(sdimd)
            if max_ax_c is NULL:
                PyMem_Free(node.childs)
                PyMem_Free(node.has_child)
                PyMem_Free(min_ax_c)
                printf('Out of memory in addPnt_SpNode')
                exit(EXIT_FAILURE)
            for i in range(node.dim):
                if pcm[i] >= node.mid_ax[i]:
                    ic += nc
                    min_ax_c[i] = node.mid_ax[i]
                    max_ax_c[i] = node.max_ax[i]
                else:
                    min_ax_c[i] = node.min_ax[i]
                    max_ax_c[i] = node.mid_ax[i]
                nc *= 2
            # Initializing the identified children
            node.has_child[ic] = True
            node.childs[ic] = cinit_SpNode(pcm, node.dim, min_ax_c, max_ax_c, node.has_sa, psa, node.n_sa)
            node.childs[ic].npt = node.npt-1
        else:
            # The previous center of mass and x are too close: stop digging the tree.
            dig = False
            if new_npt > 2:
                # Avoiding a Memory leak
                PyMem_Free(pcm)
                if node.has_sa:
                    PyMem_Free(psa)

    # Digging the tree to insert x
    if dig:
        # Determining the index of the child to process
        ic = 0
        nc = 1
        for i in range(node.dim):
            if x[i] >= node.mid_ax[i]:
                ic += nc
            nc *= 2
        # Testing whether the current child already exists or not
        if node.has_child[ic]:
            # Updating the current child
            addPnt_SpNode(node.childs[ic], x, new_suppl_attr)
        else:
            # Allocating memory for the data structures of the child
            min_ax_c = <double*> PyMem_Malloc(sdimd)
            if min_ax_c is NULL:
                printf("Out of memory in addPnt_SpNode")
                exit(EXIT_FAILURE)
            max_ax_c = <double*> PyMem_Malloc(sdimd)
            if max_ax_c is NULL:
                PyMem_Free(min_ax_c)
                printf("Out of memory in addPnt_SpNode")
                exit(EXIT_FAILURE)
            for i in range(node.dim):
                if x[i] >= node.mid_ax[i]:
                    min_ax_c[i] = node.mid_ax[i]
                    max_ax_c[i] = node.max_ax[i]
                else:
                    min_ax_c[i] = node.min_ax[i]
                    max_ax_c[i] = node.mid_ax[i]
            # Creating a new sub-cell containing x-axis
            node.childs[ic] = cinit_SpNode(x, node.dim, min_ax_c, max_ax_c, node.has_sa, new_suppl_attr, node.n_sa)
            # Indicating that the child is initialized
            node.has_child[ic] = True

cdef inline double approxInteractions_SpNode(const SpNode* node, const double* q, double theta, double acc, double* acc_v, double** acc_vv, double*** acc_vvv, int inter_fct, const double* t_h, double** t_h_v, double* qdiff, Py_ssize_t n_v, Py_ssize_t n_vv) nogil:
    """
    theta must be to the square
    """
    # Testing whether q is on node.cm or not
    cdef bint cm_eq_q
    # Index for loops over range(n_v), over range(n_vv) and over node.dim.
    cdef Py_ssize_t i, j, k
    # Integer storing node.npt-1
    cdef int nptm
    # Double storing the opposite of the squared Euclidean distance between node.cm and q, and double storing intermediate computations
    cdef double sqd, z
    sqd = sqeucl_dist_ptr(node.cm, q, node.dim)
    # Testing whether the approximation condition is met or not
    if node.is_leaf or (node.radius < theta * sqd):
        cm_eq_q = (inf_dist_ptr(node.cm, q, node.dim) < EPSILON_DBL)
        # Computing sqd, qdiff and nptm
        if cm_eq_q:
            nptm = node.npt-1
        else:
            sqd = -sqd
            for k in range(node.dim):
                qdiff[k] = q[k] - node.cm[k]
        if inter_fct == 6:
            # In this case, if q is not on the current center of mass, acc_v is updated and t_h_v and node.suppl_attr are considered.
            if not cm_eq_q:
                z = 0.0
                for i in range(n_v):
                    z += exp(sqd/t_h_v[i][n_vv]) * node.suppl_attr[i]
                z *= node.npt
                for k in range(node.dim):
                    acc_v[k] -= z*qdiff[k]
        elif inter_fct == 1:
            # In this case, acc_vv and acc_vvv are updated, and t_h_v is used.
            # Testing whether q is on the current center of mass or not
            if cm_eq_q:
                # As q is on node.cm, only acc_vv must be updated
                for i in range(n_v):
                    for j in range(n_vv):
                        acc_vv[i][j] += nptm
            else:
                # Updating acc_vv
                for i in range(n_v):
                    for j in range(n_vv):
                        z = node.npt * exp(sqd/t_h_v[i][j])
                        acc_vv[i][j] += z
                        # Updating acc_vvv
                        for k in range(node.dim):
                            acc_vvv[i][j][k] += z*qdiff[k]
        elif inter_fct == 7:
            # In this case, acc and acc_v are updated
            # Testing whether q is on the current center of mass or not
            if cm_eq_q:
                acc += nptm
            else:
                z = 1.0/(1.0 - sqd)
                sqd = node.npt * z
                acc += sqd
                z *= sqd
                for k in range(n_v):
                    acc_v[k] -= z*qdiff[k]
        elif inter_fct == 5:
            # In this case, if q is not on the current center of mass, acc_v is updated and t_h and node.suppl_attr are considered.
            if not cm_eq_q:
                z = 0.0
                for i in range(n_v):
                    z += exp(sqd/t_h[i]) * node.suppl_attr[i]
                z *= node.npt
                for k in range(node.dim):
                    acc_v[k] -= z*qdiff[k]
        elif inter_fct == 0:
            # In this case, acc_v and acc_vv are updated, and t_h is used.
            # Testing whether q is on the current center of mass or not
            if cm_eq_q:
                # As q is on node.cm, only acc_v must be updated
                for i in range(n_v):
                    acc_v[i] += nptm
            else:
                # Updating acc_v
                for i in range(n_v):
                    z = node.npt * exp(sqd/t_h[i])
                    acc_v[i] += z
                    # Updating acc_vv
                    for k in range(node.dim):
                        acc_vv[i][k] += z*qdiff[k]
        elif inter_fct == 2:
            # In this case, if q is not on the current center of mass, acc_v is updated and t_h and node.suppl_attr are considered.
            if not cm_eq_q:
                z = 0.0
                for i in range(n_v):
                    z += exp(sqd/t_h[i]) * node.suppl_attr[i]
                z *= node.npt
                for k in range(node.dim):
                    acc_v[k] += z*qdiff[k]
        elif inter_fct == 3:
            # In this case, acc_v is updated, and t_h is used.
            # Testing whether q is on the current center of mass or not
            if cm_eq_q:
                # As q is on node.cm, only acc_v must be updated
                for i in range(n_v):
                    acc_v[i] += nptm
            else:
                # Updating acc_v
                for i in range(n_v):
                    acc_v[i] += node.npt * exp(sqd/t_h[i])
        elif inter_fct == 4:
            # In this case, acc_vv is updated, and t_h_v is used.
            # Testing whether q is on the current center of mass or not
            if cm_eq_q:
                # As q is on node.cm, only acc_vv must be updated
                for i in range(n_v):
                    for j in range(n_vv):
                        acc_vv[i][j] += nptm
            else:
                # Updating acc_vv
                for i in range(n_v):
                    for j in range(n_vv):
                        acc_vv[i][j] += node.npt * exp(sqd/t_h_v[i][j])
    else:
        for i in range(node.n_childs):
            if node.has_child[i]:
                acc = approxInteractions_SpNode(node.childs[i], q, theta, acc, acc_v, acc_vv, acc_vvv, inter_fct, t_h, t_h_v, qdiff, n_v, n_vv)
    return acc

cdef inline void reset_sa_SpNode(SpNode* node):
    """
    Reset the supplementary attributes in the cells where there are more than one data point
    """
    # Resetting the supplementary attribute
    cdef Py_ssize_t i
    cdef size_t snsad = node.n_sa*sizeof(double)
    cdef double* tmp
    if node.npt == 1:
        # Need to copy to avoid modifying the original data
        if node.copy_up_sa and node.is_leaf:
            tmp = node.suppl_attr
            node.suppl_attr = <double*> PyMem_Malloc(snsad)
            if node.suppl_attr is NULL:
                printf("Out of memory in reset_sa_SpNode")
                exit(EXIT_FAILURE)
            memcpy(node.suppl_attr, tmp, snsad)
            # Setting node.copy_up_sa avoids repeatedly copying node.suppl_attr whereas it does not anymore refer to original data
            node.copy_up_sa = False
    else:
        memset(node.suppl_attr, 0, snsad)
    # Resetting the childrens
    if not node.is_leaf:
        for i in range(node.n_childs):
            if node.has_child[i]:
                reset_sa_SpNode(node.childs[i])

cdef inline void update_sa_SpNode(SpNode* node, const double* x, const double* sa, Py_ssize_t n_sa):
    """
    Recursively update the supplementary attributes
    """
    # Updating the supplementary attributes of the current node
    cdef double nptd
    cdef size_t snsad = n_sa*sizeof(double)
    cdef Py_ssize_t i
    cdef bint one_dp = (node.npt == 1)
    if not one_dp:
        nptd = <double> node.npt
    if node.has_sa:
        if one_dp:
            memcpy(node.suppl_attr, sa, snsad)
        else:
            for i in range(n_sa):
                node.suppl_attr[i] += sa[i]/nptd
    else:
        node.has_sa = True
        node.n_sa = n_sa
        if one_dp:
            if node.is_leaf:
                # Caution: remember that a cell may be a leaf while still containing more than one data point (i.e. when they have the same position).
                # No need to copy in this case as this supplementary attribute will not be modified
                node.suppl_attr = <double*> sa
            else:
                # This part should never be reached
                node.suppl_attr = <double*> PyMem_Malloc(snsad)
                if node.suppl_attr is NULL:
                    printf("Out of memory in function update_sa_SpNode.")
                    exit(EXIT_FAILURE)
                memcpy(node.suppl_attr, sa, snsad)
        else:
            node.suppl_attr = <double*> PyMem_Malloc(snsad)
            if node.suppl_attr is NULL:
                printf("Out of memory in function update_sa_SpNode.")
                exit(EXIT_FAILURE)
            memcpy(node.suppl_attr, sa, snsad)
            for i in range(n_sa):
                node.suppl_attr[i] /= nptd
    # Updating the supplementary attributes of the appropriate children
    cdef Py_ssize_t ic, nc
    if not node.is_leaf:
        # Determining the relevant children
        ic = 0
        nc = 1
        for i in range(node.dim):
            if x[i] >= node.mid_ax[i]:
                ic += nc
            nc *= 2
        if node.has_child[ic]:
            update_sa_SpNode(node.childs[ic], x, sa, n_sa)

cdef inline void free_SpNode(SpNode* node):
    """
    Recursively free
    """
    PyMem_Free(<void*> node.min_ax)
    PyMem_Free(node.mid_ax)
    PyMem_Free(<void*> node.max_ax)
    cdef Py_ssize_t i
    if node.is_leaf:
        if node.npt == 1:
            if node.has_sa and (not node.copy_up_sa):
                PyMem_Free(node.suppl_attr)
        else:
            PyMem_Free(node.cm)
            if node.has_sa:
                PyMem_Free(node.suppl_attr)
        # Otherwise do not free original data!
    else:
        for i in range(node.n_childs):
            if node.has_child[i]:
                free_SpNode(node.childs[i])
        PyMem_Free(node.childs)
        PyMem_Free(node.has_child)
        PyMem_Free(node.cm)
        if node.has_sa:
            PyMem_Free(node.suppl_attr)
    PyMem_Free(node)

cdef struct SpTree:
    # Root of the tree
    SpNode* root
    # Integer indicating the interaction function to consider
    int inter_fct
    # Number of points in the data set used to create the quadtree
    Py_ssize_t N

cdef inline SpTree* cinit_SpTree(const double* X, Py_ssize_t N, Py_ssize_t dim, bint has_sa, const double* suppl_attrs, Py_ssize_t n_sa, int inter_fct):
    """
    Initialize a space-partitioning tree.
    The data set is stored as a one-dimensional array. There are N data points of dimension dim. The coordinates of the ith data point are stored at &X[i*D].
    """
    cdef SpTree* tree = <SpTree*> PyMem_Malloc(sizeof(SpTree))
    if tree is NULL:
        printf("Out of memory in cinit_SpTree")
        exit(EXIT_FAILURE)
    tree.N = N
    tree.inter_fct = inter_fct
    # Managing supplementary attributes. Allocation is necessary even when has_sa is False, to avoid segmentation fault. Avoid None to avoid python interactions.
    cdef const double* sa = suppl_attrs
    # Small shift so that the quadtree box striclty contain all the data points
    cdef double sm_shift = 1e-8
    # Allocating memory for the data structures for the root of the tree
    cdef size_t sdimd = dim*sizeof(double)
    cdef double* min_ax = <double*> PyMem_Malloc(sdimd)
    if min_ax is NULL:
        PyMem_Free(tree)
        printf("Out of memory in cinit_SpTree")
        exit(EXIT_FAILURE)
    cdef double* max_ax = <double*> PyMem_Malloc(sdimd)
    if max_ax is NULL:
        PyMem_Free(tree)
        PyMem_Free(min_ax)
        printf("Out of memory in cinit_SpTree")
        exit(EXIT_FAILURE)
    cdef Py_ssize_t i
    cdef Py_ssize_t m = N*dim
    for i in range(dim):
        min_ax[i] = min_arr_ptr_step(X, m, i, dim) - sm_shift
        max_ax[i] = max_arr_ptr_step(X, m, i, dim) + sm_shift
    # Initializing the root of the tree
    tree.root = cinit_SpNode(X, dim, min_ax, max_ax, has_sa, sa, n_sa)
    # Inserting each data point
    i = dim
    cdef Py_ssize_t j
    if has_sa:
        j = n_sa
    while i < m:
        if has_sa:
            sa = &suppl_attrs[j]
            j += n_sa
        addPnt_SpNode(tree.root, &X[i], sa)
        i += dim
    # Returning
    return tree

cdef inline double approxInteractions_SpTree(const SpTree* tree, const double* q, double theta, double* acc_v, double** acc_vv, double*** acc_vvv, const double* t_h, double** t_h_v, double* qdiff, Py_ssize_t n_v, Py_ssize_t n_vv) nogil:
    """
    t_h = parameter for functions.
    We have:
    - t_h_v.shape[0] = t_h.shape[0] = n_v = acc_vv.shape[0] = acc_vvv.shape[0]
    - acc_vvv.shape[2] = q.shape[0] = tree.root.dim
    If tree.inter_fct == 0:
    - acc_vv.shape[1] = tree.root.dim
    If tree.inter_fct == 1:
    - acc_vv.shape[1] = acc_vvv.shape[1] = t_h_v.shape[1] = n_vv
    If tree.inter_fct == 2:
    - acc_v.shape[0] = tree.root.dim
    If tree.inter_fct != 2:
    - acc_v.shape[0] = n_v
    If tree.inter_fct == 7:
    - n_v = tree.root.dim

    use inter_fct = 3 for objective function, = 0 for first pass for gradient and = 2 for second pass for gradient.

    qdiff: must be an array with tree.root.dim elements, which can be used for writing purposes. It avoids repeatedly allocating the array in the function.

    for interaction functions 5 and 6, the acc_v is not set to 0. This allows modifying the gradient directly.
    """
    # Initializing acc
    cdef double acc = 0.0
    # Index for loops over range(n_v), over range(n_vv) and over tree.dim. The values of n_v and n_vv depend on inter_fct.
    cdef Py_ssize_t i, j, k
    cdef size_t sz, szsec
    # Initializing acc_v, acc_vv and acc_vvv to zero, except if the inter_fct is > 4
    if tree.inter_fct < 5:
        if tree.inter_fct == 1:
            sz = n_vv*sizeof(double)
            szsec = tree.root.dim*sizeof(double)
            for i in range(n_v):
                memset(acc_vv[i], 0, sz)
                for j in range(n_vv):
                    memset(acc_vvv[i][j], 0, szsec)
        elif tree.inter_fct == 0:
            memset(acc_v, 0, n_v*sizeof(double))
            sz = tree.root.dim*sizeof(double)
            for i in range(n_v):
                memset(acc_vv[i], 0, sz)
        elif tree.inter_fct == 2:
            memset(acc_v, 0, tree.root.dim*sizeof(double))
        elif tree.inter_fct == 3:
            memset(acc_v, 0, n_v*sizeof(double))
        elif tree.inter_fct == 4:
            sz = n_vv*sizeof(double)
            for i in range(n_v):
                memset(acc_vv[i], 0, sz)
    return approxInteractions_SpNode(tree.root, q, theta, acc, acc_v, acc_vv, acc_vvv, tree.inter_fct, t_h, t_h_v, qdiff, n_v, n_vv)

cdef inline void update_sa_SpTree(SpTree* tree, const double* X, const double* sa, Py_ssize_t n_sa):
    """
    Update the supplementary attributes of the quadtree without building it from scratch.
    sa = new supplementary attributes
    sa has as many rows as X
    """
    # Reset the existing supplementary attributes, if any
    if tree.root.has_sa:
        reset_sa_SpNode(tree.root)
    # Updating the supplementary attributes of the tree
    cdef Py_ssize_t i, j, m
    i = 0
    j = 0
    m = tree.N*tree.root.dim
    while i < m:
        update_sa_SpNode(tree.root, &X[i], &sa[j], n_sa)
        j += n_sa
        i += tree.root.dim

cdef inline void free_SpTree(SpTree* tree):
    """
    Recursively free
    """
    free_SpNode(tree.root)
    PyMem_Free(tree)

#######################################################
####################################################### Fast multi-scale SNE
#######################################################

cdef inline int* f_def_n_ds_h(bint isLmin1, int N, int shift_L_min, double Nd, Py_ssize_t L):
    """
    """
    cdef int* n_ds_h = <int*> PyMem_Malloc(L*sizeof(int))
    if n_ds_h is NULL:
        return NULL
    # Multiplication factor to determine the elements of n_ds_h
    cdef double mf
    if isLmin1:
        mf = 1.0
        n_ds_h[0] = N
    else:
        mf = 1.0/(<double> shift_L_min)
        n_ds_h[0] = <int> round(Nd*mf)
    # Filling n_ds_h
    cdef Py_ssize_t h
    for h in range(1, L, 1):
        mf *= 0.5
        n_ds_h[h] = <int> round(Nd*mf)
    return n_ds_h

cdef inline int* f_def_nnn_h(Py_ssize_t L, int* K_h, int* n_ds_h, bint cperp):
    """
    """
    cdef int* nnn_h = <int*> PyMem_Malloc(L*sizeof(int))
    if nnn_h is NULL:
        return NULL
    cdef Py_ssize_t h
    # Filling nnn_h
    if cperp:
        nnn_h[0] = 3*K_h[0]
        if nnn_h[0] > n_ds_h[0]:
            nnn_h[0] = n_ds_h[0]
        for h in range(1, L, 1):
            if nnn_h[0] > n_ds_h[h]:
                nnn_h[h] = n_ds_h[h]
            else:
                nnn_h[h] = nnn_h[0]
    else:
        for h in range(L):
            nnn_h[h] = 3*K_h[h]
            if nnn_h[h] > n_ds_h[h]:
                nnn_h[h] = n_ds_h[h]
    return nnn_h

cdef inline int f_nnn_tot(int* nnn_h, Py_ssize_t L) nogil:
    """
    """
    # Sum of the elements of nnn_h
    cdef int nnn_tot = 0
    cdef Py_ssize_t h
    for h in range(L):
        nnn_tot += nnn_h[h]
    return nnn_tot

cdef inline bint f_nn_ds_hdprec(int d_hds, int* K_h, int N, Py_ssize_t L, int* n_ds_h, int* all_ind, int* nnn_h, bint isLmin1, double* X_hds, Py_ssize_t n_rs, int*** arr_nn_i_rs, int** nnn_i_rs, double*** ds_nn_i_rs, double*** tau_h_i_rs, int nnn_tot, bint sym_nn_set):
    """
    Return False if everything is ok, True if memory problem.
    """
    # Defining some variables
    cdef Py_ssize_t rs, i, j, nsr, isa, k, last, h, nrs_loop
    cdef VpTree* vpt
    cdef bint build_vp, in_cur_ds
    cdef double* Xhd_cur
    cdef double* x
    cdef int* i_sds
    cdef int nnn, nnn_ub, nnn_cpy
    # Number of bytes of an HD data point
    cdef size_t shdp = d_hds*sizeof(double)
    # Logarithm of the considered perplexity and temporary variable
    cdef double log_perp, min_ds
    cdef bint clogp
    if K_h[0] == 2:
        log_perp = log(2.0)
        clogp = False
    else:
        clogp = True
    # For each scale
    for h in range(L):
        nnn_ub = nnn_h[h]+1
        if nnn_ub > n_ds_h[h]:
            nnn_ub = n_ds_h[h]
        if (h == 0) and isLmin1:
            # Vantage-point tree for the complete data set. No need to use the cython vantage-point tree class: we can directly call the C code! But a del statement (below) is necessary to avoid a memory leak.
            vpt = new VpTree(X_hds, N, d_hds)
            # The vantage-point tree must not be build anymore
            build_vp = False
            # Indicates that the data point for which the neighbors are searched is in the currently considered subsampled data set
            in_cur_ds = True
            # Number of random samplings over which we need to iterate. Only 1 since, in this case, all the random samplings leads to the same results for the first scale.
            nrs_loop = 1
        else:
            # The vantage-point tree must be created
            build_vp = True
            # Number of random samplings over which we need to iterate.
            nrs_loop = n_rs
            # Allocating memory for the subsampled data sets at scale h
            Xhd_cur = <double*> PyMem_Malloc(n_ds_h[h]*d_hds*sizeof(double))
            if Xhd_cur is NULL:
                return True
            # Allocating memory to store the indexes of the data points in the subsampled data set
            i_sds = <int*> PyMem_Malloc(n_ds_h[h]*sizeof(int))
            if i_sds is NULL:
                PyMem_Free(Xhd_cur)
                return True
        # For each random sampling
        for rs in range(nrs_loop):
            # Subsampling the data set and building the vantage-point tree
            if build_vp:
                # Subsampling the data set without replacement
                nsr = N
                j = 0
                for i in range(n_ds_h[h]):
                    isa = rand()%nsr
                    # Storing the sampled index
                    i_sds[j] = all_ind[isa]
                    # Making sure that the further samplings will be made without replacement
                    nsr -= 1
                    if isa != nsr:
                        all_ind[isa] = all_ind[nsr]
                        all_ind[nsr] = i_sds[j]
                    j += 1
                # Sorting i_sds, to be able to check whether the considered data points lie in the subsampled data set
                sort(i_sds, i_sds + n_ds_h[h])
                # Constructing Xhd_cur
                nsr = 0
                for i in range(n_ds_h[h]):
                    isa = i_sds[i]*d_hds
                    memcpy(&Xhd_cur[nsr], &X_hds[isa], shdp)
                    nsr += d_hds
                # Building the vantage-point tree for the subsampled data set. No need to call the cython vantage-point tree class: we can directly call the C code! But a del statement is necessary to avoid a memory leak.
                vpt = new VpTree(Xhd_cur, n_ds_h[h], d_hds)
                # Setting nsr back to 0 as it will be used to check whether the considered data point lie in the subsampled data set
                nsr = 0
            # Searching the nearest neighbors of all data points in the subsampled data set
            for i in range(N):
                # Checking whether the considered data point is in the currently considered subsampled data set
                if build_vp:
                    if (nsr < n_ds_h[h]) and (i == i_sds[nsr]):
                        nsr += 1
                        nnn = nnn_ub
                        in_cur_ds = True
                    else:
                        nnn = nnn_h[h]
                        in_cur_ds = False
                else:
                    # Number of neighbors to search in the vantage-point tree. Need to define it here because nnn is modified in the loop.
                    nnn = nnn_ub
                isa = nnn_i_rs[rs][i]
                # Searching the nnn nearest neighbors of i in vpt
                x = &X_hds[i*d_hds]
                vpt.search(x, nnn, &arr_nn_i_rs[rs][i][isa])
                # Converting the indexes in the range of the full data set instead of the subsampled one
                if build_vp:
                    for j in range(nnn):
                        arr_nn_i_rs[rs][i][isa] = i_sds[arr_nn_i_rs[rs][i][isa]]
                        isa += 1
                # Removing the considered data point from its nearest neighbor if it belongs to the subsampled data set
                if in_cur_ds:
                    isa = nnn_i_rs[rs][i]
                    nnn -= 1
                    for j in range(nnn):
                        if arr_nn_i_rs[rs][i][isa] == i:
                            arr_nn_i_rs[rs][i][isa] = arr_nn_i_rs[rs][i][nnn_i_rs[rs][i]+nnn]
                            break
                        isa += 1

                # Computing the squared euclidean distance between the considered data point and its nearest neighbors
                isa = nnn_i_rs[rs][i]
                min_ds = DBL_MAX
                for j in range(nnn):
                    ds_nn_i_rs[rs][i][isa] = sqeucl_dist_ptr(x, &X_hds[arr_nn_i_rs[rs][i][isa]*d_hds], d_hds)
                    if min_ds > ds_nn_i_rs[rs][i][isa]:
                        min_ds = ds_nn_i_rs[rs][i][isa]
                    isa += 1

                # Substracting the minimum squared distance and changing the sign, to avoid to do it during the computation of the bandwidths
                isa = nnn_i_rs[rs][i]
                for j in range(nnn):
                    ds_nn_i_rs[rs][i][isa] = min_ds - ds_nn_i_rs[rs][i][isa]
                    isa += 1

                # Logarithm of the current perplexity
                if clogp:
                    log_perp = log(<double> min(K_h[0], nnn - 1))

                # Computing the HD bandwith of the similarities at scale h, in random sampling rs and with respect to data point i
                tau_h_i_rs[h][rs][i] = sne_binsearch_bandwidth_fit(&ds_nn_i_rs[rs][i][nnn_i_rs[rs][i]], nnn, log_perp, 1.0)

                # Only adding the new neighbors to arr_nn_i_rs
                k = nnn_i_rs[rs][i]
                nnn_cpy = nnn
                for j in range(nnn_cpy):
                    for isa in range(nnn_i_rs[rs][i]):
                        if arr_nn_i_rs[rs][i][isa] == arr_nn_i_rs[rs][i][k]:
                            nnn -= 1
                            last = nnn_i_rs[rs][i]+nnn
                            if last > k:
                                arr_nn_i_rs[rs][i][k] = arr_nn_i_rs[rs][i][last]
                                ds_nn_i_rs[rs][i][k] = ds_nn_i_rs[rs][i][last]
                            break
                    else:
                        # If no break in inner loop, set the squared distance back to its value and increment k
                        ds_nn_i_rs[rs][i][k] = min_ds - ds_nn_i_rs[rs][i][k]
                        k += 1
                # Updating the number of considered neighbors
                nnn_i_rs[rs][i] += nnn
                # Updating for the other random samplings if they all use the same vantage-point tree at the first scale
                if not build_vp:
                    for isa in range(1, n_rs, 1):
                        tau_h_i_rs[h][isa][i] = tau_h_i_rs[h][rs][i]
                        nnn_i_rs[isa][i] = nnn
                        for j in range(nnn):
                            arr_nn_i_rs[isa][i][j] = arr_nn_i_rs[rs][i][j]
                            ds_nn_i_rs[isa][i][j] = ds_nn_i_rs[rs][i][j]
            if build_vp:
                # Call the destructor of the tree and free the ressources allocated for the object
                del vpt
        if build_vp:
            # Free the memory for the subsampled data set at the current scale
            PyMem_Free(Xhd_cur)
            PyMem_Free(i_sds)
        else:
            # Call the destructor of the tree and free the ressources allocated for the object
            del vpt
    cdef int* n_nnn
    if sym_nn_set:
        # Intermediate variable to store the new number of nearest neighbors for each data point
        shdp = N*sizeof(int)
        n_nnn = <int*> PyMem_Malloc(shdp)
        if n_nnn is NULL:
            return True
        # Symmetrizing the nearest neighbors sets
        for rs in range(n_rs):
            memcpy(n_nnn, nnn_i_rs[rs], shdp)
            # Computing the new number of neighbors to consider for each data point
            for i in range(N):
                for isa in range(nnn_i_rs[rs][i]):
                    j = arr_nn_i_rs[rs][i][isa]
                    for nsr in range(nnn_i_rs[rs][j]):
                        if arr_nn_i_rs[rs][j][nsr] == i:
                            break
                    else:
                        # i is not in the neighbors of j: we must add it
                        n_nnn[j] += 1
            # Updating the memory allocated to arr_nn_i_rs and ds_nn_i_rs according to the new number of neighbors
            for i in range(N):
                if n_nnn[i] < nnn_tot:
                    # Reallocating arr_nn_i_rs to the considered number of neighbors
                    i_sds = <int*> PyMem_Realloc(<void*> arr_nn_i_rs[rs][i], n_nnn[i]*sizeof(int))
                    if i_sds is NULL:
                        PyMem_Free(n_nnn)
                        return True
                    arr_nn_i_rs[rs][i] = i_sds
                    # Reallocating ds_nn_i_rs to the considered number of neighbors
                    Xhd_cur = <double*> PyMem_Realloc(<void*> ds_nn_i_rs[rs][i], n_nnn[i]*sizeof(double))
                    if Xhd_cur is NULL:
                        PyMem_Free(n_nnn)
                        return True
                    ds_nn_i_rs[rs][i] = Xhd_cur
                elif n_nnn[i] > nnn_tot:
                    # Allocating some space to store the new number of neighbors
                    i_sds = <int*> PyMem_Malloc(n_nnn[i]*sizeof(int))
                    if i_sds is NULL:
                        PyMem_Free(n_nnn)
                        return True
                    memcpy(i_sds, arr_nn_i_rs[rs][i], nnn_i_rs[rs][i]*sizeof(int))
                    PyMem_Free(arr_nn_i_rs[rs][i])
                    arr_nn_i_rs[rs][i] = i_sds
                    # Allocating some space to store the distances to the new number of neighbors
                    Xhd_cur = <double*> PyMem_Malloc(n_nnn[i]*sizeof(double))
                    if Xhd_cur is NULL:
                        PyMem_Free(n_nnn)
                        return True
                    memcpy(Xhd_cur, ds_nn_i_rs[rs][i], nnn_i_rs[rs][i]*sizeof(double))
                    PyMem_Free(ds_nn_i_rs[rs][i])
                    ds_nn_i_rs[rs][i] = Xhd_cur
            # Adding the new considered neighbors
            memcpy(n_nnn, nnn_i_rs[rs], shdp)
            for i in range(N):
                for isa in range(nnn_i_rs[rs][i]):
                    j = arr_nn_i_rs[rs][i][isa]
                    for nsr in range(nnn_i_rs[rs][j]):
                        if arr_nn_i_rs[rs][j][nsr] == i:
                            break
                    else:
                        # i is not in the neighbors of j: we must add it
                        arr_nn_i_rs[rs][j][n_nnn[j]] = <int> i
                        ds_nn_i_rs[rs][j][n_nnn[j]] = ds_nn_i_rs[rs][i][isa]
                        n_nnn[j] += 1
            # Substracting the minimum from the squared distances and changing the sign, to avoid doing it when computing the similarities
            for i in range(N):
                min_ds = min_arr_ptr(ds_nn_i_rs[rs][i], n_nnn[i])
                for j in range(n_nnn[i]):
                    ds_nn_i_rs[rs][i][j] = min_ds - ds_nn_i_rs[rs][i][j]
            # Updating nnn_i_rs with the new number of neighbors
            memcpy(nnn_i_rs[rs], n_nnn, shdp)
        PyMem_Free(n_nnn)
    else:
        # Reallocating arr_nn_i_rs and ds_nn_i_rs to reserve the exact amount of memory which is needed, and removing the minimum squared distances and changing their signs, to avoid doing it when computing the HD similarities.
        for rs in range(n_rs):
            for i in range(N):
                if nnn_i_rs[rs][i] < nnn_tot:
                    # Reallocating arr_nn_i_rs to the considered number of neighbors
                    i_sds = <int*> PyMem_Realloc(<void*> arr_nn_i_rs[rs][i], nnn_i_rs[rs][i]*sizeof(int))
                    if i_sds is NULL:
                        return True
                    arr_nn_i_rs[rs][i] = i_sds
                    # Reallocating ds_nn_i_rs to the considered number of neighbors
                    Xhd_cur = <double*> PyMem_Realloc(<void*> ds_nn_i_rs[rs][i], nnn_i_rs[rs][i]*sizeof(double))
                    if Xhd_cur is NULL:
                        return True
                    ds_nn_i_rs[rs][i] = Xhd_cur
                # Substracting the minimum from the squared distances and changing the sign, to avoid doing it when computing the similarities
                min_ds = min_arr_ptr(ds_nn_i_rs[rs][i], nnn_i_rs[rs][i])
                for j in range(nnn_i_rs[rs][i]):
                    ds_nn_i_rs[rs][i][j] = min_ds - ds_nn_i_rs[rs][i][j]
    # Everything ok -> return False
    return False

cdef inline Py_ssize_t** fms_nn_dld_match(Py_ssize_t*** nn_i_rs_id_dld, Py_ssize_t* ni_dld, size_t siz_ni_dld, Py_ssize_t n_rs, Py_ssize_t N_1, Py_ssize_t N, int** nnn_i_rs, int*** arr_nn_i_rs, bint sym_nn, Py_ssize_t n_components):
    """
    sym_nn is True if the neighbor sets are symmetric and False otherwise.
    """
    memset(ni_dld, 0, siz_ni_dld)
    # Index variables
    cdef Py_ssize_t i, rs, j, idj, Mij, mij
    # Storing an upperbound for the number of distances for each i in ni_dld
    if sym_nn:
        for rs in range(n_rs):
            for i in range(N_1):
                for idj in range(nnn_i_rs[rs][i]):
                    if i < arr_nn_i_rs[rs][i][idj]:
                        ni_dld[i] += 1
    else:
        for rs in range(n_rs):
            for i in range(N):
                for idj in range(nnn_i_rs[rs][i]):
                    j = arr_nn_i_rs[rs][i][idj]
                    if i < j:
                        ni_dld[i] += 1
                    else:
                        ni_dld[j] += 1
    # Temporary variable
    cdef Py_ssize_t* tmp
    # Allocating space
    cdef Py_ssize_t** ij_dld = alloc_Pysst_2dmat_varN(N_1, ni_dld)
    if ij_dld is NULL:
        return NULL
    # Filling ij_dld
    if sym_nn:
        for i in range(N_1):
            Mij = 0
            for rs in range(n_rs):
                for idj in range(nnn_i_rs[rs][i]):
                    j = arr_nn_i_rs[rs][i][idj]
                    if i<j:
                        for mij in range(Mij):
                            if ij_dld[i][mij] == j:
                                nn_i_rs_id_dld[rs][i][idj] = mij
                                break
                        else:
                            nn_i_rs_id_dld[rs][i][idj] = Mij
                            ij_dld[i][Mij] = j
                            Mij += 1
                    else:
                        for mij in range(ni_dld[j]):
                            if ij_dld[j][mij] == i:
                                nn_i_rs_id_dld[rs][i][idj] = mij
                                break
                        else:
                            free_Pysst_2dmat(ij_dld, N_1)
                            return NULL
            if Mij < ni_dld[i]:
                ni_dld[i] = Mij
                tmp = <Py_ssize_t*> PyMem_Realloc(<void*> ij_dld[i], Mij*sizeof(Py_ssize_t))
                if tmp is NULL:
                    free_Pysst_2dmat(ij_dld, N_1)
                    return NULL
                ij_dld[i] = tmp
        # Managing the i = N-1 in nn_i_rs_id_dld
        for rs in range(n_rs):
            for idj in range(nnn_i_rs[rs][N_1]):
                i = arr_nn_i_rs[rs][N_1][idj]
                for j in range(ni_dld[i]):
                    if ij_dld[i][j] == N_1:
                        nn_i_rs_id_dld[rs][N_1][idj] = j
                        break
                else:
                    free_Pysst_2dmat(ij_dld, N_1)
                    return NULL
    else:
        memset(ni_dld, 0, siz_ni_dld)
        for rs in range(n_rs):
            # i must range to N since the neighbor sets are not symmetric
            for i in range(N):
                for idj in range(nnn_i_rs[rs][i]):
                    j = arr_nn_i_rs[rs][i][idj]
                    if i<j:
                        Mij = j
                        mij = i
                    else:
                        Mij = i
                        mij = j
                    for j in range(ni_dld[mij]):
                        if ij_dld[mij][j] == Mij:
                            nn_i_rs_id_dld[rs][i][idj] = j
                            break
                    else:
                        nn_i_rs_id_dld[rs][i][idj] = ni_dld[mij]
                        ij_dld[mij][ni_dld[mij]] = Mij
                        ni_dld[mij] += 1
        # Reallocating
        for i in range(N_1):
            tmp = <Py_ssize_t*> PyMem_Realloc(<void*> ij_dld[i], ni_dld[i]*sizeof(Py_ssize_t))
            if tmp is NULL:
                free_Pysst_2dmat(ij_dld, N_1)
                return NULL
            ij_dld[i] = tmp
    # Multiplying the elements of ij_dld by n_components
    for i in range(N_1):
        for j in range(ni_dld[i]):
            ij_dld[i][j] *= n_components
    # Returning
    return ij_dld

cdef inline void f_ldprec(int n_components, double Nd, double* xlds, int prod_N_nc, bint fit_U, Py_ssize_t n_rs, Py_ssize_t L, Py_ssize_t N, double*** tau_h_i_rs, int* K_h, double** p_h_rs, double** t_h_rs, double* p_h, double* t_h, int N_1) nogil:
    """
    """
    cdef bint isnc2 = n_components == 2
    cdef double div2N, Dhmax, td, mf, ihncf, ihncfexp, n_c_f, mean_var_X_lds
    n_c_f = <double> n_components
    if isnc2:
        ihncf = 1.0
        ihncfexp = 4.0
    else:
        ihncf = 2.0/n_c_f
        ihncfexp = pow(2.0, 1.0+ihncf)
    mean_var_X_lds = eval_mean_var_X_lds(Nd, n_components, xlds, prod_N_nc, n_c_f, <double> N_1)
    cdef Py_ssize_t rs, i, k, h, L_1
    if fit_U:
        div2N = msld_def_div2N(isnc2, Nd, n_c_f)
        L_1 = L-1
        for rs in range(n_rs):
            # Computing the U and storing it in mf
            Dhmax = -DBL_MAX
            for h in range(L_1):
                mf = 0.0
                k = h+1
                for i in range(N):
                    td = log2(tau_h_i_rs[k][rs][i]) - log2(tau_h_i_rs[h][rs][i])
                    if td >= DBL_MIN:
                        mf += 1.0/td
                if mf > Dhmax:
                    Dhmax = mf
            mf = Dhmax*div2N
            if mf < 1.0:
                mf = 1.0
            elif mf > 2.0:
                mf = 2.0
            # Computing the LD precisions
            if isnc2:
                for h in range(L):
                    p_h_rs[h][rs] = pow(<double> K_h[h], mf)
            else:
                for h in range(L):
                    p_h_rs[h][rs] = pow(<double> K_h[h], mf*ihncf)
            mf = max_arr2d_col(p_h_rs, L, rs)*ihncfexp
            for h in range(L):
                p_h_rs[h][rs] *= mean_var_X_lds
                if p_h_rs[h][rs] < FLOAT64_EPS:
                    p_h_rs[h][rs] = FLOAT64_EPS
                p_h_rs[h][rs] = mf/p_h_rs[h][rs]
                if p_h_rs[h][rs] < FLOAT64_EPS:
                    t_h_rs[h][rs] = 2.0/FLOAT64_EPS
                else:
                    t_h_rs[h][rs] = 2.0/p_h_rs[h][rs]
                if t_h_rs[h][rs] < FLOAT64_EPS:
                    t_h_rs[h][rs] = FLOAT64_EPS
    else:
        ms_ldprec_nofitU(p_h, t_h, isnc2, L, K_h, ihncf, ihncfexp, mean_var_X_lds)

cdef inline double f_update_mso_step(Py_ssize_t k, Py_ssize_t h, Py_ssize_t n_rs, Py_ssize_t N, int** nnn_i_rs, double*** ds_nn_i_rs, double*** tau_h_i_rs, double*** simhd_ms_nn_i_rs, double*** simhd_h_nn_i_rs) nogil:
    """
    k refers to the number of currently considered scales, between 1 and the number of scales.
    h is the index of the current scale.
    """
    cdef Py_ssize_t rs, i, j
    cdef double kd, ikd
    # Computing the multi-scale similarities for the current multi-scale optimization step
    if k == 1:
        # Computing the similarities at the last scale and storing them in simhd_ms_nn_i_rs
        for rs in range(n_rs):
            for i in range(N):
                sne_hdpinn_nolog(ds_nn_i_rs[rs][i], tau_h_i_rs[h][rs][i], nnn_i_rs[rs][i], simhd_ms_nn_i_rs[rs][i])
        return 1.0
    else:
        # Storing the current value of k, in double
        kd = <double> k
        # Inverse of k
        ikd = 1.0/kd
        # Value of kd at the previous step
        kd -= 1.0
        # Computing the similarities at the current scale and updating simhd_ms_nn_i_rs
        for rs in range(n_rs):
            for i in range(N):
                sne_hdpinn_nolog(ds_nn_i_rs[rs][i], tau_h_i_rs[h][rs][i], nnn_i_rs[rs][i], simhd_h_nn_i_rs[rs][i])
                for j in range(nnn_i_rs[rs][i]):
                    simhd_ms_nn_i_rs[rs][i][j] = (kd*simhd_ms_nn_i_rs[rs][i][j] + simhd_h_nn_i_rs[rs][i][j])*ikd
        return ikd

cdef struct Opfmssne:
    Py_ssize_t ns                   # Number of scales which are considered in the current multi-scale optimization step
    Py_ssize_t N                    # Number of data points
    Py_ssize_t N_1                  # N-1
    Py_ssize_t n_components         # Dimension of the LDS
    size_t sstx                     # Size, in bytes, of the vector of variables and hence, of the gradient
    bint fit_U                      # Whether U is fitted or not in the LD precisions
    Py_ssize_t n_rs                 # Number of random samplings
    double inv_ns                   # Inverse of the number of scales
    double inv_n_rs_f               # Inverse of the number of random samplings
    double inv_nsrs                 # Inverse of the number of scales times the number of random samplings
    double theta_s                  # The square of the threshold parameter for the Barnes-Hut algorithm
    double*** sim_hd_ms             # Multi-scale HD similarities
    int*** arr_nn                   # Index of the considered neighbors for each data point and random sampling
    int** nnn                       # Number of considered neighbors for each data point
    double* p_h                     # LD precisions for all scales when fit_U is False (equal for all random samplings)
    double* t_h                     # 2/p_h
    double** p_h_rs                 # LD precisions for all scales and random samplings when fit_U is True
    double** t_h_rs                 # 2/p_h_rs
    double** sa                     # Suppl. attributes for the tree, when evaluating the gradient, for all random samplings.
    double* Z                       # Den. of the LD sim wrt a data point, at all scales (when fit_U is False)
    double** sX                     # Sum of the LD sim wrt a data point, at all scales (when fit_U is False)
    double** Z_rs                   # Den. of the LD sim wrt a data point, at all scales and rand samplings (when fit_U is True)
    double*** sX_rs                 # Sum of the LD sim wrt a data point, at all scales and rand samplings (when fit_U is True)
    int inter_fct_1                 # First interaction function to employ
    int inter_fct_2                 # Second interaction function to employ
    double* qdiff                   # Array with n_components elements to store intermediate computations when traversing the tree
    size_t sbqd                     # Size in bytes of qdiff
    size_t sbsa                     # Size in bytes of sa[rs], for rs in range(n_rs)
    double* sah                     # Buffer with ns elements to store intermediate computations for the supplementary attributes
    double** dij_ld                 # Buffer to store the LD distances
    Py_ssize_t** ij_dld             # Buffer indicate which Ld distances must be computed in dij_ld
    Py_ssize_t* ni_dld              # Number of LD distances to compute for the first N-1 data points (as the LD distances are symmetric)
    Py_ssize_t*** nn_i_rs_id_dld    # Matching between the neighbors in arr_nn and the LD distances in dij_ld

cdef inline lbfgsfloatval_t fmssne_evaluate(void* instance, const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int n, const lbfgsfloatval_t step):
    """
    Computes cost function and gradient for the current LD coordinates.
    See documentation on the web.
    n stores the number of variables
    """
    cdef Opfmssne* popt = <Opfmssne*> instance
    # Cost function value to return
    cdef lbfgsfloatval_t fx = 0.0
    # Initializing the gradient to 0
    memset(g, 0, popt.sstx)
    # Index variables
    cdef Py_ssize_t i, rs, j, h, k, idx, idxj, idxsa
    # Initializing the supplementary attributes to 0
    for rs in range(popt.n_rs):
        memset(popt.sa[rs], 0, popt.sbsa)
    # Intermediate variable
    cdef const double* xi
    # Computing the necessary LD distances
    idx = 0
    for i in range(popt.N_1):
        xi = &x[idx]
        idx += popt.n_components
        for j in range(popt.ni_dld[i]):
            popt.dij_ld[i][j] = -sqeucl_dist_ptr(xi, &x[popt.ij_dld[i][j]], popt.n_components)
    # Intermediate variables
    cdef double sd, sij, spshij
    cdef const double* xj
    # Creating the space-partitioning tree without supplementary attributes
    cdef SpTree* tree = cinit_SpTree(x, popt.N, popt.n_components, False, popt.sa[0], 0, popt.inter_fct_1)
    # Stores the index of the currently considered data point in x
    idx = 0
    idxsa = 0
    # For each data point
    for i in range(popt.N):
        # Currently considered data point
        xi = &x[idx]
        # Traversing the tree to estimate the denominators of the similarities with respect to i, at all scales and random samplings
        if popt.fit_U:
            # popt.Z_rs, popt.sX_rs and popt.t_h_rs are used. popt.Z and popt.t_h are not used (and are set to NULL)
            approxInteractions_SpTree(tree, xi, popt.theta_s, popt.Z, popt.Z_rs, popt.sX_rs, popt.t_h, popt.t_h_rs, popt.qdiff, popt.ns, popt.n_rs)
            # Normalizing
            for h in range(popt.ns):
                for rs in range(popt.n_rs):
                    if popt.Z_rs[h][rs] < FLOAT64_EPS:
                        popt.Z_rs[h][rs] = FLOAT64_EPS
                    for k in range(popt.n_components):
                        popt.sX_rs[h][rs][k] /= popt.Z_rs[h][rs]
        else:
            # popt.Z, popt.sX and popt.t_h are used. popt.sX_rs and popt.t_h_rs are not used (and are set to NULL)
            approxInteractions_SpTree(tree, xi, popt.theta_s, popt.Z, popt.sX, popt.sX_rs, popt.t_h, popt.t_h_rs, popt.qdiff, popt.ns, 0)
            # Normalizing
            for h in range(popt.ns):
                if popt.Z[h] < FLOAT64_EPS:
                    popt.Z[h] = FLOAT64_EPS
                for k in range(popt.n_components):
                    popt.sX[h][k] /= popt.Z[h]
        # For each random sampling
        for rs in range(popt.n_rs):
            # For each considered neighbor of i
            for j in range(popt.nnn[rs][i]):
                # Currently considered neighbor
                k = popt.arr_nn[rs][i][j]
                idxj = k*popt.n_components
                xj = &x[idxj]
                # Opposite of the squared LD distance between i and j
                if i < k:
                    sd = popt.dij_ld[i][popt.nn_i_rs_id_dld[rs][i][j]]
                else:
                    sd = popt.dij_ld[k][popt.nn_i_rs_id_dld[rs][i][j]]
                # Computing the multi-scale LD similarity between i and j
                sij = 0.0
                spshij = 0.0
                memset(popt.qdiff, 0, popt.sbqd)
                for h in range(popt.ns):
                    if popt.fit_U:
                        popt.sah[h] = exp(sd/popt.t_h_rs[h][rs])/popt.Z_rs[h][rs]
                        sij += popt.sah[h]
                        popt.sah[h] *= popt.p_h_rs[h][rs]
                        for k in range(popt.n_components):
                            popt.qdiff[k] += popt.sah[h]*popt.sX_rs[h][rs][k]
                    else:
                        popt.sah[h] = exp(sd/popt.t_h[h])/popt.Z[h]
                        sij += popt.sah[h]
                        popt.sah[h] *= popt.p_h[h]
                        for k in range(popt.n_components):
                            popt.qdiff[k] += popt.sah[h]*popt.sX[h][k]
                    spshij += popt.sah[h]
                sij *= popt.inv_ns
                if sij < FLOAT64_EPS:
                    sij = FLOAT64_EPS
                # Updating the cost function
                fx -= popt.sim_hd_ms[rs][i][j]*log(sij)
                # Updating the gradient. We use spshij and sd as intermediate variables to store computations.
                sij = popt.sim_hd_ms[rs][i][j]/sij
                spshij *= sij
                for k in range(popt.n_components):
                    sd = spshij*(xi[k]-xj[k])
                    g[idx+k] += (sd - sij*popt.qdiff[k])
                    g[idxj] -= sd
                    idxj += 1
                # Updating the supplementary attributes
                k = idxsa
                for h in range(popt.ns):
                    popt.sa[rs][k] += popt.sah[h]*sij
                    k += 1
            # Normalizing the supplementary attributes by the denominator of the LD similarities at each scale
            k = idxsa
            if popt.fit_U:
                for h in range(popt.ns):
                    popt.sa[rs][k] /= popt.Z_rs[h][rs]
                    k += 1
            else:
                for h in range(popt.ns):
                    popt.sa[rs][k] /= popt.Z[h]
                    k += 1
        # Updating idx and idxsa
        idx += popt.n_components
        idxsa += popt.ns

    # Updating the interaction function in the tree
    tree.inter_fct = popt.inter_fct_2

    # Finishing to update the gradient.
    for rs in range(popt.n_rs):
        # Updating the supplementary attributes in the tree
        update_sa_SpTree(tree, x, popt.sa[rs], popt.ns)
        # For each data point
        idx = 0
        for i in range(popt.N):
            # Updating the currently considered coordinates of the gradient directly
            approxInteractions_SpTree(tree, &x[idx], popt.theta_s, &g[idx], popt.Z_rs, popt.sX_rs, popt.t_h, popt.t_h_rs, popt.qdiff, popt.ns, rs)
            # Updating idx
            idx += popt.n_components

    # Free the ressources allocated for the tree
    free_SpTree(tree)

    # Normalizing the gradient
    for i in range(n):
        g[i] *= popt.inv_nsrs

    # Returning the cost function value
    return fx*popt.inv_n_rs_f

cpdef inline void fmssne_implem(double[::1] X_hds, double[::1] X_lds, int N, int d_hds, int n_components, bint cperp, int n_rs, bint fit_U, double ms_thetha, int nit_max, double gtol, double ftol, int maxls, int maxcor, int L_min, int rseed, bint sym_nn):
    """
    Cython implementation of Fast Multi-scale SNE.
    L_min is provided in argument.
    X_hds and X_lds must both be in a 1d array
    sym_nn: Whether to use symmetric neighbor sets or not
    """
    # Fix the random seed
    srand(rseed)
    # Number of data points in double
    cdef double Nd = <double> N

    #####
    ##### Perplexity-related quantities
    #####

    cdef int K_star = 2
    cdef bint isLmin1 = L_min == 1
    cdef bint isnotLmin1 = not isLmin1
    # Number of scales
    cdef int L = ms_def_n_scales(Nd, K_star, L_min, isLmin1)

    # Just a shift for the perplexity at first scale when L_min != 1
    cdef int sLm_nt = ms_def_shift_Lmin(isnotLmin1, L_min)

    # Perplexity at each scale
    cdef int* K_h = ms_def_Kh(K_star, isnotLmin1, sLm_nt, L)
    if K_h is NULL:
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for K_h.")
        exit(EXIT_FAILURE)

    #####
    ##### Computing the size of the subsampled data set at each scale
    #####

    # Size of the subsampled data set at each scale (except the first scale if L_min==1)
    cdef int* n_ds_h = f_def_n_ds_h(isLmin1, N, sLm_nt, Nd, L)
    if n_ds_h is NULL:
        PyMem_Free(K_h)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for n_ds_h.")
        exit(EXIT_FAILURE)

    #####
    ##### Indexes of all the examples in the data set
    #####

    cdef int* all_ind = seq_1step(N)
    if all_ind is NULL:
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for all_ind.")
        exit(EXIT_FAILURE)

    #####
    ##### Number of neighbors to compute per data point for each scale
    #####

    cdef int* nnn_h = f_def_nnn_h(L, K_h, n_ds_h, cperp)
    if nnn_h is NULL:
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for nnn_h.")
        exit(EXIT_FAILURE)
    # Sum of the elements of nnn_h
    sLm_nt = f_nnn_tot(nnn_h, L)

    #####
    ##### Computing the considered neighbors of each data point, for each scale and random sampling
    #####

    # Allocating memory to store the indexes of the considered neighbors for each data point, for each random sampling. In function f_nn_ds_hdprec, arr_nn_i_rs will be reallocated so that its third dimension may be smaller than sLm_nt.
    cdef int*** arr_nn_i_rs = alloc_int_3dmat(n_rs, N, sLm_nt)
    if arr_nn_i_rs is NULL:
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        PyMem_Free(nnn_h)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for arr_nn_i_rs.")
        exit(EXIT_FAILURE)

    # Allocating memory to store the number of considered neighbors for each data point, for each random sampling
    cdef int** nnn_i_rs = calloc_int_2dmat(n_rs, N)
    if nnn_i_rs is NULL:
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        PyMem_Free(nnn_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for nnn_i_rs.")
        exit(EXIT_FAILURE)

    # Allocating memory to store the squared distances between the considered neighbors and each data point, for each random sampling. In fact, for each random sampling rs, data point i and neighbor j, ds_nn_i_rs[rs][i][j] will contain the minimum squared distance between i and all its neighbors in random sampling rs minus the squared distance between i and j. In function f_nn_ds_hdprec, ds_nn_i_rs will be reallocated so that its third dimension may be smaller than sLm_nt.
    cdef double*** ds_nn_i_rs = alloc_dble_3dmat(n_rs, N, sLm_nt)
    if ds_nn_i_rs is NULL:
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        PyMem_Free(nnn_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for ds_nn_i_rs.")
        exit(EXIT_FAILURE)

    # Allocating memory to store the HD bandwidths for each scale, data point and random sampling
    cdef double*** tau_h_i_rs = alloc_dble_3dmat(L, n_rs, N)
    if tau_h_i_rs is NULL:
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        PyMem_Free(nnn_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for tau_h_i_rs.")
        exit(EXIT_FAILURE)

    # Computing the considered nearest neighbors of each data point for each random sampling and filling arr_nn_i_rs, nnn_i_rs, ds_nn_i_rs and tau_h_i_rs.
    if f_nn_ds_hdprec(d_hds, K_h, N, L, n_ds_h, all_ind, nnn_h, isLmin1, &X_hds[0], n_rs, arr_nn_i_rs, nnn_i_rs, ds_nn_i_rs, tau_h_i_rs, sLm_nt, sym_nn):
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        PyMem_Free(nnn_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory in function f_nn_ds_hdprec.")
        exit(EXIT_FAILURE)

    # Free stuffs which will not be used anymore
    PyMem_Free(n_ds_h)
    PyMem_Free(all_ind)
    PyMem_Free(nnn_h)

    #####
    ##### Data structures to compute the LD distances in the gradient
    #####

    cdef Py_ssize_t*** nn_i_rs_id_dld = alloc_Pysst_3dmat_varK(n_rs, N, nnn_i_rs)
    if nn_i_rs_id_dld is NULL:
        PyMem_Free(K_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        printf('Error in fmssne_implem function of fmsne_implem.pyx: out of memory for nn_i_rs_id_dld')
        exit(EXIT_FAILURE)

    # sLm_nt will now refer to N-1
    sLm_nt = N-1

    cdef size_t shdp = sLm_nt*sizeof(Py_ssize_t)
    cdef Py_ssize_t* ni_dld = <Py_ssize_t*> PyMem_Malloc(shdp)
    if ni_dld is NULL:
        PyMem_Free(K_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
        printf('Error in fmssne_implem function of fmsne_implem.pyx: out of memory for ni_dld')
        exit(EXIT_FAILURE)

    cdef Py_ssize_t** ij_dld = fms_nn_dld_match(nn_i_rs_id_dld, ni_dld, shdp, n_rs, sLm_nt, N, nnn_i_rs, arr_nn_i_rs, sym_nn, n_components)
    if ij_dld is NULL:
        PyMem_Free(K_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
        PyMem_Free(ni_dld)
        printf('Error in fmssne_implem function of fmsne_implem.pyx: out of memory for ij_dld')
        exit(EXIT_FAILURE)

    cdef double** dij_ld = alloc_dble_2dmat_varKpysst(sLm_nt, ni_dld)
    if dij_ld is NULL:
        PyMem_Free(K_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
        PyMem_Free(ni_dld)
        free_Pysst_2dmat(ij_dld, sLm_nt)
        printf('Error in fmssne_implem function of fmsne_implem.pyx: out of memory for dij_ld')
        exit(EXIT_FAILURE)

    #####
    ##### Computing the LD precisions
    #####

    # Array storing the LD precisions for each scale when fit_U is False. They are common to all random samplings.
    cdef double* p_h
    # Array storing 2/p_h
    cdef double* t_h
    # Array storing the LD precisions for each scale and random sampling when fit_U is True.
    cdef double** p_h_rs
    # Array storing 2/p_h_rs
    cdef double** t_h_rs
    if fit_U:
        p_h = NULL
        t_h = NULL
        p_h_rs = alloc_dble_2dmat(L, n_rs)
        if p_h_rs is NULL:
            PyMem_Free(K_h)
            free_int_3dmat(arr_nn_i_rs, n_rs, N)
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
            PyMem_Free(ni_dld)
            free_Pysst_2dmat(ij_dld, sLm_nt)
            free_dble_2dmat(dij_ld, sLm_nt)
            printf('Error in fmssne_implem function of fmsne_implem.pyx: out of memory for p_h_rs')
            exit(EXIT_FAILURE)
        t_h_rs = alloc_dble_2dmat(L, n_rs)
        if t_h_rs is NULL:
            PyMem_Free(K_h)
            free_int_3dmat(arr_nn_i_rs, n_rs, N)
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
            PyMem_Free(ni_dld)
            free_Pysst_2dmat(ij_dld, sLm_nt)
            free_dble_2dmat(dij_ld, sLm_nt)
            free_dble_2dmat(p_h_rs, L)
            printf('Error in fmssne_implem function of fmsne_implem.pyx: out of memory for t_h_rs')
            exit(EXIT_FAILURE)
    else:
        p_h_rs = NULL
        t_h_rs = NULL
        p_h = <double*> PyMem_Malloc(L*sizeof(double))
        if p_h is NULL:
            PyMem_Free(K_h)
            free_int_3dmat(arr_nn_i_rs, n_rs, N)
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
            PyMem_Free(ni_dld)
            free_Pysst_2dmat(ij_dld, sLm_nt)
            free_dble_2dmat(dij_ld, sLm_nt)
            printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for p_h.")
            exit(EXIT_FAILURE)
        t_h = <double*> PyMem_Malloc(L*sizeof(double))
        if t_h is NULL:
            PyMem_Free(K_h)
            free_int_3dmat(arr_nn_i_rs, n_rs, N)
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
            PyMem_Free(ni_dld)
            free_Pysst_2dmat(ij_dld, sLm_nt)
            free_dble_2dmat(dij_ld, sLm_nt)
            PyMem_Free(p_h)
            printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for t_h.")
            exit(EXIT_FAILURE)
    # Pointer toward the start of the LDS
    cdef double* xlds = &X_lds[0]
    cdef int prod_N_nc = N*n_components
    # Computing the LD precisions
    f_ldprec(n_components, Nd, xlds, prod_N_nc, fit_U, n_rs, L, N, tau_h_i_rs, K_h, p_h_rs, t_h_rs, p_h, t_h, sLm_nt)

    # Free stuff which will not be used anymore
    PyMem_Free(K_h)

    #####
    ##### Allocating memory to store the HD similarities
    #####

    # Array storing the multi-scale HD similarities, as computed during the multi-scale optimization
    cdef double*** simhd_ms_nn_i_rs = alloc_dble_3dmat_varK(n_rs, N, nnn_i_rs)
    if simhd_ms_nn_i_rs is NULL:
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
        PyMem_Free(ni_dld)
        free_Pysst_2dmat(ij_dld, sLm_nt)
        free_dble_2dmat(dij_ld, sLm_nt)
        if fit_U:
            free_dble_2dmat(p_h_rs, L)
            free_dble_2dmat(t_h_rs, L)
        else:
            PyMem_Free(p_h)
            PyMem_Free(t_h)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for simhd_ms_nn_i_rs.")
        exit(EXIT_FAILURE)

    # Array storing the HD similarities at some scale h, during the multi-scale optimization
    cdef double*** simhd_h_nn_i_rs = alloc_dble_3dmat_varK(n_rs, N, nnn_i_rs)
    if simhd_h_nn_i_rs is NULL:
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
        PyMem_Free(ni_dld)
        free_Pysst_2dmat(ij_dld, sLm_nt)
        free_dble_2dmat(dij_ld, sLm_nt)
        if fit_U:
            free_dble_2dmat(p_h_rs, L)
            free_dble_2dmat(t_h_rs, L)
        else:
            PyMem_Free(p_h)
            PyMem_Free(t_h)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for simhd_h_nn_i_rs.")
        exit(EXIT_FAILURE)

    #####
    ##### Multi-scale optimization
    #####

    # Number of bytes of the array for the optimization
    shdp = prod_N_nc*sizeof(double)
    # Variables for the optimization, initialized to the current LDS.
    cdef lbfgsfloatval_t* xopt = init_lbfgs_var(shdp, prod_N_nc, xlds)
    if xopt is NULL:
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
        PyMem_Free(ni_dld)
        free_Pysst_2dmat(ij_dld, sLm_nt)
        free_dble_2dmat(dij_ld, sLm_nt)
        if fit_U:
            free_dble_2dmat(p_h_rs, L)
            free_dble_2dmat(t_h_rs, L)
        else:
            PyMem_Free(p_h)
            PyMem_Free(t_h)
        printf('Out of memory for xopt')
        exit(EXIT_FAILURE)

    # Structure gathering the data which are necessary to evaluate the cost function and the gradient
    cdef Opfmssne* popt = <Opfmssne*> PyMem_Malloc(sizeof(Opfmssne))
    if popt is NULL:
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
        PyMem_Free(ni_dld)
        free_Pysst_2dmat(ij_dld, sLm_nt)
        free_dble_2dmat(dij_ld, sLm_nt)
        if fit_U:
            free_dble_2dmat(p_h_rs, L)
            free_dble_2dmat(t_h_rs, L)
        else:
            PyMem_Free(p_h)
            PyMem_Free(t_h)
        lbfgs_free(xopt)
        printf("Out of memory for popt")
        exit(EXIT_FAILURE)
    # Filling popt
    popt.N = N
    popt.N_1 = sLm_nt
    popt.dij_ld = dij_ld
    popt.ij_dld = ij_dld
    popt.ni_dld = ni_dld
    popt.nn_i_rs_id_dld = nn_i_rs_id_dld
    popt.n_components = n_components
    popt.sstx = shdp
    popt.n_rs = n_rs
    popt.inv_n_rs_f = 1.0/(<double> n_rs)
    popt.sim_hd_ms = simhd_ms_nn_i_rs
    popt.arr_nn = arr_nn_i_rs
    popt.nnn = nnn_i_rs
    popt.fit_U = fit_U
    popt.sbsa = 0
    # Space-partitioning trees are working with the squared threshold to save the computation time of computing the square root for the Euclidean distance
    popt.theta_s = ms_thetha*ms_thetha

    # Accumulators to traverse the space-partitioning trees
    if fit_U:
        popt.p_h = NULL
        popt.t_h = NULL
        popt.Z = NULL
        popt.sX = NULL
        popt.Z_rs = alloc_dble_2dmat(L, n_rs)
        if popt.Z_rs is NULL:
            free_int_3dmat(arr_nn_i_rs, n_rs, N)
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
            PyMem_Free(ni_dld)
            free_Pysst_2dmat(ij_dld, sLm_nt)
            free_dble_2dmat(dij_ld, sLm_nt)
            free_dble_2dmat(p_h_rs, L)
            free_dble_2dmat(t_h_rs, L)
            lbfgs_free(xopt)
            PyMem_Free(popt)
            printf("Out of memory for popt.Z_rs")
            exit(EXIT_FAILURE)
        popt.sX_rs = alloc_dble_3dmat(L, n_rs, n_components)
        if popt.sX_rs is NULL:
            free_int_3dmat(arr_nn_i_rs, n_rs, N)
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
            PyMem_Free(ni_dld)
            free_Pysst_2dmat(ij_dld, sLm_nt)
            free_dble_2dmat(dij_ld, sLm_nt)
            free_dble_2dmat(p_h_rs, L)
            free_dble_2dmat(t_h_rs, L)
            free_dble_2dmat(popt.Z_rs, L)
            lbfgs_free(xopt)
            PyMem_Free(popt)
            printf("Out of memory for popt.sX_rs.\n")
            exit(EXIT_FAILURE)
        popt.inter_fct_1 = 1
        popt.inter_fct_2 = 6
    else:
        popt.p_h_rs = NULL
        popt.t_h_rs = NULL
        popt.Z_rs = NULL
        popt.sX_rs = NULL
        popt.Z = <double*> PyMem_Malloc(L*sizeof(double))
        if popt.Z is NULL:
            free_int_3dmat(arr_nn_i_rs, n_rs, N)
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
            PyMem_Free(ni_dld)
            free_Pysst_2dmat(ij_dld, sLm_nt)
            free_dble_2dmat(dij_ld, sLm_nt)
            PyMem_Free(p_h)
            PyMem_Free(t_h)
            lbfgs_free(xopt)
            PyMem_Free(popt)
            printf("Out of memory for popt.Z.\n")
            exit(EXIT_FAILURE)
        popt.sX = alloc_dble_2dmat(L, n_components)
        if popt.sX is NULL:
            free_int_3dmat(arr_nn_i_rs, n_rs, N)
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
            PyMem_Free(ni_dld)
            free_Pysst_2dmat(ij_dld, sLm_nt)
            free_dble_2dmat(dij_ld, sLm_nt)
            PyMem_Free(p_h)
            PyMem_Free(t_h)
            PyMem_Free(popt.Z)
            lbfgs_free(xopt)
            PyMem_Free(popt)
            printf("Out of memory for popt.sX.\n")
            exit(EXIT_FAILURE)
        popt.inter_fct_1 = 0
        popt.inter_fct_2 = 5

    popt.sbqd = n_components*sizeof(double)
    popt.qdiff = <double*> PyMem_Malloc(popt.sbqd)
    if popt.qdiff is NULL:
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
        PyMem_Free(ni_dld)
        free_Pysst_2dmat(ij_dld, sLm_nt)
        free_dble_2dmat(dij_ld, sLm_nt)
        if fit_U:
            free_dble_2dmat(p_h_rs, L)
            free_dble_2dmat(t_h_rs, L)
            free_dble_2dmat(popt.Z_rs, L)
            free_dble_3dmat(popt.sX_rs, L, n_rs)
        else:
            PyMem_Free(p_h)
            PyMem_Free(t_h)
            PyMem_Free(popt.Z)
            free_dble_2dmat(popt.sX, L)
        lbfgs_free(xopt)
        PyMem_Free(popt)
        printf("Out of memory for popt.qdiff.\n")
        exit(EXIT_FAILURE)

    popt.sa = alloc_dble_2dmat(n_rs, L*N)
    if popt.sa is NULL:
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
        PyMem_Free(ni_dld)
        free_Pysst_2dmat(ij_dld, sLm_nt)
        free_dble_2dmat(dij_ld, sLm_nt)
        if fit_U:
            free_dble_2dmat(p_h_rs, L)
            free_dble_2dmat(t_h_rs, L)
            free_dble_2dmat(popt.Z_rs, L)
            free_dble_3dmat(popt.sX_rs, L, n_rs)
        else:
            PyMem_Free(p_h)
            PyMem_Free(t_h)
            PyMem_Free(popt.Z)
            free_dble_2dmat(popt.sX, L)
        lbfgs_free(xopt)
        PyMem_Free(popt.qdiff)
        PyMem_Free(popt)
        printf("Out of memory for popt.sa.\n")
        exit(EXIT_FAILURE)

    popt.sah = <double*> PyMem_Malloc(L*sizeof(double))
    if popt.sah is NULL:
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
        PyMem_Free(ni_dld)
        free_Pysst_2dmat(ij_dld, sLm_nt)
        free_dble_2dmat(dij_ld, sLm_nt)
        if fit_U:
            free_dble_2dmat(p_h_rs, L)
            free_dble_2dmat(t_h_rs, L)
            free_dble_2dmat(popt.Z_rs, L)
            free_dble_3dmat(popt.sX_rs, L, n_rs)
        else:
            PyMem_Free(p_h)
            PyMem_Free(t_h)
            PyMem_Free(popt.Z)
            free_dble_2dmat(popt.sX, L)
        lbfgs_free(xopt)
        PyMem_Free(popt.qdiff)
        free_dble_2dmat(popt.sa, n_rs)
        PyMem_Free(popt)
        printf("Out of memory for popt.sah.\n")
        exit(EXIT_FAILURE)

    # Parameters of the L-BFGS optimization
    cdef lbfgs_parameter_t param
    cdef lbfgs_parameter_t* pparam = &param
    # Initializing param with default values
    lbfgs_parameter_init(pparam)
    # Updating some parameters
    param.m = maxcor
    param.epsilon = gtol
    param.delta = ftol
    param.max_iterations = nit_max
    param.max_linesearch = maxls
    param.past = 1
    # We modify the default values of the minimum and maximum step sizes of the line search because the problem is badly scaled
    param.max_step = DBL_MAX
    param.min_step = DBL_MIN

    # Will update the number of supplementary attributes in the space-partitioning tree, which is augmenting with the number of scales which are considered
    K_star = N*sizeof(double)
    # k refers to the number of currently considered scales and h to the index of the current scale. Nd will store the inverse of the number of currently considered scales.
    cdef Py_ssize_t k, h
    h = L-1
    for k in range(1, L+1, 1):
        # Updates related to the current multi-scale optimization step
        Nd = f_update_mso_step(k, h, n_rs, N, nnn_i_rs, ds_nn_i_rs, tau_h_i_rs, simhd_ms_nn_i_rs, simhd_h_nn_i_rs)
        # Updating the data structure to evaluate the cost function and the gradient
        popt.ns = k
        popt.inv_ns = Nd
        popt.inv_nsrs = Nd*popt.inv_n_rs_f
        if fit_U:
            popt.p_h_rs = &p_h_rs[h]
            popt.t_h_rs = &t_h_rs[h]
        else:
            popt.p_h = &p_h[h]
            popt.t_h = &t_h[h]
        popt.sbsa += K_star
        # Performing the optimization
        lbfgs(prod_N_nc, xopt, NULL, fmssne_evaluate, NULL, popt, pparam)
        h -= 1

    # Gathering the optimized LD coordinates
    memcpy(xlds, xopt, shdp)

    # Free the ressources
    free_int_3dmat(arr_nn_i_rs, n_rs, N)
    free_int_2dmat(nnn_i_rs, n_rs)
    free_dble_3dmat(ds_nn_i_rs, n_rs, N)
    free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
    free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
    free_dble_3dmat(tau_h_i_rs, L, n_rs)
    free_Pysst_3dmat(nn_i_rs_id_dld, n_rs, N)
    PyMem_Free(ni_dld)
    free_Pysst_2dmat(ij_dld, sLm_nt)
    free_dble_2dmat(dij_ld, sLm_nt)
    if fit_U:
        free_dble_2dmat(p_h_rs, L)
        free_dble_2dmat(t_h_rs, L)
        free_dble_2dmat(popt.Z_rs, L)
        free_dble_3dmat(popt.sX_rs, L, n_rs)
    else:
        PyMem_Free(p_h)
        PyMem_Free(t_h)
        PyMem_Free(popt.Z)
        free_dble_2dmat(popt.sX, L)
    lbfgs_free(xopt)
    PyMem_Free(popt.qdiff)
    free_dble_2dmat(popt.sa, n_rs)
    PyMem_Free(popt.sah)
    PyMem_Free(popt)

#######################################################
####################################################### Fast multi-scale t-SNE
#######################################################

cdef inline Py_ssize_t*** fms_sym_nn_match(Py_ssize_t n_rs, Py_ssize_t N_1, int*** arr_nn_i_rs, int** nnn_i_rs, Py_ssize_t n_components):
    """
    This assumes that the nearest neighbor sets are symmetric: if i is in the neighbors of j (ie in arr_nn_i_rs[rs][j]), then j must be in the neighbors of i (ie in arr_nn_i_rs[rs][i]).
    Return NULL if problem.
    """
    cdef Py_ssize_t rs, i, j, idj, k, nnn
    # Temporarily modifying nnn_i_rs
    for rs in range(n_rs):
        for i in range(N_1):
            nnn_i_rs[rs][i] = 4*nnn_i_rs[rs][i] + 2
    # Allocate memory
    cdef Py_ssize_t*** m_nn = alloc_Pysst_3dmat_varK(n_rs, N_1, nnn_i_rs)
    if m_nn is NULL:
        return NULL
    # Setting nnn_i_rs back to its value
    for rs in range(n_rs):
        for i in range(N_1):
            nnn_i_rs[rs][i] = (nnn_i_rs[rs][i] - 2)/4
    # Filling m_nn
    cdef Py_ssize_t* tmp
    for rs in range(n_rs):
        for i in range(N_1):
            nnn = 2
            for idj in range(nnn_i_rs[rs][i]):
                j = arr_nn_i_rs[rs][i][idj]
                if j > i:
                    for k in range(nnn_i_rs[rs][j]):
                        if arr_nn_i_rs[rs][j][k] == i:
                            m_nn[rs][i][nnn] = idj
                            nnn += 1
                            m_nn[rs][i][nnn] = j*n_components
                            nnn += 1
                            m_nn[rs][i][nnn] = k
                            nnn += 1
                            m_nn[rs][i][nnn] = j
                            nnn += 1
                            break
                    else:
                        free_Pysst_3dmat(m_nn, n_rs, N_1)
                        return NULL
            m_nn[rs][i][0] = nnn
            # m_nn[rs][i][1] = number of j > i in the neighbors of i in random sampling rs
            m_nn[rs][i][1] = (nnn-2)/4
            if nnn < nnn_i_rs[rs][i]:
                tmp = <Py_ssize_t*> PyMem_Realloc(<void*> m_nn[rs][i], nnn*sizeof(Py_ssize_t))
                if tmp is NULL:
                    free_Pysst_3dmat(m_nn, n_rs, N_1)
                    return NULL
                m_nn[rs][i] = tmp
    return m_nn

cdef inline Py_ssize_t** gather_nn_all_rs(Py_ssize_t* nnn_all_rs, Py_ssize_t n_rs, Py_ssize_t N_1, Py_ssize_t*** m_nn):
    """
    """
    # Will store the indexes of the neighbors j of i over all random samplings, such that j>i
    cdef Py_ssize_t** inn_all_rs = <Py_ssize_t**> PyMem_Malloc(N_1*sizeof(Py_ssize_t*))
    if inn_all_rs is NULL:
        return NULL
    cdef Py_ssize_t* tmp
    cdef Py_ssize_t i, rs, j, k, n, nel
    for i in range(N_1):
        # Counting the total number of neighbors across all random samplings (possibly with duplicates)
        nel = 0
        for rs in range(n_rs):
            nel += m_nn[rs][i][1]
        # Allocate memory
        inn_all_rs[i] = <Py_ssize_t*> PyMem_Malloc(nel*sizeof(Py_ssize_t))
        if inn_all_rs[i] is NULL:
            free_Pysst_2dmat(inn_all_rs, i)
            return NULL
        # Add the neighbors of the first random sampling
        nnn_all_rs[i] = m_nn[0][i][1]
        k = 0
        for j in range(3, m_nn[0][i][0], 4):
            inn_all_rs[i][k] = m_nn[0][i][j]
            k += 1
        # Adding the neighbors of the other random samplings
        for rs in range(1, n_rs, 1):
            n = nnn_all_rs[i]
            for j in range(3, m_nn[rs][i][0], 4):
                for k in range(nnn_all_rs[i]):
                    if inn_all_rs[i][k] == m_nn[rs][i][j]:
                        break
                else:
                    inn_all_rs[i][n] = m_nn[rs][i][j]
                    n += 1
            nnn_all_rs[i] = n
        # Reallocating inn_all_rs[i]
        if nnn_all_rs[i] < nel:
            tmp = <Py_ssize_t*> PyMem_Realloc(<void*> inn_all_rs[i], nnn_all_rs[i]*sizeof(Py_ssize_t))
            if tmp is NULL:
                free_Pysst_2dmat(inn_all_rs, i)
                return NULL
            inn_all_rs[i] = tmp
    return inn_all_rs

cdef inline Py_ssize_t*** fms_nn_rs_match_all_rs(Py_ssize_t n_rs, Py_ssize_t N_1, Py_ssize_t** inn_all_rs, Py_ssize_t* nnn_all_rs, Py_ssize_t*** m_nn):
    """
    """
    cdef Py_ssize_t*** idnn_in_ars = alloc_Pysst_3dmat_varK_3dK(n_rs, N_1, m_nn, 1)
    if idnn_in_ars is NULL:
        return NULL
    cdef Py_ssize_t rs, i, j, k, n
    for rs in range(n_rs):
        for i in range(N_1):
            k = 0
            for j in range(3, m_nn[rs][i][0], 4):
                for n in range(nnn_all_rs[i]):
                    if inn_all_rs[i][n] == m_nn[rs][i][j]:
                        idnn_in_ars[rs][i][k] = n
                        k += 1
                        break
                else:
                    free_Pysst_3dmat(idnn_in_ars, n_rs, N_1)
                    return NULL
    return idnn_in_ars

cdef inline void fmstsne_symmetrize(Py_ssize_t n_rs, Py_ssize_t N_1, double*** sim_hd, Py_ssize_t*** m_nn, double*** sim_hd_sym) nogil:
    """
    This assumes that the nearest neighbor sets are symmetric: if i is in the neighbors of j (ie in arr_nn_i_rs[rs][j]), then j must be in the neighbors of i (ie in arr_nn_i_rs[rs][i]).
    Be careful that only the similarities between i and j such that j>i are actually symmetrized, since only these are used in the evaluate function.
    """
    cdef double tot
    cdef Py_ssize_t rs, i, idj, inn
    for rs in range(n_rs):
        tot = 0.0
        for i in range(N_1):
            for inn in range(2, m_nn[rs][i][0], 4):
                idj = m_nn[rs][i][inn]
                sim_hd_sym[rs][i][idj] = sim_hd[rs][i][idj] + sim_hd[rs][m_nn[rs][i][inn+3]][m_nn[rs][i][inn+2]]
                tot += sim_hd_sym[rs][i][idj]
        tot = 1.0/(2.0*tot)
        for i in range(N_1):
            for inn in range(2, m_nn[rs][i][0], 4):
                sim_hd_sym[rs][i][m_nn[rs][i][inn]] *= tot

cdef struct Opfmstsne:
    Py_ssize_t N                # Number of data points
    Py_ssize_t N_1              # N-1
    Py_ssize_t n_components     # Dimension of the LDS
    size_t sstx                 # Size, in bytes, of the vector of variables and hence, of the gradient
    Py_ssize_t n_rs             # Number of random samplings
    bint n_rs_geq1              # n_rs > 1
    double n_rs_f               # Number of random samplings in double
    double inv_n_rs_2f          # 2.0/n_rs_f
    double inv_n_rs_4f          # 4.0/n_rs_f
    double theta_s              # The square of the threshold parameter for the Barnes-Hut algorithm
    double*** sim_hd_ms         # Multi-scale HD similarities
    Py_ssize_t*** m_nn          # As returned by fms_sym_nn_match
    Py_ssize_t* nnn_all_rs      # nnn_all_rs[i] contains the total number of neighbors considered for i over all random samplings.
    Py_ssize_t** inn_all_rs     # inn_all_rs[i] contains the neighbors considered for i over all random samplings.
    double* dsld_all_rs         # Allows to store the squared LD distances between a data point and its considered neighbors across all random samplings.
    Py_ssize_t*** idnn_in_ars   # idnn_in_ars[rs][i][j] contains the index of the distance between i and its jth neighbor in random sampling rs in dsld_all_rs.
    int inter_fct               # Interaction function to employ in the space-partitioning tree
    double* qdiff               # Array with n_components elements to store intermediate computations when traversing the tree

cdef inline lbfgsfloatval_t fmstsne_evaluate(void* instance, const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int n, const lbfgsfloatval_t step):
    """
    Computes cost function and gradient for the current LD coordinates.
    See documentation on the web.
    n stores the number of variables
    We exploit the fact that the nearest neighbor sets are symmetric.
    """
    cdef Opfmstsne* popt = <Opfmstsne*> instance
    # Initializing the gradient to 0
    memset(g, 0, popt.sstx)
    # Creating the space-partitioning tree without supplementary attributes
    cdef SpTree* tree = cinit_SpTree(x, popt.N, popt.n_components, False, NULL, 0, popt.inter_fct)
    # Index variables
    cdef Py_ssize_t i, rs, idx, k, inn, j, idj
    # Intermediate variables. Z will store the denominator of the LD similarities, as computed by the Barnes-Hut algorithm
    cdef double Z, cfx, a
    Z = 0.0
    # Stores the index of the currently considered data point in x
    idx = 0
    # For each data point
    for i in range(popt.N):
        Z += approxInteractions_SpTree(tree, &x[idx], popt.theta_s, &g[idx], NULL, NULL, NULL, NULL, popt.qdiff, popt.n_components, 0)
        idx += popt.n_components
    # Free the ressources allocated for the tree
    free_SpTree(tree)
    # Check whether Z is not too small
    if Z < FLOAT64_EPS:
        Z = FLOAT64_EPS
    cfx = log(Z)
    Z = popt.n_rs_f/Z
    # Normalizing the repulsive forces by Z
    for i in range(n):
        g[i] *= Z
    # Pointer toward the considered data point
    cdef const double* xi
    # Cost function value to return
    cdef lbfgsfloatval_t fx = 0.0
    # For each data point
    idx = 0
    if popt.n_rs_geq1:
        for i in range(popt.N_1):
            # Currently considered data point
            xi = &x[idx]
            # Computing the LD distances which are needed
            for k in range(popt.nnn_all_rs[i]):
                # Storing the distance
                popt.dsld_all_rs[k] = 1.0 + sqeucl_dist_ptr(xi, &x[popt.inn_all_rs[i][k]], popt.n_components)
            # For each random sampling
            for rs in range(popt.n_rs):
                k = 0
                for inn in range(2, popt.m_nn[rs][i][0], 4):
                    a = popt.dsld_all_rs[popt.idnn_in_ars[rs][i][k]]
                    Z = popt.sim_hd_ms[rs][i][popt.m_nn[rs][i][inn]]
                    # Updating the cost function
                    fx += Z * log(a)
                    # Updating the gradient
                    Z /= a
                    idj = popt.m_nn[rs][i][inn+1]
                    for j in range(popt.n_components):
                        a = Z * (xi[j] - x[idj])
                        g[idx+j] += a
                        g[idj] -= a
                        idj += 1
                    k += 1
            idx += popt.n_components
    else:
        for i in range(popt.N_1):
            # Currently considered data point
            xi = &x[idx]
            for inn in range(2, popt.m_nn[0][i][0], 4):
                idj = popt.m_nn[0][i][inn+1]
                a = 1.0 + sqeucl_dist_ptr(xi, &x[idj], popt.n_components)
                Z = popt.sim_hd_ms[0][i][popt.m_nn[0][i][inn]]
                # Updating the cost function
                fx += Z * log(a)
                # Updating the gradient
                Z /= a
                for j in range(popt.n_components):
                    a = Z * (xi[j] - x[idj])
                    g[idx+j] += a
                    g[idj] -= a
                    idj += 1
            idx += popt.n_components
    # Normalizing the gradient
    for i in range(n):
        g[i] *= popt.inv_n_rs_4f
    # Returning the cost function value
    return fx*popt.inv_n_rs_2f + cfx

cpdef inline void fmstsne_implem(double[::1] X_hds, double[::1] X_lds, int N, int d_hds, int n_components, bint cperp, int n_rs, double ms_thetha, int nit_max, double gtol, double ftol, int maxls, int maxcor, int L_min, int rseed):
    """
    Cython implementation of FMs t-SNE.
    L_min is provided in argument.
    X_hds and X_lds must both be in a 1d array
    """
    # Fix the random seed
    srand(rseed)
    # Number of data points in double
    cdef double Nd = <double> N

    #####
    ##### Perplexity-related quantities
    #####

    cdef int K_star = 2
    cdef bint isLmin1 = L_min == 1
    cdef bint isnotLmin1 = not isLmin1
    # Number of scales
    cdef int L = ms_def_n_scales(Nd, K_star, L_min, isLmin1)

    # Just a shift for the perplexity at first scale when L_min != 1
    cdef int sLm_nt = ms_def_shift_Lmin(isnotLmin1, L_min)

    # Perplexity at each scale
    cdef int* K_h = ms_def_Kh(K_star, isnotLmin1, sLm_nt, L)
    if K_h is NULL:
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for K_h.")
        exit(EXIT_FAILURE)

    #####
    ##### Computing the size of the subsampled data set at each scale
    #####

    # Size of the subsampled data set at each scale (except the first scale if L_min==1)
    cdef int* n_ds_h = f_def_n_ds_h(isLmin1, N, sLm_nt, Nd, L)
    if n_ds_h is NULL:
        PyMem_Free(K_h)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for n_ds_h.")
        exit(EXIT_FAILURE)

    #####
    ##### Indexes of all the examples in the data set
    #####

    cdef int* all_ind = seq_1step(N)
    if all_ind is NULL:
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for all_ind.")
        exit(EXIT_FAILURE)

    #####
    ##### Number of neighbors to compute per data point for each scale
    #####

    cdef int* nnn_h = f_def_nnn_h(L, K_h, n_ds_h, cperp)
    if nnn_h is NULL:
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for nnn_h.")
        exit(EXIT_FAILURE)
    # Sum of the elements of nnn_h
    sLm_nt = f_nnn_tot(nnn_h, L)

    #####
    ##### Computing the considered neighbors of each data point, for each scale and random sampling
    #####

    # Allocating memory to store the indexes of the considered neighbors for each data point, for each random sampling. In function f_nn_ds_hdprec, arr_nn_i_rs will be reallocated so that its third dimension may be smaller than sLm_nt.
    cdef int*** arr_nn_i_rs = alloc_int_3dmat(n_rs, N, sLm_nt)
    if arr_nn_i_rs is NULL:
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        PyMem_Free(nnn_h)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for arr_nn_i_rs.")
        exit(EXIT_FAILURE)

    # Allocating memory to store the number of considered neighbors for each data point, for each random sampling
    cdef int** nnn_i_rs = calloc_int_2dmat(n_rs, N)
    if nnn_i_rs is NULL:
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        PyMem_Free(nnn_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for nnn_i_rs.")
        exit(EXIT_FAILURE)

    # Allocating memory to store the squared distances between the considered neighbors and each data point, for each random sampling. In fact, for each random sampling rs, data point i and neighbor j, ds_nn_i_rs[rs][i][j] will contain the minimum squared distance between i and all its neighbors in random sampling rs minus the squared distance between i and j. In function f_nn_ds_hdprec, ds_nn_i_rs will be reallocated so that its third dimension may be smaller than sLm_nt.
    cdef double*** ds_nn_i_rs = alloc_dble_3dmat(n_rs, N, sLm_nt)
    if ds_nn_i_rs is NULL:
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        PyMem_Free(nnn_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for ds_nn_i_rs.")
        exit(EXIT_FAILURE)

    # Allocating memory to store the HD bandwidths for each scale, data point and random sampling
    cdef double*** tau_h_i_rs = alloc_dble_3dmat(L, n_rs, N)
    if tau_h_i_rs is NULL:
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        PyMem_Free(nnn_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for tau_h_i_rs.")
        exit(EXIT_FAILURE)

    # Computing the considered nearest neighbors of each data point for each random sampling and filling arr_nn_i_rs, nnn_i_rs, ds_nn_i_rs and tau_h_i_rs. The considered nearest neighbors of each data point are also symmetrized for each random sampling (i.e. if i is in the considered nearest neighbors of j, than j must also be in the considered nearest neighbors of i).
    if f_nn_ds_hdprec(d_hds, K_h, N, L, n_ds_h, all_ind, nnn_h, isLmin1, &X_hds[0], n_rs, arr_nn_i_rs, nnn_i_rs, ds_nn_i_rs, tau_h_i_rs, sLm_nt, True):
        PyMem_Free(K_h)
        PyMem_Free(n_ds_h)
        PyMem_Free(all_ind)
        PyMem_Free(nnn_h)
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory in function f_nn_ds_hdprec.")
        exit(EXIT_FAILURE)

    # Free stuffs which will not be used anymore
    PyMem_Free(K_h)
    PyMem_Free(n_ds_h)
    PyMem_Free(all_ind)
    PyMem_Free(nnn_h)

    #####
    ##### Data structure facilitating the symmetrization of the HD similarities
    #####

    # sLm_nt now refers to N-1
    sLm_nt = N-1
    cdef Py_ssize_t*** m_nn = fms_sym_nn_match(n_rs, sLm_nt, arr_nn_i_rs, nnn_i_rs, n_components)
    if m_nn is NULL:
        free_int_3dmat(arr_nn_i_rs, n_rs, N)
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        printf('Error in function fmstsne_implem of module fmsne_implem.pyx: out of memory in function fms_sym_nn_match.')
        exit(EXIT_FAILURE)

    # Free resources which are not needed anymore
    free_int_3dmat(arr_nn_i_rs, n_rs, N)

    #####
    ##### Allocating memory to store the HD similarities
    #####

    # Array storing the multi-scale HD similarities, as computed during the multi-scale optimization
    cdef double*** simhd_ms_nn_i_rs = alloc_dble_3dmat_varK(n_rs, N, nnn_i_rs)
    if simhd_ms_nn_i_rs is NULL:
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for simhd_ms_nn_i_rs.")
        exit(EXIT_FAILURE)

    # Array storing the HD similarities at some scale h, during the multi-scale optimization
    cdef double*** simhd_h_nn_i_rs = alloc_dble_3dmat_varK(n_rs, N, nnn_i_rs)
    if simhd_h_nn_i_rs is NULL:
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
        printf("Error in fmssne_implem function of fmsne_implem.pyx: out of memory for simhd_h_nn_i_rs.")
        exit(EXIT_FAILURE)

    #####
    ##### Data structures to compute the LD distances when evaluating the cost function and its gradient
    #####

    # isLmin1 now refers to n_rs > 1
    isLmin1 = n_rs > 1

    # nnn_all_rs[i] will contain the total number of neighbors considered for i over all random samplings.
    cdef Py_ssize_t* nnn_all_rs
    # inn_all_rs[i] will contain the neighbors considered for i over all random samplings.
    cdef Py_ssize_t** inn_all_rs
    # dsld_all_rs will allow to store the squared LD distances between a data point and its neighbors over all random samplings.
    cdef double* dsld_all_rs
    # idnn_in_ars[rs][i][j] contains the index of the distance between i and its jth neighbor in random sampling rs in dsld_all_rs.
    cdef Py_ssize_t*** idnn_in_ars
    if isLmin1:
        # nnn_all_rs will be filled in function gather_nn_all_rs
        nnn_all_rs =  <Py_ssize_t*> PyMem_Malloc(sLm_nt*sizeof(Py_ssize_t))
        if nnn_all_rs is NULL:
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
            printf('Error in function fmstsne_implem of module fmsne_implem.pyx: out of memory for nnn_all_rs.')
            exit(EXIT_FAILURE)
        inn_all_rs = gather_nn_all_rs(nnn_all_rs, n_rs, sLm_nt, m_nn)
        if inn_all_rs is NULL:
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
            PyMem_Free(nnn_all_rs)
            printf('Error in function fmstsne_implem of module fmsne_implem.pyx: out of memory in function gather_nn_all_rs.')
            exit(EXIT_FAILURE)
        dsld_all_rs = <double*> PyMem_Malloc(max_arr_ptr_Pysst(nnn_all_rs, sLm_nt)*sizeof(double))
        if dsld_all_rs is NULL:
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
            PyMem_Free(nnn_all_rs)
            free_Pysst_2dmat(inn_all_rs, sLm_nt)
            printf('Error in function fmstsne_implem of module fmsne_implem.pyx: out of memory for dsld_all_rs.')
            exit(EXIT_FAILURE)
        idnn_in_ars = fms_nn_rs_match_all_rs(n_rs, sLm_nt, inn_all_rs, nnn_all_rs, m_nn)
        if idnn_in_ars is NULL:
            free_int_2dmat(nnn_i_rs, n_rs)
            free_dble_3dmat(ds_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
            free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
            free_dble_3dmat(tau_h_i_rs, L, n_rs)
            free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
            PyMem_Free(nnn_all_rs)
            free_Pysst_2dmat(inn_all_rs, sLm_nt)
            PyMem_Free(dsld_all_rs)
            printf('Error in function fmstsne_implem of module fmsne_implem.pyx: out of memory in function fms_nn_rs_match_all_rs.')
            exit(EXIT_FAILURE)
    else:
        nnn_all_rs = NULL
        inn_all_rs = NULL
        dsld_all_rs = NULL
        idnn_in_ars = NULL

    #####
    ##### Multi-scale optimization
    #####

    # Pointer toward the start of the LDS
    cdef double* xlds = &X_lds[0]
    cdef int prod_N_nc = N*n_components
    # Number of bytes of the array for the optimization
    cdef size_t shdp = prod_N_nc*sizeof(double)
    # Variables for the optimization, initialized to the current LDS.
    cdef lbfgsfloatval_t* xopt = init_lbfgs_var(shdp, prod_N_nc, xlds)
    if xopt is NULL:
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
        if isLmin1:
            PyMem_Free(nnn_all_rs)
            free_Pysst_2dmat(inn_all_rs, sLm_nt)
            PyMem_Free(dsld_all_rs)
            free_Pysst_3dmat(idnn_in_ars, n_rs, sLm_nt)
        printf('Out of memory for xopt')
        exit(EXIT_FAILURE)

    # Structure gathering the data which are necessary to evaluate the cost function and the gradient
    cdef Opfmstsne* popt = <Opfmstsne*> PyMem_Malloc(sizeof(Opfmstsne))
    if popt is NULL:
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
        if isLmin1:
            PyMem_Free(nnn_all_rs)
            free_Pysst_2dmat(inn_all_rs, sLm_nt)
            PyMem_Free(dsld_all_rs)
            free_Pysst_3dmat(idnn_in_ars, n_rs, sLm_nt)
        lbfgs_free(xopt)
        printf("Out of memory for popt")
        exit(EXIT_FAILURE)
    # Filling popt
    popt.N = N
    popt.N_1 = sLm_nt
    popt.n_components = n_components
    popt.sstx = shdp
    popt.n_rs = n_rs
    popt.n_rs_geq1 = isLmin1
    popt.n_rs_f = <double> n_rs
    popt.inv_n_rs_2f = 2.0/popt.n_rs_f
    popt.inv_n_rs_4f = 2.0*popt.inv_n_rs_2f
    popt.sim_hd_ms = simhd_h_nn_i_rs
    popt.m_nn = m_nn
    popt.nnn_all_rs = nnn_all_rs
    popt.inn_all_rs = inn_all_rs
    popt.dsld_all_rs = dsld_all_rs
    popt.idnn_in_ars = idnn_in_ars
    # Space-partitioning trees are working with the squared threshold to save the computation time of computing the square root for the Euclidean distance
    popt.theta_s = ms_thetha*ms_thetha
    popt.inter_fct = 7
    K_star = n_components*sizeof(double)
    popt.qdiff = <double*> PyMem_Malloc(K_star)
    if popt.qdiff is NULL:
        free_int_2dmat(nnn_i_rs, n_rs)
        free_dble_3dmat(ds_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
        free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
        free_dble_3dmat(tau_h_i_rs, L, n_rs)
        free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
        if isLmin1:
            PyMem_Free(nnn_all_rs)
            free_Pysst_2dmat(inn_all_rs, sLm_nt)
            PyMem_Free(dsld_all_rs)
            free_Pysst_3dmat(idnn_in_ars, n_rs, sLm_nt)
        lbfgs_free(xopt)
        PyMem_Free(popt)
        printf("Out of memory for popt.qdiff.\n")
        exit(EXIT_FAILURE)

    # Parameters of the L-BFGS optimization
    cdef lbfgs_parameter_t param
    cdef lbfgs_parameter_t* pparam = &param
    # Initializing param with default values
    lbfgs_parameter_init(pparam)
    # Updating some parameters
    param.m = maxcor
    param.epsilon = gtol
    param.delta = ftol
    param.max_iterations = nit_max
    param.max_linesearch = maxls
    param.past = 1
    # We modify the default values of the minimum and maximum step sizes of the line search because the problem is badly scaled
    param.max_step = DBL_MAX
    param.min_step = DBL_MIN

    # k refers to the number of currently considered scales and h to the index of the current scale. Nd will store the inverse of the number of currently considered scales.
    cdef Py_ssize_t k, h
    h = L-1
    for k in range(1, L+1, 1):
        # Updates related to the current multi-scale optimization step
        f_update_mso_step(k, h, n_rs, N, nnn_i_rs, ds_nn_i_rs, tau_h_i_rs, simhd_ms_nn_i_rs, simhd_h_nn_i_rs)
        # Symmetrizing the multi-scale HD similarities. Be careful that only the similarities between i and j such that j>i are actually symetrized, since only these are used in the evaluate function.
        fmstsne_symmetrize(n_rs, sLm_nt, simhd_ms_nn_i_rs, m_nn, simhd_h_nn_i_rs)
        # Performing the optimization
        lbfgs(prod_N_nc, xopt, NULL, fmstsne_evaluate, NULL, popt, pparam)
        h -= 1

    # Gathering the optimized LD coordinates
    memcpy(xlds, xopt, shdp)

    # Free the ressources
    free_int_2dmat(nnn_i_rs, n_rs)
    free_dble_3dmat(ds_nn_i_rs, n_rs, N)
    free_dble_3dmat(simhd_ms_nn_i_rs, n_rs, N)
    free_dble_3dmat(simhd_h_nn_i_rs, n_rs, N)
    free_dble_3dmat(tau_h_i_rs, L, n_rs)
    free_Pysst_3dmat(m_nn, n_rs, sLm_nt)
    if isLmin1:
        PyMem_Free(nnn_all_rs)
        free_Pysst_2dmat(inn_all_rs, sLm_nt)
        PyMem_Free(dsld_all_rs)
        free_Pysst_3dmat(idnn_in_ars, n_rs, sLm_nt)
    lbfgs_free(xopt)
    PyMem_Free(popt.qdiff)
    PyMem_Free(popt)

#######################################################
####################################################### Quality criteria Q_NX and R_NX
#######################################################

cdef struct nnRank:
    int nn                # Index of a sample
    int rank              # Rank of the sample

cdef inline bint sortByInd(const nnRank v, const nnRank w) nogil:
    """
    Returns True when v.nn is < than w.nn.
    """
    return v.nn < w.nn

cpdef inline double drqa_qnx_rnx_auc(double[::1] X_hds, double[::1] X_lds, int N, int d_hds, int d_lds, int Kup, double[::1] qnxk, double[::1] rnxk, int rnxk_size):
    """
    Compute the quality criteria curves Q_NX(K) and R_NX(K) with the neighborhood size K ranging from 1 to Kup. The AUC of the reduced R_NX(K) curve is returned.
    In:
    - X_hds: one-dimensional array with the HD samples stacked one after the other.
    - X_lds: one-dimensional array with the LD samples stacked one after the other.
    - N: number of samples.
    - d_hds: dimension of the HDS.
    - d_lds: dimension of the LDS.
    - Kup: greatest neighborhood size to consider.
    - qnxk: array to store the Q_NX(K) values for K = 1, ..., Kup.
    - rnxk: array to store the R_NX(K) values for K = 1, ..., min(N-2, Kup).
    - rnxk_size: min(N-2, Kup).
    This function modifies the arrays qnxk and rnxk.
    Out:
    - A double being the AUC of the reduced R_NX curve.
    Remark:
    - the time complexity to compute these criteria scales as O(N*Kup*log(N)).
    - the Euclidean distance is employed to compute the quality criteria.
    """
    # Initializing qnxk to zero
    memset(&qnxk[0], 0, Kup*sizeof(double))

    # Constructing the VP trees in the HDS and LDS
    cdef VpTree* vpt_hd = new VpTree(&X_hds[0], N, d_hds)
    cdef VpTree* vpt_ld = new VpTree(&X_lds[0], N, d_lds)

    # Kup + 1
    cdef int Kupadd = Kup + 1

    # Allocating an array to store the Kupadd nearest HD neighbor of a data point
    cdef int* nn_hd = <int*> PyMem_Malloc(Kupadd*sizeof(int))
    if nn_hd is NULL:
        del vpt_hd
        del vpt_ld
        printf("Error in drqa_qnx_rnx_auc function of fmsne_implem.pyx: out of memory for nn_hd.")
        exit(EXIT_FAILURE)
    # Allocating an array to store the Kupadd nearest LD neighbor of a data point
    cdef int* nn_ld = <int*> PyMem_Malloc(Kupadd*sizeof(int))
    if nn_ld is NULL:
        del vpt_hd
        del vpt_ld
        PyMem_Free(nn_hd)
        printf("Error in drqa_qnx_rnx_auc function of fmsne_implem.pyx: out of memory for nn_ld.")
        exit(EXIT_FAILURE)

    # Allocating an array of structure to store the indexes of the HD neighbors and their ranks
    cdef nnRank* nnrk_hd = <nnRank*> PyMem_Malloc(Kup*sizeof(nnRank))
    if nnrk_hd is NULL:
        del vpt_hd
        del vpt_ld
        PyMem_Free(nn_hd)
        PyMem_Free(nn_ld)
        printf("Error in drqa_qnx_rnx_auc function of fmsne_implem.pyx: out of memory for nnrk_hd.")
        exit(EXIT_FAILURE)

    # Variable to iterate over the samples
    cdef Py_ssize_t i, ihd, ild, j, lb, ub, mid
    ihd = 0
    ild = 0
    cdef int jr, Kupsub
    Kupsub = Kup - 1

    # For each data point
    for i in range(N):
        # Searching the Kupadd nearest neighbors of sample i in HDS and LDS
        vpt_hd.search(&X_hds[ihd], Kupadd, nn_hd)
        vpt_ld.search(&X_lds[ild], Kupadd, nn_ld)

        # Filling nnrk_hd
        jr = 0
        for j in range(Kupadd):
            if nn_hd[j] != i:
                nnrk_hd[jr].nn = nn_hd[j]
                nnrk_hd[jr].rank = jr
                jr += 1

        # Sorting nnrk_hd according to the nn keys
        sort(nnrk_hd, nnrk_hd + Kup, sortByInd)

        # LD rank
        jr = 0
        # For each LD neighbor
        for j in range(Kupadd):
            if nn_ld[j] != i:
                # If nn_ld[j] is in the range of nnrk_hd
                if (nn_ld[j] >= nnrk_hd[0].nn) and (nn_ld[j] <= nnrk_hd[Kupsub].nn):
                    # Searching for nn_ld[j] in nnrk_hd using binary search
                    lb = 0
                    ub = Kup
                    while ub - lb > 1:
                        mid = (ub + lb)//2
                        if nn_ld[j] == nnrk_hd[mid].nn:
                            lb = mid
                            break
                        elif nn_ld[j] < nnrk_hd[mid].nn:
                            ub = mid
                        else:
                            lb = mid + 1
                    # Updating qnxk only if nn_ld[j] == nnrk_hd[lb].nn
                    if nn_ld[j] == nnrk_hd[lb].nn:
                        # Updating at the biggest rank between the HD and LD ones
                        if jr < nnrk_hd[lb].rank:
                            qnxk[nnrk_hd[lb].rank] += 1.0
                        else:
                            qnxk[jr] += 1.0
                # Incrementing the LD rank
                jr += 1

        # Updating ihd and ild
        ihd += d_hds
        ild += d_lds

    # Free the ressources
    del vpt_hd
    del vpt_ld
    PyMem_Free(nn_hd)
    PyMem_Free(nn_ld)
    PyMem_Free(nnrk_hd)

    # Computing the cumulative sum of qnxk and normalizing it
    cdef double cs = 0.0
    cdef double Nd = <double> N
    for i in range(Kup):
        cs += qnxk[i]
        qnxk[i] = cs/Nd
        Nd += N

    # Computing rnxk and its AUC
    Nd = <double> (N-1)
    cs = Nd - 1.0
    cdef double K = 1.0
    cdef double iK = 1.0
    cdef double siK = 0.0
    cdef double auc = 0.0
    for i in range(rnxk_size):
        siK += iK
        rnxk[i] = (Nd*qnxk[i] - K)/cs
        auc += (rnxk[i]*iK)
        K += 1.0
        iK = 1.0/K
        cs -= 1.0

    # Normalizing the AUC
    auc /= siK

    # Returning the AUC
    return auc
