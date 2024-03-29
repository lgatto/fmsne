## ============================================================
## scater:::.get_mat_for_reddim()
##
## - Picking the 'ntop' most highly variable features or just using a
##   pre-specified set of features.
## - Also removing zero-variance columns and scaling the variance of
##   each column.
## - Finally, transposing for downstream use (cells are now rows).


## ============================================================
## scater:::.get_mat_from_sce()
##
## Either returns an assay (defined by exprs_values) or the reducedDim
## matrix (defined by dimred) with all or (optional) 'n_dimred'
## dimensions. The default is to return the "logcounts" assay.


##' Test function returning python function names
##'
##' @noRd
##'
##' @importFrom reticulate import
##' @importFrom basilisk basiliskStart basiliskRun basiliskStop
fmsnePythonNames <- function() {
    cl <- basiliskStart(fmsneenv,
                        testload = "numba")
    fmsne.names <- basiliskRun(cl, function() {
        X <- reticulate::import("fmsne")
        names(X)
    })
    basiliskStop(cl)
    list(fmnsne=fmsne.names)
}

fmsnePythonVersion <- function() {
    basiliskRun(env = fmsneenv,
                fun = function() {
                    fmsne <- reticulate::import("fmsne")
                    fmsne$`__version__`
                },
                testload = "numba")
}
