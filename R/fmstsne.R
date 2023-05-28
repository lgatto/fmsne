##' @export
##'
##' @importFrom reticulate import
##'
##' @import SingleCellExperiment
##'
##' @rdname fmsne
calculateFMSTSNE <- function(x,
                             ncomponents = 2L,
                             ntop = 500,
                             subset_row = NULL,
                             scale = FALSE,
                             transposed = FALSE,
                             init = 'pca',
                             ## rand_state = NA,
                             nit_max = 30,
                             gtol = 1e-5,
                             ftol = 2.2204460492503131e-09,
                             maxls = 50,
                             maxcor = 10,
                             bht = 0.45,
                             fseed = 1L) {
    if (!transposed)
        x <- scater:::.get_mat_for_reddim(x,
                                          subset_row = subset_row,
                                          ntop = ntop,
                                          scale = scale)
    x <- as.matrix(x)
    ncomponents <- as.integer(ncomponents)
    fseed <- as.integer(fseed)
    stopifnot(fseed >= 1)
    ans <- basiliskRun(env = fmsneenv,
                       fun = .run_fmstsne,
                       X = x,
                       n_components = ncomponents,
                       init = init,
                       rand_state = NA,
                       nit_max = nit_max,
                       gtol = gtol,
                       ftol = ftol,
                       maxls = maxls,
                       maxcor = maxcor,
                       bht = bht,
                       fseed = fseed)
    rownames(ans) <- rownames(x)
    colnames(ans) <- paste0("FMSTSNE", seq_len(ncomponents))
    reducedDim(x, name) <- ans
    x
}

##' @export
##'
##' @rdname fmsne
runFMSTSNE <- function(x, ...,
                       name = "FMSTSNE") {
    stopifnot(inherits(x, "SingleCellExperiment"))
    reducedDim(x, name) <- calculateFMSTSNE(assay(x), ...)
    x
}

.run_fmstsne <- function(X,
                         n_components = 2L,
                         init = 'pca',
                         rand_state = NA,
                         nit_max = 30,
                         gtol = 1e-5,
                         ftol = 2.2204460492503131e-09,
                         maxls = 50,
                         maxcor = 10,
                         bht = 0.45,
                         fseed = 1L) {
    fmsne <- reticulate::import("fmsne")
    fmsne$fmstsne(X_hds = X,
                  n_components = n_components,
                  init = init,
                  rand_state = rand_state,
                  nit_max = nit_max,
                  gtol = gtol,
                  ftol = ftol,
                  maxls = maxls,
                  maxcor = maxcor,
                  bht = bht,
                  fseed = fseed)
}
