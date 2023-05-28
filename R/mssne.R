##' @export
##'
##' @rdname fmsne
calculateMSSNE <- function(x,
                           ncomponents = 2L,
                           topn = 500,
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
                           fit_U = TRUE) {
    if (!transposed)
        x <- scater:::.get_mat_for_reddim(x,
                                          subset_row = subset_row,
                                          ntop = ntop,
                                          scale = scale)
    x <- as.matrix(x)
    ncomponents <- as.integer(ncomponents)
    ans <- basiliskRun(env = fmsneenv,
                       fun = .run_mssne,
                       X = x,
                       n_components = ncomponents,
                       init = init,
                       rand_state = NA,
                       nit_max = nit_max,
                       gtol = gtol,
                       ftol = ftol,
                       maxls = maxls,
                       maxcor = maxcor,
                       fit_U = fit_U)
    rownames(ans) <- rownames(x)
    colnames(ans) <- paste0("MSSNE", seq_len(ncomponents))
    x
}

##' @export
##'
##' @param ... additional parameters passed to the respective
##'     'calculate*()' functions.
##'
##' @rdname fmsne
runMSSNE <- function(x, ...,
                      name = "MSSNE") {
    stopifnot(inherits(x, "SingleCellExperiment"))
    reducedDim(x, name) <- calculateMSSNE(assay(x), ...)
    x
}

.run_mssne <- function(X,
                       n_components = 2L,
                       init = 'pca',
                       rand_state = NA,
                       nit_max = 30,
                       gtol = 1e-5,
                       ftol = 2.2204460492503131e-09,
                       maxls = 50,
                       maxcor = 10,
                       fit_U = TRUE) {
    fmsne <- reticulate::import("fmsne")
    fmsne$mssne(X_hds = t(X),
                n_components = n_components,
                init = init,
                rand_state = rand_state,
                nit_max = nit_max,
                gtol = gtol,
                ftol = ftol,
                maxls = maxls,
                maxcor = maxcor,
                fit_U = fit_U)
}
