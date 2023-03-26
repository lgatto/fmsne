.run_fmstsne <- function(X,
                         n_components = 2L,
                         init = 'pca',
                         rand_state = NA,
                         nit_max = 30,
                         gtol = 1e-5,
                         ftol = 2.2204460492503131e-09,
                         maxls = 50,
                         maxcor = 10,
                         fit_U = TRUE,
                         bht = 0.45,
                         fseed = 1L) {
    fmsnepy <- reticulate::import("fmsnepy")
    ans <- fmsnepy$fmstsne(X_hds = t(X),
                           n_components = n_components,
                           init = init,
                           rand_state = rand_state,
                           nit_max = nit_max,
                           gtol = gtol,
                           ftol = ftol,
                           maxls = maxls,
                           maxcor = maxcor,
                           fit_U = fit_U,
                           bht = bht,
                           fseed = fseed)
    ans
}

##' @export
runFMSTSNE <- function(x,
                       n_components = 2L,
                       init = 'pca',
                       rand_state = NA,
                       nit_max = 30,
                       gtol = 1e-5,
                       ftol = 2.2204460492503131e-09,
                       maxls = 50,
                       maxcor = 10,
                       fit_U = TRUE,
                       bht = 0.45,
                       fseed = 1L,
                       subset_row = NULL,
                       name = "FMSTSNE") {
    stopifnot(inherits(x, "SingleCellExperiment"))
    X <- as.matrix(assay(x))
    if (!is.null(subset_row))
        X <- X[subset_row, ]
    n_components <- as.integer(n_components)
    fseed <- as.integer(fseed)

    ans <- basiliskRun(env = fmsneenv,
                       fun = .run_fmstsne,
                       X = X,
                       n_components = n_components,
                       init = init,
                       rand_state = rand_state,
                       nit_max = nit_max,
                       gtol = gtol,
                       ftol = ftol,
                       maxls = maxls,
                       maxcor = maxcor,
                       fit_U = fit_U,
                       bht = bht,
                       fseed = fseed)
    rownames(ans) <- colnames(x)
    reducedDim(x, name) <- ans
    x
}


##' @export
plotFMSTSNE <- function(object, ..., ncomponents = 2) {
    plotReducedDim(object,
                   ncomponents = ncomponents,
                   dimred = "FMSTSNE",
                   ...)
}
