##' @export
runMSTSNE <- function(x,
                      n_components = 2L,
                      init = 'pca',
                      ## rand_state = NA,
                      nit_max = 30,
                      gtol = 1e-5,
                      ftol = 2.2204460492503131e-09,
                      maxls = 50,
                      maxcor = 10,
                      fit_U = TRUE,
                      subset_row = NULL,
                      name = "MSTSNE") {
    stopifnot(inherits(x, "SingleCellExperiment"))
    X <- as.matrix(assay(x))
    if (!is.null(subset_row))
        X <- X[subset_row, ]
    n_components <- as.integer(n_components)

    ans <- basiliskRun(env = fmsneenv,
                       fun = .run_mstsne,
                       X = X,
                       n_components = n_components,
                       init = init,
                       rand_state = rand_state,
                       nit_max = nit_max,
                       gtol = gtol,
                       ftol = ftol,
                       maxls = maxls,
                       maxcor = maxcor,
                       fit_U = fit_U)
    rownames(ans) <- colnames(x)
    reducedDim(x, name) <- ans
    x
}

.run_mstsne <- function(X,
                        n_components = 2L,
                        init = 'pca',
                        rand_state = NA,
                        nit_max = 30,
                        gtol = 1e-5,
                        ftol = 2.2204460492503131e-09,
                        maxls = 50,
                        maxcor = 10) {
    fmsne <- reticulate::import("fmsne")
    ans <- fmsne$mstsne(X_hds = t(X),
                       n_components = n_components,
                       init = init,
                       rand_state = rand_state,
                       nit_max = nit_max,
                       gtol = gtol,
                       ftol = ftol,
                       maxls = maxls,
                       maxcor = maxcor)
    ans
}
