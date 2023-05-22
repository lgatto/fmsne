##' @export
##'
##' @rdname fmsne
runFMSSNE <- function(x,
                      n_components = 2L,
                      init = 'pca',
                      ## rand_state = NA,
                      nit_max = 30,
                      gtol = 1e-5,
                      ftol = 2.2204460492503131e-09,
                      maxls = 50,
                      maxcor = 10,
                      fit_U = TRUE,
                      bht = 0.45,
                      fseed = 1L,
                      subset_row = NULL,
                      name = "FMSSNE") {
    stopifnot(inherits(x, "SingleCellExperiment"))
    fseed <- as.integer(fseed)
    stopifnot(fseed >= 1)
    X <- as.matrix(assay(x))
    if (!is.null(subset_row))
        X <- X[subset_row, , drop = FALSE]
    n_components <- as.integer(n_components)
    fseed <- as.integer(fseed)

    ans <- basiliskRun(env = fmsneenv,
                       fun = .run_fmssne,
                       X = X,
                       n_components = n_components,
                       init = init,
                       rand_state = NA,
                       nit_max = nit_max,
                       gtol = gtol,
                       ftol = ftol,
                       maxls = maxls,
                       maxcor = maxcor,
                       fit_U = fit_U,
                       bht = bht,
                       fseed = fseed)
    rownames(ans) <- colnames(x)
    colnames(ans) <- paste0(name, seq_len(n_components))
    reducedDim(x, name) <- ans
    x
}

.run_fmssne <- function(X,
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
    fmsne <- reticulate::import("fmsne")
    ans <- fmsne$fmssne(X_hds = t(X),
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
