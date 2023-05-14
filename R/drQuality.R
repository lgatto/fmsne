.run_eval_dr_quality_from_data <- function(x, y) {
    fmsne <- reticulate::import("fmsne")
    ans <- fmsne$eval_dr_quality_from_data(X = x,
                                             Y = y)
    names(ans) <- c("Rk", "AUC")
    ans
}

##' @export
drQuality <- function(object, dimred = "PCA") {
    stopifnot(inherits(object, "SingleCellExperiment"))
    x <- t(as.matrix(assay(object)))
    y <- reducedDim(object, dimred)
    basiliskRun(env = fmsneenv,
                fun = .run_eval_dr_quality_from_data,
                x = x,
                y = y)
}
