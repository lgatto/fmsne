.run_eval_dr_quality_from_data <- function(x, y) {
    fmsnepy <- reticulate::import("fmsnepy")
    ans <- fmsnepy$eval_dr_quality_from_data(X = x,
                                             Y = y)
    ans
}

##' @export
drQuality <- function(object, dimred = "PCA") {
    stopifnot(inherits(object, "SingleCellExperiment"))
    NULL
}
