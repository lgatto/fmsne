#' Test function returning python function names
#'
#' @export
#'
#' @importFrom reticulate import
#' @importFrom basilisk basiliskStart basiliskRun basiliskStop
fmsnePythonNames <- function() {
    cl <- basiliskStart(fmsneenv)
    fmsne.names <- basiliskRun(cl, function() {
        X <- reticulate::import("fmsne")
        names(X)
    })
    basiliskStop(cl)
    list(fmnsne=fmsne.names)
}
