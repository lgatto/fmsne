#' Test function
#'
#' @export
#'
#' @importFrom reticulate import
#' @importFrom basilisk basiliskStart basiliskRun basiliskStop
test <- function() {
    cl <- basiliskStart(fmsneenv)
    fmsne.names <- basiliskRun(cl, function() {
        X <- reticulate::import("fmsne")
        names(X)
    })
    basiliskStop(cl)

    list(fmnsne=fmsne.names)
}
