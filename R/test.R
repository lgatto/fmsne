#' Test function
#'
#' @export
#'
#' @importFrom reticulate import
#' @importFrom basilisk basiliskStart basiliskRun basiliskStop
test <- function() {

    cl <- basiliskStart(fmsneenv)
    fmsnepy.names <- basiliskRun(cl, function() {
        X <- reticulate::import("fmsnepy")
        names(X)
    })
    basiliskStop(cl)

    list(fmnsne=fmsnepy.names)
}
