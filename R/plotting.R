##' @title Plot reduced dimensions
##'
##' @description Wrapper fonctions to create plots for specific types
##'     of reduced dimension results produced by the various `fmsne`
##'     dimensionality reductions. The function follow the `scater`
##'     package's syntax.
##'
##' @details
##'
##' As for the equivalent functions from the `scater` package, each
##' function is a convenient wrapper around [scater::plotReducedDim()]
##' that searches the [SingleCellExperiment::reducedDims()] slot for
##' an appropriately named dimensionality reduction result:
##'
##' - "FMSTSNE" for `plotFMSTNSE()`
##'
##' - "FMSSNE" for `plotFMSNSE()`
##'
##' - "MSTSNE" for `plotMSTNSE()`
##'
##' - "MSTSNE" for `plotMSNSE()`
##'
##' Its only purpose is to streamline workflows to avoid the need to
##' specify the ‘dimred’ argument.
##'
##' @param object A `SingleCellExperiment` object.
##'
##' @param ... Additional arguments to pass to
##'     [scater::plotReducedDim()].
##'
##' @param ncomponents `numeric(1)` indicating the number of
##'     dimensions components to plot This can also be a numeric
##'     vector, see [scater::plotReducedDim()] for details.
##'
##' @return A `ggplot` object.
##'
##' @export
##'
##' @importFrom scater  plotReducedDim
##'
##' @rdname plotFMSNE
##'
##' @aliases plotMSSNE plotFMSSNE plotMSTSNE plotFMSTSNE
##'
##' @seealso
##'
##' [runFMSSNE()], [runFMSSNE()], [runMSTSNE()] and [runMSSNE()] for
##' the functions that actually perform the calculations.
##'
##' [scater::plotReducedDim()] for the underlying plotting function.
##'
##' @author Laurent Gatto
plotMSSNE <- function(object, ..., ncomponents = 2) {
    plotReducedDim(object,
                   ncomponents = ncomponents,
                   dimred = "MSSNE")
}


##' @export
##'
##' @rdname plotFMSNE
plotFMSSNE <- function(object, ..., ncomponents = 2) {
    plotReducedDim(object,
                   ncomponents = ncomponents,
                   dimred = "FMSSNE",
                   ...)
}


##' @export
##'
##' @rdname plotFMSNE
plotFMSTSNE <- function(object, ..., ncomponents = 2) {
    plotReducedDim(object,
                   ncomponents = ncomponents,
                   dimred = "FMSTSNE",
                   ...)
}

##' @export
##'
##' @rdname plotFMSNE
plotMSTSNE <- function(object, ..., ncomponents = 2) {
    plotReducedDim(object,
                   ncomponents = ncomponents,
                   dimred = "FMSTSNE",
                   ...)
}
