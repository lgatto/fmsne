% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plotting.R
\name{plotMSSNE}
\alias{plotMSSNE}
\alias{plotFMSSNE}
\alias{plotMSTSNE}
\alias{plotFMSTSNE}
\title{Plot reduced dimensions}
\usage{
plotMSSNE(object, ..., ncomponents = 2)

plotFMSSNE(object, ..., ncomponents = 2)

plotFMSTSNE(object, ..., ncomponents = 2)

plotMSTSNE(object, ..., ncomponents = 2)
}
\arguments{
\item{object}{A \code{SingleCellExperiment} object.}

\item{...}{Additional arguments to pass to
\code{\link[scater:plotReducedDim]{scater::plotReducedDim()}}.}

\item{ncomponents}{\code{numeric(1)} indicating the number of
dimensions components to plot This can also be a numeric
vector, see \code{\link[scater:plotReducedDim]{scater::plotReducedDim()}} for details.}
}
\value{
A \code{ggplot} object.
}
\description{
Wrapper fonctions to create plots for specific types
of reduced dimension results produced by the various \code{fmsne}
dimensionality reductions. The function follow the \code{scater}
package's syntax.
}
\details{
As for the equivalent functions from the \code{scater} package, each
function is a convenient wrapper around \code{\link[scater:plotReducedDim]{scater::plotReducedDim()}}
that searches the \code{\link[SingleCellExperiment:reducedDims]{SingleCellExperiment::reducedDims()}} slot for
an appropriately named dimensionality reduction result:
\itemize{
\item "FMSTSNE" for \code{plotFMSTNSE()}
\item "FMSSNE" for \code{plotFMSNSE()}
\item "MSTSNE" for \code{plotMSTNSE()}
\item "MSTSNE" for \code{plotMSNSE()}
}

Its only purpose is to streamline workflows to avoid the need to
specify the ‘dimred’ argument.
}
\seealso{
\code{\link[=runFMSSNE]{runFMSSNE()}}, \code{\link[=runFMSSNE]{runFMSSNE()}}, \code{\link[=runMSTSNE]{runMSTSNE()}} and \code{\link[=runMSSNE]{runMSSNE()}} for
the functions that actually perform the calculations.

\code{\link[scater:plotReducedDim]{scater::plotReducedDim()}} for the underlying plotting function.
}
\author{
Laurent Gatto
}
