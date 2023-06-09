% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/AllGenerics.R, R/fmsne.R, R/fmssne.R,
%   R/fmstsne.R, R/mssne.R, R/mstnse.R
\docType{package}
\name{calculateMSSNE}
\alias{calculateMSSNE}
\alias{calculateMSTSNE}
\alias{calculateFMSSNE}
\alias{calculateFMSTSNE}
\alias{fmsne}
\alias{runMSSNE}
\alias{runMSTSNE}
\alias{runFMSSNE}
\alias{runFMSTSNE}
\alias{calculateFMSSNE,ANY-method}
\alias{calculateFMSSNE,SingleCellExperiment-method}
\alias{calculateFMSTSNE,ANY-method}
\alias{calculateFMSTSNE,SingleCellExperiment-method}
\alias{calculateMSSNE,ANY-method}
\alias{calculateMSSNE,SingleCellExperiment-method}
\alias{calculateMSTSNE,ANY-method}
\alias{calculateMSTSNE,SingleCellExperiment-method}
\title{Multi-scale stochastic neighbour embedding}
\usage{
calculateMSSNE(x, ...)

calculateMSTSNE(x, ...)

calculateFMSSNE(x, ...)

calculateFMSTSNE(x, ...)

\S4method{calculateFMSSNE}{ANY}(
  x,
  ncomponents = 2L,
  ntop = 500,
  subset_row = NULL,
  scale = FALSE,
  transposed = FALSE,
  init = "pca",
  nit_max = 30,
  gtol = 1e-05,
  ftol = 2.22044604925031e-09,
  maxls = 50,
  maxcor = 10,
  fit_U = TRUE,
  bht = 0.45,
  fseed = 1L
)

\S4method{calculateFMSSNE}{SingleCellExperiment}(
  x,
  ...,
  exprs_values = "logcounts",
  dimred = NULL,
  n_dimred = NULL
)

runFMSSNE(x, ..., name = "FMSSNE")

\S4method{calculateFMSTSNE}{ANY}(
  x,
  ncomponents = 2L,
  ntop = 500,
  subset_row = NULL,
  scale = FALSE,
  transposed = FALSE,
  init = "pca",
  nit_max = 30,
  gtol = 1e-05,
  ftol = 2.22044604925031e-09,
  maxls = 50,
  maxcor = 10,
  bht = 0.45,
  fseed = 1L
)

\S4method{calculateFMSTSNE}{SingleCellExperiment}(
  x,
  ...,
  exprs_values = "logcounts",
  dimred = NULL,
  n_dimred = NULL
)

runFMSTSNE(x, ..., name = "FMSTSNE")

\S4method{calculateMSSNE}{ANY}(
  x,
  ncomponents = 2L,
  ntop = 500,
  subset_row = NULL,
  scale = FALSE,
  transposed = FALSE,
  init = "pca",
  nit_max = 30,
  gtol = 1e-05,
  ftol = 2.22044604925031e-09,
  maxls = 50,
  maxcor = 10,
  fit_U = TRUE
)

\S4method{calculateMSSNE}{SingleCellExperiment}(
  x,
  ...,
  exprs_values = "logcounts",
  dimred = NULL,
  n_dimred = NULL
)

runMSSNE(x, ..., name = "MSSNE")

\S4method{calculateMSTSNE}{ANY}(
  x,
  ncomponents = 2L,
  ntop = 500,
  subset_row = NULL,
  scale = FALSE,
  transposed = FALSE,
  init = "pca",
  nit_max = 30,
  gtol = 1e-05,
  ftol = 2.22044604925031e-09,
  maxls = 50,
  maxcor = 10
)

\S4method{calculateMSTSNE}{SingleCellExperiment}(
  x,
  ...,
  exprs_values = "logcounts",
  dimred = NULL,
  n_dimred = NULL
)

runMSTSNE(x, ..., name = "MSTSNE")
}
\arguments{
\item{x}{Matrix (for \verb{calculate*()}) or object of class
\code{SingleCellExperiment} (for both \verb{calculate*()} and \verb{run*()})
containing a numeric assay with log-expression values where
rows are features and columns are cells.}

\item{...}{additional parameters passed to the respective
'calculate*()' functions.}

\item{ncomponents}{\code{integer(1)} indicating the number of t-SNE
dimensions to obtain. Default is 2L.}

\item{ntop}{\code{integer(1)} specifying the number of features with
the highest variances to use for dimensionality
reduction. Default is 500.}

\item{subset_row}{Vector specifying the subset of features to use
for dimensionality reduction. This can be a character vector
of row names, an integer vector of row indices or a logical
vector. Default is \code{NULL} that takes all features.}

\item{scale}{\code{logical(1)} indicating whether the expression values
should be standardized? Default is \code{FALSE}.}

\item{transposed}{\code{logical(1)} indicating whether \code{x} is
transposed with cells in rows? Default is \code{FALSE}.}

\item{init}{\code{character(1)}. If equal to "pca" (default), the LD
embedding is initialized with the first \code{n_components}
principal components computed on \code{x}. If equal to "random",
the LD embedding is initialized randomly, using a uniform
Gaussian distribution with a variance equal to
var. Alternatively, \code{init} can also be a number of cells by
\code{n_components} matrix of dimension (not tested - please file
an issue in case of problems.).}

\item{nit_max}{\code{numeric(1)} defining the maximum number of L-BFGS
steps at each stage of the multi-scale optimization, which is
defined in Lee et al. (2015). Default is 30.}

\item{gtol}{\code{numeric(1)} defining the tolerance for the infinite
norm of the gradient in the L-BFGS algorithm. The L-BFGS
iterations hence stop when $max{|g_i | i = 1, ..., n} <= gtol$
where $g_i$ is the i-th component of the gradient.}

\item{ftol}{\code{numeric(1)} defining the tolerance for the relative
updates of the cost function value in L-BFGS. Default is
2.2204460492503131e-09.}

\item{maxls}{\code{numeric(1)} maximum number of line search steps per
L-BFGS-B iteration. Default is 50.}

\item{maxcor}{\code{numeric(1)} defining the maximum number of variable
metric corrections used to define the limited memory matrix in
L-BFGS. Default is 10.}

\item{fit_U}{\code{logical(1)} indicating whether to fit the U in the
definition of the LD similarities in Lee et al. (2015). If
TRUE (default), the U is tuned as in Lee et
al. (2015). Otherwise, it is forced to 1. Setting \code{fit_U} to
TRUE usually tends to slightly improve DR quality at the
expense of slightly increasing computation time.}

\item{bht}{\code{logical(1)} indicating whether to fit the U in the
definition of the LD similarities in Lee et al. (2015). If
\code{TRUE}, the U is tuned as in Lee et al. (2015). Otherwise, it
is forced to 1. Setting \code{fit_U} to \code{TRUE} usually tends to
slightly improve dimensionality reduction quality at the
expense of slightly increasing computation time.}

\item{fseed}{strictly positive \code{integer(1)} defining the random
seed used to perform the random sampling of the
high-dimensional data set at the different scales.}

\item{exprs_values}{\code{integer(1)} or \code{character(1)} indicating
which assay of ‘x’ contains the expression values. Default is
\code{"logcounts"}.}

\item{dimred}{\code{character(1)} specifying the optional
dimensionality reduction results to use. Default is \code{NULL}.}

\item{n_dimred}{\code{interger(1)} specifying the dimensions to use if
\code{dimred} is specified. Default is \code{NULL}, to use all
components.}

\item{name}{\code{character(1)} specifying the name to be used to store
the result in the \code{reducedDims} of the output. Default is
\code{"MSSNE"}, \code{"MSTSNE"}, \code{"FMSSNE"} or \code{"FMSTSNE"} depending on
the function.}
}
\value{
The \verb{calculate*()} functions retrun a reduced dimension
matrix of dimensions \code{ncol(x)} (i.e. cells) by
\code{ncomponents}. The \verb{run*()} functions return a modified
\code{SingleCellExperiment} that contains the reduced dimension
coordinates in ‘reducedDim(x, name)’.
}
\description{
The \code{fmsne} package offers various functions to perform nonlinear
dimensionality reduction through multi-scale (MS) stochastic
neighbor embedding (SNE) or t-distributed SNE (t-SNE), including
fast versions thereof.
\itemize{
\item The \code{calculateMSSNE()} function performs a nonlinear
dimensionality reduction through multi-scale SNE, as presented
in Lee et al. (2015) below and summarized in de Bodt et
al. (2020).  Given a data set with N samples, it has \eqn{O(N^2
  log(N))} time complexity.
\item The \code{calculateMSTSNE()} function performs nonlinear
dimensionality reduction through multi-scale t-SNE, as presented
in the reference de Bodt et al. (2018) below and summarized in
de Bodt et al. (2020). Given a data set with N samples, it has
\eqn{O(N^2 log(N))} time complexity.
\item The \code{calculateFMSSNE()} function performs nonlinear
dimensionality reduction through fast multi-scale SNE, as
presented in Lee et al. (2015) below. Given a data set with N
samples, it has \eqn{O(N log(N)^2)} time complexity.
\item The \code{calculateFMSTSNE()} function performs nonlinear
dimensionality reduction through fast multi-scale t-SNE, as
presented in the de Bodt et al. (2020) below.  Given a data set
with N samples, it has \eqn{O(N log(N))^2} time complexity.
}

Each method can also be called with \verb{run[F]MS[T]SNE()} to store
the result as a new \code{SingleCellExperiment} reduced dimension
\code{reducedDim} instance.

See the vignette for further details.
}
\section{Feature selection}{


This section is adapted from the \code{scater} package manual and is
relevant if \code{x} is a numeric matrix of (log-)expression values
with features in rows and cells in columns; or if \code{x} is a
\code{SingleCellExperiment} and \code{dimred = NULL}.  In the latter, the
expression values are obtained from the assay specified by
\code{exprs_values}.

The \code{subset_row} argument specifies the features to use for
dimensionality reduction.  The aim is to allow users to specify
highly variable features to improve the signal/noise ratio, or to
specify genes in a pathway of interest to focus on particular
aspects of heterogeneity.

If \code{subset_row = NULL}, the \code{ntop} features with the largest
variances are used instead.  Using the same underlying function as
in the \code{scater} package, we literally compute the variances from
the expression values without considering any mean-variance trend,
so often a more considered choice of genes is possible, e.g., with
\code{scran} functions.  Note that the value of \code{ntop} is ignored if
\code{subset_row} is specified.

If \code{scale = TRUE}, the expression values for each feature are
standardized so that their variance is unity.  This will also
remove features with standard deviations below 1e-8.
}

\section{Using reduced dimensions}{


This section is adapted from the \code{scater} package manual.

If \code{x} is a \code{SingleCellExperiment}, the neighbour embedding
methods can be applied on existing dimensionality reduction
results in \code{x} by setting the \code{dimred} argument.  This is
typically used to run slower non-linear algorithms (t-SNE, UMAP)
on the results of fast linear decompositions (PCA).  We might also
use this with existing reduced dimensions computed from \emph{a
priori} knowledge (e.g., gene set scores), where further
dimensionality reduction could be applied to compress the data.

The matrix of existing reduced dimensions is taken from
\code{reducedDim(x, dimred)}.  By default, all dimensions are used to
compute the second set of reduced dimensions.  If \code{n_dimred} is
also specified, only the first \code{n_dimred} columns are used.
Alternatively, \code{n_dimred} can be an integer vector specifying the
column indices of the dimensions to use.

When \code{dimred} is specified, no additional feature selection or
standardization is performed.  This means that any settings of
\code{ntop}, \code{subset_row} and \code{scale} are ignored.

If \code{x} is a numeric matrix, setting \code{transposed=TRUE} will treat
the rows as cells and the columns as the variables/diemnsions.
This allows users to manually pass in dimensionality reduction
results without needing to wrap them in a \code{SingleCellExperiment}.
As such, no feature selection or standardization is performed,
i.e., \code{ntop}, \code{subset_row} and \code{scale} are ignored.
}

\references{
\itemize{
\item Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen,
M. (2015). Multi-scale similarities in stochastic neighbour
embedding: Reducing dimensionality while preserving both local
and global structure. Neurocomputing, 169, 246-261.
\item C. de Bodt, D. Mulders, M. Verleysen and J. A. Lee, Fast
Multiscale Neighbor Embedding, in IEEE Transactions on Neural
Networks and Learning Systems, 2020, doi:
10.1109/TNNLS.2020.3042807.
\item de Bodt, C., Mulders, D., Verleysen, M., & Lee,
J. A. (2018). Perplexity-free t-SNE and twice Student tt-SNE. In
ESANN (pp. 123-128).
\item Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen,
(2015). Multi-scale similarities in stochastic neighbour
embedding: Reducing dimensionality while preserving both local
and global structure. Neurocomputing, 169, 246-261.
}
}
\seealso{
The \code{\link[=plotFMSSNE]{plotFMSSNE()}}, \code{\link[=plotFMSTSNE]{plotFMSTSNE()}}, \code{\link[=plotMSSNE]{plotMSSNE()}} and
\code{\link[=plotMSTSNE]{plotMSTSNE()}} functions to visualise the low dimension embeddings
and \code{\link[=drQuality]{drQuality()}} the ynsupervised dimensionality reduction
quality assessment.
}
\author{
Laurent Gatto
}
