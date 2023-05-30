##' @title Multi-scale stochastic neighbour embedding
##'
##' @description
##'
##' The `fmsne` package offers various functions to perform nonlinear
##' dimensionality reduction through multi-scale (MS) stochastic
##' neighbor embedding (SNE) or t-distributed SNE (t-SNE), including
##' fast versions thereof.
##'
##' - The `calculateMSSNE()` function performs a nonlinear
##'   dimensionality reduction through multi-scale SNE, as presented
##'   in Lee et al. (2015) below and summarized in de Bodt et
##'   al. (2020).  Given a data set with N samples, it has \eqn{O(N^2
##'   log(N))} time complexity.
##'
##' - The `calculateMSTSNE()` function performs nonlinear
##'   dimensionality reduction through multi-scale t-SNE, as presented
##'   in the reference de Bodt et al. (2018) below and summarized in
##'   de Bodt et al. (2020). Given a data set with N samples, it has
##'   \eqn{O(N^2 log(N))} time complexity.
##'
##' - The `calculateFMSSNE()` function performs nonlinear
##'   dimensionality reduction through fast multi-scale SNE, as
##'   presented in Lee et al. (2015) below. Given a data set with N
##'   samples, it has \eqn{O(N log(N)^2)} time complexity.
##'
##' - The `calculateFMSTSNE()` function performs nonlinear
##'   dimensionality reduction through fast multi-scale t-SNE, as
##'   presented in the de Bodt et al. (2020) below.  Given a data set
##'   with N samples, it has \eqn{O(N log(N))^2} time complexity.
##'
##' Each method can also be called with `run[F]MS[T]SNE()` to store
##' the result as a new `SingleCellExperiment` reduced dimension
##' `reducedDim` instance.
##'
##' See the vignette for further details.
##'
##' @section Feature selection:
##'
##' This section is adapted from the `scater` package manual and is
##' relevant if `x` is a numeric matrix of (log-)expression values
##' with features in rows and cells in columns; or if `x` is a
##' `SingleCellExperiment` and `dimred = NULL`.  In the latter, the
##' expression values are obtained from the assay specified by
##' `exprs_values`.
##'
##' The `subset_row` argument specifies the features to use for
##' dimensionality reduction.  The aim is to allow users to specify
##' highly variable features to improve the signal/noise ratio, or to
##' specify genes in a pathway of interest to focus on particular
##' aspects of heterogeneity.
##'
##' If `subset_row = NULL`, the `ntop` features with the largest
##' variances are used instead.  Using the same underlying function as
##' in the `scater` package, we literally compute the variances from
##' the expression values without considering any mean-variance trend,
##' so often a more considered choice of genes is possible, e.g., with
##' `scran` functions.  Note that the value of `ntop` is ignored if
##' `subset_row` is specified.
##'
##' If `scale = TRUE`, the expression values for each feature are
##' standardized so that their variance is unity.  This will also
##' remove features with standard deviations below 1e-8.
##'
##' @section Using reduced dimensions:
##'
##' This section is adapted from the `scater` package manual.
##'
##' If `x` is a `SingleCellExperiment`, the neighbour embedding
##' methods can be applied on existing dimensionality reduction
##' results in `x` by setting the `dimred` argument.  This is
##' typically used to run slower non-linear algorithms (t-SNE, UMAP)
##' on the results of fast linear decompositions (PCA).  We might also
##' use this with existing reduced dimensions computed from *a
##' priori* knowledge (e.g., gene set scores), where further
##' dimensionality reduction could be applied to compress the data.
##'
##' The matrix of existing reduced dimensions is taken from
##' `reducedDim(x, dimred)`.  By default, all dimensions are used to
##' compute the second set of reduced dimensions.  If `n_dimred` is
##' also specified, only the first `n_dimred` columns are used.
##' Alternatively, `n_dimred` can be an integer vector specifying the
##' column indices of the dimensions to use.
##'
##' When `dimred` is specified, no additional feature selection or
##' standardization is performed.  This means that any settings of
##' `ntop`, `subset_row` and `scale` are ignored.
##'
##' If `x` is a numeric matrix, setting `transposed=TRUE` will treat
##' the rows as cells and the columns as the variables/diemnsions.
##' This allows users to manually pass in dimensionality reduction
##' results without needing to wrap them in a `SingleCellExperiment`.
##' As such, no feature selection or standardization is performed,
##' i.e., `ntop`, `subset_row` and `scale` are ignored.
##'
##' @param x Matrix (for `calcualate*()`) or object of class
##'     `SingleCellExperiment` (for `run*()`) containing a numeric
##'     assay with log-expression values where rows are features and
##'     columns are cells.
##'
##' @param ncomponents `integer(1)` indicating the number of t-SNE
##'     dimensions to obtain. Default is 2L.
##'
##' @param ntop `integer(1)` specifying the number of features with
##'     the highest variances to use for dimensionality
##'     reduction. Default is 500.
##'
##' @param scale `logical(1)` indicating whether the expression values
##'     should be standardized? Default is `FALSE`.
##'
##' @param transposed `logical(1)` indicating whether `x` is
##'     transposed with cells in rows? Default is `FALSE`.
##'
##' @param exprs_values `integer(1)` or `character(1)` indicating
##'     which assay of ‘x’ contains the expression values. Default is
##'     `"logcounts"`.
##'
##' @param dimred `character(1)` specifying the optional
##'     dimensionality reduction results to use. Default is `NULL`.
##'
##' @param n_dimred `interger(1)` specifying the dimensions to use if
##'     `dimred` is specified. Default is `NULL`, to use all
##'     components.
##'
##' @param init `character(1)`. If equal to "pca" (default), the LD
##'     embedding is initialized with the first `n_components`
##'     principal components computed on `x`. If equal to "random",
##'     the LD embedding is initialized randomly, using a uniform
##'     Gaussian distribution with a variance equal to
##'     var. Alternatively, `init` can also be a number of cells by
##'     `n_components` matrix of dimension (not tested - please file
##'     an issue in case of problems.).
##'
##' @param nit_max `numeric(1)` defining the maximum number of L-BFGS
##'     steps at each stage of the multi-scale optimization, which is
##'     defined in Lee et al. (2015). Default is 30.
##'
##' @param gtol `numeric(1)` defining the tolerance for the infinite
##'     norm of the gradient in the L-BFGS algorithm. The L-BFGS
##'     iterations hence stop when $max{|g_i | i = 1, ..., n} <= gtol$
##'     where $g_i$ is the i-th component of the gradient.
##'
##' @param ftol `numeric(1)` defining the tolerance for the relative
##'     updates of the cost function value in L-BFGS. Default is
##'     2.2204460492503131e-09.
##'
##' @param maxls `numeric(1)` maximum number of line search steps per
##'     L-BFGS-B iteration. Default is 50.
##'
##' @param maxcor `numeric(1)` defining the maximum number of variable
##'     metric corrections used to define the limited memory matrix in
##'     L-BFGS. Default is 10.
##'
##' @param fit_U `logical(1)` indicating whether to fit the U in the
##'     definition of the LD similarities in Lee et al. (2015). If
##'     TRUE (default), the U is tuned as in Lee et
##'     al. (2015). Otherwise, it is forced to 1. Setting `fit_U` to
##'     TRUE usually tends to slightly improve DR quality at the
##'     expense of slightly increasing computation time.
##'
##' @param bht `logical(1)` indicating whether to fit the U in the
##'     definition of the LD similarities in Lee et al. (2015). If
##'     `TRUE`, the U is tuned as in Lee et al. (2015). Otherwise, it
##'     is forced to 1. Setting `fit_U` to `TRUE` usually tends to
##'     slightly improve dimensionality reduction quality at the
##'     expense of slightly increasing computation time.
##'
##' @param fseed strictly positive `integer(1)` defining the random
##'     seed used to perform the random sampling of the
##'     high-dimensional data set at the different scales.
##'
##' @param subset_row Vector specifying the subset of features to use
##'     for dimensionality reduction. This can be a character vector
##'     of row names, an integer vector of row indices or a logical
##'     vector. Default is `NULL` that takes all features.
##'
##' @param name `character(1)` specifying the name to be used to store
##'     the result in the `reducedDims` of the output. Default is
##'     `"MSSNE"`, `"MSTSNE"`, `"FMSSNE"` or `"FMSTSNE"` depending on
##'     the function.
##'
##' @return The `calculate*()` functions reduced a reduced dimension
##'     matrix of dimensions `ncol(x)` (i.e. cells) by
##'     `ncomponents`. The `run*()` functions return a modified ‘x’
##'     that contains the reduced dimension coordinates in
##'     ‘reducedDim(x, name)’.
##'
##' @seealso
##'
##' The [plotFMSSNE()], [plotFMSTSNE()], [plotMSSNE()] and
##' [plotMSTSNE()] functions to visualise the low dimension embeddings
##' and [drQuality()] the ynsupervised dimensionality reduction
##' quality assessment.
##'
##' @aliases runMSSNE runMSTSNE runFMSSNE runFMSTSNE
##'
##' @aliases calculateMSSNE calculateMSTSNE calculateFMSSNE calculateFMSTSNE
##'
##' @name fmsne
##'
##' @references
##'
##' - Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen,
##'   M. (2015). Multi-scale similarities in stochastic neighbour
##'   embedding: Reducing dimensionality while preserving both local
##'   and global structure. Neurocomputing, 169, 246-261.
##'
##' - C. de Bodt, D. Mulders, M. Verleysen and J. A. Lee, Fast
##'   Multiscale Neighbor Embedding, in IEEE Transactions on Neural
##'   Networks and Learning Systems, 2020, doi:
##'   10.1109/TNNLS.2020.3042807.
##'
##' - de Bodt, C., Mulders, D., Verleysen, M., & Lee,
##'   J. A. (2018). Perplexity-free t-SNE and twice Student tt-SNE. In
##'   ESANN (pp. 123-128).
##'
##' - Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen,
##'   (2015). Multi-scale similarities in stochastic neighbour
##'   embedding: Reducing dimensionality while preserving both local
##'   and global structure. Neurocomputing, 169, 246-261.
##'
##' @docType package
##'
##' @author Laurent Gatto
NULL
