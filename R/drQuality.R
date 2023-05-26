##' @title Unsupervised dimensionality reduction quality assessment
##'
##' @description
##'
##' Rank-based criteria measuring the high-dimensional neighborhood
##' preservation in the low-dimensional embedding from Lee et
##' al. (2009, 2010). These criteria are used in the experiments
##' reported in de Bodt et al. (2020).
##'
##' @details
##'
##' The `drQuality()` function computes the dimensionality reduction
##' quality assessment criteria \eqn{R_{NX}(K)} (`Rx`) and area under
##' the curve (`AUC`), as defined in Lee et al. (2009, 2010, 2103) and
##' Lee and Verleysen (2009, 2010) and as used in the experiments
##' reported in de Bodt et al. (2020).
##'
##' These criteria measure the neighborhood preservation around the
##' data points from the high-dimensional space to the
##' low-dimensional space.
##'
##' The `plotDrQuality()` fonction can be used to visualise the
##' results of the quality assessment produced by `drQuality()`.
##'
##' Based on the high-dimensional and low-dimensional Euclidean
##' distances, the sets \eqn{v_i^K} (resp. \eqn{n_i^K}) of the K
##' nearest neighbours of data point i in the high-dimensional space
##' (resp. low-dimensional space) can first be computed.
##'
##' Their average normalized agreement develops as
##'
##' \deqn{Q_{NX}(K) = \frac{1}{N} \times \sum_{i=1}^{N} \frac{|v_i^K \cap n_i^K|}{K}}{%%
##'       Q_{NX}(K) = (1/N) * Sum_{i=1}^{N} |intersect(v_i^K, n_i^K)|/K}
##'
##' where N refers to the number of data points and
##' \eqn{\cap}{intersect()} to the set intersection
##' operator. \eqn{Q_{NX}(K)} ranges between 0 and 1; the closer to 1,
##' the better.
##'
##' As the expectation of \eqn{Q_{NX}(K)} with random low-dimensional
##' coordinates is equal to \eqn{\frac{K}{N-1}}{K/(N-1}, which is
##' increasing with K
##'
##' \deqn{R_{NX}(K) = \frac{(N-1) \times Q_{NX}(K)-K}{N-1-K}}{%%
##'       R_{NX}(K) = ((N-1)*Q_{NX}(K)-K)/(N-1-K)}
##'
##' enables to more easily compare different neighbourhood sizes
##' K. \eqn{R_{NX}(K)} ranges between -1 and 1, but a negative value
##' indicates that the embedding performs worse than
##' random. Therefore, \eqn{R_{NX}(K)} typically lies between 0 and 1.
##' The \eqn{R_{NX}(K)} values of K ranging from 1 to N-2 can be
##' displayed as a curve with a log scale for K, as closer neighbours
##' typically prevail.
##'
##' The area under the resulting curve (AUC) is a scalar score that
##' grows with dimensionality reduction quality, quantified at all
##' scales with an emphasis on small ones. The AUC lies between -1 and
##' 1, but a negative value implies performances which are worse than
##' random.
##'
##' Given a dataset with N cells, the function has \eqn{O(N^2 \times log(N))}
##' time complexity. The `Kup` parameter sets the maximum neighborhood
##' size when computing the quality criteria, that is computed only
##' for the neighborhood sizes of K equal to 1 up to Kup, as opposed
##' to all possible neighborhood sizes (for `Kup` set to `NA`). This
##' implementation reduces the time complexity to \eqn{O(N \times Kup
##' \times log(N))}{O(N * Kup * log(N))}.  Setting `Kup` can hence be
##' run using much larger dataset, provided that `Kup` is small
##' compared to the total number of cells N.
##'
##' Note however that when using `Kup`, in particlar a small one, one
##' does not quantify the quality of the low-dimension embedding in
##' terms of global structure preservation (larger neighborhood
##' sizes), and instead focuses on local aspects (smaller
##' neighborhoods), which favors local DR methods over global ones and
##' misses, parly at least) the advantage of multi-scale
##' approaches. Experiment in de Bodt et al. (2020) indicate that
##' multi-scale methods are superior to single-scale ones, on both
##' moderate-sized databases (N < 10e4; DR quality is quantified using
##' K = 1, 2, ... N-2) and large-scale data sets (N >> 10e; DR quality is
##' quantified using K = 1, 2, ... Kup).
##'
##' @param object A `SingleCellExperiment` object.
##'
##' @param dimred `character()` with the `reducedDims` low dimension
##'     embeddings to be assessed. Default is to use all available
##'     with `reducedDimNames(object)`.
##'
##' @param Kup `numeric(1)` defining the maximum number of nearest
##'     neighbours to compute the quality metrics `Rx` for. Default is
##'     `ceiling(ncol(object)/2)`, i.e. neighborhood sizes of 1 up to
##'     half the number of cells. Setting `Kup` to NA will compute the
##'     `Rx` metric for all values (i.e. 1 to N-2). This will however
##'     be at a considerable cost in computation time.
##'
##' @return A `data.frame` containing the `Rx` values for the low
##'     dimensional embeddings defined by `dimred` (along the
##'     columns). The number of rows will be be `Kup` (is set) or
##'     `ncol(object)` if `Kup` was `NA` (see details). The areas
##'     under the respective curves (AUC) are stored as an attribute,
##'     named `"AUC"`.
##'
##' @export
##'
##' @seealso
##'
##' [runFMSSNE()], [runFMSSNE()], [runMSTSNE()] and [runMSSNE()] for
##' the functions that perform the multi-scale stochastic neighbour
##' embeddings.
##'
##' @references
##'
##' - Lee, J. A., & Verleysen, M. (2009). Quality assessment of
##'   dimensionality reduction: Rank-based criteria. Neurocomputing,
##'   72(7-9), 1431-1443.
##'
##' - Lee, J. A., & Verleysen, M. (2010). Scale-independent quality
##'   criteria for dimensionality reduction. Pattern Recognition
##'   Letters, 31(14), 2248-2257.
##'
##' - C. de Bodt, D. Mulders, M. Verleysen and J. A. Lee,
##'   "Fast Multiscale Neighbor Embedding," in IEEE Transactions on
##'   Neural Networks and Learning Systems, 2020, doi:
##'   10.1109/TNNLS.2020.3042807.
##'
##' - Lee, J. A., & Verleysen, M. (2009). Quality assessment of
##'   dimensionality reduction: Rank-based criteria. Neurocomputing,
##'   72(7-9), 1431-1443.
##'
##' - Lee, J. A., & Verleysen, M. (2010). Scale-independent quality
##'   criteria for dimensionality reduction. Pattern Recognition
##'   Letters, 31(14), 2248-2257.
##'
##' - Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen,
##'   M. (2013). Type 1 and 2 mixtures of Kullback–Leibler divergences
##'   as cost functions in dimensionality reduction based on
##'   similarity preservation. Neurocomputing, 112, 92-108.
##'
##' @author Laurent Gatto
##'
##' @importFrom reticulate import
##'
##' @importFrom SummarizedExperiment assay
drQuality <- function(object, dimred = reducedDimNames(object),
                      Kup = ceiling(ncol(object)/2)) {
    stopifnot(inherits(object, "SingleCellExperiment"))
    stopifnot(length(dimred) > 0)
    x <- t(as.matrix(assay(object)))
    if (is.na(Kup)) {
        ## named list -> Python Dict
        ys <- as.list(reducedDims(object))
        res <- basiliskRun(env = fmsneenv,
                           fun = .run_eval_dr_quality_from_list,
                           x = x,
                           y = ys)

    } else {
        Kup <- as.integer(Kup)
        ## Kup must be a scalar between 1 and the number of
        ## cells. Note that here, x has already been transposed, hence
        ## nrow(x) rather than ncol(x).
        stopifnot(length(Kup) == 1, Kup > 0, Kup < (nrow(x) - 1))
        res <- lapply(dimred, function(rd) {
            y <- reducedDim(object, rd)
            basiliskRun(env = fmsneenv,
                        fun = .run_eval_red_rnx_auc_from_data,
                        x = x,
                        y = y,
                        Kup = Kup)
        })
    }
    ## Convert list output to a data.frame
    ans <- sapply(res, "[[", 1)
    auc <- sapply(res, "[[", 2)
    names(auc) <- colnames(ans) <- dimred
    attr(ans,"AUC") <- auc
    ans
}

##' @param x A `data.frame`, as produced by `drQuality()`.
##'
##' @rdname drQuality
##'
##' @export
plotDrQuality <- function(x) {
    matplot(x, type = "l", lty = 1, log = "x")
    legend("topleft",
           paste(colnames(x), "-", round(attr(x, "AUC"), 2)),
           lty = 1, col = seq_len(ncol(x)))
}


.run_eval_dr_quality_from_list <- function(x, ys) {
    fmsne <- reticulate::import("fmsne")
    ## Euclidean distances are calulated in Python
    ans <- fmsne$eval_dr_quality_from_list(X = x,
                                           Ys = ys)
    names(ans) <- names(ys)
    ans <- lapply(ans, function(x) {
        names(x) <- c("Rk", "AUC")
        x
    })
    ans

}

.run_eval_red_rnx_auc_from_data <- function(x, y, Kup) {
    fmsne <- reticulate::import("fmsne")
    ## No Euclidean distances computed here
    ans <- fmsne$eval_red_rnx_auc_from_data(X = x,
                                            Y = y,
                                            Kup = Kup)
    names(ans) <- names(y)
    names(ans) <- c("Rk", "AUC")
    names(ans[["Rk"]]) <- seq(1, Kup, 1)
    ans
}
