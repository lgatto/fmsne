##' @title Unsupervised dimensionality reduction quality assessment
##'
##' @description
##'
##' Rank-based criteria measuring the high-dimensional neighborhood
##' preservation in the low-dimensional embedding Lee et al. (2009,
##' 2010). These criteria are used in the experiments reported in de
##' Bodt et al. (2020).
##'
##' @details
##'
##' The `drQuality()` function computes the dimensionality reduction
##' quality assessment criteria $R_{NX}(K)$ and AUC, as defined in Lee
##' et al. (2009, 2010, 2103) and Lee and Verleysen (2009, 2010) and
##' as used in the experiments reported in de Bodt et al. (2020).
##'
##' These criteria measure the neighborhood preservation around the
##' data points from the high-dimensional space to the
##' low-dimensional space.
##'
##' Based on the high-dimensional and low-dimensional Euclidean
##' distances, the sets $v_i^K$ (resp. $n_i^K$) of the K nearest
##' neighbours of data point i in the high-dimensional space
##' (resp. low-dimensional space) can first be computed.
##'
##' Their average normalized agreement develops as $Q_{NX}(K) = (1/N)
##' * \sum_{i=1}^{N} |v_i^K \cap n_i^K|/K$, where N refers to the
##' number of data points and $\cap$ to the set intersection
##' operator. $Q_{NX}(K)$ ranges between 0 and 1; the closer to 1, the
##' better.
##'
##' As the expectation of $Q_{NX}(K)$ with random low-dimensional
##' coordinates is equal to $K/(N-1)$, which is increasing with $K$,
##' $R_{NX}(K) = ((N-1)*Q_{NX}(K)-K)/(N-1-K)$ enables to more easily
##' compare different neighbourhood sizes $K$. $R_{NX}(K)$ ranges
##' between -1 and 1, but a negative value indicates that the
##' embedding performs worse than random. Therefore, $R_{NX}(K)$
##' typically lies between 0 and 1. The $R_{NX}(K)$ values for K=1 to
##' N-2 can be displayed as a curve with a log scale for K, as closer
##' neighbours typically prevail.
##'
##' The area under the resulting curve (AUC) is a scalar score which
##' grows with dimensionality reduction quality, quantified at all
##' scales with an emphasis on small ones. The AUC lies between -1 and
##' 1, but a negative value implies performances which are worse than
##' random.
##'
##' Given a dataset with N cells, the function has $O(N**2 log(N))$
##' time complexity. It can hence run using databases with up to a few
##' thousands of cells.
##'
##' The `Kup` parameter sets the maximum neighborhood size when
##' computing the quality criteria, that is computed only for the
##' neighborhood sizes K up to Kup, as opposed to all possible
##' neighborhood sizes when `Kup` is `NA`. This reduces the time
##' complexity to $O(N*Kup*log(N))$. Setting `Kup` can hence be run
##' using much larger dataset, provided that `Kup` is small compared
##' to the total number of cells N.
##'
##' @param object A `SingleCellExperiment` object.
##'
##' @param dimred `character(1)` with the `reducedDims` low dimension
##'     embeddings to be assessed.
##'
##' @return
##'
##' @export
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
drQuality <- function(object, dimred = "PCA", Kup = NA) {
    stopifnot(inherits(object, "SingleCellExperiment"))
    x <- t(as.matrix(assay(object)))
    y <- reducedDim(object, dimred)
    if (is.na(Kup)) {
        ans <- basiliskRun(env = fmsneenv,
                           fun = .run_eval_dr_quality_from_data,
                           x = x,
                           y = y)
    } else {
        Kup <- as.integer(Kup)
        stopifnot(length(Kup) == 1, Kup > 0)
        ans <- basiliskRun(env = fmsneenv,
                           fun = .run_eval_red_rnx_auc_from_data,
                           x = x,
                           y = y,
                           Kup = Kup)
    }
    ans
}

.run_eval_dr_quality_from_data <- function(x, y) {
    fmsne <- reticulate::import("fmsne")
    ans <- fmsne$eval_dr_quality_from_data(X = x,
                                           Y = y)
    names(ans) <- c("Rk", "AUC")
    names(ans[["Rk"]]) <- seq(1, nrow(x)-2, 1)
    ans
}

.run_eval_red_rnx_auc_from_data <- function(x, y, Kup) {
    fmsne <- reticulate::import("fmsne")
    ans <- fmsne$eval_red_rnx_auc_from_data(X = x,
                                            Y = y,
                                            Kup = Kup)
    names(ans) <- c("Rk", "AUC")
    names(ans[["Rk"]]) <- seq(1, Kup, 1)
    ans
}
