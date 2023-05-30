test_that("drQuality() works", {
    sce <- runFMSTSNE(sce0) |>
        runFMSSNE() |>
        runMSTSNE() |>
        runMSSNE()
    rx <- drQuality(sce)
    qx <- drQuality(sce, Kup = NA)
    ndimred <- length(reducedDimNames(sce))
    expect_equal(dim(rx), c(ncol(sce) / 2, ndimred))
    expect_equal(dim(qx), c(ncol(sce) - 2, ndimred))
    expect_identical(names(rx), names(qx))
    expect_identical(order(attr(rx, "AUC")),
                     order(attr(qx, "AUC")))
})


test_that("plotDrQuality() works", {
    expect_null(plotDrQuality(rx), "list")
    expect_null(plotDrQuality(qx), "list")
})
