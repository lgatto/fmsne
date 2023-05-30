test_that("drQuality() and plotDrQuality work", {
    sce <- runFMSTSNE(sce0) |>
        runFMSSNE()
    rx <- drQuality(sce)
    qx <- drQuality(sce, Kup = NA)
    ndimred <- length(reducedDimNames(sce))
    expect_equal(dim(rx), c(ncol(sce) / 2, ndimred))
    expect_equal(dim(qx), c(ncol(sce) - 2, ndimred))
    expect_identical(names(rx), names(qx))
    expect_null(plotDrQuality(rx), "list")
    expect_null(plotDrQuality(qx), "list")
})
