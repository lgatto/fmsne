test_that("calculateFMSTSNE() works", {
    res1 <- reducedDim(runFMSTSNE(sce0), "FMSTSNE")
    res2 <- calculateFMSTSNE(sce0)
    expect_identical(res1, res2)
    res3 <- calculateFMSTSNE(logcounts(sce0))
    expect_identical(res2, res3)
})


test_that("calculateFMSSNE() works", {
    res1 <- reducedDim(runFMSSNE(sce0), "FMSSNE")
    res2 <- calculateFMSSNE(sce0)
    expect_identical(res1, res2)
    res3 <- calculateFMSSNE(logcounts(sce0))
    expect_identical(res2, res3)
})

test_that("calculateMSTSNE() works", {
    res1 <- reducedDim(runMSTSNE(sce0), "MSTSNE")
    res2 <- calculateMSTSNE(sce0)
    expect_identical(res1, res2)
    res3 <- calculateMSTSNE(logcounts(sce0))
    expect_identical(res2, res3)
})

test_that("calculateMSSNE() works", {
    res1 <- reducedDim(runMSSNE(sce0), "MSSNE")
    res2 <- calculateMSSNE(sce0)
    expect_identical(res1, res2)
    res3 <- calculateMSSNE(logcounts(sce0))
    expect_identical(res2, res3)
})
