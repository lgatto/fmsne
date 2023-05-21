ncells <- 100L
u <- matrix(rpois(20000, 5), ncol=ncells)
colnames(u) <- paste0("Cell", 1:100)
v <- log2(u + 1)
suppressWarnings(
    sce0 <- SingleCellExperiment(assays=list(counts=u, logcounts=v)) |>
    runPCA() |>
        scater::runTSNE())
rd0 <- reducedDims(sce0)

test_that("runFMSTSNE() works", {
    sce <- runFMSTSNE(sce0)
    expect_identical(names(reducedDims(sce)),
                     c(names(rd0), "FMSTSNE"))
    expect_identical(dim(reducedDim(sce, "FMSTSNE")),
                     c(ncells, 2L))
    expect_identical(colnames(reducedDim(sce, "FMSTSNE")),
                     c("FMSTSNE1", "FMSTSNE2"))
})

test_that("runFMSTSNE(n_components) works", {
 sce <- runFMSTSNE(sce0, n_components = 3)
    expect_identical(names(reducedDims(sce)),
                     c(names(rd0), "FMSTSNE"))
    expect_identical(dim(reducedDim(sce, "FMSTSNE")),
                     c(ncells, 3L))
    expect_identical(colnames(reducedDim(sce, "FMSTSNE")),
                     c("FMSTSNE1", "FMSTSNE2", "FMSTSNE3"))
})

test_that("runFMSSNE() works", {
    sce <- runFMSSNE(sce0)
    expect_identical(names(reducedDims(sce)),
                     c(names(rd0), "FMSSNE"))
    expect_identical(dim(reducedDim(sce, "FMSSNE")),
                     c(ncells, 2L))
    expect_identical(colnames(reducedDim(sce, "FMSSNE")),
                     c("FMSSNE1", "FMSSNE2"))
})

test_that("runFMSSNE(n_components) works", {
 sce <- runFMSSNE(sce0, n_components = 3)
    expect_identical(names(reducedDims(sce)),
                     c(names(rd0), "FMSSNE"))
    expect_identical(dim(reducedDim(sce, "FMSSNE")),
                     c(ncells, 3L))
    expect_identical(colnames(reducedDim(sce, "FMSSNE")),
                     c("FMSSNE1", "FMSSNE2", "FMSSNE3"))
})

test_that("runMSTSNE() works", {
    sce <- runMSTSNE(sce0)
    expect_identical(names(reducedDims(sce)),
                     c(names(rd0), "MSTSNE"))
    expect_identical(dim(reducedDim(sce, "MSTSNE")),
                     c(ncells, 2L))
    expect_identical(colnames(reducedDim(sce, "MSTSNE")),
                     c("MSTSNE1", "MSTSNE2"))
})

test_that("runMSTSNE(n_components) works", {
    sce <- runMSTSNE(sce0, n_components = 3)
    expect_identical(names(reducedDims(sce)),
                     c(names(rd0), "MSTSNE"))
    expect_identical(dim(reducedDim(sce, "MSTSNE")),
                     c(ncells, 3L))
    expect_identical(colnames(reducedDim(sce, "MSTSNE")),
                     c("MSTSNE1", "MSTSNE2", "MSTSNE3"))
})

test_that("runMSSNE() works", {
    sce <- runMSSNE(sce0)
    expect_identical(names(reducedDims(sce)),
                     c(names(rd0), "MSSNE"))
    expect_identical(dim(reducedDim(sce, "MSSNE")),
                     c(ncells, 2L))
    expect_identical(colnames(reducedDim(sce, "MSSNE")),
                     c("MSSNE1", "MSSNE2"))
})

test_that("runMSSNE(n_components) works", {
 sce <- runMSSNE(sce0, n_components = 3)
    expect_identical(names(reducedDims(sce)),
                     c(names(rd0), "MSSNE"))
    expect_identical(dim(reducedDim(sce, "MSSNE")),
                     c(ncells, 3L))
    expect_identical(colnames(reducedDim(sce, "MSSNE")),
                     c("MSSNE1", "MSSNE2", "MSSNE3"))
})
