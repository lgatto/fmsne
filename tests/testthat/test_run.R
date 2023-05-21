################################################################################
## FMSTSNE

test_that("runFMSTSNE() works", {
    sce <- runFMSTSNE(sce0)
    expect_identical(reducedDimNames(sce),
                     c(names(rd0), "FMSTSNE"))
    expect_identical(dim(reducedDim(sce, "FMSTSNE")),
                     c(ncells, 2L))
    expect_identical(colnames(reducedDim(sce, "FMSTSNE")),
                     c("FMSTSNE1", "FMSTSNE2"))
})

test_that("runFMSTSNE(n_components) works", {
 sce <- runFMSTSNE(sce0, n_components = 3)
    expect_identical(reducedDimNames(sce),
                     c(names(rd0), "FMSTSNE"))
    expect_identical(dim(reducedDim(sce, "FMSTSNE")),
                     c(ncells, 3L))
    expect_identical(colnames(reducedDim(sce, "FMSTSNE")),
                     c("FMSTSNE1", "FMSTSNE2", "FMSTSNE3"))
})

test_that("runFMSTSNE(subset_row) works", {
    sce <- runFMSTSNE(sce0, subset_row = 1:100, name = "FMSTSNE_1")
    sce <- runFMSTSNE(sce, subset_row = rep(c(TRUE, FALSE), each = 100),
                      name = "FMSTSNE_2")
    sce <- runFMSTSNE(sce, subset_row = paste0("Gene", 1:100),
                      name = "FMSTSNE_3")
    expect_equivalent(reducedDim(sce, "FMSTSNE_1"),
                      reducedDim(sce, "FMSTSNE_2")) ## Different colnames
    expect_equivalent(reducedDim(sce, "FMSTSNE_1"),
                      reducedDim(sce, "FMSTSNE_3")) ## Different colnames
    expect_identical(colnames(reducedDim(sce, "FMSTSNE_1")),
                     c("FMSTSNE_11", "FMSTSNE_12"))
    expect_identical(colnames(reducedDim(sce, "FMSTSNE_2")),
                     c("FMSTSNE_21", "FMSTSNE_22"))
    expect_identical(colnames(reducedDim(sce, "FMSTSNE_3")),
                     c("FMSTSNE_31", "FMSTSNE_32"))
    expect_identical(dim(reducedDim(sce, "FMSTSNE_1")),
                     c(ncells, 2L))
    expect_identical(reducedDimNames(sce),
                     c(names(rd0), "FMSTSNE_1", "FMSTSNE_2", "FMSTSNE_3"))
})

################################################################################
## FMSSNE

test_that("runFMSSNE() works", {
    sce <- runFMSSNE(sce0)
    expect_identical(reducedDimNames(sce),
                     c(names(rd0), "FMSSNE"))
    expect_identical(dim(reducedDim(sce, "FMSSNE")),
                     c(ncells, 2L))
    expect_identical(colnames(reducedDim(sce, "FMSSNE")),
                     c("FMSSNE1", "FMSSNE2"))
})

test_that("runFMSSNE(n_components) works", {
 sce <- runFMSSNE(sce0, n_components = 3)
    expect_identical(reducedDimNames(sce),
                     c(names(rd0), "FMSSNE"))
    expect_identical(dim(reducedDim(sce, "FMSSNE")),
                     c(ncells, 3L))
    expect_identical(colnames(reducedDim(sce, "FMSSNE")),
                     c("FMSSNE1", "FMSSNE2", "FMSSNE3"))
})

test_that("runFMSSNE(subset_row) works", {
    sce <- runFMSSNE(sce0, subset_row = 1:100, name = "FMSSNE_1")
    sce <- runFMSSNE(sce, subset_row = rep(c(TRUE, FALSE), each = 100),
                      name = "FMSSNE_2")
    sce <- runFMSSNE(sce, subset_row = paste0("Gene", 1:100),
                      name = "FMSSNE_3")
    expect_equivalent(reducedDim(sce, "FMSSNE_1"),
                      reducedDim(sce, "FMSSNE_2")) ## Different colnames
    expect_equivalent(reducedDim(sce, "FMSSNE_1"),
                      reducedDim(sce, "FMSSNE_3")) ## Different colnames
    expect_identical(colnames(reducedDim(sce, "FMSSNE_1")),
                     c("FMSSNE_11", "FMSSNE_12"))
    expect_identical(colnames(reducedDim(sce, "FMSSNE_2")),
                     c("FMSSNE_21", "FMSSNE_22"))
    expect_identical(colnames(reducedDim(sce, "FMSSNE_3")),
                     c("FMSSNE_31", "FMSSNE_32"))
    expect_identical(dim(reducedDim(sce, "FMSSNE_1")),
                     c(ncells, 2L))
    expect_identical(reducedDimNames(sce),
                     c(names(rd0), "FMSSNE_1", "FMSSNE_2", "FMSSNE_3"))
})


################################################################################
## MSTSNE

test_that("runMSTSNE() works", {
    sce <- runMSTSNE(sce0)
    expect_identical(reducedDimNames(sce),
                     c(names(rd0), "MSTSNE"))
    expect_identical(dim(reducedDim(sce, "MSTSNE")),
                     c(ncells, 2L))
    expect_identical(colnames(reducedDim(sce, "MSTSNE")),
                     c("MSTSNE1", "MSTSNE2"))
})

test_that("runMSTSNE(n_components) works", {
    sce <- runMSTSNE(sce0, n_components = 3)
    expect_identical(reducedDimNames(sce),
                     c(names(rd0), "MSTSNE"))
    expect_identical(dim(reducedDim(sce, "MSTSNE")),
                     c(ncells, 3L))
    expect_identical(colnames(reducedDim(sce, "MSTSNE")),
                     c("MSTSNE1", "MSTSNE2", "MSTSNE3"))
})

test_that("runMSTSNE(subset_row) works", {
    sce <- runMSTSNE(sce0, subset_row = 1:100, name = "MSTSNE_1")
    sce <- runMSTSNE(sce, subset_row = rep(c(TRUE, FALSE), each = 100),
                      name = "MSTSNE_2")
    sce <- runMSTSNE(sce, subset_row = paste0("Gene", 1:100),
                      name = "MSTSNE_3")
    expect_equivalent(reducedDim(sce, "MSTSNE_1"),
                      reducedDim(sce, "MSTSNE_2")) ## Different colnames
    expect_equivalent(reducedDim(sce, "MSTSNE_1"),
                      reducedDim(sce, "MSTSNE_3")) ## Different colnames
    expect_identical(colnames(reducedDim(sce, "MSTSNE_1")),
                     c("MSTSNE_11", "MSTSNE_12"))
    expect_identical(colnames(reducedDim(sce, "MSTSNE_2")),
                     c("MSTSNE_21", "MSTSNE_22"))
    expect_identical(colnames(reducedDim(sce, "MSTSNE_3")),
                     c("MSTSNE_31", "MSTSNE_32"))
    expect_identical(dim(reducedDim(sce, "MSTSNE_1")),
                     c(ncells, 2L))
    expect_identical(reducedDimNames(sce),
                     c(names(rd0), "MSTSNE_1", "MSTSNE_2", "MSTSNE_3"))
})


################################################################################
## MSSNE

test_that("runMSSNE() works", {
    sce <- runMSSNE(sce0)
    expect_identical(reducedDimNames(sce),
                     c(names(rd0), "MSSNE"))
    expect_identical(dim(reducedDim(sce, "MSSNE")),
                     c(ncells, 2L))
    expect_identical(colnames(reducedDim(sce, "MSSNE")),
                     c("MSSNE1", "MSSNE2"))
})

test_that("runMSSNE(n_components) works", {
 sce <- runMSSNE(sce0, n_components = 3)
    expect_identical(reducedDimNames(sce),
                     c(names(rd0), "MSSNE"))
    expect_identical(dim(reducedDim(sce, "MSSNE")),
                     c(ncells, 3L))
    expect_identical(colnames(reducedDim(sce, "MSSNE")),
                     c("MSSNE1", "MSSNE2", "MSSNE3"))
})

test_that("runMSSNE(subset_row) works", {
    sce <- runMSSNE(sce0, subset_row = 1:100, name = "MSSNE_1")
    sce <- runMSSNE(sce, subset_row = rep(c(TRUE, FALSE), each = 100),
                      name = "MSSNE_2")
    sce <- runMSSNE(sce, subset_row = paste0("Gene", 1:100),
                      name = "MSSNE_3")
    expect_equivalent(reducedDim(sce, "MSSNE_1"),
                      reducedDim(sce, "MSSNE_2")) ## Different colnames
    expect_equivalent(reducedDim(sce, "MSSNE_1"),
                      reducedDim(sce, "MSSNE_3")) ## Different colnames
    expect_identical(colnames(reducedDim(sce, "MSSNE_1")),
                     c("MSSNE_11", "MSSNE_12"))
    expect_identical(colnames(reducedDim(sce, "MSSNE_2")),
                     c("MSSNE_21", "MSSNE_22"))
    expect_identical(colnames(reducedDim(sce, "MSSNE_3")),
                     c("MSSNE_31", "MSSNE_32"))
    expect_identical(dim(reducedDim(sce, "MSSNE_1")),
                     c(ncells, 2L))
    expect_identical(reducedDimNames(sce),
                     c(names(rd0), "MSSNE_1", "MSSNE_2", "MSSNE_3"))
})
