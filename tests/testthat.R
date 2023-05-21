# This file is part of the standard setup for testthat.
# It is recommended that you do not modify it.
#
# Where should you do additional test configuration?
# Learn more about the roles of various files in:
# * https://r-pkgs.org/tests.html
# * https://testthat.r-lib.org/reference/test_package.html#special-files

library(testthat)
library(fmsne)
library(SingleCellExperiment)
library(scater)

ncells <- 100L
u <- matrix(rpois(20000, 5), ncol=ncells)
colnames(u) <- paste0("Cell", 1:100)
rownames(u) <- paste0("Gene", 1:200)
v <- log2(u + 1)
suppressWarnings(
    sce0 <- SingleCellExperiment(assays=list(counts=u, logcounts=v)) |>
        runPCA() |>
        runTSNE())
rd0 <- reducedDims(sce0)

test_check("fmsne")
