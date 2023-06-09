library(scater)
library(scran)
library(scRNAseq)
library(BiocSingular)
library(AnnotationHub)

## source: https://bioconductor.org/books/3.16/OSCA.workflows/bach-mouse-mammary-gland-10x-genomics.html

######################################
## 12.2 Data loading
sce.mam <- BachMammaryData(samples="G_1")

rownames(sce.mam) <- uniquifyFeatureNames(
    rowData(sce.mam)$Ensembl, rowData(sce.mam)$Symbol)

ens.mm.v97 <- AnnotationHub()[["AH73905"]]
rowData(sce.mam)$SEQNAME <- mapIds(ens.mm.v97, keys=rowData(sce.mam)$Ensembl,
                                   keytype="GENEID", column="SEQNAME")

######################################
## 12.3 Quality control

unfiltered <- sce.mam

is.mito <- rowData(sce.mam)$SEQNAME == "MT"
stats <- perCellQCMetrics(sce.mam, subsets=list(Mito=which(is.mito)))
qc <- quickPerCellQC(stats, percent_subsets="subsets_Mito_percent")
sce.mam <- sce.mam[,!qc$discard]

colData(unfiltered) <- cbind(colData(unfiltered), stats)
unfiltered$discard <- qc$discard

######################################
## 12.4 Normalization

set.seed(101000110)
clusters <- quickCluster(sce.mam)
sce.mam <- computeSumFactors(sce.mam, clusters=clusters)
sce.mam <- logNormCounts(sce.mam)

######################################
## 12.5 Variance modelling

nset.seed(00010101)
dec.mam <- modelGeneVarByPoisson(sce.mam)
top.mam <- getTopHVGs(dec.mam, prop=0.1)

######################################
## 12.6 Dimensionality reduction

set.seed(101010011)
sce.mam <- runPCA(sce.mam, subset_row=top.mam)

sce.mam <- runTSNE(sce.mam, dimred="PCA",
                   perplexity = 30,
                   name = "TSNE30")

sce.mam <- runTSNE(sce.mam, dimred="PCA",
                   perplexity = 200,
                   name = "TSNE200")

######################################
## 12.7 Clustering

snn.gr <- buildSNNGraph(sce.mam, use.dimred="PCA", k=25)
colLabels(sce.mam) <- factor(igraph::cluster_walktrap(snn.gr)$membership)

######################################
## Fast multi-scale neighbour embeddig
library(fmsne)

sce.mam <- fmsne::runFMSSNE(sce.mam, subset_row = top.mam)
sce.mam <- fmsne::runFMSTSNE(sce.mam, subset_row = top.mam)

reducedDims(sce.mam)

sce <- sce.mam

gridExtra::grid.arrange(
               plotReducedDim(sce, colour_by="label",
                              dimred = "TSNE30"),
               plotReducedDim(sce, colour_by="label",
                              dimred = "TSNE200"),
               plotPCA(sce, colour_by = "label"),
               plotFMSSNE(sce, colour_by="label"),
               plotFMSTSNE(sce, colour_by="label"))g


reducedDim(sce, "PCA")
reducedDim(sce, "TSNE30")

i <- sample(ncol(sce), 500)

rk <- drQuality(sce[, i])

sapply(rk, "[[", 1) |>
    matplot(type = "l", lty = 1, lwd = 2, log = "x")
legend("topleft",
       paste(names(rk), round(sapply(rk, "[[", 2), 3)),
       lty = 1, col = 1:5,
       bty = "n")

## --------------------------------------------------------------
library("fmsne")

ref1 <- readRDS("~/tmp/ref.rds")
ref1 <- runTSNE(ref1, dimred = "PCA",
                perplexity = 30,
                name = "TSNE30")


ref2 <- readRDS("~/tmp/ref2.rds")
names(colData(ref2))[13] <- "Trophoblast"
ref2 <- runPCA(ref2)
ref2 <- runTSNE(ref2, dimred = "PCA",
                perplexity = 30,
                name = "TSNE30")

gridExtra::grid.arrange(
               plotReducedDim(ref1, colour_by = "cellType",
                              dimred = "TSNE30"),
               plotPCA(ref1, colour_by = "cellType"),
               plotReducedDim(ref2, colour_by = "Trophoblast",
                              dimred = "TSNE30"),
               plotPCA(ref2, colour_by = "Trophoblast"))


gridExtra::grid.arrange(
               plotReducedDim(ref1, colour_by = "cellType",
                              dimred = "TSNE30"),
               plotPCA(ref1, colour_by = "cellType"),
               plotReducedDim(ref1, colour_by = "GA",
                              dimred = "TSNE30"),
               plotPCA(ref1, colour_by = "GA"))



ref1 <- fmsne::runFMSSNE(ref1)
## ref2 <- fmsne::runFMSSNE(ref2)

ref1 <- readRDS("~/tmp/ref.rds")
sce <- ref1[, ref1$GA == "E12.5"]
reducedDims(sce) <- NULL

## set.seed(123)
## i <- sample(ncol(sce), ncol(sce)/10)
## sce <- sce[, i]
## reducedDimNames(sce) <- paste0("00", reducedDimNames(sce))


## ===================================================
## Placental data (ref1)
## ===================================================

## ----------------------------------------------------
## Environment
library("fmsne")
library("scater")


## ----------------------------------------------------
## Load data
## ref1 <- readRDS("~/tmp/ref.rds")
## reducedDims(ref1) <- NULL
ref1 <- readRDS("fmsneRef1.rds")
ref1

## ----------------------------------------------------
## DR from PCA, 50 PCs, constructed from top 500
ref1 <- runPCA(ref1)
ref1 <- runTSNE(ref1, dimred = "PCA")
ref1 <- runFMSTSNE(ref1, dimred = "PCA")
ref1 <- runFMSSNE(ref1, dimred = "PCA")
## Verify that TSNE without dimred shows similar result
ref1 <- runTSNE(ref1, name = "PCA+TSNE")


## ----------------------------------------------------
## DR from top 500 (default)
ref1 <- runFMSTSNE(ref1, name = "FMSTSNE500")
ref1 <- runFMSSNE(ref1,  name = "FMSSNE500")

## ----------------------------------------------------
## DR from all (top 2000)
ref1 <- runPCA(ref1, ntop = 2000, name = "PCA2000")
ref1 <- runFMSTSNE(ref1, ntop = 2000, name = "FMSTSNE2000")

## ----------------------------------------------------
## Save results
## saveRDS(ref1, file = "fmsneRef1.rds")

## ----------------------------------------------------
## Quality assessment
rxRef1 <- drQuality(ref1)
saveRDS(rxRef1, file = "rxRef1.rds")

## rxRef1 <- readRDS("rxRef1.rds")

gridExtra::grid.arrange(
               plotPCA(ref1, colour_by = "cellType") + ggtitle("PCA (top 500)"),
               plotTSNE(ref1, colour_by = "cellType") + ggtitle("TSNE (from PCA)"),
               plotReducedDim(ref1, dimred = "PCA+TSNE", colour_by = "cellType") +
               ggtitle("PCA+TSNE"),
               plotFMSTSNE(ref1, colour_by = "cellType") + ggtitle("FMSTSNE (from PCA)"),
               ## plotFMSTSNE(ref1, colour_by = "cellType") + ggtitle("FMSSNE (from PCA)"),
               plotReducedDim(ref1, dimred = "FMSTSNE500", colour_by = "cellType") +
               ggtitle("FMSTSNE (top 500)"),
               ## plotReducedDim(ref1, dimred = "FMSSNE500", colour_by = "cellType") +
               ##                  ggtitle("FMSSNE (top 500)"),
               plotReducedDim(ref1, dimred = "PCA2000",
                              colour_by = "cellType") +
               ggtitle("PCA (top 2000)"),
               plotReducedDim(ref1, dimred = "FMSTSNE2000",
                              colour_by = "cellType") +
               ggtitle("FMSTSNE (top 2000)"),
               ncol = 2)


## ===================================================
## Testis data
## ===================================================

## ----------------------------------------------------
## Environment
library("fmsne")
library("scater")
library("scran")


## ----------------------------------------------------
## Load data
sce <- readRDS("sceTestis.rds")

set.seed(100)
clust.testis <- quickCluster(sce)
sce <- computeSumFactors(sce, cluster=clust.testis, min.mean=0.1)
sce <- logNormCounts(sce)
sce

## ----------------------------------------------------
## DR from PCA, 50 PCs, constructed from top 500
sce <- runPCA(sce)
sce <- runTSNE(sce, dimred = "PCA")
sce <- runFMSTSNE(sce, dimred = "PCA")

## ----------------------------------------------------
## DR from top 500 (default)
sce <- runFMSTSNE(sce, name = "FMSTSNE500")

## ----------------------------------------------------
## DR from (top 5000)
sce <- runPCA(sce, ntop = 5000, name = "PCA5000")
sce <- runFMSTSNE(sce, ntop = 5000, name = "FMSTSNE5000")

## ----------------------------------------------------
## DR from (all)
sce <- runPCA(sce, ntop = ncol(sce), name = "PCAall")
sce <- runFMSTSNE(sce, ntop = ncol(sce), name = "FMSTSNEall")

## ----------------------------------------------------
## Save results
saveRDS(sce, file = "sceTestis.rds")

## ----------------------------------------------------
## Quality assessment
rxSce <- drQuality(sce)
saveRDS(rxSce, file = "rxSce.rds")


## Check also Donor
gridExtra::grid.arrange(
               plotPCA(sce, colour_by = "type") + ggtitle("PCA (top 500)"),
               plotTSNE(sce, colour_by = "type") + ggtitle("TSNE (from PCA)"),
               plotFMSTSNE(sce, colour_by = "type") + ggtitle("FMSTSNE (from PCA)"),
               plotReducedDim(sce, dimred = "FMSTSNE500", colour_by = "type")
               + ggtitle("FMSTSNE (top 500)"),
               plotReducedDim(sce, dimred = "PCA5000", colour_by = "type")
               + ggtitle("PCA (top 5000)"),
               plotReducedDim(sce, dimred = "FMSTSNE5000", colour_by = "type")
               + ggtitle("FMSTNSE (top 5000)"),
               plotReducedDim(sce, dimred = "PCAall", colour_by = "type")
               + ggtitle("PCA (all)"),
               plotReducedDim(sce, dimred = "FMSTSNEall", colour_by = "type")
               + ggtitle("FMSTNSE (all)"),
               ncol = 2)
