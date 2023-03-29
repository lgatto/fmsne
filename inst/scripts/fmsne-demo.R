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

set.seed(00010101)
dec.mam <- modelGeneVarByPoisson(sce.mam)
top.mam <- getTopHVGs(dec.mam, prop=0.1)

######################################
## 12.6 Dimensionality reduction

set.seed(101010011)
sce.mam <- denoisePCA(sce.mam, technical=dec.mam, subset.row=top.mam)

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
               plotFMSTSNE(sce, colour_by="label"))


reducedDim(sce, "PCA")
reducedDim(sce, "TSNE30")

i <- sample(ncol(sce), 500)

rk <- BiocParallel::bplapply(names(reducedDims(sce)),
                             function(x) drQuality(sce[, i], dimred = x))
names(rk) <- names(reducedDims(sce))

sapply(rk, "[[", 2)

sapply(rk, "[[", 1) |>
    matplot(type = "l", lty = 1, lwd = 2, log = "x")


legend("topleft",
       paste(names(rk), round(sapply(rk, "[[", 2), 3)),
       lty = 1, col = 1:5,
       bty = "n")
