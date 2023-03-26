library(scater)
library(scran)
library(scRNAseq)
library(org.Mm.eg.db)
library(BiocSingular)

## Load data
sce.zeisel <- ZeiselBrainData()
sce.zeisel <- aggregateAcrossFeatures(sce.zeisel,
                                      id=sub("_loc[0-9]+$", "",
                                             rownames(sce.zeisel)))
rowData(sce.zeisel)$Ensembl <- mapIds(org.Mm.eg.db,
                                      keys=rownames(sce.zeisel),
                                      keytype="SYMBOL",
                                      column="ENSEMBL")

## Quality control
unfiltered <- sce.zeisel
stats <- perCellQCMetrics(sce.zeisel, subsets=list(
    Mt=rowData(sce.zeisel)$featureType=="mito"))
qc <- quickPerCellQC(stats, percent_subsets=c("altexps_ERCC_percent",
    "subsets_Mt_percent"))
sce.zeisel <- sce.zeisel[,!qc$discard]
colData(unfiltered) <- cbind(colData(unfiltered), stats)
unfiltered$discard <- qc$discard


## Normalisation
set.seed(1000)
clusters <- quickCluster(sce.zeisel)
sce.zeisel <- computeSumFactors(sce.zeisel, cluster=clusters)
sce.zeisel <- logNormCounts(sce.zeisel)

## Variance modelling
dec.zeisel <- modelGeneVarWithSpikes(sce.zeisel, "ERCC")
top.hvgs <- getTopHVGs(dec.zeisel, prop=0.1)

## Dimensionality reduction
set.seed(101011001)
sce.zeisel <- denoisePCA(sce.zeisel,
                         technical=dec.zeisel, subset.row=top.hvgs)
sce.zeisel <- runTSNE(sce.zeisel, dimred="PCA")

## Clustering
snn.gr <- buildSNNGraph(sce.zeisel, use.dimred="PCA")
colLabels(sce.zeisel) <- factor(igraph::cluster_walktrap(snn.gr)$membership)


plotTSNE(sce.zeisel, colour_by="label")

## Fast multi-scale neighbour embeddig
library(fmsne)

sce.zeisel <- fmsne::runMSSNE(sce.zeisel)
plotMSSNE(sce.zeisel, colour_by="label")
