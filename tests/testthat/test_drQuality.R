sce <- runFMSTSNE(sce0) |>
    runFMSSNE()

drq <- lapply(reducedDimNames(sce),
              function(x) drQuality(sce, x))
names(drq) <- reducedDimNames(sce)
