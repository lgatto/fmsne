## List of packages exported from working environment
.all_deps <- c(
    "python==3.9.10",
    "numpy==1.23.5",
    "numba==0.56.4",
    "scipy==1.10.0",
    "matplotlib==3.7.0",
    "scikit-learn==1.2.1",
    "Cython==0.29.33"
)

#' @importFrom basilisk BasiliskEnvironment
fmsneenv <- BasiliskEnvironment(
    envname = "fmsne", pkgname = "fmsne",
    packages = .all_deps,
    pip = "fmsne==0.6.1",
    channels = c("bioconda", "conda-forge")
    ## paths = c("fmsne") ## in ./inst/fmsne
)
