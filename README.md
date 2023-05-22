The `fmsne` R package implements the [fast multi-scale neighbour
embedding](https://github.com/lgatto/Fast_Multi-scale_NE) methods
developed by [Cyril de Bodt](https://github.com/cdebodt).

# Fast Multi-scale Neighbor Embedding

This project and the codes in this repository implement fast
multi-scale neighbor embedding algorithms for nonlinear dimensionality
reduction (DR).

The fast algorithms which are implemented are described in the article
[Fast Multiscale Neighbor
Embedding](https://ieeexplore.ieee.org/document/9308987), from Cyril
de Bodt, Dounia Mulders, Michel Verleysen and John A. Lee, published
in IEEE Transactions on Neural Networks and Learning Systems, in 2020.

The implementations are provided using the python programming
language, but involve some C and Cython codes for performance
purposes.

If you use the codes in this repository or the article, please cite
as:

> C. de Bodt, D. Mulders, M. Verleysen and J. A. Lee, "Fast Multiscale
> Neighbor Embedding," in IEEE Transactions on Neural Networks and
> Learning Systems, 2020, doi: 10.1109/TNNLS.2020.3042807.

## Installation

To install this R package:

```r
BiocManager::install("lgatto/fmsne")
```

The package depends on the following Bioconductor packages:

- [SingleCellExperiment](https://bioconductor.org/packages/SingleCellExperiment)
  for the infrastructure to hold the single-cell and reduced dimension
  data.

- [basilisk](https://bioconductor.org/packages/basilisk) to install
  and run the underlying Python implementation.

If you are looking to apply fast multi-scale neighbor embedding in
Pyhton, you can install the `fmsne` [python
package](https://pypi.org/project/fmsne/) with

```r
pip install fmsne
```
