# teg_get_best_n
Estimate the optimal number of components or clusters

# PCA
Selects the number of components based on comparing eigenvectors between split-halves of the data. I.e., this doesn't use the shape of the eigenvalue curve, but makes a split between components with high versus low split-half similarity.

Usage is shown in example.py, which also contains tests with simulated data to check how well the true number of latent variables is recovered.

[![DOI](https://zenodo.org/badge/621991078.svg)](https://zenodo.org/badge/latestdoi/621991078)
