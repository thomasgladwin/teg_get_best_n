# teg_get_best_n
Estimate the optimal number of components in a PCA, using the SHEM procedure: Split-Half Eigenvector Matching.

The get_n_components function estimates the true (or "generating") number of principal components. While scree/elbow/knee criteria for the eigenvalues curve is common, this is known to be a very fallible heuristic. The rationale of this alternative procedure is that true principal components should be found in random split halves of the data. The estimate is therefore based on measuring the similarity of eigenvectors between a set of split-halves; i.e., the procedure doesn't use the shape of the eigenvalue curve. Instead, a separation is made between components with high versus low split-half similarity.

Detail: For an nd-array X, with shape == (nObservations, nVariables), a number of random splits are performed. For each split separately, a PCA is performed, via eigendecomposition of the covariance matrix of X. Each of the first split's eigenvectors is matched to the most-similar of the second split's eigenvectors. Similarity is measured via the dot product. The vector of similarities is sorted from high to low, and the vectors are averaged over all random splits. Finally, the optimal seperation between the high versus low similarities is determined by a basic between-within variance criterion. An estimated zero components is possible.

Usage is shown in example.py. This also contains tests with simulated data to check how well the true number of latent variables, used to generate simulated data, is recovered.

[![DOI](https://zenodo.org/badge/621991078.svg)](https://zenodo.org/badge/latestdoi/621991078)
