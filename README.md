# Global Economy – PCA & Clustering Analysis

Statistical analysis of country-level economic indicators for 2021 (212 countries), written in Python. It reduces the data with **Principal Component Analysis (PCA), implemented from scratch**, and then groups countries with **K-means clustering**.

## What it does

**Principal component analysis**
1. Loads the indicators (population, GNI per capita, value added by sector, exports, imports, ...) from `analysis/dataIN/data_2021.csv`.
2. Replaces missing values with the column mean and standardizes the data.
3. Runs PCA with the `ACP` class in `analysis/pca/pca.py`: covariance matrix, eigenvalues and eigenvectors, sorting by explained variance, principal components and factor loadings.
4. Plots the eigenvalues (scree plot with the Kaiser threshold at 1), the countries on the first two components and the factor loadings as a correlogram.
5. Saves the first two components to `analysis/dataOUT/pca_results.csv`.

**Cluster analysis**
1. Standardizes exports and imports of goods and services.
2. Uses the elbow method (k = 1 to 10) to choose the number of clusters, then fits K-means with k = 3.
3. Plots the clusters and saves the labeled data to `analysis/dataOUT/cluster_results.csv`.

## Project structure

```
analysis/
  dataIN/data_2021.csv          input data
  pca/pca.py                    PCA implementation
  dataOUT/main.py               full analysis
  dataOUT/plots.py              chart helpers
  dataOUT/missing_values.py     mean imputation
  dataOUT/pca_results.csv       output: principal components
  dataOUT/cluster_results.csv   output: data with cluster labels
```

## Getting started

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
cd analysis
```

Run from the `analysis` folder so the data path and the `pca` package are found:

```bash
# macOS / Linux
PYTHONPATH=. python dataOUT/main.py

# Windows PowerShell
$env:PYTHONPATH = "."; python dataOUT/main.py
```

The charts open one after another. Close each window to continue.

## Notes
- The code comments and chart labels are in Romanian.
