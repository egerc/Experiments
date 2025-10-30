from numpy import ndarray
from sklearn.decomposition import NMF
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from numpy.typing import NDArray
from exp_runner.runner import Variable, run
import warnings
import scanpy as sc

warnings.filterwarnings("ignore")

def n_components_generator():
    for i in [1,2,3,4,5,6,7,8,9,10,12,16,24,32]:
        yield Variable(i, i)


def data_generator():
    adata = sc.read_h5ad("/Users/egerc/Downloads/livercellatlas.h5ad")
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=500)
    adata = adata[:, adata.var["highly_variable"] == True]
    celltypes = list(adata.obs["annot"].unique())
    for celltype in celltypes:
        data = adata[adata.obs["annot"] == celltype].X.toarray()
        n_obs, n_var = data.shape
        yield Variable(data, name=celltype, metadata={"n_obs": n_obs, "n_var": n_var})


# --- Experiment ---
@run
def experiment(X: ndarray, n_components: int):
    """
    Runs NMF on X with n_components and returns CSV-compatible metrics.
    """
    nmf = NMF(n_components=n_components).fit(X)
    X_recon = nmf.inverse_transform(nmf.transform(X))
    return {
        "mae": float(mean_absolute_error(X, X_recon)),
        "mse": float(mean_squared_error(X, X_recon)),
        "explained_variance": float(explained_variance_score(X, X_recon))
    }


if __name__ == "__main__":
    experiment()