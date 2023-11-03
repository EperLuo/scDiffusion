import numpy as np
import anndata as ad
import scanpy as sc
import celltypist
from sklearn.model_selection import train_test_split  
from imblearn.over_sampling import RandomOverSampler
  
def split_adata(adata, train_ratio=0.8, random_state=42):  
    indexes = np.arange(adata.shape[0])  
    train_indexes, test_indexes = train_test_split(indexes, train_size=train_ratio, random_state=random_state)  
  
    train_adata = adata[train_indexes].copy()  
    test_adata = adata[test_indexes].copy()  
  
    return train_adata, test_adata  

adata = sc.read_h5ad('/data1/lep/Workspace/guided-diffusion/data/tabula_muris/all.h5ad')
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=10)
adata.var_names_make_unique()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# rebalance
adata, test_adata = split_adata(adata, train_ratio=0.8, random_state=42) 
celltype = adata.obs['celltype'].values
ros = RandomOverSampler(random_state=42)  
X_resampled, y_resampled = ros.fit_resample(adata.X, celltype)
adata_resampled = ad.AnnData(X_resampled[:80000])
adata_resampled.var_names = adata.var_names
print(adata_resampled)
# if you want to save the testset
test_adata.write_h5ad('data/testset_muris_all.h5ad')

new_model = celltypist.train(adata_resampled, labels = y_resampled[:80000], n_jobs=32)

new_model.write('checkpoint/my_celltypist.pkl')
