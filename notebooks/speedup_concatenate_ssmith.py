#!/usr/bin/env python
# coding: utf-8


# In[2]:


import anndata
from glob import glob
import pandas as pd
import scanpy as sc
import scipy.sparse
from glob import glob
import dask

# In[3]:


# update the paths here with the 50+ files I shared with you
paths = glob("/mnt/ebs_gp2/moredata/*")



# In[5]:

var_names = None
gexs = []
obss = []
for i, path in enumerate(paths):
    sample_name = f"Sample{i+1:0>3}"

    cur_adata = sc.read_10x_h5(path, dask=True)
    print(cur_adata)

    # this filter is currently missing in our version but will almost certainly speed up access to
    # the raw data just by adding these two lines of code
    cur_adata.obs["umi_counts"] = cur_adata.X.sum(axis=1).A.flatten()
    cur_adata = cur_adata[
        cur_adata.obs["umi_counts"]>0
    ]

    cur_adata.obs_names = sample_name + ":" + cur_adata.obs_names.str.rsplit("-").str[0]
    cur_adata.obs["sample"] = sample_name

    print(path, cur_adata.shape)

    gexs.append(cur_adata.X.tocsr())
    obss.append(cur_adata.obs)

    if var_names is None:
        var_names = cur_adata.var_names
    else:
        assert (var_names == cur_adata.var_names).all()


# In[ ]:





# In[34]:


gex = scipy.sparse.vstack(gexs)

obs = pd.concat(obss)

adata = anndata.AnnData(gex)
adata.var = cur_adata.var
adata.var_names = var_names
adata.obs = obs


# In[35]:


adata


# In[36]:


adata.obs.sample(5)


# ## What we want
#
# * We want to be able to easily create a new filtered_adata matrix, eg changing the umi_counts threshold from 1000 to 500.
# * We want to have relatively easy access to the unfiltered data from each sample, primarily for qc purposes (ie, we don't really need to access all samples at once, we need to access each unfiltered sample separately but efficiently)

# In[23]:


filtered_adata = adata[adata.obs.eval("umi_counts>1000")]
filtered_adata

