#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ctxbio.imports import *


# In[2]:


import glob


# In[3]:


# adjust as appropriate for your system
basepath = "/mnt/ssd/moredata/"


# In[ ]:





# The next 3 sections split up the original notebook so we can see exactly which parts of the code are slow.  The original timings were:
# ```
# CPU times: user 27 s, sys: 2.34 s, total: 29.3 s
# Wall time: 29.2 s
# ```
# The code below takes 24.5s total, so the original code takes 20% longer.  This could be because of NFS, or because the test server has faster CPUs.

# In[4]:


get_ipython().run_cell_magic('time', '', 'adata_raw = {}\npaths = glob.glob(os.path.join(basepath, "*.h5ad"))\nfor i, path in enumerate(paths):#[:4]:\n    print(path, i+1, "of", len(paths))\n    name = os.path.basename(path).split(".")[0]\n    cur_adata = sc.read_10x_h5(path)\n    adata_raw[name] = cur_adata')


# In[ ]:





# Timings: 
# ```
# CPU times: user 21.5 s, sys: 1.82 s, total: 23.4 s
# Wall time: 23.3 s
# ```
# Reading the files CPU-bound when accesssing `/mnt/ebs_gp2/`, and consumes most of the time.  Switching to `ssd` provides no benefit, as expected.

# In[5]:


get_ipython().run_cell_magic('time', '', 'paths = glob.glob(os.path.join(basepath, "*.h5ad"))\nfor i, path in enumerate(paths):#[:4]:\n    print(path, i+1, "of", len(paths))\n    name = os.path.basename(path).split(".")[0]\n    cur_adata = adata_raw[name]\n    cur_adata.var_names_make_unique()')


# Timings: 
# ```
# CPU times: user 145 ms, sys: 4.42 ms, total: 149 ms
# Wall time: 143 ms
# ```
# Making the names unique takes negligible time.

# In[6]:


get_ipython().run_cell_magic('time', '', 'adata_collection = {}\npaths = glob.glob(os.path.join(basepath, "*.h5ad"))\nfor i, path in enumerate(paths):#[:4]:\n    print(path, i+1, "of", len(paths))\n    name = os.path.basename(path).split(".")[0]\n    cur_adata = adata_raw[name]\n    cur_adata = cur_adata[cur_adata.X.sum(axis=1).A.flatten()>250].copy()\n    cur_adata.obs["sample"] = name\n    print("  ", cur_adata.shape)\n    if cur_adata.shape[0] < 500: continue\n    adata_collection[name] = cur_adata    ')


# Timings: 
# ```
# /mnt/ssd_gp2/moredata/*
# CPU times: user 1.03 s, sys: 199 ms, total: 1.23 s
# Wall time: 1.22 s
# ```
# The flattening operation takes a small but measurable amount of time.

# (in our current pipeline, we actually do a single concatenation of a really big matrix, which is kind of slow but better than the alternative of having to re-do the concatenations which is truly cumbersome with our current infrastructure)

# In[7]:


get_ipython().run_cell_magic('time', '', 'samples = list(adata_collection.values())\ntogether = samples.pop()\ntogether = together.concatenate(*samples)')


# Timings:
# ```
# /mnt/ssd_gp2/moredata/*
# CPU times: user 2.94 s, sys: 360 ms, total: 3.3 s
# Wall time: 3.3 s
# ```
# Concatenation takes just over 3s here.  But in real world scenarios this is a barrier.
# 
# When the samples list is 5x the size, the time goes up >7x?
# ```
# CPU times: user 21.7 s, sys: 2.31 s, total: 24 s
# Wall time: 24 s
# ```

# In[ ]:





# In[8]:


get_ipython().run_cell_magic('time', '', 'together = SCData(together)\n\ntogether.raw = together\ntogether.uns["species"] = "Human"\ntogether')


# In[9]:


together.obs["batch"] = together.obs["batch"].astype(int)


# I haven't carefully profiled the preprocess/reprocess step below, but I believe about half the time is spent using an inefficient clustering algorithm that we're in the process of upgrading. The other big step is training the deep learning model, and I don't think that'll be something we can do a whole lot to improve.

# In[10]:


get_ipython().run_cell_magic('time', '', 'reprocessed = SCData(together).reprocess()')


# Timings:
# ```
# CPU times: user 2h 50min 6s, sys: 7min 43s, total: 2h 57min 50s
# Wall time: 40min 18s
# ```
# 

# In[11]:


reprocessed


# In[12]:


reprocessed.X


# In[13]:


"density:", reprocessed.X.nnz/(reprocessed.X.shape[0]*reprocessed.X.shape[1])


# After processing the data, we typically do some exploratory data analysis to identify cell populations of interest.
# 
# We use the `vaex` library to summarize data in a gridded fashion prior to plotting using matplotlib, as matplotlib alone is incapable of rendering large numbers of points.

# In[14]:


reprocessed.qplot("PTPRC", vaex=True, pointsize=1, scale=1.5)


# In[ ]:


_, ax = plt.subplots(figsize=(8,8))
reprocessed.catplot("clusters", vaex=True, ax=ax, facet=False)


# In[ ]:





# One thing we do frequently is zoom in on a subset of cells by retraining our dimensionality reduction functions on only that subset of cells (eg T cells or endothelial cells). Then, we'd like to carry annotations made in those subviews back to the main dataset.

# In[ ]:


cluster2_adata = reprocessed[reprocessed.obs.query("clusters=='12'").index].reprocess()


# In[ ]:


cluster2_adata.catplot("clusters")


# In[ ]:


cluster2_adata.diff_exp().head(10)


# In[ ]:


cluster2_adata.assign_labels("clusters", "cell_type", {
    ("0","1","2"):"Cell Type X",
    ("3","4","5"):"Cell Type Y",
    ("6","7"):"Cell Type Z",
    "8":"W Cells"
})


# In[ ]:


cluster2_adata.catplot("cell_type")


# In[ ]:


reprocessed.obs["cell_type"] = "Unknown"
reprocessed.obs.loc[cluster2_adata.obs_names, "cell_type"] = cluster2_adata.obs["cell_type"]


# In[ ]:


reprocessed.catplot("cell_type", vaex=True)


# In[ ]:




