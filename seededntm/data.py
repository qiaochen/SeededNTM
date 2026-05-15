import numpy as np
import pandas as pd
from typing import Union
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import csr_matrix, diags, identity, issparse, spmatrix, sparray
from dataclasses import dataclass

@dataclass
class ExpData:
    input_rep: Union[np.ndarray, list]=None
    init_bg_rna: Union[np.array, list]=None
    init_bg_gau: Union[np.array, list]=None
    out_counts: Union[sparray, spmatrix, np.ndarray, None]=None
    out_normal: Union[sparray, spmatrix, np.ndarray, None]=None
    batch_labels: Union[np.array, list, pd.Series, None]=None
    var_names: Union[np.array, list, pd.Series, pd.Index, None]=None
    obs_names: Union[np.array, list, pd.Series, pd.Index, None]=None
    condition_mask: Union[np.array, list, None]=None
    topic_prior: Union[np.array, list, None]=None
    spatial_adj: Union[sparray, spmatrix, None]=None

class MyDataset(Dataset):
    def __init__(self,
                 input,
                 out_counts=None,
                 out_normal=None,
                 batch_labels=None,
                 top_cls_labels=None,
                 topic_prior=None,
                 ):
        """
        Basic Dataset class
        input:     Input representation (e.g., counts, PCs, histology embeddings), N X D
        output:    Output expression count matrix (ensure count data) that would be modeled as Multinomial distribution
        """
        self.input = input
        if issparse(out_counts):
            out_counts = out_counts.toarray()
            
        if issparse(out_normal):
            out_normal = out_normal.toarray()
        self.out_counts = out_counts
        self.out_normal = out_normal
        self.batch_labels = batch_labels
        self.top_cls_labels = top_cls_labels
        self.topic_prior = topic_prior
    
    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        batch_data = {}
        
        input = self.input[idx]
        if issparse(input):
            input = input.toarray().squeeze()
        batch_data['input'] = input
        
        if not self.out_counts is None:
            out_counts = self.out_counts[idx]
            if issparse(out_counts):
                out_counts = out_counts.toarray().squeeze()
            batch_data['out_counts'] = out_counts
            
        if not self.out_normal is None:
            batch_data['out_normal'] = self.out_normal[idx]
            
        if not self.batch_labels is None:
            batch_data['batch_labels'] = self.batch_labels[idx]
            
        if not self.top_cls_labels is None:
            batch_data['top_cls_labels'] = self.top_cls_labels[idx]
            
        if not self.topic_prior is None:
            batch_data['topic_prior'] = self.topic_prior[idx]
            
        return batch_data