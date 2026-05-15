import numpy as np
import torch
import pyro
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import os
import re
import pandas as pd
import json

from tqdm import tqdm
from torch.utils.data import DataLoader
from pyro.infer import SVI, TraceMeanField_ELBO
from scipy.sparse import issparse
import random
from .util import EarlyStopper
from .data import MyDataset, ExpData
from .model import SeededNTM
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if "CUBLAS_WORKSPACE_CONFIG" in os.environ and os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8":
        logger.info('use_deterministic_algorithms')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

def preprocess_ST(adata,
               n_topics=None,
               input_rep=None,
               out_counts=None,
               out_normal=None,
               n_pcs=50,
               key_obs_batch_label=None,               
               var_names=None,
               obs_names=None,
               condition_feat_path=None,
               topic_prior = None,               
    ):   
    """
    Select local spatial neighors
    """        
    if input_rep is None:
        adata.layers['raw'] = adata.X.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        if n_pcs != -1:
            logger.info(f"input_rep is None, compute {n_pcs} pcs as input, added as 'embedding' to adata.obsm")
            sc.pp.pca(adata, n_pcs)
        else:
            logger.info(f"input_rep is None, use normalized X as input, added as 'embedding' to adata.obsm")
            adata.obsm['embedding'] = adata.X.copy()
        adata.X = adata.layers['raw'] # model original counts
        input_rep = adata.obsm['embedding']
    
    """
    Compute background noises
    """
    if issparse(adata.X):
        init_bg_rna = adata.X / adata.X.sum(axis=1)
    else:
        init_bg_rna = adata.X / adata.X.sum(axis=1).reshape(-1, 1)
    init_bg_rna = init_bg_rna.mean(axis=0)
    init_bg_rna = np.log(init_bg_rna + 1e-15)
    
    init_bg_rna = None
    init_bg_gau = None
    if not out_counts is None:
        if issparse(out_counts):
            init_bg_rna = out_counts / out_counts.sum(axis=1)
        else:
            init_bg_rna = out_counts / out_counts.sum(axis=1).reshape(-1, 1)
        init_bg_rna = init_bg_rna.mean(axis=0)
        init_bg_rna = np.log(init_bg_rna + 1e-15)
        
    if not out_normal is None:
        init_bg_gau = np.mean(out_normal, axis=0)        
        
    ########
    ### Create conditional mask
    ########
    if not condition_feat_path is None and not os.path.exists(condition_feat_path):
        logger.warning(f"Cannot find path {condition_feat_path}, does it exits? Falling back to non-condition mode...")
    
    condition_mask = None
    if not condition_feat_path is None and os.path.exists(condition_feat_path):
        logger.info(f'Compute conditional mask based on groupped features in path {condition_feat_path}')
        with open(condition_feat_path, 'r') as infile:
            feature_groups = json.load(infile)
            assert(len(feature_groups) <= n_topics)
            out = out_counts if out_counts is not None else out_normal
            condition_mask = np.zeros((n_topics, out.shape[1]), dtype=bool)
            for celltype_name, record in feature_groups.items():
                feat_list = record['features']
                assert(adata.var_names.isin(feat_list).sum() == len(feat_list))
                condition_mask[record['topic_index'], adata.var_names.isin(feat_list)] = True
    
    if key_obs_batch_label is None:
        batch_labels = None
    else:
        batch_labels = adata.obs[key_obs_batch_label].values
        
    exp_data = ExpData(
        init_bg_rna=init_bg_rna,
        init_bg_gau=init_bg_gau,
        input_rep=input_rep,
        out_counts=out_counts,
        out_normal=out_normal,
        batch_labels=batch_labels,
        var_names=var_names,
        obs_names=obs_names,
        condition_mask=condition_mask,
        topic_prior=topic_prior,
        spatial_adj=None,
    ) 
    return exp_data
  

def do_experiment(
        exp_data: ExpData,
        n_topics: int,
        exp_outdir: str='./exp_out',
        model_hid_dim: int=64,
        batch_size: int=4096,
        learning_rate: float=0.01,
        n_epochs: int=800,
        early_stop: bool=False,
        early_stop_tolerance: int=20,
        early_stop_miniepochs: int=100,
        clip_norm: float=1,
        clamp_logvar_max: float=None,
        scale_normal_feat: float=-1,
        wt_fusion_top_seed: float=0.5,
        reg_topic_prior: float=0.5,
        spatial_reg_lambda: float=0.0,
        n_workers: int=4,
        use_nb_obs: bool=True,
        is_group_mode: bool=False,
        pos_scale: float=0.5,
        seed: int=0,
        device=None
    ):
    
    if not os.path.exists(exp_outdir):
        logger.info(f'Creating output dir {exp_outdir}')
        os.makedirs(exp_outdir, exist_ok=True)
        
    pyro.clear_param_store()
    seed_everything(seed)
    pyro.set_rng_seed(seed)
    
    params = None
    adam = None
    elbo = TraceMeanField_ELBO(num_particles=1)
    
    if exp_data.batch_labels is None:
        n_batches = 1
    else:
        n_batches = np.unique(exp_data.batch_labels).shape[0]
    
    model = SeededNTM(
            input_dim = exp_data.input_rep.shape[1],
            out_dim_rna = exp_data.out_counts.shape[1] if not exp_data.out_counts is None else 0,
            out_dim_normal = exp_data.out_normal.shape[1] if not exp_data.out_normal is None else 0,
            init_bg_count = exp_data.init_bg_rna,
            init_bg_normal = exp_data.init_bg_gau,
            condition_mask = exp_data.condition_mask,
            n_topics = n_topics,
            enc_hid_dim=model_hid_dim,
            clamp_logvar_max=clamp_logvar_max,
            scale_normal_feat=scale_normal_feat,
            wt_fusion_top_seed=wt_fusion_top_seed,
            is_group_mode=is_group_mode,
            pos_scale=pos_scale,
            device=device,
            n_batches=n_batches,
            use_nb_obs=use_nb_obs
    )
    
    model.to(device)

    dataset = MyDataset(
        input=exp_data.input_rep,
        out_counts=exp_data.out_counts,
        out_normal=exp_data.out_normal,
        batch_labels=exp_data.batch_labels,
        topic_prior=exp_data.topic_prior,
    )
    
    
    ## run graph and register parameters
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False
    )
    
    for ith_batch, batch_data in enumerate(train_loader):  
        bs = batch_data['input'].shape[0]
        elbo.differentiable_loss(
            model.model, 
            model.guide, 
            **{key:val.to(device) for key, val in batch_data.items() if key in {'input', 'out_counts', 
                                                                                'out_normal', 'batch_labels'}}
        )  
        if reg_topic_prior > 0:
            log_theta = model.encode(
                **{key:value.to(model.device) for key, value in batch_data.items() if key in {'input',
                                                                                     'batch_labels'}})
            loss_top_prior = model.regularizer_topic_prior(
                logtheta_loc=log_theta,
                topic_prior=batch_data['topic_prior'].to(model.device),
            )
        
        params = [value for name, value in pyro.get_param_store().named_parameters()]
        adam = torch.optim.AdamW(params, lr=learning_rate, 
                                    betas=(0.90, 0.999)) 
        break
    # pyro.render_model(model.model, model_args=(batch_data['input'].to(model.device) ,batch_data['out_counts'].to(model.device) , None, None), 
    #                   filename=os.path.join(exp_outdir, 'plate.pdf'), render_distributions=True, render_params=True)
        
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        generator=g,
        shuffle=True
    )
    
    losses = []

    p_bar = tqdm(range(n_epochs))
    es = EarlyStopper(early_stop_tolerance)
    min_epochs = early_stop_miniepochs
    
    # Prepare spatial regularization if enabled
    spatial_adj_torch = None
    if spatial_reg_lambda > 0 and exp_data.spatial_adj is not None:
        from scipy.sparse import coo_matrix
        adj_coo = coo_matrix(exp_data.spatial_adj)
        spatial_adj_torch = torch.sparse_coo_tensor(
            torch.LongTensor(np.vstack([adj_coo.row, adj_coo.col])),
            torch.FloatTensor(adj_coo.data),
            adj_coo.shape,
        ).to(device)
        logger.info(f"Spatial regularization enabled: lambda={spatial_reg_lambda}, "
                    f"edges={adj_coo.nnz}, avg_degree={adj_coo.nnz / adj_coo.shape[0]:.1f}")
    
    # Full input tensor for spatial reg (all spots)
    full_input = torch.FloatTensor(exp_data.input_rep).to(device) if spatial_adj_torch is not None else None
    full_batch_labels = None
    if spatial_adj_torch is not None and exp_data.batch_labels is not None:
        full_batch_labels = torch.LongTensor(exp_data.batch_labels).to(device)
    
    for epoch in p_bar:
        loss_dict = train_step(
            train_loader, 
            model, 
            elbo,
            adam,
            clip_norm,
            params,
            reg_topic_prior,
            epoch,
         )
        if isinstance(loss_dict, tuple):
            logging.error('training error')
            raise Exception('training error')
        
        # Spatial regularization step (every epoch, full-batch)
        if spatial_adj_torch is not None and spatial_reg_lambda > 0:
            model.train()
            adam.zero_grad()
            with torch.no_grad():
                theta = torch.softmax(
                    model.encoder(full_input, full_batch_labels)[0], dim=-1
                )
            theta_for_grad = model.encoder(full_input, full_batch_labels)[0]
            theta_soft = torch.softmax(theta_for_grad, dim=-1)
            
            # Graph Laplacian smoothness: sum_ij A_ij * ||theta_i - theta_j||^2
            # Efficient: Tr(theta^T L theta) = Tr(theta^T D theta) - Tr(theta^T A theta)
            # = sum_i d_i * ||theta_i||^2 - sum_ij A_ij * theta_i . theta_j
            Atheta = torch.sparse.mm(spatial_adj_torch, theta_soft)  # (N, K)
            smooth_loss = (theta_soft * (theta_soft * spatial_adj_torch.sum(dim=1).to_dense().unsqueeze(1) - Atheta)).sum()
            smooth_loss = spatial_reg_lambda * smooth_loss / theta_soft.shape[0]
            
            smooth_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip_norm)
            adam.step()
            adam.zero_grad()
            
            loss_dict['spatial'] = smooth_loss.item()
        
        loss_info = "  ".join([f"{k}({v:.4f})" for k,v in loss_dict.items()])
        
        p_bar.set_description(f"Epoch loss: {loss_info}")
        losses.append(loss_dict['total'])
        if early_stop:
            if epoch > min_epochs:
                if es.early_stop(loss_dict['total']):
                    logging.info("Early Stopping")
                    break
                
    loader = DataLoader(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=False
    )
    
    topics = []
    for ith_batch, batch_data in enumerate(loader):
        model.eval()
        with torch.no_grad():
            theta = model.infer_topic(**{key:val.cuda() for key, val in batch_data.items() if key in {'input', 'batch_labels'}})
            topics.append(theta.cpu())
            
    topics = torch.concat(topics, dim=0)
    
    return_beta_rna, return_beta_normal = True, False
    if not exp_data.out_normal is None:
        return_beta_rna, return_beta_normal = False, True
    topic_vocab = model.beta(return_beta_rna=return_beta_rna, return_beta_normal=return_beta_normal).detach().cpu().numpy()
    top_names = [ f'topic_{i}' for i in range(n_topics)]
    
    df_topic = pd.DataFrame(topics, columns=top_names)
    if not exp_data.obs_names is None:
        df_topic.index = pd.Series(exp_data.obs_names, name='obs_names')
        
    df_top_vocab = pd.DataFrame(topic_vocab, index=pd.Series(top_names, name='topic'))
    if not exp_data.var_names is None:
        df_top_vocab.index = exp_data.var_names
    
    path_topic = os.path.join(exp_outdir, 'df_topic.csv')
    path_top_vocab = os.path.join(exp_outdir, 'df_topic_vocab.csv')
    path_model_ckpt = os.path.join(exp_outdir, 'model.ckpt')
    path_loss_plot = os.path.join(exp_outdir, 'loss.png')
    
    df_topic.to_csv(path_topic)
    df_top_vocab.to_csv(path_top_vocab)
    
    torch.save(model.cpu().state_dict(), path_model_ckpt)
    
    plt.figure(figsize=(5, 2))
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("Loss")
    plt.savefig(path_loss_plot)
    
    return model, df_topic, df_top_vocab, losses
    

def train_step(
        dataloader,
        model, 
        elbo,
        adam,
        clip_norm,
        params,
        reg_topic_prior,
        epoch
    ):
    """
    A train step with MyBatchEffectDataset setting
    """
    losses = []
    elbo_losses = []
    topic_prior_losses = []
    for ith_batch, batch_data in enumerate(dataloader):            
        bs = batch_data['input'].shape[0]
        elbo_loss = elbo.differentiable_loss(
            model.model, 
            model.guide, 
            **{key:value.to(model.device) for key, value in batch_data.items() if key in {'input', 'out_counts', 
                                                                                'out_normal', 'batch_labels'}}
        )
        
        if reg_topic_prior > 0 and 'topic_prior' in batch_data:
            log_theta = model.encode(
                **{key:value.to(model.device) for key, value in batch_data.items() if key in {'input',
                                                                                     'batch_labels'}})
            loss_top_prior = model.regularizer_topic_prior(
                logtheta_loc=log_theta,
                topic_prior=batch_data['topic_prior'].to(model.device),
            )
            elbo_loss = elbo_loss / bs
            loss = (1 - reg_topic_prior) * elbo_loss + reg_topic_prior * loss_top_prior  
            elbo_losses.append(elbo_loss.item())
            topic_prior_losses.append(loss_top_prior.item())
            
        else:
            loss_top_prior = torch.zeros([])
            elbo_loss = elbo_loss / bs
            loss = elbo_loss 
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip_norm) 
        adam.step()
        adam.zero_grad()        
        loss = loss.item()
        
        ith_batch = ith_batch + 1
        losses.append(loss)
        
    
    if len(topic_prior_losses) > 0:
        return {"total": np.mean(losses), 
                "elbo": np.mean(elbo_losses),
                "topReg": np.mean(topic_prior_losses)
            }
    return {"total": np.mean(losses)}