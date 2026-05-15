import warnings
warnings.filterwarnings("ignore")
import argparse
import logging
import torch
import numpy as np

import scanpy as sc
import os
from typing import Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

from .experiment import preprocess_ST, do_experiment

def none_or_str(value):
    if value == 'None':
        return None
    return value

def do_exp(
        adata_h5ad_path: str,
        key_input: str='input_rep',
        key_count_out: str=None,
        key_normal_out: str=None,
        key_obs_batch_label: str=None,
        key_topic_prior: str=None,
        n_topics: int=10,
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
        scale_normal_feat: float=0.0,
        wt_fusion_top_seed: float=0.5,
        reg_topic_prior: float=0.5,
        spatial_reg_lambda: float=0.0,
        spatial_k_neighbors: int=6,
        seed: int=0,
        condition_feat_path: str=None,
        use_nb_obs: bool=False,
        is_group_mode: bool=False,
        pos_scale: float=0.5,
        device=None
    ):
    
    logger.info('Reading adata...')
    adata = sc.read_h5ad(adata_h5ad_path)
    
    out_counts = None
    out_normal = None
    
    if not key_count_out is None:
        out_counts = adata.obsm[key_count_out]
        logger.info(f'Output Count data key is: {key_count_out}')
    
    if not key_normal_out is None:
        out_normal = adata.obsm[key_normal_out]
        logger.info(f'Output Normal data key is: {key_normal_out}')
            
    logger.info('Processing experiment data ...')
         
    topic_prior = None   
    if not key_topic_prior is None:
        logger.info('applying topic prior')
        topic_prior = adata.obsm[key_topic_prior]

    exp_data = preprocess_ST(
        adata=adata,
        n_topics=n_topics,
        input_rep = adata.obsm[key_input],
        out_counts=out_counts,
        out_normal=out_normal,
        key_obs_batch_label=key_obs_batch_label,
        condition_feat_path=condition_feat_path,
        topic_prior = topic_prior,
    )
    
    # Build spatial graph if spatial regularization is requested
    if spatial_reg_lambda > 0 and 'spatial' in adata.obsm:
        from .util import build_spatial_graph
        coords = adata.obsm['spatial']
        logger.info(f'Building spatial k-NN graph (k={spatial_k_neighbors}) from {coords.shape[0]} spots')
        exp_data.spatial_adj = build_spatial_graph(coords, k_neighbors=spatial_k_neighbors)
    elif spatial_reg_lambda > 0:
        logger.warning('spatial_reg_lambda > 0 but no spatial coordinates found in adata.obsm["spatial"]. Disabling spatial reg.')
        spatial_reg_lambda = 0.0
    
    logger.info("Model fitting start ...")
    model, df_topic, df_top_vocab, losses = do_experiment(
                exp_data=exp_data,
                n_topics=n_topics,
                exp_outdir=exp_outdir,
                model_hid_dim=model_hid_dim,
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                early_stop=early_stop,
                early_stop_tolerance=early_stop_tolerance,
                early_stop_miniepochs=early_stop_miniepochs,
                clip_norm=clip_norm,
                clamp_logvar_max=clamp_logvar_max,
                scale_normal_feat=scale_normal_feat,
                wt_fusion_top_seed=wt_fusion_top_seed,
                reg_topic_prior=reg_topic_prior,
                spatial_reg_lambda=spatial_reg_lambda,
                use_nb_obs=use_nb_obs,
                is_group_mode=is_group_mode,
                pos_scale=pos_scale,
                seed=seed,
                device=device
    )
    logger.info(f'Finished model fitting, please check output in {exp_outdir}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adata_h5ad_path', type=str, required=True, help='h5ad adata with input and other data stored in adata.obsm')
    parser.add_argument('--condition_feat_path', type=str, default=None, help='path to seed gene json file, {"celltype": {"features": [marker genes], "topic_index": idx(int)}')
    parser.add_argument('--key_input', type=str, default='input_rep', help='key to input representation stored in adata.obsm, default: input_rep')
    parser.add_argument('--key_count_out', type=str, default=None, help='key to output count representation stored in adata.obsm, default: None')
    parser.add_argument('--key_normal_out', type=str, default=None, help='key to output normal representation stored in adata.obsm, default: None')
    parser.add_argument('--key_obs_batch_label', type=none_or_str, default=None, help='key to batch effect labels, default: None, i.e., only one batch')
    parser.add_argument('--key_topic_prior', type=none_or_str, default=None, help='key to topic priors stored in adata.obsm, default: None,')
    
    parser.add_argument('--num_topics', type=int, default=10, help='Number of topics, default: 10')
    parser.add_argument('--exp_outdir', type=str, default='./exp_out', help='Experiment result output dir, default: "./exp_out"')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed for reproducable experiment, default: 0')
    parser.add_argument('--num_epochs', type=int, default=800, help='Model training epochs, default: 800')
    parser.add_argument('--batch_size', type=int, default=4096, help='Model training batch size, default: 4096')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Model training learning rate, default: 0.01')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Gradient clip norm in training, default: 1.0')
    parser.add_argument('--clamp_logvar_max', type=float, default=None, help='clamp maximum value in logvar, to avoid extreme values during exponential, default: None, not clamp')
    
    parser.add_argument('--model_hid_dim', type=int, default=64, help='Model hidding layer dimension, default: 64')
    parser.add_argument('--wt_fusion_top_seed', type=float, default=1.0, help='weight of gene seeds, non-seed gene weight is (1-wt_fusion_top_seed), default: 1.0')
    parser.add_argument('--reg_topic_prior', type=float, default=0.9, help='regularization strength of topic alignment loss, a search in range [0.0, 0.5, 0.8, 0.9, 0.95 0.99] may help find the best value, default: 0.9')
    parser.add_argument('--scale_normal_feat', type=float, default=0.0, help='scale weight for normal out, default: 0.0')
    parser.add_argument('--device', type=str, default='auto', help='Device e.g., "cuda:0", "cpu", default: "auto"')
    
    parser.add_argument('--use_nb_obs', default=True, action=argparse.BooleanOptionalAction, help='use negative binomial observation loss for counts data')
    parser.add_argument('--is_group_mode', default=False, action=argparse.BooleanOptionalAction, help='whether to group expression by mean as observation for elbo loss, default False')
    parser.add_argument('--scale_binary_mode_positive', type=float, default=0.5, help='scale weight for positive topic in binary mode, default: 0.5')
    parser.add_argument('--use_multinomial_obs', default=False, action=argparse.BooleanOptionalAction, help='use multinomial observation loss for counts data')
    parser.add_argument('--early_stop', action=argparse.BooleanOptionalAction)
    parser.add_argument('--early_stop_tolerance', type=int, default=20, help='Tolerance steps of early stop during training, if early_stop is turned on, default: 20')
    parser.add_argument('--early_stop_minimum_steps', type=int, default=100, help='Minimum number of steps before turning on early stop during training, if early_stop is turned on, default: 100')
    parser.add_argument('--spatial_reg_lambda', type=float, default=0.0, help='Spatial graph Laplacian regularization strength. 0 disables. Suggested range: 0.001-0.1. default: 0.0')
    parser.add_argument('--spatial_k_neighbors', type=int, default=6, help='Number of nearest neighbors for spatial graph. default: 6')
    
    args = parser.parse_args()
    logger.info(f"Input parameters: {args}")
    
    print(args)
    
    assert(args.wt_fusion_top_seed >= 0 and args.wt_fusion_top_seed <=1, "wt_fusion_top_seed should be in the range [0, 1]")
    assert(args.reg_topic_prior >= 0 and args.reg_topic_prior <=1, "reg_topic_prior should be in the range [0, 1]")
    
    do_exp(
        adata_h5ad_path=args.adata_h5ad_path,
        wt_fusion_top_seed=args.wt_fusion_top_seed,
        reg_topic_prior=args.reg_topic_prior,
        key_input=args.key_input,
        key_count_out=args.key_count_out,
        key_normal_out=args.key_normal_out,
        key_obs_batch_label=args.key_obs_batch_label,
        key_topic_prior=args.key_topic_prior,
        n_topics=args.num_topics,
        exp_outdir=args.exp_outdir,
        model_hid_dim=args.model_hid_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_epochs=args.num_epochs,
        early_stop=args.early_stop,
        early_stop_tolerance=args.early_stop_tolerance,
        early_stop_miniepochs=args.early_stop_minimum_steps,
        clip_norm=args.clip_norm,
        clamp_logvar_max=args.clamp_logvar_max,
        scale_normal_feat = args.scale_normal_feat,
        seed=args.random_seed,
        condition_feat_path=args.condition_feat_path,
        use_nb_obs= not args.use_multinomial_obs,
        is_group_mode = args.is_group_mode,
        pos_scale = args.scale_binary_mode_positive,
        spatial_reg_lambda=args.spatial_reg_lambda,
        spatial_k_neighbors=args.spatial_k_neighbors,
        device=torch.device(args.device) if not args.device == 'auto' else (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
    )
    