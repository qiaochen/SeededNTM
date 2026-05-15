## Modelling
import pyro
import torch
import math
import logging
from torch import nn
import numpy as np
from collections import namedtuple
from pyro import distributions as dist
from pyro.distributions import constraints
from torch.nn import functional as F
from dataclasses import dataclass
from torch.distributions.utils import broadcast_all
import warnings
warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(False)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

ModUnit = namedtuple('ModUnit', 'start_id, end_id')

@dataclass
class MetaModInfo:
    count_obs: ModUnit=None
    normal_obs: ModUnit=None
    n_modalities: int=0
    n_units: int=0
    full_dim: int=0
    

class Encoder(nn.Module):
    """
    SpaNTM encoder with optional batch effect removal mechanism
    """
    def __init__(self, 
                 input_dim: int,
                 hid_dim: int,
                 n_topics: int,
                 n_batches: int=2,
                 dropout: float=0.2,
                 clamp_logvar_max: float=None,
                ):
        """
        input_dim: dimension of the input features
        hid_dim:   dimension of the hidden layers
        n_topics:  specified number of topics
        n_batches: number of data batches (not training batch)
        dropout:   dropout probability
        """
        super().__init__()
        self.n_batches = n_batches
        self.hid_dim = hid_dim
        self.clamp_logvar_max = clamp_logvar_max
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(
            input_dim, 
            hid_dim, 
            bias=False)
        self.tanh = nn.Tanh()
        self.silu = nn.SiLU()
        self.bn = nn.BatchNorm1d(hid_dim, affine=False)
        self.fcmu = nn.Linear(hid_dim, n_topics, 
                              bias=False
                             )
        self.fclv = nn.Linear(hid_dim, n_topics, 
                              bias=False
                             )
        self.gelu = nn.GELU()
        self.normlv = nn.BatchNorm1d(n_topics, affine=False)

        if n_batches > 1:
            self.normmu = nn.ModuleList([
                nn.Sequential(
                    nn.BatchNorm1d(n_topics, affine=False)
                )
                for i in range(n_batches)])
        else:
            self.normmu = nn.BatchNorm1d(n_topics, affine=False)
        
    def forward(self, 
                X,
                batch_labels=None):
        """
        X: input to encoder
        masks: masks for valid local spatial neighbors, 1 for valid, 0 invalid (number of valid neighbors may differ)
        batch_labels: the batch assignment lable of the spot/cell 0-T
        """
        z = self.fc1(self.drop(X))
        h = self.gelu(z)
        h = self.bn(h)

        logtheta_loc = self.fcmu(h)
        logtheta_logvar = self.normlv(self.fclv(h))
        
        if self.n_batches > 1:
            tmp_logtheta_loc = logtheta_loc.new_zeros(logtheta_loc.shape)
            for i in batch_labels.unique():
                indices = torch.where(batch_labels == i)
                if len(indices[0]) > 1:
                    tmp_logtheta_loc[indices] = self.normmu[i](logtheta_loc[indices])
                else:
                    tmp_logtheta_loc[indices] = logtheta_loc[indices]
                
            logtheta_loc = tmp_logtheta_loc
        else:
            logtheta_loc = self.normmu(logtheta_loc)

        # theta_scale = (0.5 * logtheta_logvar.clamp(max=self.clamp_logvar_max)).exp()
        theta_scale = F.softplus(logtheta_logvar if self.clamp_logvar_max is None 
                                 else logtheta_logvar.clamp(-self.clamp_logvar_max, self.clamp_logvar_max))
        return logtheta_loc, theta_scale
    

class SeededNTM(nn.Module):
    """
    Seeded Neural Topic Modelling
    """
    def __init__(self, 
                 input_dim,
                 out_dim_rna=0,                 
                 init_bg_count=None,
                 init_bg_normal=None,
                 out_dim_normal=0,
                 condition_mask=None,
                 n_topics=10,
                 enc_hid_dim=32,
                 dropout=0.2,
                 clamp_logvar_max=None,
                 scale_normal_feat=1.0, # deprecated, to be removed
                 n_batches=1,
                 wt_fusion_top_seed = 0.5,
                 use_nb_obs = True,
                 is_group_mode = False,
                 pos_scale = 0.5, 
                 device=None
                ):
        """
        input_dim: dimension of input features for encoding
        out_dim_rna: dimension of output count data
        init_bg_count: background noise of observations (log mean of raw counts)
        init_bg_feat: background noise of continuous observations (normally distributed)
        out_dim_normal: dimensionality of normal output results 
        condition_mask: (K by out_dim) mask for seeded topic learning (0 for masking out features)
        n_topics: number of topics
        enc_hid_dim: encoding hidden dimension
        dropout: dropout rate for neural layers
        clamp_top_var_max: clampping upper bound for topic variance term (lower model capacity for colapse or nan values)
        # scale_normal_feat: weight of normal feature loss, larger value will weigh normal observations more (value to be tested in new dataset)
        n_batches: number of experiment batches (not training batch size)
        # prior_scale_of_nb_disp: hyperparameter scale of the halfcauchy distribution for modelling dispersion parameter of negative binomial observation distribution
        wt_fusion_top_seed: strength of imposing seeded topics over background topics, [0, 1], p would be applied to seed topics, 1-p to background topics
        # prior_top_lambda_tilde: supervised prior lambda_tilde learned from a reference dataset (can miss some topics)
        use_nb_obs: True to use negative binomial observation distribution, False to use Mulitnomial
        device: computational device
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_topics = n_topics
        self.out_dim_normal = out_dim_normal
        self.out_dim_rna = out_dim_rna
        self.scale_normal_feat = scale_normal_feat
        self.use_nb_obs = use_nb_obs
        self.n_batches = n_batches
        self.wt_fusion_top_seed = wt_fusion_top_seed
        self.device = device
        self.pos_scale = pos_scale
        self.is_group_mode = is_group_mode
        
        assert(out_dim_rna > 0 or out_dim_normal > 0, 
               "At least one type of output should be specified, count or normal.")
        
        assert(wt_fusion_top_seed >= 0 and wt_fusion_top_seed <= 1, 
               "Fusion weight for topic seeding should be in range [0, 1]")
        
        self.is_seeding = False
            
        if out_dim_rna > 0:
            self.init_bg_count = torch.Tensor(init_bg_count).to(device)
        if out_dim_normal > 0:
            self.init_bg_feat = torch.Tensor(init_bg_normal).to(device)

        self.condition_mask = condition_mask if condition_mask is None else torch.BoolTensor(condition_mask).to(device)
        
        if not self.condition_mask is None:
            self.valid_condition_genes = (self.condition_mask.sum(dim=0) > 0)
            self.seed_mask = torch.BoolTensor(condition_mask).to(device)
            self.condition_mask[:, ~self.valid_condition_genes] = torch.BoolTensor([True]).to(device) # allow non-seed gene dimentions to be trainable
            self.seed_gene_dim = self.valid_condition_genes.sum().item()
            
            if self.wt_fusion_top_seed > 0:
                self.is_seeding = True
                
        self.encoder = Encoder(
                self.input_dim,
                enc_hid_dim,
                self.n_topics,
                n_batches=n_batches,
                dropout=dropout,
                clamp_logvar_max=clamp_logvar_max,
            ).to(device)  
        
        self.obs_info = MetaModInfo()
        
        if self.out_dim_rna > 0:
            self.obs_info.n_modalities += 1
            self.obs_info.count_obs = ModUnit(0, self.out_dim_rna*2)
            self.obs_info.full_dim += (self.out_dim_rna * 2)
            self.obs_info.n_units += 2
            
        if self.out_dim_normal > 0:
            start, end = 0, self.out_dim_normal*2
            if not self.obs_info.count_obs is None:
                start += self.obs_info.count_obs.end_id
                end   += self.obs_info.count_obs.end_id
            self.obs_info.normal_obs = ModUnit(start, end)
            self.obs_info.n_modalities += 1
            self.obs_info.n_units += 2
            self.obs_info.full_dim += (self.out_dim_normal * 2)
        
        logger.info(f'{self.obs_info}')
    
    def _compute_lambda_tilde(self,caux, tau, delta, lambda_):
        """
        Compute lambda_tilde from sampled model parameters
        """
        tau = tau.unsqueeze(1)
        delta = delta.unsqueeze(0)
        if lambda_.shape[0] != self.n_topics:
            lambda_ = lambda_.t()
        
        # Horseshoe prior
        lambda_tilde = torch.sqrt(
            (caux**2 * tau**2 * delta**2 * lambda_**2)
            / (caux**2 + tau**2 * delta**2 * lambda_**2)
        )
        return lambda_tilde
    
    def _compute_lambda_tilde_batch(self, caux, tau, delta, lambda_):
        """
        Compute lambda_tilde from sampled model parameters for batched seeded and background topics
        caux: 2
        tau:  n_decomposed_topics or ( 2 * n_topics)
        delta: 2 * dim_out
        lambda_: n_topics X (2 * dim_out)
        
        return: 2 X n_topics X 
        """
        caux = caux.view(2, 1, 1)
        tau = tau.view(2, -1, 1)
        delta = delta.view(2, 1, -1)
        lambda_ = lambda_.view(2, lambda_.shape[0], lambda_.shape[1]//2)
        
        # Horseshoe prior
        lambda_tilde = torch.sqrt(
            (caux**2 * tau**2 * delta**2 * lambda_**2)
            / (caux**2 + tau**2 * delta**2 * lambda_**2)
        )
        
        return lambda_tilde
    
    def _compute_lambda_tilde_single(self, caux, tau, delta, lambda_):
        """
        Compute lambda_tilde from sampled model parameters for batched seeded and background topics
        caux: 2
        tau:  n_decomposed_topics or ( 2 * n_topics)
        delta: 2 * dim_out
        lambda_: n_topics X (2 * dim_out)
        
        return: 2 X n_topics X 
        """
        caux = caux.view(1, 1)
        tau = tau.view(-1, 1)
        delta = delta.view(1, -1)
        
        # Horseshoe prior
        lambda_tilde = torch.sqrt(
            (caux**2 * tau**2 * delta**2 * lambda_**2)
            / (caux**2 + tau**2 * delta**2 * lambda_**2)
        )
        return lambda_tilde
                 
    def _compute_topic_params(self, caux, tau, delta, lambda_, bg, noise, is_batch=True,
                              ):
        """
        Compute the topic-observation logits
        """
        if is_batch:
            lambda_tilde = self._compute_lambda_tilde_batch(caux, tau, delta, lambda_)
            beta = 0 + noise.view(2, noise.shape[0], noise.shape[1]//2) * lambda_tilde
        else:
            lambda_tilde = self._compute_lambda_tilde_single(caux, tau, delta, lambda_)
            beta = 0 + noise * lambda_tilde
        beta = beta + bg
        return beta
    
    
    def _compute_with_adding_batch_effect_batch(
                                 self,
                                 theta, 
                                 batch_tau, 
                                 batch_delta, 
                                 beta, 
                                 batch_labels,
                                 ):
        
        lkl_param = torch.zeros(
                    batch_labels.shape[0], batch_delta.shape[1]//2, device=self.device
                )
        for i in range(self.n_batches):
            indices = torch.where(batch_labels == i)[0]
            offset = batch_tau.view(2, -1, 1) * batch_delta[i,:].view(2, 1, -1)
            beta_offset = beta + offset
            _theta = theta[indices]
            if not self.condition_mask is None and self.wt_fusion_top_seed > 0 and self.valid_condition_genes.sum() < self.condition_mask.shape[1]:
                top_dist_seed = torch.softmax(beta_offset[0].masked_fill(self.condition_mask == 0, float('-inf')), dim=-1)
                top_dist_bg   = torch.softmax(beta_offset[1], dim=-1)
                lkl_param[indices] = self.wt_fusion_top_seed * _theta @ top_dist_seed + (1 - self.wt_fusion_top_seed) * _theta @ top_dist_bg
            else:
                lkl_param[indices] = _theta @ torch.softmax(beta_offset, dim=-1)
        return lkl_param
    
    def _compute_with_adding_batch_effect_batch_normal(
                                 self,
                                 theta, 
                                 batch_tau, 
                                 batch_delta, 
                                 beta, 
                                 batch_labels,
                                 ):
        
        lkl_param = torch.zeros(
                    batch_labels.shape[0], batch_delta.shape[1]//2, device=self.device
                )
        for i in range(self.n_batches):
            indices = torch.where(batch_labels == i)[0]
            offset = batch_tau.view(2, -1, 1) * batch_delta[i,:].view(2, 1, -1)
            beta_offset = torch.clamp(beta + offset, 0)
            _theta = theta[indices]
            if not self.condition_mask is None and self.wt_fusion_top_seed > 0 and self.valid_condition_genes.sum() < self.condition_mask.shape[1]:
                top_dist_seed = beta_offset[0].masked_fill(self.condition_mask == 0, 0)
                top_dist_bg   = beta_offset[1]
                lkl_param[indices] = self.wt_fusion_top_seed * _theta @ top_dist_seed + (1 - self.wt_fusion_top_seed) * _theta @ top_dist_bg
            else:
                lkl_param[indices] = _theta @ beta_offset
        return lkl_param
    
    def _compute_with_adding_batch_effect_single(self,
                                 theta,
                                 batch_tau,  # n_topics
                                 batch_delta, # n_batches X out_dim
                                 beta, 
                                 batch_labels,
                                 bg_seed,
                                 softmax=False
                                 ):
        
        lkl_param = torch.zeros(
                    batch_labels.shape[0], batch_delta.shape[1], device=self.device
                )
        for i in range(self.n_batches):
            indices = torch.where(batch_labels == i)[0]
            offset = batch_tau.view(-1, 1) * batch_delta[i,:].view(1, -1)
            beta_offset = beta + offset
            _theta = theta[indices]
            
            if softmax:                
                beta_seeded = torch.zeros_like(beta_offset)
                
                if not self.condition_mask is None and self.wt_fusion_top_seed > 0 and self.valid_condition_genes.sum() < self.condition_mask.shape[1]:
                    beta_seeded = beta_seeded.masked_fill(self.condition_mask == 0, float('-inf'))
                    beta_seeded[:, ~self.valid_condition_genes] = bg_seed.view(1, -1).expand(beta_seeded.shape[0], -1)
                    beta_seeded[self.seed_mask] = beta_offset[self.seed_mask]
                    top_dist_seed = torch.softmax(beta_seeded, dim=-1)
                    top_dist_bg   = torch.softmax(beta_offset, dim=-1)    
                    lkl_param[indices] = self.wt_fusion_top_seed * _theta @ top_dist_seed + (1 - self.wt_fusion_top_seed) * _theta @ top_dist_bg
                else:
                    lkl_param[indices] = _theta @ torch.softmax(beta_offset, dim=-1)
            else:
                lkl_param[indices] = _theta @ beta_offset
        return lkl_param
            

    def model(self, input, out_counts=None, out_normal=None, batch_labels=None):
        """
        Generative process
        input: input representation
        out_counts: output count
        out_normal: output normal features
        batch_labels: label of batch effect source
        """
        # Params for Batchsize X Topic distribution
        logtheta_loc = torch.zeros(
                            input.shape[0], 
                            self.n_topics,  
                            device=self.device
                            )  
        
        logtheta_scale = torch.ones(
                    input.shape[0], 
                    self.n_topics, 
                    device=self.device
        )
        
        if self.use_nb_obs and not self.obs_info.count_obs is None:
            with pyro.plate(f"Dim_rna", self.out_dim_rna):
                disp = pyro.sample(
                    'disp',
                    dist.HalfCauchy(
                        torch.ones(1, device=self.device)
                    )
            )
            
        if not self.condition_mask is None and self.wt_fusion_top_seed > 0 and self.valid_condition_genes.sum() < self.condition_mask.shape[1]:
            with pyro.plate(f"Dim_seed_bg", 
                            self.valid_condition_genes.shape[0] - self.seed_gene_dim):
                bg_seed = pyro.sample(
                    'bg_seed',
                    dist.Normal(
                        torch.zeros(1, device=self.device),
                        torch.ones(1, device=self.device)
                    )
                )
                
        if not self.obs_info.normal_obs is None:
            with pyro.plate(f"Dim_normal", self.out_dim_normal):
                normal_scale = pyro.sample(
                    'normal_scale',
                    dist.InverseGamma(
                        torch.ones(1, device=self.device) * 0.5,
                        torch.ones(1, device=self.device) * 0.5,
                    )
                )
                
        
        with pyro.plate(f'N_modality', self.obs_info.n_units):
            # 1 X (n_modalith)
            caux = pyro.sample(
                "caux",
                dist.InverseGamma(
                    torch.ones(1, device=self.device) * 0.5,
                    torch.ones(1, device=self.device) * 0.5,
                ),
            )
            
        with pyro.plate(f'N_feature', self.obs_info.full_dim):
            # 1 X (full out dim)
            delta = pyro.sample(
                "delta",
                dist.HalfCauchy(torch.ones(1, 
                        device=self.device)),)
            
            bg = pyro.sample(
                'bg',
                dist.Normal(
                    torch.zeros(1, device=self.device),
                    torch.ones(1, device=self.device)
                )
            )
            
            if self.n_batches > 1:
                with pyro.plate(f'N_batches', self.n_batches):
                    batch_delta = pyro.sample(
                        "batch_delta",
                        dist.StudentT(
                            10,
                            torch.zeros(1, device=self.device),
                            torch.ones(1, device=self.device) * 0.01,
                        ),
                )
                    
            with pyro.plate(f"N_real_top", self.n_topics):
                lambda_ = pyro.sample(
                        "lambda_",
                        dist.HalfCauchy(torch.ones(1, 
                                        device=self.device)),)
                    
                noise = pyro.sample(
                        "noise",
                        dist.Normal(
                            torch.zeros(
                                1,
                                device=self.device,
                            ),
                            torch.ones(1, device=self.device),
                        ),
            )
            
        # Normal distribution, reparameterization trick
        # beta ~ N(0, lambda_tilde^2)
        with pyro.plate(f'N_all_top', self.n_topics * self.obs_info.n_units):
            # 1 X (n_topics * n_modalities)
            tau = pyro.sample(
                "tau",
                dist.HalfCauchy(torch.ones(1, 
                                device=self.device
                                )),
            )
            
            if self.n_batches > 1:
                batch_tau = pyro.sample(
                        "batch_tau",
                        dist.Beta(
                            torch.ones(1, device=self.device) * 0.5,
                            torch.ones(1, device=self.device) * 0.5,
                        ),
                )        
        
        
        with pyro.plate(f'N_obs', input.shape[0]):            
            logtheta = pyro.sample(
                "logtheta", dist.MultivariateNormal(
                    logtheta_loc,
                    covariance_matrix=torch.diag_embed(logtheta_scale)
                ).to_event(0)
            )
       
            theta = torch.softmax(logtheta, dim=-1)
            
            if self.out_dim_rna > 0:
                bg_count = bg[self.obs_info.count_obs.start_id:self.obs_info.count_obs.end_id].view(2, 1, -1) + self.init_bg_count.view(1, 1, -1)
                caux_count = caux[:2]
                tau_count = tau[:self.n_topics*2]
                delta_count = delta[self.obs_info.count_obs.start_id:self.obs_info.count_obs.end_id]
                lambda_count = lambda_[:, self.obs_info.count_obs.start_id:self.obs_info.count_obs.end_id]
                noise_count =  noise[:, self.obs_info.count_obs.start_id:self.obs_info.count_obs.end_id]
                beta_rna = self._compute_topic_params(caux_count, tau_count, delta_count, lambda_count, bg_count, noise_count, is_batch=True)
                
                
                
                if self.n_batches > 1:
                    batch_tau_count = batch_tau[:self.n_topics*2]
                    batch_delta_count = batch_delta[:, self.obs_info.count_obs.start_id:self.obs_info.count_obs.end_id]
                    lkl_param_count = self._compute_with_adding_batch_effect_batch(theta, batch_tau_count, 
                                                                                    batch_delta_count, beta_rna, batch_labels,
                                                                                    bg_seed,
                                                                                    softmax=True)
                else:
                    if not self.condition_mask is None and self.wt_fusion_top_seed > 0 and self.valid_condition_genes.sum() < self.condition_mask.shape[1]:
                        _beta_seeded = beta_rna[0].masked_fill(self.seed_mask == 0, float('-inf'))
                        if self.use_nb_obs and not self.obs_info.count_obs is None:
                            _beta_seeded[:, ~self.valid_condition_genes] = bg_seed.view(1, -1).expand(_beta_seeded.shape[0], -1)
                        top_dist_seed = torch.softmax(_beta_seeded, dim=-1)
                        top_dist_bg   = torch.softmax(beta_rna[1].masked_fill(self.seed_mask == 1, float('-inf')), dim=-1)
                        lkl_param_count = self.wt_fusion_top_seed * theta @ top_dist_seed + (1 - self.wt_fusion_top_seed) * theta @ top_dist_bg
                    else:
                        lkl_param_count = theta @ torch.softmax(beta_rna[0], dim=-1)
                        
                logger.debug(f"beta_rna shape: {beta_rna.shape}")
                # logger.debug(f"top_dist_seed shape: {top_dist_seed.shape}")
                # logger.debug(f"top_dist_bg shape: {top_dist_bg.shape}")
                logger.debug(f"theta shape: {theta.shape}")    
                logger.debug(f"lkl_param_count shape: {lkl_param_count.shape}")    
                logger.debug(f'caux {caux_count}, tau max {tau_count.max()}, tau min {tau_count.min()}')
                logger.debug(f'delta max {delta_count.max()}, delta min {delta_count.min()}')
                logger.debug(f'lambda_count max {lambda_count.max()}, lambda_count min {lambda_count.min()}')
                logger.debug(f'bg_count max {bg_count.max()}, bg_count min {bg_count.min()}')
                logger.debug(f'noise_count max {noise_count.max()}, noise_count min {noise_count.min()}')
                logger.debug(f'beta_rna max {beta_rna.max()}, beta_rna min {beta_rna.min()}')
                logger.debug(f'caux_loc max {self.caux_loc}, caux_scale min {self.caux_scale}')
            
            if self.out_dim_normal > 0:
                bg_feat = bg[self.obs_info.normal_obs.start_id:self.obs_info.normal_obs.end_id].view(2, 1, -1)+ self.init_bg_feat.view(1, 1, -1)
                caux_feat = caux[:2] if self.obs_info.count_obs is None else caux[2:]
                tau_feat = tau[:self.n_topics*2] if self.obs_info.count_obs is None else tau[2*self.n_topics:]
                delta_feat = delta[self.obs_info.normal_obs.start_id:self.obs_info.normal_obs.end_id]
                lambda_feat = lambda_[:, self.obs_info.normal_obs.start_id:self.obs_info.normal_obs.end_id]
                noise_feat =  noise[:, self.obs_info.normal_obs.start_id:self.obs_info.normal_obs.end_id]
                beta_feat = self._compute_topic_params(caux_feat, 
                                                        tau_feat, 
                                                        delta_feat, 
                                                        lambda_feat, 
                                                        bg_feat, 
                                                        noise_feat, 
                                                        is_batch=True)
                beta_feat = torch.clamp(beta_feat, min=0)
                
                if self.n_batches > 1:
                    batch_tau_feat = batch_tau[:self.n_topics*2] if self.obs_info.count_obs is None else batch_tau[2*self.n_topics:]
                    batch_delta_feat = batch_delta[:, self.obs_info.normal_obs.start_id:self.obs_info.normal_obs.end_id]
                    lkl_param_feat = self._compute_with_adding_batch_effect_batch_normal(theta, batch_tau_feat,
                                                                                    batch_delta_feat, beta_feat, 
                                                                                    batch_labels, bg_seed)
                else:
                    if not self.condition_mask is None and self.wt_fusion_top_seed > 0 and self.valid_condition_genes.sum() < self.condition_mask.shape[1]:
                        _beta_seeded = beta_feat[0].masked_fill(self.seed_mask == 0, 0)
                        _beta_seeded[:, ~self.valid_condition_genes] = bg_seed.view(1, -1).expand(_beta_seeded.shape[0], -1)
                        top_dist_seed = _beta_seeded
                        top_dist_bg   = beta_feat[1].masked_fill(self.seed_mask == 1, 0)
                        lkl_param_feat = self.wt_fusion_top_seed * theta @ top_dist_seed + (1 - self.wt_fusion_top_seed) * theta @ top_dist_bg
                    else:
                        lkl_param_feat = theta @ beta_feat[0]
                        
                        
            if self.is_group_mode:
                out = out_counts if self.out_dim_rna > 0 else out_normal
                assert(not self.condition_mask is None, 'No gene seeds provided, group mode requires gene seeds...')
                assert(self.wt_fusion_top_seed > 0, 'group mode requires wt_fusion_top_seed > 0.0 ...')
                # out_group_seed = []
                # lkl_param_group_seed = [] 
                # group_param_scale_seed = []
                for ith, signature in enumerate(self.condition_mask):
                    out_normal_seed = out[:, signature].mean(dim=-1, keepdim=True)
                    lkl_param_seed = lkl_param_count[:, signature].mean(dim=-1, keepdim=True) if self.out_dim_rna > 0 else lkl_param_feat[:, signature].mean(dim=-1, keepdim=True)
                    # out_group_seed.append(out_normal_seed)
                    # lkl_param_group_seed.append(lkl_param_seed)
                
                    normal_scale_seed = disp[signature].sum(dim=-1, keepdim=True) if self.out_dim_rna > 0 else normal_scale[signature].sum(dim=-1, keepdim=True) 
                    # group_param_scale_seed.append(normal_scale_seed)
                    with pyro.poutine.scale(scale=self.pos_scale if ith == 0 else 1 - self.pos_scale):
                        feat  = pyro.sample(f"obs_feat_seed{ith}",
                            dist.Normal(lkl_param_seed, normal_scale_seed).to_event(1), 
                            obs=out_normal_seed
                        ) 
                
                # out_normal_seed = torch.hstack(out_group_seed)
                # lkl_param_seed = torch.hstack(lkl_param_group_seed)
                # logger.debug(f"{group_param_scale_seed}")
                # logger.debug(f"{group_param_scale_seed[0]}")
                # logger.debug(f"{len(group_param_scale_seed)}")
                
                # normal_scale_seed = torch.hstack(group_param_scale_seed).view(1, -1)
                
                # logger.debug(f"Size of group outs: out_normal {out_normal_seed.shape}, lkl_param {lkl_param_seed.shape}, scale: {normal_scale_seed.shape}")
                # with pyro.poutine.scale(scale=self.wt_fusion_top_seed):
                #     feat  = pyro.sample("obs_feat_seed",
                #         dist.Normal(lkl_param_seed, normal_scale_seed).to_event(1), 
                #         obs=out_normal_seed
                #     ) 
                    
                # if self.wt_fusion_top_seed < 1.0:                     
                #     out_normal_bg = out[:, ~self.valid_condition_genes].mean(dim=-1, keepdim=True)
                #     lkl_param_bg = lkl_param_feat[:, ~self.valid_condition_genes].mean(dim=-1, keepdim=True)
                #     normal_scale_bg = normal_scale[~self.valid_condition_genes].sum(dim=-1, keepdim=True)
                    
                #     with pyro.poutine.scale(scale= 1 - self.wt_fusion_top_seed):
                #         feat  = pyro.sample("obs_feat_bg",
                #             dist.Normal(lkl_param_bg, normal_scale_bg).to_event(1), 
                #             obs=out_normal_bg
                #     )
                    
            else:
                if self.out_dim_rna > 0:                    
                    if self.use_nb_obs:
                        
                        if self.condition_mask is None or self.wt_fusion_top_seed == 0:
                            total_count = torch.sum(out_counts, -1, keepdim=True)
                            
                            inv_disp = 1 / disp**2
                            mean, inv_disp = broadcast_all(total_count * lkl_param_count + 1e-15, inv_disp)
                            count = pyro.sample('obs_rna',
                                                dist.GammaPoisson(inv_disp, inv_disp / mean).to_event(1),
                                                obs=out_counts)
                        elif self.wt_fusion_top_seed == 1:
                            out_counts = out_counts[:, self.valid_condition_genes]
                            total_count = torch.sum(out_counts, -1, keepdim=True)
                            lkl_param_count = lkl_param_count[:, self.valid_condition_genes]
                            disp = disp[self.valid_condition_genes]
                            
                            inv_disp = 1 / disp**2
                            mean, inv_disp = broadcast_all(total_count * lkl_param_count + 1e-15, inv_disp)
                            count = pyro.sample('obs_rna',
                                                dist.GammaPoisson(inv_disp, inv_disp / mean).to_event(1),
                                                obs=out_counts)
                        else:
                            out_counts_seed = out_counts[:, self.valid_condition_genes]
                            total_count_seed = torch.sum(out_counts_seed, -1, keepdim=True)
                            lkl_param_seed = lkl_param_count[:, self.valid_condition_genes]
                            disp_seed = disp[self.valid_condition_genes]
                            
                            inv_disp_seed = 1 / disp_seed**2
                            mean_seed, inv_disp_seed = broadcast_all(total_count_seed * lkl_param_seed + 1e-15, inv_disp_seed)
                            
                            with pyro.poutine.scale(scale=self.wt_fusion_top_seed):
                                count = pyro.sample('obs_rna_seed',
                                                    dist.GammaPoisson(inv_disp_seed, inv_disp_seed / mean_seed).to_event(1),
                                                    obs=out_counts_seed)
                                
                            out_counts_bg = out_counts[:, ~self.valid_condition_genes]
                            total_count_bg = torch.sum(out_counts_bg, -1, keepdim=True)
                            lkl_param_bg = lkl_param_count[:, ~self.valid_condition_genes]
                            disp_bg = disp[~self.valid_condition_genes]
                            
                            inv_disp_bg = 1 / disp_bg**2
                            mean_bg, inv_disp_bg = broadcast_all(total_count_bg * lkl_param_bg + 1e-15, inv_disp_bg)
                                
                            with pyro.poutine.scale(scale= 1 - self.wt_fusion_top_seed):
                                count = pyro.sample('obs_rna_bg',
                                                    dist.GammaPoisson(inv_disp_bg, inv_disp_bg / mean_bg).to_event(1),
                                                    obs=out_counts_bg)                    
                    else:
                        if self.condition_mask is None or self.wt_fusion_top_seed == 0:
                            total_count = int(out_counts.sum(-1).max())
                            count = pyro.sample('obs_rn',
                                            dist.Multinomial(total_count, lkl_param_count),
                                            obs=out_counts)
                        elif self.wt_fusion_top_seed == 1:
                            total_count = int(out_counts[:, self.valid_condition_genes].sum(-1).max())
                            count = pyro.sample('obs_rna',
                                            dist.Multinomial(total_count, lkl_param_count[:, self.valid_condition_genes]),
                                            obs=out_counts[:, self.valid_condition_genes])
                        else:
                            total_count_seed = int(out_counts[:, self.valid_condition_genes].sum(-1).max())
                            total_count_bg = int(out_counts[:, ~self.valid_condition_genes].sum(-1).max())
                            with pyro.poutine.scale(scale=self.wt_fusion_top_seed):
                                count = pyro.sample('obs_seed',
                                                dist.Multinomial(total_count_seed, lkl_param_count[:, self.valid_condition_genes]),
                                                obs=out_counts[:, self.valid_condition_genes])
                            with pyro.poutine.scale(scale=1-self.wt_fusion_top_seed):
                                count = pyro.sample('obs_bg',
                                                dist.Multinomial(total_count_bg, lkl_param_count[:, ~self.valid_condition_genes]),
                                                obs=out_counts[:, ~self.valid_condition_genes])
                            
                                
                if self.out_dim_normal > 0:
                    # with pyro.poutine.scale(scale=self.scale_normal_feat):
                    if self.condition_mask is None or self.wt_fusion_top_seed == 0:
                        feat  = pyro.sample("obs_feat",
                                dist.Normal(lkl_param_feat, normal_scale).to_event(1), 
                                obs=out_normal
                        )
                    elif self.wt_fusion_top_seed == 1:
                        out_normal = out_normal[:, self.valid_condition_genes]
                        lkl_param_feat = lkl_param_feat[:, self.valid_condition_genes]
                        normal_scale = normal_scale[self.valid_condition_genes]
                        
                        feat  = pyro.sample("obs_feat",
                                dist.Normal(lkl_param_feat, normal_scale).to_event(1), 
                                obs=out_normal
                            )
                    else:
                        out_normal_seed = out_normal[:, self.valid_condition_genes]
                        lkl_param_seed = lkl_param_feat[:, self.valid_condition_genes]
                        
                        normal_scale_seed = normal_scale[self.valid_condition_genes]
                        
                        with pyro.poutine.scale(scale=self.wt_fusion_top_seed):
                            feat  = pyro.sample("obs_feat_seed",
                                dist.Normal(lkl_param_seed, normal_scale_seed).to_event(1), 
                                obs=out_normal_seed
                            )                        
                        out_normal_bg = out_normal[:, ~self.valid_condition_genes]
                        lkl_param_bg = lkl_param_feat[:, ~self.valid_condition_genes]
                        normal_scale_bg = normal_scale[~self.valid_condition_genes]
                            
                        with pyro.poutine.scale(scale= 1 - self.wt_fusion_top_seed):
                            feat  = pyro.sample("obs_feat_bg",
                                dist.Normal(lkl_param_bg, normal_scale_bg).to_event(1), 
                                obs=out_normal_bg
                            )
        
                    
    def guide(self, input, out_counts=None, out_normal=None, batch_labels=None):
        pyro.module("encoder", self.encoder)

        if self.use_nb_obs and not self.obs_info.count_obs is None:
            self.disp_loc = pyro.param(
                    'disp_loc',
                    self._ones_init((self.out_dim_rna)),
                )
            self.disp_scale = pyro.param(
                'disp_scale',
                self._ones_init((self.out_dim_rna)),
                constraint=constraints.positive
            )
            
            with pyro.plate(f"Dim_rna", self.out_dim_rna):
                pyro.sample("disp", dist.LogNormal(self.disp_loc, self.disp_scale))
            
        if not self.condition_mask is None and self.wt_fusion_top_seed > 0 and self.valid_condition_genes.sum() < self.condition_mask.shape[1]:
            self.seed_bg_loc = pyro.param(
                    'seed_bg_loc',
                    self._ones_init((self.valid_condition_genes.shape[0] - self.seed_gene_dim)),
                )
            self.seed_bg_scale = pyro.param(
                'seed_bg_scale',
                self._ones_init((self.valid_condition_genes.shape[0] - self.seed_gene_dim)),
                constraint=constraints.positive
            )
            with pyro.plate(f"Dim_seed_bg", 
                        self.valid_condition_genes.shape[0] - self.seed_gene_dim):
                pyro.sample("bg_seed", dist.Normal(self.seed_bg_loc, self.seed_bg_scale))
        
            
                
            
                
                
        if not self.obs_info.normal_obs is None:
            self.feat_scale_loc = pyro.param(
                    'feat_scale_loc',
                    self._ones_init((self.out_dim_normal)),
                )
            self.feat_scale_scale = pyro.param(
                    'feat_scale_scale',
                    self._ones_init((self.out_dim_normal)),
                    constraint=constraints.positive
                )
            
            with pyro.plate(f"Dim_normal", self.out_dim_normal):
                pyro.sample("normal_scale", dist.LogNormal(self.feat_scale_loc, self.feat_scale_scale))
                
        
        ### caux
        self.caux_loc = pyro.param("caux_loc",
            self._ones_init((self.obs_info.n_units), multiplier=1),
            )
        
        self.caux_scale = pyro.param(
            "caux_scale",
            self._ones_init((self.obs_info.n_units)),
            constraint=constraints.positive,
        )
        
        with pyro.plate(f'N_modality', self.obs_info.n_units):
            caux = pyro.sample("caux", dist.LogNormal(self.caux_loc, self.caux_scale))
            
            
        ### delta, bg, batch_delta, lambda_ and noise with outdim
        self.delta_loc = pyro.param(
                "delta_loc",
                self._zeros_init((self.obs_info.full_dim)))
        
        self.delta_scale = pyro.param(
            "delta_scale",
            self._ones_init((self.obs_info.full_dim)),
            constraint=constraints.positive,
        )
        
        self.bg_loc = pyro.param(
                "bg_loc",
                self._zeros_init(self.obs_info.full_dim),
            )

        self.bg_scale = pyro.param(
            "bg_scale",
            self._ones_init(self.obs_info.full_dim),
            constraint=constraints.positive
        )
        
        if self.n_batches > 1:
            self.batch_delta_loc = pyro.param(
                "batch_delta_loc",
                self._zeros_init((self.n_batches, self.obs_info.full_dim)))
            self.batch_delta_scale = pyro.param(
                "batch_delta_scale",
                self._ones_init((self.n_batches, self.obs_info.full_dim)),
                constraint=constraints.positive,
            )
        
        self.lambda_loc = pyro.param(
                "lambda_loc",
                self._zeros_init(
                    (self.n_topics,self.obs_info.full_dim),
                ),
            )
        self.lambda_scale = pyro.param(
            "lambda_scale",
            self._ones_init(
                (self.n_topics, self.obs_info.full_dim),
            ),
            constraint=constraints.positive,
        )

        self.noise_loc = pyro.param(
            "noise_loc",
            self._zeros_init((self.n_topics, self.obs_info.full_dim)),
        )

        self.noise_scale = pyro.param(
            'noise_scale',
            self._ones_init(
                (self.n_topics, self.obs_info.full_dim),
            ),
            constraint=constraints.positive,
        )
        
        # tau, batch_tau
        self.tau_loc = pyro.param(
            "tau_loc",
            self._zeros_init((self.n_topics * self.obs_info.n_units)),
        )
        
        self.tau_scale = pyro.param(
            "tau_scale",
            self._ones_init((self.n_topics * self.obs_info.n_units)),
            constraint=constraints.positive,
        )
        
        if self.n_batches > 1:
        
            self.batch_tau_loc = pyro.param(
                    "batch_tau_loc",
                    self._zeros_init((self.n_topics * self.obs_info.n_units)),
                )
            self.batch_tau_scale = pyro.param(
                "batch_tau_scale",
                self._ones_init((self.n_topics * self.obs_info.n_units)),
                constraint=constraints.positive,
            )
        
        with pyro.plate(f'N_feature', self.obs_info.full_dim):
            pyro.sample("delta", dist.LogNormal(self.delta_loc, self.delta_scale))
            pyro.sample("bg", dist.Normal(self.bg_loc, self.bg_scale))
            if self.n_batches > 1:
                with pyro.plate(f'N_batches', self.n_batches):
                    pyro.sample(
                        "batch_delta",
                        dist.Normal(self.batch_delta_loc, 
                                    self.batch_delta_scale),
                    )
                    
            with pyro.plate(f"N_real_top", self.n_topics):
                pyro.sample(
                        "lambda_",
                        dist.LogNormal(self.lambda_loc, self.lambda_scale),
                    )

                pyro.sample(
                            "noise",
                            dist.Normal(self.noise_loc, self.noise_scale),
                )
                
        with pyro.plate(f'N_all_top', self.n_topics * self.obs_info.n_units):
            pyro.sample("tau", dist.LogNormal(self.tau_loc, self.tau_scale))
            if self.n_batches > 1:
                pyro.sample(
                            "batch_tau",
                            dist.TransformedDistribution(
                                dist.Normal(self.batch_tau_loc, self.batch_tau_scale),
                                dist.transforms.SigmoidTransform(),
                            ),
                        )

        with pyro.plate(f'N_obs', input.shape[0]):
            logtheta_loc, logtheta_scale = self.encoder(input, batch_labels)            
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            
    def regularizer_topic_prior(self, logtheta_loc, topic_prior=None):
        loss = 0
        if not topic_prior is None:
            valid_sel = (topic_prior.sum(dim=1) > 0.5)
            if valid_sel.sum() > 0:
                loss = torch.nn.CrossEntropyLoss(
                                                 label_smoothing=0.01, reduction='sum')(
                        logtheta_loc[valid_sel], 
                        topic_prior[valid_sel].float(),
                    )
                loss = loss / valid_sel.sum()
        return loss

    def infer_topic(self, input, batch_labels=None):
        """
        Get topic from model
        """
        # X: BS x PCA_dim
        with torch.no_grad():
            logtheta_loc, logtheta_scale = self.encoder(input, batch_labels)
            theta = F.softmax(logtheta_loc, -1)
        return theta

    def _xavier_init(self, shape, device):
        return torch.randn(shape, device=device) * (math.sqrt(2 / np.sum(shape)))

    def _zeros_init(self, shape):
        return torch.zeros(shape, device=self.device)

    def _ones_init(self, shape, multiplier=0.1):
        return torch.ones(shape, device=self.device) * multiplier
    
    def encode(self, input, batch_labels=None):
        logtheta_loc, _ = self.encoder(input, batch_labels)
        return logtheta_loc
    
    def beta_rna(self, pseudocount=0.1):
        """
        Get topic X out_feature matrix
        """
        if self.obs_info.count_obs is None:
            logger.warning('No rna output is modelled, no beta for rna, return None')
            return None
        
        tau = self.mean(self.tau_loc[:self.n_topics*2], self.tau_scale[:self.n_topics*2])
        delta = self.mean(self.delta_loc[self.obs_info.count_obs.start_id:self.obs_info.count_obs.end_id], 
                          self.delta_scale[self.obs_info.count_obs.start_id:self.obs_info.count_obs.end_id])
        lambda_ = self.mean(self.lambda_loc[:, self.obs_info.count_obs.start_id:self.obs_info.count_obs.end_id], 
                            self.lambda_scale[:, self.obs_info.count_obs.start_id:self.obs_info.count_obs.end_id])
        caux = self.mean(self.caux_loc[:2], self.caux_scale[:2])
        noise = self.noise_loc[:, self.obs_info.count_obs.start_id:self.obs_info.count_obs.end_id]

        lambda_tilde = self._compute_lambda_tilde_batch(caux, tau, delta, lambda_)
            
        beta = 0 + noise.view(2, noise.shape[0], noise.shape[1]//2) * lambda_tilde
                    
        bg = (self.bg_loc[self.obs_info.count_obs.start_id:self.obs_info.count_obs.end_id].view(2, 1, -1) 
              + self.init_bg_count.view(1, 1, -1)).exp()

        if pseudocount > 0:
            pseudocount0 = torch.quantile(bg[0], q=pseudocount)
            pseudocount1 = torch.quantile(bg[1], q=pseudocount)
            pseudocount = torch.concat([pseudocount0.view(1,1,1),pseudocount1.view(1,1,1) ], dim=0)
        else:
            pseudocount = 0
        
        adjust = torch.log(bg + pseudocount) - torch.log(bg)
        beta = beta - adjust
        
        if self.condition_mask is not None and self.wt_fusion_top_seed > 0:
            top_dist_seed = torch.softmax(beta[0].masked_fill(self.condition_mask == 0, float('-inf')), dim=-1)
            top_dist_bg   = torch.softmax(beta[1], dim=-1)
            top_dist = self.wt_fusion_top_seed * top_dist_seed + (1 - self.wt_fusion_top_seed) * top_dist_bg
        else:
            top_dist = torch.softmax(beta[0], dim=-1)
        return top_dist.cpu()
    
    def beta_normal(self, pseudocount=0.1):
        """
        Get topic X out_feature matrix
        """
        if self.obs_info.normal_obs is None:
            logger.warning('No normal feature output is modelled, no beta for normal features, return None')
            return None
        
        tau = self.mean(self.tau_loc[:self.n_topics*2] if self.obs_info.count_obs is None else self.tau_loc[2*self.n_topics:], 
                        self.tau_scale[:self.n_topics*2] if self.obs_info.count_obs is None else self.tau_scale[2*self.n_topics:])
        delta = self.mean(self.delta_loc[self.obs_info.normal_obs.start_id:self.obs_info.normal_obs.end_id], 
                          self.delta_scale[self.obs_info.normal_obs.start_id:self.obs_info.normal_obs.end_id])
        lambda_ = self.mean(self.lambda_loc[:, self.obs_info.normal_obs.start_id:self.obs_info.normal_obs.end_id], 
                            self.lambda_scale[:, self.obs_info.normal_obs.start_id:self.obs_info.normal_obs.end_id])
        caux = self.mean(self.caux_loc[:2] if self.obs_info.count_obs is None else self.caux_loc[2:], 
                         self.caux_scale[:2] if self.obs_info.count_obs is None else self.caux_scale[2:])

        noise = self.noise_loc[:, self.obs_info.normal_obs.start_id:self.obs_info.normal_obs.end_id]
        
        lambda_tilde = self._compute_lambda_tilde_batch(caux, tau, delta, lambda_)
            
        beta = 0 + noise.view(2, noise.shape[0], noise.shape[1]//2) * lambda_tilde
        
        beta = torch.clamp(beta, min=0)
        
        bg = (self.bg_loc[self.obs_info.normal_obs.start_id:self.obs_info.normal_obs.end_id].view(2, 1, -1) 
              + self.init_bg_feat.view(1, 1, -1)).exp()

        if pseudocount > 0:
            pseudocount0 = torch.quantile(bg[0], q=pseudocount)
            pseudocount1 = torch.quantile(bg[1], q=pseudocount)
            pseudocount = torch.concat([pseudocount0.view(1,1,1),pseudocount1.view(1,1,1) ], dim=0)
        else:
            pseudocount = 0
            
        adjust = torch.log(bg + pseudocount) - torch.log(bg)
        beta = beta - adjust
        
        if self.condition_mask is not None and self.wt_fusion_top_seed > 0:
            top_dist_seed = beta[0].masked_fill(self.condition_mask == 0, 0)
            top_dist_bg   = beta[1]
            top_dist = self.wt_fusion_top_seed * top_dist_seed + (1 - self.wt_fusion_top_seed) * top_dist_bg
        else:
            top_dist = beta[0]
        
        return top_dist.cpu()


    def beta(self, pseudocount=0.1, return_beta_rna=True, return_beta_normal=False):
        """
        Get topic X out_feature matrix
        """
        if not return_beta_rna and not return_beta_normal:
            logger.warning('Not returning any beta, as specified')
            return None
        
        results = []
        if return_beta_rna:
            results.append(self.beta_rna(pseudocount=pseudocount))
        
        if return_beta_normal:
            results.append(self.beta_normal(pseudocount=pseudocount))

        if len(results) == 1:
            return results[0]
        return results

    def mean(self, loc, scale):
        return dist.LogNormal(loc, scale).mean

    def train(self):
        self.encoder.train()

    def eval(self):
        self.encoder.eval()        
        