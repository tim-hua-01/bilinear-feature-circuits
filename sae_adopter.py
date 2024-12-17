import os
import sys
parent_dir = os.path.abspath('.')
sys.path.append(parent_dir + '/bilinear_interp_tim')
from einops import einsum
from sae import SAEConfig, SAE
from shared.hub import from_pretrained
from dictionary_learning.dictionary import Dictionary
import torch
class DictionarySAE(SAE, Dictionary):
    """
    An SAE that maintains most of the functionality of SAEs from the bilinear interp package, but with encode/decode/forward methods
    that are more inline with the format used by 
    """
    def __init__(self, config):
        SAE.__init__(self, config)
        self.activation_dim = config.d_model
        self.dict_size = config.d_features
        
    def encode(self, x):
        """
        Dictionary interface encode method that wraps SAE's encode method
        """
        return super().encode(self.preprocess(x))
    
    def decode(self, f):
        """
        Dictionary interface decode method that wraps SAE's decode method
        """
        return self.postprocess(super().decode(f))
    
    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass implementing both SAE and Dictionary interfaces
        """
        if ghost_mask is None:
            x_processed = self.preprocess(x)
            x_hat, f = super().forward(x_processed)
            x_hat = self.postprocess(x_hat) + einsum(self.passthrough, x, "f, ... f -> ... f")
            
            if output_features:
                return x_hat, f
            return x_hat
        else:
            # Implement ghost mode similar to AutoEncoder
            x_processed = self.preprocess(x)
            f_pre = self.w_enc(x_processed - self.b_dec)
            f_ghost = torch.exp(f_pre) * ghost_mask.to(f_pre)
            
            # Normal encode-decode path
            f = self.encode(x_processed)
            x_hat = self.decode(f)
            
            # Ghost decode path (without bias)
            x_ghost = self.w_dec(f_ghost)
            
            x_hat = self.postprocess(x_hat)+ einsum(self.passthrough, x, "f, ... f -> ... f")
            x_ghost = self.postprocess(x_ghost)+ einsum(self.passthrough, x, "f, ... f -> ... f")
            
            if output_features:
                return x_hat, x_ghost, f
            return x_hat, x_ghost
    @staticmethod
    def from_config(*args, **kwargs):
        return DictionarySAE(SAEConfig(*args, **kwargs))
    
    @staticmethod
    def from_pretrained(repo_id_or_model, point, expansion, k, tag=None):
        repo_id = repo_id_or_model if isinstance(repo_id_or_model, str) else f"{repo_id_or_model.config.repo}-scope"
        config = SAEConfig(point=point, expansion=expansion, k=k, d_model=0, tag=tag)
        return from_pretrained(DictionarySAE, repo_id, config.name)
    