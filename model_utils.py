
import os
import sys
parent_dir = os.path.abspath('.')
sys.path.append(parent_dir + '/bilinear_interp_tim')
sys.path.append(parent_dir + '/dictionary_learning')
sys.path.append(parent_dir)
import argparse
import gc
import json
import math
from collections import defaultdict
import torch as t
from einops import rearrange, repeat, reduce,einsum
from tqdm import tqdm
from sae import Tracer, Visualizer
from activation_utils import SparseAct
from attribution import patching_effect, jvp
from circuit_plotting import plot_circuit, plot_circuit_posaligned
from dictionary_learning import AutoEncoder
from loading_utils import load_examples, load_examples_nopair
from nnsight import LanguageModel
from nnsight.envoy import Envoy
from language import Transformer, Sight
from sae_adopter import DictionarySAE
from bilinear_circuits_v0 import initialize_model_and_dictionaries, get_circuit
import numpy as np
import pandas as pd
from plotly import express as px
from typing import Callable, Optional, Union, Tuple


device, model, embed, attns, mlps, resids, dictionaries, _, _, _, _, _, _ = initialize_model_and_dictionaries(
    device='cuda:0',
    model_name="tdooms/fw-nano",
    dict_id='10',  # Note: This was originally an int, but the function expects a string.
    d_model=1024,
    dict_path='tdooms/fw-nano-scope',
    dataset='simple_train',
    num_examples=20,
    example_length=None,
    batch_size=4,
    aggregation='sum',
    nopair=False,
)
DESIGNED = model._model.w_e[:,5682]
all_submods = [embed] + [submod for layer_submods in zip(mlps, attns, resids) for submod in layer_submods]
def single_tok_logprob(tok_ind:int) -> Callable[[LanguageModel],t.Tensor]:
    def metric_fn(model: LanguageModel):
        # Get the logits for the last token in the sequence
        logits = model.lm_head.output[:, -1, :]
        
        # Apply log-softmax to convert logits to log probabilities
        log_probs = t.nn.functional.log_softmax(logits, dim=-1)
        
        # Gather the log probability for the specified token index
        log_prob = t.gather(log_probs, dim=-1, index=t.tensor([tok_ind], device=model.device).view(-1, 1)).squeeze(-1)
        
        return log_prob
    return metric_fn
input_tensor = t.tensor(model.tokenizer("A trigger is designed to activate a task of the virus, as display ing strange messages, deleting files, sending emails  begin the replicate process or whatever the programmer write in his malicious code.")['input_ids'], device=device).unsqueeze(0)
short_input = input_tensor[:, :5]

model._model.config.dataset = 'fineweb'

effects, deltas, grads, total_effect = patching_effect(
        clean = short_input,
        patch = None,
        model = model,
        submodules = all_submods,
        dictionaries = dictionaries,
        metric_fn = single_tok_logprob(298),
        metric_kwargs=dict(),
        method='ig' # get better approximations for early layers by using ig
    )




def sort_by_interaction_then_plot(q_mat: np.ndarray, labels: list[str], sort_index:int = 0) -> None:
    sorted_indices = np.argsort(q_mat[:,sort_index])
    sorted_indices = sorted_indices[::-1]
    sorted_q_mat = q_mat[sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    
    # Create the figure
    fig = px.imshow(sorted_q_mat, color_continuous_midpoint=0, color_continuous_scale="RdBu", x=labels, y=labels)
    fig.show()  # Explicitly show the figure

baseline_out = model.forward(short_input)


def compare_distributions(new_out, reference_out = baseline_out, correct_token=298, top_k=5, tokenizer=model._model.tokenizer) -> None:
    """
    Compare two probability distributions (from two causal LM outputs) by checking:
      - The log probability of the correct token.
      - The top-k tokens (IDs, decoded strings, and log probabilities) for the reference distribution.
      - The top-k tokens for the new distribution.
      - The overlap between the top-k tokens of the two distributions.
    
    Parameters:
      new_out: A model output from a modified run. Must have an attribute 'logits' of shape [batch, 
               sequence_length, vocab_size].
      reference_out: The reference (baseline) model output.
      correct_token: The token ID of the "correct" token (default is 298).
      top_k: The number of top tokens to consider from each distribution (default is 5).
      tokenizer: The tokenizer to convert token ids to strings. (Must implement a .decode method)
      
    Returns:
      A dictionary containing:
         - correct_token: Information on the correct token (id, decoded token, and its log probs in both distributions)
         - reference_top5: A list of dicts for the top_k tokens from the reference distribution.
         - new_top5: A list of dicts for the top_k tokens from the new distribution.
         - overlap: The overlapping tokens (both IDs and decoded tokens) between the two top_k lists.
    """
    import torch as t

    if tokenizer is None:
        raise ValueError("A tokenizer must be provided to decode token ids.")
    
    # Extract log probabilities from the logits for the last token.
    new_log_probs = t.nn.functional.log_softmax(new_out.logits, dim=-1)[0, -1]
    ref_log_probs = t.nn.functional.log_softmax(reference_out.logits, dim=-1)[0, -1]
    
    # Get the log probabilities for the correct token.
    correct_log_prob_new = new_log_probs[correct_token].item()
    correct_log_prob_ref = ref_log_probs[correct_token].item()
    
    # Get top-k tokens for both distributions.
    new_topk = t.topk(new_log_probs, k=top_k)
    ref_topk = t.topk(ref_log_probs, k=top_k)
    
    new_topk_ids = new_topk.indices.tolist()
    ref_topk_ids = ref_topk.indices.tolist()
    
    new_topk_values = new_topk.values.tolist()
    ref_topk_values = ref_topk.values.tolist()
    
    # Decode tokens using the provided tokenizer.
    new_topk_tokens = [tokenizer.decode([tid]) for tid in new_topk_ids]
    ref_topk_tokens = [tokenizer.decode([tid]) for tid in ref_topk_ids]
    correct_token_decoded = tokenizer.decode([correct_token])
    
    # Compute the overlap (common tokens by ID) between the two top_k lists.
    overlap_ids = set(new_topk_ids).intersection(ref_topk_ids)
    overlap_tokens = [tokenizer.decode([tid]) for tid in sorted(list(overlap_ids))]
    
    # Print comparison results.
    print("Correct Token:")
    print(f"  Token ID: {correct_token}, Token: {correct_token_decoded}")
    print(f"  Reference Log Prob: {correct_log_prob_ref:.4f}")
    print(f"  New Log Prob:       {correct_log_prob_new:.4f}")
    print(f"  Change:             {correct_log_prob_new - correct_log_prob_ref:.4f}")
    print("\nReference Distribution Top 5 Tokens:")
    for rank, (tid, token_str, prob) in enumerate(zip(ref_topk_ids, ref_topk_tokens, ref_topk_values), start=1):
        print(f"  Rank {rank}: ID {tid}, Token: {token_str}, Log Prob: {prob:.4f}")
    print("\nNew Distribution Top 5 Tokens:")
    for rank, (tid, token_str, prob) in enumerate(zip(new_topk_ids, new_topk_tokens, new_topk_values), start=1):
        print(f"  Rank {rank}: ID {tid}, Token: {token_str}, Log Prob: {prob:.4f}")
    print("\nOverlap in Top 5 Tokens:")
    if overlap_tokens:
        print("  " + ", ".join(overlap_tokens))
    else:
        print("  None")
    
    # Package results into a dictionary.
    result = {
        "correct_token": {
            "id": correct_token,
            "token": correct_token_decoded,
            "log_prob_ref": correct_log_prob_ref,
            "log_prob_new": correct_log_prob_new
        },
        "reference_top5": [
            {"id": tid, "token": token_str, "log_prob": prob}
            for tid, token_str, prob in zip(ref_topk_ids, ref_topk_tokens, ref_topk_values)
        ],
        "new_top5": [
            {"id": tid, "token": token_str, "log_prob": prob}
            for tid, token_str, prob in zip(new_topk_ids, new_topk_tokens, new_topk_values)
        ],
        "overlap": {
            "ids": list(overlap_ids),
            "tokens": overlap_tokens
        }
    }
    #return result



with model.trace(short_input):
    latent_acts = dict()
    for submod in all_submods:
        subout = submod.output
        sub_latents = dictionaries[submod].encode(subout)
        sub_latents.save()
        latent_acts[submod] = sub_latents


visualizers = dict()
for m in all_submods:
    try:
        visualizers[m] = Visualizer(model._model, dictionaries[m])
    except Exception as e:
        print(f"Error creating visualizer for {m}: {e}")

def print_effects_and_residual_effects(module: Envoy, last_token_only = True, negative_effects = False):
    if last_token_only:
        print(f"Positive effects at {module}:")
        print(t.topk(-effects[module].act[0,-1], k = 31)) #For some weird reason the effect signs are flipped
        if negative_effects:
            print(f"Negative effects at {module}:")
            print(t.topk(effects[module].act[0,-1], k = 31))
        print(f"Latent activations at {module}:")
        print(t.topk(latent_acts[module][0,-1], k = 31))
        print(f"Residual effects at {module}:")
    else:
        print(f"Positive effects at {module}:")
        print(t.topk(-effects[module].act, k = 31)) #For some weird reason the effect signs are flipped
        if negative_effects:
            print(f"Negative effects at {module}:")
            print(t.topk(effects[module].act, k = 31))
        print(f"Latent activations at {module}:")
        print(t.topk(latent_acts[module], k = 31))
        print(f"Residual effects at {module}:")
    print(effects[module].resc)

def compute_aggregated_latents(
    module,
    position_indices: dict[int, list[int]],
    target_norms: Optional[Union[float, dict[int, float]]] = None,
    norm_method: Optional[str] = None,  # "scale" or "noise"
    latent_store: dict = latent_acts,
    dictionary_store: dict = dictionaries,
) -> Tuple[t.Tensor, t.Tensor]:
    """
    Aggregates a weighted sum of decoder directions using latent activations for a given module.
    
    The function retrieves the latent activations and the corresponding decoder weight matrix
    (w_dec) from the provided stores. For each sequence position (ignoring position 0), it then:
    
      - Looks up the specified latent indices (from `position_indices`).
      - For each latent index, extracts the latent activation value at that sequence position 
        (a scalar) and the associated decoder direction (i.e. the column from w_dec).
      - Aggregates (sums) the products of these scalars and decoder directions to produce a vector.
      - If no indices are specified for a position, that position will have a zero vector.
    
    Optionally, the function adjusts the aggregated vector at each sequence position to match
    a target norm. Two methods are implemented:
      - "scale": Multiplying the vector by (target / current_norm) (or using a random unit 
                 vector if the aggregate is zero).
      - "noise": Adding calibrated Gaussian noise (in a random direction) if the current norm is less 
                 than the target.
    
    Parameters:
      module: The module key used to access the latent activations and the dictionary.
      position_indices: A dict mapping sequence positions (positions ≥ 1) to a list of latent indices.
      target_norms: Either a single float (target norm across all positions) or a dict mapping positions 
                    to desired norm values. (Optional)
      norm_method: "scale" or "noise" to specify adjustment method for matching target norms (Optional).
      latent_store: A dictionary mapping modules to latent activation tensors (default: latent_acts).
      dictionary_store: A dictionary mapping modules to dictionary objects (default: dictionaries).

    Returns:
      aggregated: A tensor of shape (seq_len, d_model) containing the aggregated vectors per sequence position.
      norms: A tensor of shape (seq_len,) containing the norm of each aggregated vector.
    """
    # Retrieve latent activations for this module.
    latent_tensor = latent_store[module]
    # If latent_tensor has a batch dimension of 1, remove it.
    if latent_tensor.dim() == 3 and latent_tensor.size(0) == 1:
        latent_tensor = latent_tensor.squeeze(0)
    # Expected shape: (seq_len, latent_dim)
    seq_len, latent_dim = latent_tensor.shape

    # Retrieve the decoder weight matrix from the provided dictionary store.
    # Expected shape: (d_model, latent_dim)
    w_dec = dictionary_store[module].w_dec.weight.data
    d_model = w_dec.shape[0]

    # Prepare an output tensor: one vector per sequence position.
    aggregated = t.zeros(seq_len, d_model, device=latent_tensor.device, dtype=latent_tensor.dtype)

    # For each specified sequence position (positions are assumed >= 1),
    # sum contributions from the selected latent indices.
    for pos, indices in position_indices.items():
        if pos < seq_len:
            # Loop over specified latent indices
            for idx in indices:
                # Ensure latent index is in the valid range.
                if 0 <= idx < latent_dim and idx < w_dec.shape[1]:
                    weight = latent_tensor[pos, idx]  # scalar latent activation at this pos for this index
                    direction = w_dec[:, idx]           # decoder direction (vector of size d_model)
                    aggregated[pos] += weight * direction
                else:
                    print(f"Warning: latent index {idx} out of range for position {pos}.")
        else:
            print(f"Warning: sequence position {pos} is out of range (max pos {seq_len-1}).")

    # Compute the Euclidean norm (L2 norm) for each sequence position.
    norms = aggregated.norm(dim=-1)

    # If target_norms and a norm_method are provided, adjust each vector accordingly.
    if target_norms is not None and norm_method is not None:
        # Handle the case where a single float is provided.
        if isinstance(target_norms, (int, float)):
            target_norms_dict = {pos: target_norms for pos in range(seq_len)}
        else:
            target_norms_dict = target_norms

        # Adjust each position's vector.
        for pos in range(seq_len):
            if pos in target_norms_dict:
                target = target_norms_dict[pos]
                current_norm = aggregated[pos].norm().item()
                if norm_method == "scale":
                    if current_norm > 0:
                        aggregated[pos] = aggregated[pos] * (target / current_norm)
                    else:
                        # If the current aggregated vector is zero, generate a random unit vector.
                        new_vec = t.randn(d_model, device=aggregated.device, dtype=aggregated.dtype)
                        new_vec = new_vec / new_vec.norm() * target
                        aggregated[pos] = new_vec
                elif norm_method == "noise":
                    raise NotImplementedError("Noise method not implemented")
                    if current_norm < target:
                        diff = target - current_norm
                        noise_vec = t.randn(d_model, device=aggregated.device, dtype=aggregated.dtype)
                        noise_vec = noise_vec / noise_vec.norm() * diff #This doesn't actually achieve the desired norm
                        aggregated[pos] = aggregated[pos] + noise_vec
                else:
                    print(f"Warning: Unknown norm_method '{norm_method}'. No adjustment made at position {pos}.")

        # Recompute norms after adjustment.
        norms = aggregated.norm(dim=-1)

    return aggregated[1:], norms[1:]