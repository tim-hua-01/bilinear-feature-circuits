import os
import sys
parent_dir = os.path.abspath('.')
sys.path.append(parent_dir + '/bilinear_interp_tim')
import argparse
import gc
import json
import math
from collections import defaultdict
import torch as t
from einops import rearrange
from tqdm import tqdm

from activation_utils import SparseAct
from attribution import patching_effect, jvp
from circuit_plotting import plot_circuit, plot_circuit_posaligned
from dictionary_learning import AutoEncoder
from loading_utils import load_examples, load_examples_nopair
from nnsight import LanguageModel
from language import Transformer, Sight
from sae_adopter import DictionarySAE
from dictionary_learning.dictionary import IdentityDict


###### utilities for dealing with sparse COO tensors ######
def flatten_index(idxs, shape):
    """
    index : a tensor of shape [n, len(shape)]
    shape : a shape
    return a tensor of shape [n] where each element is the flattened index
    """
    idxs = idxs.t()
    # get strides from shape
    strides = [1]
    for i in range(len(shape)-1, 0, -1):
        strides.append(strides[-1]*shape[i])
    strides = list(reversed(strides))
    strides = t.tensor(strides).to(idxs.device)
    # flatten index
    return (idxs * strides).sum(dim=1).unsqueeze(0)

def prod(l):
    out = 1
    for x in l: out *= x
    return out

def sparse_flatten(x):
    x = x.coalesce()
    return t.sparse_coo_tensor(
        flatten_index(x.indices(), x.shape),
        x.values(),
        (prod(x.shape),)
    )

def reshape_index(index, shape):
    """
    index : a tensor of shape [n]
    shape : a shape
    return a tensor of shape [n, len(shape)] where each element is the reshaped index
    """
    multi_index = []
    for dim in reversed(shape):
        multi_index.append(index % dim)
        index //= dim
    multi_index.reverse()
    return t.stack(multi_index, dim=-1)

def sparse_reshape(x, shape):
    """
    x : a sparse COO tensor
    shape : a shape
    return x reshaped to shape
    """
    # first flatten x
    x = sparse_flatten(x).coalesce()
    new_indices = reshape_index(x.indices()[0], shape)
    return t.sparse_coo_tensor(new_indices.t(), x.values(), shape)

def sparse_mean(x, dim):
    if isinstance(dim, int):
        return x.sum(dim=dim) / x.shape[dim]
    else:
        return x.sum(dim=dim) / prod(x.shape[d] for d in dim)

######## end sparse tensor utilities ########


def get_circuit(
        clean,
        patch,
        model,
        embed,
        attns,
        mlps,
        resids,
        dictionaries,
        metric_fn,
        metric_kwargs=dict(),
        aggregation='sum', # or 'none' for not aggregating across sequence position
        nodes_only=False,
        node_threshold=0.1,
        edge_threshold=0.01,
):
    t.cuda.empty_cache()
    all_submods = [embed] + [submod for layer_submods in zip(mlps, attns, resids) for submod in layer_submods]
    
    # first get the patching effect of everything on y
    effects, deltas, grads, total_effect = patching_effect(
        clean,
        patch,
        model,
        all_submods,
        dictionaries,
        metric_fn,
        metric_kwargs=metric_kwargs,
        method='ig' # get better approximations for early layers by using ig
    )

    def unflatten(tensor): # will break if dictionaries vary in size between layers
        b, s, f = effects[resids[0]].act.shape
        unflattened = rearrange(tensor, '(b s x) -> b s x', b=b, s=s)
        return SparseAct(act=unflattened[...,:f], res=unflattened[...,f:])
    
    features_by_submod = {
        submod : (effects[submod].to_tensor().flatten().abs() > node_threshold).nonzero().flatten().tolist() for submod in all_submods
    }

    n_layers = len(resids)

    nodes = {'y' : total_effect}
    nodes['embed'] = effects[embed]
    for i in range(n_layers):
        nodes[f'attn_{i}'] = effects[attns[i]]
        nodes[f'mlp_{i}'] = effects[mlps[i]]
        nodes[f'resid_{i}'] = effects[resids[i]]

    if nodes_only:
        if aggregation == 'sum':
            for k in nodes:
                if k != 'y':
                    nodes[k] = nodes[k].sum(dim=1)
        nodes = {k : v.mean(dim=0) for k, v in nodes.items()}
        return nodes, None

    edges = defaultdict(lambda:{})
    edges[f'resid_{len(resids)-1}'] = { 'y' : effects[resids[-1]].to_tensor().flatten().to_sparse() }

    def N(upstream, downstream):
        return jvp(
            clean,
            model,
            dictionaries,
            downstream,
            features_by_submod[downstream],
            upstream,
            grads[downstream],
            deltas[upstream],
            return_without_right=True,
        )

    #print(f"Upstream deltas{deltas}")
    # now we work backward through the model to get the edges
    print('Edges time')
    for layer in reversed(range(len(resids))):
        print(f'\nAt layer {layer}\n')
        resid = resids[layer]
        mlp = mlps[layer]
        attn = attns[layer]
        
        MR_effect, MR_grad = N(mlp, resid)
        AR_effect, AR_grad = N(attn, resid)

        edges[f'mlp_{layer}'][f'resid_{layer}'] = MR_effect
        edges[f'attn_{layer}'][f'resid_{layer}'] = AR_effect

        if layer > 0:
            prev_resid = resids[layer-1]
        else:
            prev_resid = embed

        RM_effect, _ = N(prev_resid, mlp)
        RA_effect, _ = N(prev_resid, attn)

        MR_grad = MR_grad.coalesce()
        AR_grad = AR_grad.coalesce()

        RMR_effect = jvp(
            clean,
            model,
            dictionaries,
            mlp,
            features_by_submod[resid],
            prev_resid,
            {feat_idx : unflatten(MR_grad[feat_idx].to_dense()) for feat_idx in features_by_submod[resid]},
            deltas[prev_resid],
        )
        RAR_effect = jvp(
            clean,
            model,
            dictionaries,
            attn,
            features_by_submod[resid],
            prev_resid,
            {feat_idx : unflatten(AR_grad[feat_idx].to_dense()) for feat_idx in features_by_submod[resid]},
            deltas[prev_resid],
        )
        RR_effect, _ = N(prev_resid, resid)

        if layer > 0: 
            edges[f'resid_{layer-1}'][f'mlp_{layer}'] = RM_effect
            edges[f'resid_{layer-1}'][f'attn_{layer}'] = RA_effect
            edges[f'resid_{layer-1}'][f'resid_{layer}'] = RR_effect - RMR_effect - RAR_effect
        else:
            edges['embed'][f'mlp_{layer}'] = RM_effect
            edges['embed'][f'attn_{layer}'] = RA_effect
            edges['embed'][f'resid_0'] = RR_effect - RMR_effect - RAR_effect

    # rearrange weight matrices
    t.cuda.empty_cache()
    for child in edges:
        # get shape for child
        bc, sc, fc = nodes[child].act.shape
        for parent in edges[child]:
            weight_matrix = edges[child][parent]
            if parent == 'y':
                weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
            else:
                bp, sp, fp = nodes[parent].act.shape
                assert bp == bc
                weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
            edges[child][parent] = weight_matrix
    t.cuda.empty_cache()
    if aggregation == 'sum':
        # aggregate across sequence position
        for child in edges:
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=1)
                else:
                    weight_matrix = weight_matrix.sum(dim=(1, 4))
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].sum(dim=1)

        # aggregate across batch dimension
        for child in edges:
            bc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = weight_matrix.sum(dim=(0,2)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != 'y':
                nodes[node] = nodes[node].mean(dim=0)
    
    elif aggregation == 'none':

        # aggregate across batch dimensions
        for child in edges:
            # get shape for child
            bc, sc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == 'y':
                    weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc+1))
                    weight_matrix = weight_matrix.sum(dim=0) / bc
                else:
                    bp, sp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = sparse_reshape(weight_matrix, (bp, sp, fp+1, bc, sc, fc+1))
                    weight_matrix = weight_matrix.sum(dim=(0, 3)) / bc
                edges[child][parent] = weight_matrix
        for node in nodes:
            nodes[node] = nodes[node].mean(dim=0)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    t.cuda.empty_cache()
    return nodes, edges


def get_circuit_cluster(dataset,
                        model_name="EleutherAI/pythia-70m-deduped",
                        d_model=512,
                        dict_id=10,
                        dict_size=32768,
                        max_length=64,
                        max_examples=100,
                        batch_size=2,
                        node_threshold=0.1,
                        edge_threshold=0.01,
                        device="cuda:0",
                        dict_path="dictionaries/pythia-70m-deduped/",
                        dataset_name="cluster_circuit",
                        circuit_dir="circuits/",
                        plot_dir="circuits/figures/",
                        model=None,
                        dictionaries=None,):
    
    model = LanguageModel(model_name, device_map=device, dispatch=True)

    embed = model.gpt_neox.embed_in
    attns = [layer.attention for layer in model.gpt_neox.layers]
    mlps = [layer.mlp for layer in model.gpt_neox.layers]
    resids = [layer for layer in model.gpt_neox.layers]
    dictionaries = {}
    dictionaries[embed] = AutoEncoder.from_pretrained(
        os.path.join(dict_path, f'embed/{dict_id}_{dict_size}/ae.pt'),
        device=device
    )
    for i in range(len(model.gpt_neox.layers)):
        dictionaries[attns[i]] = AutoEncoder.from_pretrained(
            os.path.join(dict_path, f'attn_out_layer{i}/{dict_id}_{dict_size}/ae.pt'),
            device=device
        )
        dictionaries[mlps[i]] = AutoEncoder.from_pretrained(
            os.path.join(dict_path, f'mlp_out_layer{i}/{dict_id}_{dict_size}/ae.pt'),
            device=device
        )
        dictionaries[resids[i]] = AutoEncoder.from_pretrained(
            os.path.join(dict_path, f'resid_out_layer{i}/{dict_id}_{dict_size}/ae.pt'),
            device=device
        )

    examples = load_examples_nopair(dataset, max_examples, model, length=max_length)

    num_examples = min(len(examples), max_examples)
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch*batch_size:(batch+1)*batch_size] for batch in range(n_batches)
    ]
    if num_examples < max_examples: # warn the user
        print(f"Total number of examples is less than {max_examples}. Using {num_examples} examples instead.")

    running_nodes = None
    running_edges = None

    for batch in tqdm(batches, desc="Batches"):
        clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(device)
        clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch], dtype=t.long, device=device)

        patch_inputs = None
        def metric_fn(model):
            return (
                -1 * t.gather(
                    t.nn.functional.log_softmax(model.embed_out.output[:,-1,:], dim=-1), dim=-1, index=clean_answer_idxs.view(-1, 1)
                ).squeeze(-1)
            )
        
        nodes, edges = get_circuit(
            clean_inputs,
            patch_inputs,
            model,
            embed,
            attns,
            mlps,
            resids,
            dictionaries,
            metric_fn,
            aggregation="sum",
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
        )

        if running_nodes is None:
            running_nodes = {k : len(batch) * nodes[k].to('cpu') for k in nodes.keys() if k != 'y'}
            running_edges = { k : { kk : len(batch) * edges[k][kk].to('cpu') for kk in edges[k].keys() } for k in edges.keys()}
        else:
            for k in nodes.keys():
                if k != 'y':
                    running_nodes[k] += len(batch) * nodes[k].to('cpu')
            for k in edges.keys():
                for v in edges[k].keys():
                    running_edges[k][v] += len(batch) * edges[k][v].to('cpu')
        
        # memory cleanup
        del nodes, edges
        gc.collect()

    nodes = {k : v.to(device) / num_examples for k, v in running_nodes.items()}
    edges = {k : {kk : 1/num_examples * v.to(device) for kk, v in running_edges[k].items()} for k in running_edges.keys()}

    save_dict = {
        "examples" : examples,
        "nodes": nodes,
        "edges": edges
    }
    save_basename = f"{dataset_name}_dict{dict_id}_node{node_threshold}_edge{edge_threshold}_n{num_examples}_aggsum"
    with open(f'{circuit_dir}/{save_basename}.pt', 'wb') as outfile:
        t.save(save_dict, outfile)

    nodes = save_dict['nodes']
    edges = save_dict['edges']

    # feature annotations
    try:
        annotations = {}
        with open(f'annotations/{dict_id}_{dict_size}.jsonl', 'r') as f:
            for line in f:
                line = json.loads(line)
                annotations[line['Name']] = line['Annotation']
    except:
        annotations = None

    plot_circuit(
        nodes, 
        edges, 
        layers=len(model.gpt_neox.layers), 
        node_threshold=node_threshold, 
        edge_threshold=edge_threshold, 
        pen_thickness=1, 
        annotations=annotations, 
        save_dir=os.path.join(plot_dir, save_basename))




import os
import math

def initialize_model_and_dictionaries(
    device: str,
    model_name: str,
    dict_id: str | int,
    d_model: int,
    dict_path: str,
    dataset: str,
    num_examples: int,
    example_length: int,
    batch_size: int,
    aggregation: str,
    nopair: bool,
):
    """
    Initializes a transformer model, associated dictionaries, and data batches.

    Args:
        device: The device to run on (e.g., "cuda" or "cpu").
        model_name: The name of the pre-trained transformer model.
        dict_id: The type of dictionary ('id' for identity, otherwise uses DictionarySAE).
        d_model: The dimensionality of the model's embeddings.
        dict_path: Path to pre-trained dictionaries if dict_id is not 'id'.
        dataset: The name of the dataset to load examples from.
        num_examples: The number of examples to load.
        example_length: The length of each example sequence.
        batch_size: The batch size for processing examples.
        aggregation: The type of example loading ('sum' uses load_examples, otherwise another method).
        nopair: Whether to use load_examples_nopair.

    Returns:
        A tuple containing variables mirroring the original code's state:
        - device: The device being used.
        - model: The initialized Transformer model.
        - embed: The model's embedding layer.
        - attns: A list of attention layers.
        - mlps: A list of MLP layers.
        - resids: A list of residual connections.
        - dictionaries: A dictionary mapping layers to their corresponding dictionaries.
        - save_basename: The base name for saving files (derived from the dataset).
        - examples: The loaded data examples.
        - batch_size: The batch size used.
        - num_examples: The actual number of examples used.
        - n_batches: The number of batches.
        - batches: A list of data batches.

    """

    model = Transformer.from_pretrained(model_name, device=device)
    model = Sight(model)
    embed = model._envoy.transformer.wte
    attns = [layer.attn for layer in model._envoy.transformer.h]
    mlps = [layer.mlp for layer in model._envoy.transformer.h]
    resids = [layer for layer in model._envoy.transformer.h]

    dictionaries = {}
    if dict_id == 'id':
        dictionaries[embed] = IdentityDict(d_model)
        for i in range(len(model._envoy.transformer.h)):
            dictionaries[attns[i]] = IdentityDict(d_model)
            dictionaries[mlps[i]] = IdentityDict(d_model)
            dictionaries[resids[i]] = IdentityDict(d_model)
    else:
        dictionaries[embed] = DictionarySAE.from_pretrained(
            repo_id_or_model=dict_path, point=('resid-pre', 0), expansion=8, k=5
        ).to(device=device)
        for i in range(len(model._envoy.transformer.h)):
            dictionaries[attns[i]] = DictionarySAE.from_pretrained(
                repo_id_or_model=dict_path, point=('attn-out', i), expansion=8, k=30
            ).to(device=device)
            dictionaries[mlps[i]] = DictionarySAE.from_pretrained(
                repo_id_or_model=dict_path, point=('mlp-out', i), expansion=8, k=30
            ).to(device=device)
            dictionaries[resids[i]] = DictionarySAE.from_pretrained(
                repo_id_or_model=dict_path, point=('resid-post', i), expansion=8, k=30
            ).to(device=device)

    if nopair:
        save_basename = os.path.splitext(os.path.basename(dataset))[0]
        examples = load_examples_nopair(dataset, num_examples, model, length=example_length)
    else:
        data_path = f"data/{dataset}.json"
        save_basename = dataset
        if aggregation == "sum":
            examples = load_examples(data_path, num_examples, model, length=example_length)
        else:
            examples = load_examples(data_path, num_examples, model, length=example_length)

    num_examples = min([num_examples, len(examples)])
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch * batch_size : (batch + 1) * batch_size]
        for batch in range(n_batches)
    ]

    return device, model, embed, attns, mlps, resids, dictionaries, save_basename, examples, batch_size, num_examples, n_batches, batches

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='simple_train',
                        help="A subject-verb agreement dataset in data/, or a path to a cluster .json.")
    parser.add_argument('--num_examples', '-n', type=int, default=20,
                        help="The number of examples from the --dataset over which to average indirect effects.")
    parser.add_argument('--example_length', '-l', type=int, default=None,
                        help="The max length (if using sum aggregation) or exact length (if not aggregating) of examples.")
    parser.add_argument('--model', type=str, default="tdooms/fw-nano",
                        help="The Huggingface ID of the model you wish to test.")
    parser.add_argument("--dict_path", type=str, default='tdooms/fw-nano-scope',
                        help="Path to all dictionaries for your language model.")
    parser.add_argument('--d_model', type=int, default=1024,
                        help="Hidden size of the language model.")
    parser.add_argument('--dict_id', type=str, default=10,
                        help="ID of the dictionaries. Use `id` to obtain circuits on neurons/heads directly.")
    parser.add_argument('--dict_size', type=int, default=32768,
                        help="The width of the dictionary encoder.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Number of examples to process at once when running circuit discovery.")
    parser.add_argument('--aggregation', type=str, default='sum',
                        help="Aggregation across token positions. Should be one of `sum` or `none`.")
    parser.add_argument('--node_threshold', type=float, default=0.2,
                        help="Indirect effect threshold for keeping circuit nodes.")
    parser.add_argument('--edge_threshold', type=float, default=0.02,
                        help="Indirect effect threshold for keeping edges.")
    parser.add_argument('--pen_thickness', type=float, default=1,
                        help="Scales the width of the edges in the circuit plot.")
    parser.add_argument('--nopair', default=False, action="store_true",
                        help="Use if your data does not contain contrastive (minimal) pairs.")
    parser.add_argument('--plot_circuit', default=False, action='store_true',
                        help="Plot the circuit after discovering it.")
    parser.add_argument('--nodes_only', default=False, action='store_true',
                        help="Only search for causally implicated features; do not draw edges.")
    parser.add_argument('--plot_only', action="store_true",
                        help="Do not run circuit discovery; just plot an existing circuit.")
    parser.add_argument("--circuit_dir", type=str, default="circuits/",
                        help="Directory to save/load circuits.")
    parser.add_argument("--plot_dir", type=str, default="circuits/figures/",
                        help="Directory to save figures.")
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()


    device, model, embed, attns, mlps, resids, dictionaries, save_basename, examples, batch_size, num_examples, n_batches, batches = initialize_model_and_dictionaries(
    device=args.device,
    model_name=args.model,
    dict_id=args.dict_id,
    d_model=args.d_model,
    dict_path=args.dict_path,
    dataset=args.dataset,
    num_examples=args.num_examples,
    example_length=args.example_length,
    batch_size=args.batch_size,
    aggregation=args.aggregation,
    nopair=args.nopair,
)
    
    if num_examples < args.num_examples: # warn the user
        print(f"Total number of examples is less than {args.num_examples}. Using {num_examples} examples instead.")

    if not args.plot_only:
        running_nodes = None
        running_edges = None

        for batch in tqdm(batches, desc="Batches"):
                
            clean_inputs = t.cat([e['clean_prefix'] for e in batch], dim=0).to(device)
            clean_answer_idxs = t.tensor([e['clean_answer'] for e in batch], dtype=t.long, device=device)

            if args.nopair:
                patch_inputs = None
                def metric_fn(model):
                    return (
                        -1 * t.gather(
                            t.nn.functional.log_softmax(model.lm_head.output[:,-1,:], dim=-1), dim=-1, index=clean_answer_idxs.view(-1, 1)
                        ).squeeze(-1)
                    )
            else:
                patch_inputs = t.cat([e['patch_prefix'] for e in batch], dim=0).to(device)
                patch_answer_idxs = t.tensor([e['patch_answer'] for e in batch], dtype=t.long, device=device)
                def metric_fn(model):
                    return (
                        t.gather(model.lm_head.output[:,-1,:], dim=-1, index=patch_answer_idxs.view(-1, 1)).squeeze(-1) - \
                        t.gather(model.lm_head.output[:,-1,:], dim=-1, index=clean_answer_idxs.view(-1, 1)).squeeze(-1)
                    )
            
            nodes, edges = get_circuit(
                clean_inputs,
                patch_inputs,
                model,
                embed,
                attns,
                mlps,
                resids,
                dictionaries,
                metric_fn,
                nodes_only=args.nodes_only,
                aggregation=args.aggregation,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
            )

            if running_nodes is None:
                running_nodes = {k : len(batch) * nodes[k].to('cpu') for k in nodes.keys() if k != 'y'}
                if not args.nodes_only: running_edges = { k : { kk : len(batch) * edges[k][kk].to('cpu') for kk in edges[k].keys() } for k in edges.keys()}
            else:
                for k in nodes.keys():
                    if k != 'y':
                        running_nodes[k] += len(batch) * nodes[k].to('cpu')
                if not args.nodes_only:
                    for k in edges.keys():
                        for v in edges[k].keys():
                            running_edges[k][v] += len(batch) * edges[k][v].to('cpu')
            
            # memory cleanup
            del nodes, edges
            gc.collect()

        nodes = {k : v.to(device) / num_examples for k, v in running_nodes.items()}
        if not args.nodes_only: 
            edges = {k : {kk : 1/num_examples * v.to(device) for kk, v in running_edges[k].items()} for k in running_edges.keys()}
        else: edges = None

        save_dict = {
            "examples" : examples,
            "nodes": nodes,
            "edges": edges
        }
        with open(f'{args.circuit_dir}/{save_basename}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}.pt', 'wb') as outfile:
            t.save(save_dict, outfile)

    else:
        with open(f'{args.circuit_dir}/{save_basename}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}.pt', 'rb') as infile:
            save_dict = t.load(infile)
        nodes = save_dict['nodes']
        edges = save_dict['edges']

    # feature annotations
    try:
        annotations = {}
        with open(f"annotations/{args.dict_id}_{args.dict_size}.jsonl", 'r') as annotations_data:
            for annotation_line in annotations_data:
                annotation = json.loads(annotation_line)
                annotations[annotation["Name"]] = annotation["Annotation"]
    except:
        annotations = None

    if args.aggregation == "none":
        example = model.tokenizer.batch_decode(examples[0]["clean_prefix"])[0]
        plot_circuit_posaligned(
            nodes, 
            edges,
            layers=len(model._envoy.transformer.h), 
            length=args.example_length,
            example_text=example,
            node_threshold=args.node_threshold, 
            edge_threshold=args.edge_threshold, 
            pen_thickness=args.pen_thickness, 
            annotations=annotations, 
            save_dir=f'{args.plot_dir}/{save_basename}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}'
        )
    else:
        plot_circuit(
            nodes, 
            edges, 
            layers=len(model._envoy.transformer.h), 
            node_threshold=args.node_threshold, 
            edge_threshold=args.edge_threshold, 
            pen_thickness=args.pen_thickness, 
            annotations=annotations, 
            save_dir=f'{args.plot_dir}/{save_basename}_dict{args.dict_id}_node{args.node_threshold}_edge{args.edge_threshold}_n{num_examples}_agg{args.aggregation}'
        )