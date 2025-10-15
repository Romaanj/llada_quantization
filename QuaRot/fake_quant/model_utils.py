import torch
import typing
import transformers
import utils
import os
import logging

OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer
LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer


# type LLaDAModelLM(
#   (model): LLaDAModel(
#     (transformer): ModuleDict(
#       (wte): Embedding(126464, 4096)
#       (emb_drop): Dropout(p=0.0, inplace=False)
#       (ln_f): RMSLayerNorm()
#       (blocks): ModuleList(
#         (0-31): 32 x LLaDALlamaBlock(
#           (dropout): Dropout(p=0.0, inplace=False)
#           (act): SiLU()
#           (attn_out): Linear(in_features=4096, out_features=4096, bias=False)
#           (ff_out): Linear(in_features=12288, out_features=4096, bias=False)
#           (rotary_emb): RotaryEmbedding()
#           (attn_norm): RMSLayerNorm()
#           (ff_norm): RMSLayerNorm()
#           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (ff_proj): Linear(in_features=4096, out_features=12288, bias=False)
#           (up_proj): Linear(in_features=4096, out_features=12288, bias=False)
#         )
#       )
#       (ff_out): Linear(in_features=4096, out_features=126464, bias=False)
#     )
#   )
# )

# mapping of the modules in LLaDA model to the original LLaMA model:
# q_proj->q_proj
# k_proj->k_proj
# v_proj->v_proj
# attn_out->o_proj
# up_proj->up_proj
# ff_proj->gate_proj
# ff_out->down_proj

def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    elif 'llada' in model.__class__.__name__.lower():
        return model.__class__

    elif 'dream' in model.__class__.__name__.lower():
        return model.__class__
    else:
        raise ValueError(f'Unknown model type {model}')

def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass

def get_rope_function_name(model):
    if isinstance(model, LLAMA_MODEL) or 'dream' in model.__class__.__name__.lower():
        return "apply_rotary_pos_emb"
    # elif 'llada' in model.__class__.__name__.lower():
    #     return "apply_rotary_pos_emb"
    raise NotImplementedError


def get_layers(model):
    if isinstance(model, OPT_MODEL):
        return model.model.decoder.layers
    if isinstance(model, LLAMA_MODEL) or 'dream' in model.__class__.__name__.lower():
        return model.model.layers
    if 'llada' in model.__class__.__name__.lower():
        return model.model.transformer.blocks
    raise NotImplementedError


def get_llama(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model

def get_dream(model_name):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.AutoModel.from_pretrained(model_name, torch_dtype='auto',
                                                          low_cpu_mem_usage=True, trust_remote_code=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_opt(model_name):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.OPTForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                        low_cpu_mem_usage=True)
    model.seqlen = model.config.max_position_embeddings
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model

def get_llada(model_name):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto',   trust_remote_code=True, low_cpu_mem_usage=True)
    logging.info('---> Loading {} Model'.format(model_name))
    model.seqlen = 4096
    return model

# monkey patching the forward method of the RoPE module to add quantization in dream model
@torch.no_grad()
def dream_rope_forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # import ipdb; ipdb.set_trace()
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def money_patch(model):
    rotary_emb_cls = model.model.layers[0].self_attn.rotary_emb.__class__
    rotary_emb_cls.forward = dream_rope_forward

def get_model(
    model_name, hf_token=None
):
    if 'llama' in model_name:
        return get_llama(model_name, hf_token)
    elif 'opt' in model_name:
        return get_opt(model_name)
    elif 'llada' in model_name.lower():
        return get_llada(model_name)
    elif 'dream' in model_name.lower():
        model = get_dream(model_name)
        money_patch(model)
        return model
                
    else:
        raise ValueError(f'Unknown model {model_name}')


def get_model_type(model):
    if isinstance(model, OPT_MODEL):
        model_type = OPT_MODEL
    elif isinstance(model, LLAMA_MODEL) :
        model_type = LLAMA_MODEL
    elif 'llada' in model.__class__.__name__.lower():
        model_type = model.__class__
    elif 'dream' in model.__class__.__name__.lower():
        model_type = model.__class__
    else:
        raise ValueError(f'Unknown model type {model}')
    return model_type

def get_embeddings(model, model_type) -> list[torch.nn.Module]:
    if model_type == LLAMA_MODEL or 'dream' in model.__class__.__name__.lower():
        return [model.model.embed_tokens]
    elif model_type == OPT_MODEL:
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    elif 'llada' in model_type.__name__.lower():
        return [model.model.transformer.wte]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_transformer_layers(model, model_type):
    if model_type == LLAMA_MODEL or 'dream' in model.__class__.__name__.lower():
        return [layer for layer in model.model.layers]
    elif model_type == OPT_MODEL:
        return [layer for layer in model.model.decoder.layers]
    elif 'llada' in model_type.__name__.lower():
        return [layer for layer in model.model.transformer.blocks]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_lm_head(model, model_type):
    if model_type == LLAMA_MODEL or 'dream' in model.__class__.__name__.lower():
        return model.lm_head
    elif model_type == OPT_MODEL:
        return model.lm_head
    elif 'llada' in model_type.__name__.lower():
        return model.model.transformer.ff_out
    else:
        raise ValueError(f'Unknown model type {model_type}')

def get_pre_head_layernorm(model, model_type):
    if model_type == LLAMA_MODEL or 'dream' in model.__class__.__name__.lower():
        pre_head_layernorm = model.model.norm
        # assert isinstance(pre_head_layernorm,
        #                   transformers.models.llama.modeling_llama.LlamaRMSNorm)
    elif model_type == OPT_MODEL:
        pre_head_layernorm = model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
    elif 'llada' in model_type.__name__.lower():
        pre_head_layernorm = model.model.transformer.ln_f

    else:
        raise ValueError(f'Unknown model type {model_type}')
    return pre_head_layernorm

def get_mlp_bottleneck_size(model):
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL or 'dream' in model.__class__.__name__.lower():
        return model.config.intermediate_size
    elif model_type == OPT_MODEL:
        return model.config.ffn_dim
    elif 'llada' in model_type.__name__.lower():
        return model.config.mlp_hidden_size
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
def get_hidden_size(model):
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL or 'dream' in model.__class__.__name__.lower():
        return model.config.hidden_size
    elif model_type == OPT_MODEL:
        return model.config.hidden_size
    elif 'llada' in model_type.__name__.lower():
        return model.config.d_model
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
def get_num_attention_heads(model):
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL or 'dream' in model.__class__.__name__.lower():
        return model.config.num_attention_heads
    elif model_type == OPT_MODEL:
        return model.config.num_attention_heads
    elif 'llada' in model_type.__name__.lower():
        return model.config.n_heads
    else:
        raise ValueError(f'Unknown model type {model_type}')

def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


def get_layer_io_save_path(args):
    return os.path.join(args.save_path, 'layer_io', f'{args.layer_idx:03d}.pt')

def capture_layer_io(model_type, layer, layer_input):
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                captured_vals[module_name].append(output.detach().cpu())
        return hook

    handles = []

    if model_type == LLAMA_MODEL or 'dream' in model.__class__.__name__.lower():
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'o_proj': [],
            'gate_proj': [],  # up_proj has the same input as gate_proj
            'down_proj': []
        }

        captured_outputs = {
            'v_proj': [],
        }

        for name in captured_inputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))

    elif model_type == OPT_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'out_proj': [],
            'fc1': [],
            'fc2': []
        }
        captured_outputs = {
            'v_proj': [],
        }
        for name in captured_inputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))
    elif 'llada' in model_type.__name__.lower():
        
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'attn_out': [],
            'ff_proj': [],  # up_proj has the same input as gate_proj
            'ff_out': []
        }

        captured_outputs = {
            'v_proj': [],
        }

        for name in captured_inputs.keys():
            module = getattr(layer)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))
    else:
        raise ValueError(f'Unknown model type {model_type}')

    # Process each sequence in the batch one by one to avoid OOM.
    for seq_idx in range(layer_input.shape[0]):
        # Extract the current sequence across all dimensions.
        seq = layer_input[seq_idx:seq_idx + 1].to(utils.DEV)
        # Perform a forward pass for the current sequence.
        layer(seq)

    # After processing all sequences, concatenate the accumulated inputs for each sub-layer across the batch.
    for module_name in captured_inputs:
        captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in captured_outputs:
        captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    # Cleanup.
    for h in handles:
        h.remove()

    return {
        'input': captured_inputs,
        'output': captured_outputs
    }
