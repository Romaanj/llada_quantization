import torch
from torch import nn
from typing import Optional, Tuple, List
from DuQuant.quantize.int_linear import QuantLinear
from DuQuant.quantize.int_matmul import QuantMatMul
import torch.nn.functional as F
from DuQuant.quantize.du_norm import DuLladaRMSNorm
from collections import OrderedDict
import math
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,apply_rotary_pos_emb,LlamaRMSNorm,repeat_kv
from transformers.activations import ACT2FN
import pdb
import copy
from DuQuant.models.configuration_llada import LLaDAConfig
from DuQuant.models.transformation import *
from DuQuant.models.modeling_llada import LLaDABlock
from DuQuant.models.modeling_llada import ModelConfig, LayerNorm, init_weights, BufferCache
from dataclasses import fields

def create_model_config_from_pretrained_config(config: LLaDAConfig):
    """
    Utility function
    """

    kwargs = {}
    for field in fields(ModelConfig):
        kwargs[field.name] = getattr(config, field.name)

    model_config = ModelConfig(**kwargs)
    return model_config

class QuantLLadaDecoderLayer(nn.Module):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `LLaDASequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    # def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
    def __init__(self,
                 config: LLaDAConfig,
                 ori_layer,
                 args):
        super().__init__()
        config = create_model_config_from_pretrained_config(config)
        if not isinstance(config, ModelConfig):
            raise TypeError(
                f"Expected `config` to be an instance of `ModelConfig`, but got {type(config)}."
            )
        self.config = config
        self.ori_layer = ori_layer
        self.args = args
        self.dropout = nn.Dropout(config.residual_dropout)

        self.attn_norm = DuLladaRMSNorm(ori_layer.attn_norm, eps=config.rms_norm_eps)
        self.ff_norm = DuLladaRMSNorm(ori_layer.ff_norm, eps=config.rms_norm_eps)

        self.q_proj = QuantLinear(ori_layer.q_proj,
                                args.q_weight_quant_params,
                                args.q_act_quant_params)
        self.k_proj = QuantLinear(ori_layer.k_proj,
                                args.k_weight_quant_params,
                                args.k_act_quant_params)
        self.v_proj = QuantLinear(ori_layer.v_proj,
                                args.v_weight_quant_params,
                                args.v_act_quant_params)
        self.attn_out = QuantLinear(ori_layer.attn_out,
                                args.o_weight_quant_params,
                                args.o_act_quant_params)
        self.ff_proj = QuantLinear(ori_layer.ff_proj,
                                args.gate_weight_quant_params,
                                args.gate_act_quant_params)
        self.up_proj = QuantLinear(ori_layer.up_proj,
                                args.up_weight_quant_params,
                                args.up_act_quant_params)
        self.ff_out = QuantLinear(ori_layer.ff_out,            
                                args.down_weight_quant_params,
                                args.down_act_quant_params)

        # self.q_proj = copy.deepcopy(ori_layer.q_proj)
        # self.k_proj = copy.deepcopy(ori_layer.k_proj)
        # self.v_proj = copy.deepcopy(ori_layer.v_proj)
        # self.attn_out = copy.deepcopy(ori_layer.attn_out)
        # self.ff_proj = copy.deepcopy(ori_layer.ff_proj)
        # self.up_proj = copy.deepcopy(ori_layer.up_proj)
        # self.ff_out = copy.deepcopy(ori_layer.ff_out)

        
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul, rotate=None
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul, rotate=None
        )

        self.act = copy.deepcopy(ori_layer.act)
        self.ff_out._is_residual = True
        self.rotary_emb = copy.deepcopy(ori_layer.rotary_emb) if hasattr(ori_layer, 'rotary_emb') else None

        self.flash_attn_func = None

        self._activation_checkpoint_fn = copy.deepcopy(ori_layer._activation_checkpoint_fn) if hasattr(ori_layer, '_activation_checkpoint_fn') else None

        if config.flash_attention:
            try:
                from flash_attn import flash_attn_func  # type: ignore

                self.flash_attn_func = flash_attn_func
            except ModuleNotFoundError:
                pass
        
        self.init_duquant_params = torch.tensor(0) if args.gate_weight_quant_params['quant_method'] == 'duquant' else torch.tensor(1)
        
        # super().__init__(layer_id, config, cache)
        # # Layer norms.
        # self.attn_norm = LayerNorm.build(config)
        # self.ff_norm = LayerNorm.build(config)
        # self.__cache = cache

        # # Attention input projection. Projects x -> (q, k, v)
        # head_dim = config.d_model // config.n_heads
        # q_proj_out_dim = config.d_model
        # k_proj_out_dim = config.effective_n_kv_heads * head_dim
        # v_proj_out_dim = config.effective_n_kv_heads * head_dim
        # self.q_proj = nn.Linear(
        #     config.d_model, q_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        # )
        # self.k_proj = nn.Linear(
        #     config.d_model, k_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        # )
        # self.v_proj = nn.Linear(
        #     config.d_model, v_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        # )

        # # Feed-forward input projection.
        # self.ff_proj = nn.Linear(
        #     config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        # )
        # # new add
        # self.up_proj = nn.Linear(
        #     config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        # )

        # self.ff_out = nn.Linear(
        #     int(self.act.output_multiplier * self.hidden_size),
        #     config.d_model,
        #     bias=config.include_bias,
        #     device=config.init_device,
        # )

        # self.attn_out = nn.Linear(
        #     config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        # )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None)  # new add

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs: Optional[dict]
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)
        # import ipdb; ipdb.set_trace()
        x_normed = self.attn_norm(x)
        # import ipdb; ipdb.set_trace()
        q = self.q_proj(x_normed)
        
        if not self.init_duquant_params:
            self.k_proj.copy_quantizers_duquant_params(self.q_proj)
            self.v_proj.copy_quantizers_duquant_params(self.q_proj)
            
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x_ff = self.ff_proj(x)
        if not self.init_duquant_params:
            self.up_proj.copy_quantizers_duquant_params(self.ff_proj)
        x = self.up_proj(x)  # new add
        # x, x_up = self.ff_proj(x), self.up_proj(x) # new add
        if self._activation_checkpoint_fn is not None:
            x_ff = self._activation_checkpoint_fn(self.act, x_ff)  # type: ignore
        else:
            x_ff = self.act(x_ff)
        x = x_ff * x # new add
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x
    
        self.init_duquant_params = torch.tensor(1)

        return x, cache

    
    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.config.attention_layer_norm:
            if self.q_norm is not None and self.k_norm is not None:
                q = self.q_norm(q).to(dtype=dtype)
                k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if self.config.rope:
            # Apply rotary embeddings.
            q, k = self.rotary_emb(q, k)

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if self.flash_attn_func is not None and attn_mask is None:
            assert 0
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=False
            )
            return r.transpose(1, 2)
        else:
            # torch's sdpa doesn't support GQA, so we're doing this
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            # import ipdb; ipdb.set_trace()
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

            # Modify: MDM set causal to False, and with no attn_mask.
            # return F.scaled_dot_product_attention(
            #     q,
            #     k,
            #     v,
            #     attn_mask=None,
            #     dropout_p=dropout_p,
            #     is_causal=False,
            # )

            L, S = q.size(-2), k.size(-2)
            scale_factor = 1 / math.sqrt(q.size(-1)) 
            attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)

            # attn_weight = q @ k.transpose(-2, -1) * scale_factor
            q = self.qkt_matmul.quant_x1(q)
            k = self.qkt_matmul.quant_x2(k).transpose(-2, -1)
            attn_weight = self.qkt_matmul(q, k) * scale_factor

            attn_weight += attn_bias
            attn_weight = torch.softmax(attn_weight, dim=-1)
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

            # return attn_weight @ v
            attn_weight = self.pv_matmul.quant_x1(attn_weight)
            v = self.pv_matmul.quant_x2(v)
            return self.pv_matmul(attn_weight, v)

    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)
      

    def clear_temp_variable(self):
       for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)  

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
        return iter(params)  

    def duquant_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)  
    
    def duquant_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination
    
    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()
    
    def register_duquant_params(self):        
        for name, module in self.named_modules():
            # if isinstance(module, QuantLladaMLP) or isinstance(module, QuantLlamaAttention):
            #     delattr(module, 'init_duquant_params')
            #     module.register_buffer('init_duquant_params', torch.tensor(1))
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_duquant_params()
                module.act_quantizer.register_duquant_params()
    
    def load_duquant_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('R') > -1 or k.find('permutation_list') > -1 or k.find('init_duquant_params') > -1:
                exec(f'self.{k} = v.to(device)')
    
    def load_smooth_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('smooth') > -1:
                # exec(f'self.{k} = v')
                self.register_parameter(k, torch.nn.Parameter(v.to(device), requires_grad=False))
    
    def load_post_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('post') > -1:
                # exec(f'self.{k} = v')
                rg = False if k.find('down') > -1 else True
                self.register_parameter(k, torch.nn.Parameter(v.to(device), requires_grad=rg))

    def load_lwc_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('bound_factor') > -1:
                v = torch.nn.Parameter(v.to(device))
                exec(f'self.{k} = v.to(device)')