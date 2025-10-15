from ._base import BaseGPTQForCausalLM

class LladaGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "LLaDALlamaBlock"
    layers_block_name = "model.transformer.blocks"
    outside_layer_modules = [
        "model.transformer.wte",
        "model.transformer.ln_f",
    ]
    inside_layer_modules = [
        ["attn_out"],
        ["attn_q", "attn_k", "attn_v"],
        ["ff_out"],
        ["up_proj", "ff_proj"],
    ]


__all__ = ["LladaGPTQForCausalLM"]
