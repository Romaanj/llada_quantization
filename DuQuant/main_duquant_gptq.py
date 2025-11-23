import os
import sys
import random
import numpy as np
import torch
import time
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.duquant import duquant
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from quantize.utils import apply_rotation_to_weights_inplace

# Import necessary functions from main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    move_to_device,
    test_output,
    evaluate,
    net_choices,
)

torch.backends.cudnn.benchmark = True


def apply_gptq_to_model(lm, args, dataloader, logger):
    """Apply GPTQ quantization directly to DuQuant-rotated model in-place.
    
    Critical: We use the existing lm.model that already has DuQuant rotation applied.
    This ensures:
    1. Model structure (QuantLinear modules) is preserved
    2. Activation transformations (rotation) are already in place
    3. Weight rotations are already applied
    4. GPTQ calibration uses rotated activations (correct Hessian calculation)
    """
    logger.info("=== Starting GPTQ quantization on DuQuant-rotated model ===")
    
    from auto_gptq.quantization.gptq import GPTQ
    from quantize.int_linear import QuantLinear
    
    model = lm.model
    dev = lm.device
    
    # Helper function to find QuantLinear modules
    def find_quant_linear_modules(module, name=""):
        """Find all QuantLinear modules in the model."""
        res = {}
        if isinstance(module, QuantLinear):
            return {name: module}
        for name1, child in module.named_children():
            child_name = name + "." + name1 if name != "" else name1
            res.update(find_quant_linear_modules(child, child_name))
        return res
    
    # Prepare calibration dataset
    # Note: We'll use the model's forward pass which already applies DuQuant rotation to activations
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()
    
    # Get layers
    layers = None
    class_name = args.model.lower()
    if 'llada' in class_name:
        layers = model.model.transformer.blocks
        # Move embedding layer to device (duquant() moves it to CPU at the end)
        # For llada, embedding is model.model.transformer.wte
        # duquant() sets embed_tokens = wte, so they point to the same module
        embedding_layer = model.model.transformer.wte
        embedding_layer = embedding_layer.to(dev)
        model.model.transformer.wte = embedding_layer
        # Ensure embed_tokens also points to the same module on correct device
        if hasattr(model.model.transformer, 'embed_tokens'):
            model.model.transformer.embed_tokens = embedding_layer
    elif 'dream' in class_name:
        layers = model.model.layers
        # Move embedding layer to device (duquant() moves it to CPU at the end)
        if hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens = model.model.embed_tokens.to(dev)
    else:
        logger.error(f"Unsupported model type: {class_name}")
        return lm
    
    # Prepare initial layer inputs (these will have DuQuant rotation applied during forward pass)
    layer_inputs = []
    attention_masks = []
    position_ids_list = []
    
    logger.info(f"Collecting initial calibration data with DuQuant rotation applied...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.nsamples:
                break
            
            input_ids = batch[0].to(dev)
            
            # Get first layer input by running embedding
            # Each layer's input will have rotation applied (if act_quant is enabled)
            # CRITICAL: Embedding layer must be on the same device as input_ids
            if 'llada' in class_name:
                hidden_states = model.model.transformer.wte(input_ids)
            else:
                hidden_states = model.model.embed_tokens(input_ids)
            
            # Store first layer input
            layer_inputs.append([hidden_states])
            
            # Create attention mask
            attention_mask = torch.ones_like(input_ids)
            attention_masks.append(attention_mask)
            position_ids_list.append(None)  # Can be extended if needed
    
    # Apply GPTQ to each layer sequentially (Layer-wise Quantization)
    logger.info(f"Applying GPTQ quantization to {len(layers)} layers sequentially...")
    
    for i, layer in enumerate(layers):
        logger.info(f"Quantizing layer {i+1}/{len(layers)}...")
        layer = layer.to(dev)
        
        # Find all QuantLinear modules in this layer
        quant_linear_modules = find_quant_linear_modules(layer)
        
        if not quant_linear_modules:
            logger.info(f"  No QuantLinear modules found in layer {i+1}")
            # Still need to propagate layer output even if no quantization
            layer_outputs = []
            for j, layer_inp in enumerate(layer_inputs):
                if j < args.nsamples:
                    layer_inp = [x.to(dev) for x in layer_inp]
                    additional_inputs = {}
                    if attention_masks[j] is not None:
                        additional_inputs["attention_mask"] = attention_masks[j].to(dev)
                    if position_ids_list[j] is not None:
                        additional_inputs["position_ids"] = position_ids_list[j].to(dev)
                    
                    try:
                        output = layer(*layer_inp, **additional_inputs)
                        if isinstance(output, tuple):
                            output = output[0]
                        layer_outputs.append([output.cpu()])
                    except Exception as e:
                        try:
                            output = layer(layer_inp[0])
                            if isinstance(output, tuple):
                                output = output[0]
                            layer_outputs.append([output.cpu()])
                        except:
                            logger.warning(f"  Failed forward pass for sample {j}: {e}")
                            layer_outputs.append(layer_inp)  # Keep original input
            layer_inputs = layer_outputs  # Propagate to next layer
            layer = layer.cpu()
            continue
        
        # Create GPTQ instances for each QuantLinear module
        gptq_instances = {}
        for name, quant_linear_module in quant_linear_modules.items():
            # Get the rotated weight (already rotated by DuQuant)
            linear_weight = quant_linear_module.weight.data.clone()
            
            # Create a temporary nn.Linear for GPTQ (GPTQ expects nn.Linear)
            temp_linear = nn.Linear(
                quant_linear_module.in_features,
                quant_linear_module.out_features,
                bias=quant_linear_module.bias is not None
            ).to(dev)
            temp_linear.weight.data = linear_weight
            if quant_linear_module.bias is not None:
                temp_linear.bias.data = quant_linear_module.bias.data.clone()
            
            # Initialize GPTQ
            gptq = GPTQ(temp_linear)
            gptq.quantizer.configure(
                args.wbits,
                perchannel=True,
                sym=False,
                mse=False,
            )
            gptq_instances[name] = (gptq, temp_linear, quant_linear_module)
        
        # Collect activation statistics using forward hooks
        # Activations will have DuQuant rotation already applied via act_quantizer
        def make_add_batch(name):
            gptq, _, quant_linear_module = gptq_instances[name]
            def add_batch(_, inp, out):
                # inp[0] is the input activation (already rotated by DuQuant act_quantizer)
                # GPTQ's add_batch expects input for nn.Linear:
                #   - 2D: (SeqLen, Hidden) -> adds batch dim -> (1, SeqLen, Hidden)
                #   - 3D: (Batch, SeqLen, Hidden) -> reshape to (Batch*SeqLen, Hidden) -> transpose to (Hidden, Batch*SeqLen)
                #   Then computes H = X @ X.T where X is (Hidden, Batch*SeqLen)
                input_act = inp[0].data
                
                # QuantLinear receives input in various shapes depending on context:
                # - MLP: (Batch, SeqLen, Hidden) - OK, GPTQ handles this
                # - Attention: might be reshaped already
                # - If 4D+, reshape to 2D: (Any, Hidden)
                if len(input_act.shape) == 2:
                    # Already 2D (SeqLen, Hidden) - GPTQ will add batch dim
                    pass
                elif len(input_act.shape) == 3:
                    # 3D (Batch, SeqLen, Hidden) - GPTQ will handle reshape
                    pass
                elif len(input_act.shape) > 3:
                    # 4D+ (e.g., Attention: Batch, NumHeads, SeqLen, HeadDim)
                    # Reshape to (Any, Hidden) - GPTQ will treat as 2D and add batch dim
                    input_act = input_act.reshape(-1, input_act.shape[-1])
                else:
                    # 1D or unexpected shape
                    logger.warning(f"Unexpected input shape {input_act.shape} for {name}, attempting to reshape...")
                    input_act = input_act.reshape(-1, quant_linear_module.in_features)
                
                # GPTQ doesn't use output for Hessian calculation, but we pass it anyway
                # (It's used in DEBUG mode only)
                output_act = out.data if isinstance(out, torch.Tensor) else out
                if len(output_act.shape) > 3:
                    output_act = output_act.reshape(-1, output_act.shape[-1])
                
                # Call GPTQ's add_batch which will:
                # 1. Ensure input is 3D: (Batch, SeqLen, Hidden)
                # 2. Reshape to 2D: (Batch*SeqLen, Hidden)
                # 3. Transpose to: (Hidden, Batch*SeqLen)
                # 4. Compute Hessian: H += X @ X.T where X is (Hidden, Batch*SeqLen)
                gptq.add_batch(input_act, output_act)
            return add_batch
        
        handles = []
        for name in quant_linear_modules.keys():
            handles.append(quant_linear_modules[name].register_forward_hook(make_add_batch(name)))
        
        # Forward pass to collect activations (with DuQuant rotation already applied)
        # IMPORTANT: Store outputs for next layer input propagation
        layer_outputs = []
        for j, layer_inp in enumerate(layer_inputs):
            if j < args.nsamples:
                layer_inp = [x.to(dev) for x in layer_inp]
                additional_inputs = {}
                if attention_masks[j] is not None:
                    additional_inputs["attention_mask"] = attention_masks[j].to(dev)
                if position_ids_list[j] is not None:
                    additional_inputs["position_ids"] = position_ids_list[j].to(dev)
                
                try:
                    output = layer(*layer_inp, **additional_inputs)
                    if isinstance(output, tuple):
                        output = output[0]
                    layer_outputs.append([output.cpu()])  # Store output for next layer
                except Exception as e:
                    try:
                        output = layer(layer_inp[0])
                        if isinstance(output, tuple):
                            output = output[0]
                        layer_outputs.append([output.cpu()])  # Store output for next layer
                    except:
                        logger.warning(f"  Failed forward pass for sample {j}: {e}")
                        layer_outputs.append(layer_inp)  # Keep original input if failed
        
        # Remove hooks
        for h in handles:
            h.remove()
        
        # Apply GPTQ quantization to each QuantLinear module
        for name, (gptq, temp_linear, quant_linear_module) in gptq_instances.items():
            logger.info(f"  Quantizing {name}...")
            try:
                scale, zero, g_idx = gptq.fasterquant(
                    blocksize=128,
                    percdamp=0.01,
                    group_size=128,
                    actorder=False,
                    static_groups=False,
                )
                
                # Replace QuantLinear weight with quantized weight
                quant_linear_module.weight.data = temp_linear.weight.data.clone()
                
                gptq.free()
                del gptq, temp_linear
            except Exception as e:
                logger.error(f"  Failed to quantize {name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # CRITICAL: Propagate layer outputs to next layer inputs
        # This is essential for layer-wise quantization
        layer_inputs = layer_outputs
        
        layer = layer.cpu()
        torch.cuda.empty_cache()
    
    model.config.use_cache = use_cache
    logger.info("GPTQ quantization completed.")
    
    return lm


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="./duquant_gptq_log/", type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument('--test_dataset', type=str, default='wikitext2', help='dataset for testing')
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="", type=str, help="Tasks to evaluate on, split by comma.")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_alpha", type=float, default=0.8)
    parser.add_argument("--act_group_size", type=int, default=None)
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--smooth_lr", type=float, default=1e-4)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--smooth_epochs", type=int, default=0)
    parser.add_argument("--smooth",default=False, action="store_true")
    parser.add_argument("--let",default=False, action="store_true")
    parser.add_argument("--lwc",default=False, action="store_true")
    parser.add_argument("--aug_loss", default=False, action="store_true")
    parser.add_argument("--symmetric",default=False, action="store_true")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--multigpu", action="store_true")
    parser.add_argument("--deactive_amp", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)
    
    # DuQuant arguments
    parser.add_argument("--max_rotation_step", type=int, default=256)
    parser.add_argument("--permutation_times", type=int, default=1)
    parser.add_argument("--lac", type=float, default=None)
    parser.add_argument("--swc", type=float, default=None)
    parser.add_argument("--block_size", type=int, default=128)
    
    # MMLU arguments
    parser.add_argument("--mmlu_data_dir", default="./mmlu/data", type=str)
    parser.add_argument("--eval_mmlu", action="store_true")
    parser.add_argument("--eval_mtbench", action="store_true")
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str)
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--quant_method", type=str, default="duquant", choices=["duquant", "hadamard", None])
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--mc_num", type=int, default=128)
    parser.add_argument("--gen_length", type=int, default=1024)
    parser.add_argument("--block_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--diffusion_steps", type=int, default=512)
    parser.add_argument("--get_wa", action="store_true")
    parser.add_argument("--use_gptq", action="store_true", help="Use GPTQ for weight quantization after DuQuant rotation")
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if args.epochs > 0:
        assert args.lwc or args.let
    
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

    # Initialize logger
    args.output_dir = os.path.join(args.output_dir, f"{args.model.split('/')[-1]}_w{args.wbits}a{args.abits}")
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    
    # Load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    args.model_family = args.net.split('-')[0]
    
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    from lm_eval.api.registry import get_model

    class_name = args.model.lower()

    if 'llada' in class_name:
        model_cls = get_model('llada_dist')
        model_args = dict(
            steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, 
            temperature=0., cfg_scale=0., remasking='low_confidence', mc_num=args.mc_num, 
            batch_size=args.batch_size
        )
        lm = model_cls(model_path=args.model, **model_args)
    elif 'dream' in class_name and 'base' in args.model.lower():
        model_cls = get_model('dream_base')
        model_args = dict(
            diffusion_steps=args.diffusion_steps, max_new_tokens=args.max_new_tokens, 
            mc_num=args.mc_num, batch_size=args.batch_size
        )
        lm = model_cls(pretrained=args.model, **model_args)
    else:
        raise NotImplementedError
        
    args.model_path = args.model

    lm.seqlen = 2048
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    # Set quantization parameters
    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "swc":args.swc,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.act_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "lac":args.lac,
        "act_group_size": args.act_group_size,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,  # Keep DuQuant for activation quantization
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    FILE_PATH = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(FILE_PATH)
    if args.act_scales is None:
        args.act_scales = f'{BASE_DIR}/act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'{BASE_DIR}/act_shifts/{args.net}.pt'
    
    rot_path = f'{BASE_DIR}/Rot.pkl'
    if not os.path.exists(rot_path) and args.quant_method == "duquant":
        import get_rot
        get_rot.main()

    # Step 1: Apply DuQuant rotation (without weight quantization)
    if args.wbits < 16 or args.abits < 16:
        logger.info("=== Step 1: Apply DuQuant rotation and permutation ===")
        tick = time.time()
        
        # Load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(dataloader, cache_dataloader)
        
        act_scales = None
        act_shifts = None
        if args.smooth:
            act_scales = torch.load(args.act_scales)
            act_shifts = torch.load(args.act_shifts)
        
        # Apply DuQuant rotation (weight quantization will be skipped)
        duquant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        logger.info(f"DuQuant rotation completed in {time.time() - tick:.2f} seconds")
        
        # Step 2: Apply GPTQ for weight quantization
        if args.use_gptq and args.wbits < 16:
            logger.info("=== Step 2: Apply GPTQ for weight quantization ===")
            tick = time.time()
            lm = apply_gptq_to_model(lm, args, dataloader, logger)
            logger.info(f"GPTQ quantization completed in {time.time() - tick:.2f} seconds")
    
    move_to_device(lm, args, logger)
    evaluate(lm, args, logger)


if __name__ == "__main__":
    print(sys.argv)
    main()


