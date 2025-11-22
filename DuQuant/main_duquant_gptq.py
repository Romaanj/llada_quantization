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
    """Apply GPTQ quantization to model weights that have been rotated by DuQuant."""
    logger.info("=== Starting GPTQ quantization ===")
    
    # Prepare calibration dataset for GPTQ
    traindataset = []
    for batch in dataloader:
        input_ids = batch[0]
        attention_mask = torch.ones_like(input_ids)
        traindataset.append({"input_ids": input_ids, "attention_mask": attention_mask})
        if len(traindataset) >= args.nsamples:
            break
    
    # Convert model to AutoGPTQ format
    model = lm.model
    
    # Check model type
    class_name = args.model.lower()
    if 'llada' in class_name:
        from auto_gptq.modeling.llada import LladaGPTQForCausalLM
        gptq_model_class = LladaGPTQForCausalLM
    else:
        logger.error(f"Unsupported model type for GPTQ: {class_name}")
        return lm
    
    # Configure GPTQ quantization
    quantize_config = BaseQuantizeConfig(
        bits=args.wbits,
        group_size=128,  # Default group size
        desc_act=False,
        sym=False,
    )
    
    logger.info(f"Creating GPTQ model wrapper with {args.wbits} bits quantization...")
    
    # Convert to GPTQ model format
    # We need to create a GPTQ model from the existing model
    # The model already has DuQuant rotation applied to weights
    try:
        # Get the model's state dict
        model_state_dict = model.state_dict()
        
        # Create GPTQ model and load rotated weights
        gptq_model = gptq_model_class.from_pretrained(
            args.model,
            quantize_config=quantize_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        # Replace model weights with rotated weights from DuQuant
        logger.info("Applying DuQuant rotated weights to GPTQ model...")
        for name, param in model.named_parameters():
            if 'weight' in name and name in gptq_model.model.state_dict():
                gptq_model.model.state_dict()[name].copy_(param.data)
        
        # Apply GPTQ quantization
        logger.info(f"Applying GPTQ quantization with {args.nsamples} calibration samples...")
        gptq_model.quantize(traindataset)
        
        # Update lm.model with quantized model
        lm.model = gptq_model.model
        logger.info("GPTQ quantization completed.")
        
    except Exception as e:
        logger.error(f"Error during GPTQ quantization: {e}")
        import traceback
        traceback.print_exc()
        logger.info("Falling back to DuQuant-only quantization...")
        # If GPTQ fails, we still have the rotated model
        pass
    
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

