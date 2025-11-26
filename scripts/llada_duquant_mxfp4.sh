#!/usr/bin/env bash

export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

DIRPATH="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODEL_PATH='GSAI-ML/LLaDA-8B-Base'
W_BIT=4
A_BIT=4
MX_BLOCK=32

# Generate activation statistics required by DuQuant smoothing
python "$DIRPATH/DuQuant/generate_act_scale_shift.py" --model "$MODEL_PATH"

# DuQuant rotation + MXFP4 quantization (general eval)
python "$DIRPATH/DuQuant/main_duquant_mxfp4.py" \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits "$W_BIT" \
    --abits "$A_BIT" \
    --model "$MODEL_PATH" \
    --quant_method mxfp4 \
    --mxfp4_block_size "$MX_BLOCK" \
    --alpha 0.5 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --tasks piqa

# MMLU
python "$DIRPATH/DuQuant/main_duquant_mxfp4.py" \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits "$W_BIT" \
    --abits "$A_BIT" \
    --model "$MODEL_PATH" \
    --quant_method mxfp4 \
    --mxfp4_block_size "$MX_BLOCK" \
    --alpha 0.5 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --task mmlu --num_fewshot 5 --mc_num 1

# GSM8K
python "$DIRPATH/DuQuant/main_duquant_mxfp4.py" \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits "$W_BIT" \
    --abits "$A_BIT" \
    --model "$MODEL_PATH" \
    --quant_method mxfp4 \
    --mxfp4_block_size "$MX_BLOCK" \
    --alpha 0.5 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --task gsm8k --gen_length 256 --steps 256 --block_length 32 --num_fewshot 4

# MATH
python "$DIRPATH/DuQuant/main_duquant_mxfp4.py" \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits "$W_BIT" \
    --abits "$A_BIT" \
    --model "$MODEL_PATH" \
    --quant_method mxfp4 \
    --mxfp4_block_size "$MX_BLOCK" \
    --alpha 0.5 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --tasks minerva_math --num_fewshot 0 --gen_length 256 --steps 256 --block_length 64

# HumanEval
python "$DIRPATH/DuQuant/main_duquant_mxfp4.py" \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits "$W_BIT" \
    --abits "$A_BIT" \
    --model "$MODEL_PATH" \
    --quant_method mxfp4 \
    --mxfp4_block_size "$MX_BLOCK" \
    --alpha 0.5 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --task humaneval --gen_length 512 --steps 512 --block_length 32 --num_fewshot 0

# MBPP
python "$DIRPATH/DuQuant/main_duquant_mxfp4.py" \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits "$W_BIT" \
    --abits "$A_BIT" \
    --model "$MODEL_PATH" \
    --quant_method mxfp4 \
    --mxfp4_block_size "$MX_BLOCK" \
    --alpha 0.5 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --task mbpp --gen_length 512 --steps 512 --block_length 32 --num_fewshot 3

