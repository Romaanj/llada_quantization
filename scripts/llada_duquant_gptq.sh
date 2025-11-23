
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

DIRPATH="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODEL_PATH='GSAI-ML/LLaDA-8B-Base' 
W_BIT=4
A_BIT=4

# model_path: the path to the pretrained model
# This script applies DuQuant rotation + permutation first, then GPTQ for weight quantization
# Activation quantization still uses DuQuant (with rotation applied)

python $DIRPATH/DuQuant/generate_act_scale_shift.py --model $MODEL_PATH

# Step 1: Apply DuQuant rotation and permutation (without weight quantization)
# Step 2: Apply GPTQ for weight quantization
# general tasks
python $DIRPATH/DuQuant/main_duquant_gptq.py \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits $W_BIT \
    --abits $A_BIT \
    --model $MODEL_PATH \
    --quant_method duquant \
    --alpha 0.5 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --use_gptq \
    --tasks piqa \


# MMLU
python $DIRPATH/DuQuant/main_duquant_gptq.py \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits $W_BIT \
    --abits $A_BIT \
    --model $MODEL_PATH \
    --quant_method duquant \
    --alpha 0.5 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --use_gptq \
    --task mmlu --num_fewshot 5 --mc_num 1 \


# GSM8K
python $DIRPATH/DuQuant/main_duquant_gptq.py \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits $W_BIT \
    --abits $A_BIT \
    --model $MODEL_PATH \
    --quant_method duquant \
    --alpha 0.5 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --use_gptq \
    --task gsm8k --gen_length 256 --steps 256 --block_length 32 --num_fewshot 4 \


# MATH
python $DIRPATH/DuQuant/main_duquant_gptq.py \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits $W_BIT \
    --abits $A_BIT \
    --model $MODEL_PATH \
    --quant_method duquant \
    --alpha 0.5 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --use_gptq \
    --tasks minerva_math  --num_fewshot 0 --gen_length 256 --steps 256 --block_length 64 \


# HumanEval
python $DIRPATH/DuQuant/main_duquant_gptq.py \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits $W_BIT \
    --abits $A_BIT \
    --model $MODEL_PATH \
    --quant_method duquant \
    --alpha 0.5 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --use_gptq \
    --task humaneval --gen_length 512 --steps 512 --block_length 32 --num_fewshot 0 \


# MBPP
python $DIRPATH/DuQuant/main_duquant_gptq.py \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits $W_BIT \
    --abits $A_BIT \
    --model $MODEL_PATH \
    --quant_method duquant \
    --alpha 0.5 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --use_gptq \
    --task mbpp --gen_length 512 --steps 512 --block_length 32 --num_fewshot 3 \
