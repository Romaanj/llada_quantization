'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
# from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
# from generate import generate

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x




def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        model=None,
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        if model is None or isinstance(model, str):
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs)
        else:
            self.model = model
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif model is None or isinstance(model, str): 
            self.model = self.model.to(device)
        else:
            self.device = next(model.parameters()).device

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking    
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index).to(self.device)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index).to(self.device)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        for elem in tqdm(ds, desc="Generating..."):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]
 
            try:
                generated_answer = generate(self.model, prompt, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                            temperature=0, cfg_scale=self.cfg, remasking=self.remasking, mask_id=self.mask_id)
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"Out of memory error while generating. Please try reducing the batch size or using a smaller model. Batch size: {self.batch_size}")
                generated_answer = torch.tensor([], device=self.device).unsqueeze(0)
                assert self.batch_size == 1, 'Out of memory error should only happen for one request'
            
            generated_answer = self.tokenizer.decode(generated_answer[0][prompt.shape[1]:], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
        return out
    
    def generate(self, *args, **kwargs):
        return generate(self.model, *args, **kwargs)


# if __name__ == "__main__":
#     set_seed(1234)
#     cli_evaluate()
    