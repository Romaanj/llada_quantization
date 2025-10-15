import utils
import torch
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import eval_utils
import hadamard_utils

def test_output(lm, args):
    with torch.cuda.amp.autocast():
        prompt = "Recent advances in neural language modeling have highlighted the importance of scaling laws. In this work, we present a new theoretical framework that—"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # m = [{"role": "user", "content": prompt}, ]
        # prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).cuda().unsqueeze(0)

        out = lm.generate(input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
        print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])

def main():
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)
        
    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    
    
    # Rotate the weights
    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)
            
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)

        # import ipdb; ipdb.set_trace()
        for name in qlayers:
            if 'down_proj' in name or ('ff_out' in name and 'block' in name):
                had_K, K = hadamard_utils.get_hadK(model_utils.get_mlp_bottleneck_size(model))
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'o_proj' in name or 'attn_out' in name:
                num_attention_heads = model_utils.get_num_attention_heads(model)
                hidden_size = model_utils.get_hidden_size(model)
                had_K, K = hadamard_utils.get_hadK(num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = hidden_size//num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present
        
    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path: # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path, weights_only=False)
            model.load_state_dict(save_dict["model"])
            
        elif not args.w_rtn: # GPTQ Weight Quantization
            # assert "llama" in args.model, "Only llama is supported for GPTQ!"
            
            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
            
        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)


    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
        
        for name in qlayers:            
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not(args.a_asym)
            layer_a_clip = args.a_clip_ratio
            
            if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                              groupsize=args.v_groupsize,
                                              sym=not(args.v_asym),
                                              clip_ratio=args.v_clip_ratio)
            
            if 'lm_head' in name or ('ff_out' in name and 'block' not in name): #Skip lm_head quantization   
                layer_input_bits = 16
            
            if 'down_proj' in name or ('ff_out' in name and 'block' in name): #Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

                
            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip)

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            layers = model_utils.get_layers(model)
            k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                          "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
            for layer in layers:
                if 'llada' in model.__class__.__name__.lower():
                    rotation_utils.add_qk_rotation_wrapper_after_function_call_in_submodule(
                            layer.rotary_emb, 
                            'forward',
                            config=model.config,
                            **k_quant_config)
                else:
                    rope_function_name = model_utils.get_rope_function_name(model)
                    rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                                layer.self_attn, 
                                rope_function_name, 
                                config=model.config,
                                **k_quant_config)
        
    # Evaluating on dataset
    testloader = data_utils.get_loaders(
            args.eval_dataset,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True
        )

    model.eval()
    # dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
    # if args.wandb:
    #         wandb.log({'ppl/{}'.format(args.eval_dataset.upper()): dataset_ppl})

    # if not args.lm_eval:
    #     return
    # else:
        # Import lm_eval utils
    import lm_eval
    # from lm_eval import utils as lm_eval_utils
    # from lm_eval.api.registry import ALL_TASKS
    # from lm_eval.models.huggingface import HFLM

        
    
    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token, trust_remote_code=True)
    # hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

    from lm_eval.api.registry import get_model
    if 'llada' in args.model.lower():
        model_cls = get_model('llada_dist')

        model_args = dict(
            steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, temperature=0., cfg_scale=0., remasking='low_confidence', mc_num=args.mc_num, batch_size=args.batch_size
        )

        model = model_cls(model=model, model_path=args.model, **model_args)
    else:
        model_cls = get_model('dream_base')
        model_args = dict(
            diffusion_steps=args.diffusion_steps, max_new_tokens=args.max_new_tokens, mc_num=args.mc_num, batch_size=args.batch_size
        )
        model = model_cls(model=model, pretrained=args.model_path, **model_args)

    test_output(model, args)

    # task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
    # results = lm_eval.simple_evaluate(hflm, tasks=task_names, batch_size=args.lm_eval_batch_size)['results']
    from lm_eval import evaluator

    results = {}
    task_names = args.tasks.split(",")
    with torch.cuda.amp.autocast():
        t_results = evaluator.simple_evaluate(
            model,
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
            model_args=model_args,
            confirm_run_unsafe_code=True
        )
    results.update(t_results)
    # print(results.keys())
    print(args)
    print(results['results'])

    # save results
    import json
    import os   
    if not os.path.exists('results'):
        os.makedirs('results')
    with open(f'results/{args.model.split("/")[-1]}-{args.a_bits}a{args.v_bits}v{args.k_bits}k{args.w_bits}w.json', 'a') as f:
        try:
            json.dump(results['results'], f)
        except Exception as e:
            print(f"Error writing results to {f}: {e}")
    
    exit(0)

    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    print(metric_vals)

    if args.wandb:
        wandb.log(metric_vals)


if __name__ == '__main__':
    main()
