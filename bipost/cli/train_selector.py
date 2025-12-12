import argparse
import math
import os
import torch
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime

from transformers.trainer import get_scheduler

from bipost.datasets import RewardDataset, SFTDataset, SFTDatasetIndexed
from bipost.models import Actor, TrainableTensorModule
from bipost.trainer import SelectorTrainer_BDR, SelectorTrainer_BDR_Online
from bipost.utils import blending_datasets, get_strategy, get_tokenizer

def get_dataloaders(strategy, obj_index, tokenizer):
    args = strategy.args
    obj_type = getattr(args, f"obj_{obj_index}")
    if obj_type == "SFT" or obj_type=="KD":
        train_data, eval_data = blending_datasets(
            getattr(args, f"dataset_{obj_index}"),
            getattr(args, f"dataset_probs_{obj_index}"),
            strategy,
            args.seed,
            max_count=getattr(args, f"max_samples_{obj_index}"),
            train_split=getattr(args, f"train_split_{obj_index}"),
            eval_split=getattr(args, f"eval_split_{obj_index}"),
        )
        train_data = train_data.select(range(min(getattr(args, f"max_samples_{obj_index}"), len(train_data))))
        eval_data = eval_data.select(range(min(getattr(args, f"max_samples_{obj_index}"), len(eval_data))))
        train_dataset = SFTDataset(
            train_data,
            tokenizer,
            getattr(args, f"max_len_{obj_index}"),
            strategy,
            input_template=getattr(args, f"input_template_{obj_index}"),
            obj_index=obj_index
        )
        eval_dataset = SFTDataset(
            eval_data,
            tokenizer,
            getattr(args, f"max_len_{obj_index}"),
            strategy,
            input_template=getattr(args, f"input_template_{obj_index}"),
            obj_index=obj_index
        )
        train_dataloader = strategy.setup_dataloader(
            train_dataset,
            # getattr(args, f"micro_train_batch_size_{obj_index}"),
            getattr(args, "micro_train_batch_size"),
            True,
            True,
            train_dataset.collate_fn,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            # getattr(args, f"micro_train_batch_size_{obj_index}"),
            getattr(args, "micro_train_batch_size"),
            True,
            False,
            eval_dataset.collate_fn,
        )

    if obj_type == "DPO":
        train_data, eval_data = blending_datasets(
            getattr(args, f"dataset_{obj_index}"),
            getattr(args, f"dataset_probs_{obj_index}"),
            strategy,
            args.seed,
            max_count=getattr(args, f"max_samples_{obj_index}"),
            stopping_strategy="all_exhausted",
            train_split=getattr(args, f"train_split_{obj_index}"),
            eval_split=getattr(args, f"eval_split_{obj_index}"),
        )
        train_data = train_data.select(range(min(getattr(args, f"max_samples_{obj_index}"), len(train_data))))
        eval_data = eval_data.select(range(min(getattr(args, f"max_samples_{obj_index}"), len(eval_data))))
        train_dataset = RewardDataset(
            train_data, 
            tokenizer, 
            getattr(args, f"max_len_{obj_index}"), 
            strategy, 
            input_template=getattr(args, f"input_template_{obj_index}"), 
            is_dpo=True, 
            obj_index=obj_index
        )
        eval_dataset = RewardDataset(
            eval_data, 
            tokenizer, 
            getattr(args, f"max_len_{obj_index}"), 
            strategy, 
            input_template=getattr(args, f"input_template_{obj_index}"), 
            is_dpo=True, 
            obj_index=obj_index
        )

        train_dataloader = strategy.setup_dataloader(
            train_dataset,
            # getattr(args, f"micro_train_batch_size_{obj_index}"),
            getattr(args, "micro_train_batch_size"),
            True,
            True,
            train_dataset.collate_fn,
        )

        eval_dataloader = strategy.setup_dataloader(
            eval_dataset, 
            # getattr(args, f"micro_train_batch_size_{obj_index}"), 
            getattr(args, "micro_train_batch_size"),
            True, False, eval_dataset.collate_fn
        )

    return train_dataloader, eval_dataloader

def train(args):
    # Multiple debugging methods to ensure we see output
    try:
        debug_file = f"debug_rank_{args.local_rank}.txt"
        with open(debug_file, "w") as f:
            f.write(f"=== TRAINING SCRIPT STARTED ===\n")
            f.write(f"Selector mode: {args.selector_mode}\n")
            f.write(f"Local rank: {args.local_rank}\n")
            f.write(f"World size: {os.environ.get('WORLD_SIZE', 'N/A')}\n")
    except Exception as e:
        print(f"[RANK {args.local_rank}] Could not write to debug file: {e}", file=sys.stderr, flush=True)
    
    print(f"[RANK {args.local_rank}] Selector mode: {args.selector_mode}", flush=True)
    
    try:
        print(f"[RANK {args.local_rank}] Setting up strategy...", flush=True)
        strategy = get_strategy(args)
        print(f"[RANK {args.local_rank}] Strategy setup completed", flush=True)
        
        print(f"[RANK {args.local_rank}] Setting up distributed...", flush=True)
        strategy.setup_distributed()
        print(f"[RANK {args.local_rank}] Distributed setup completed", flush=True)
            
    except Exception as e:
        print(f"[RANK {args.local_rank}] Error in strategy setup: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    try:
        print("Setting up model...", flush=True)
        model = Actor(
            pretrain_or_model=args.pretrain, 
            load_in_4bit=args.load_in_4bit, 
            lora_rank=args.lora_rank, 
            lora_alpha=args.lora_alpha, 
            target_modules=args.target_modules, 
            lora_dropout=args.lora_dropout,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
        )
        print("Model creation completed", flush=True)
        strategy.print(model)
        print("Model printing completed", flush=True)
    except Exception as e:
        print(f"Error in model setup: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    try:
        print("Setting up tokenizer...", flush=True)
        tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
        print("Tokenizer setup completed", flush=True)
        
        print("Setting up optimizer...", flush=True)
        optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)
        print("Optimizer setup completed", flush=True)
        
        print("Setting up reference model...", flush=True)
        # For DPO/KTO, we always need a reference model
        obj_1_type = getattr(args, 'obj_1', 'SFT')
        obj_2_type = getattr(args, 'obj_2', 'SFT')
        needs_ref_model = (obj_1_type in ["DPO", "KTO"]) or (obj_2_type in ["DPO", "KTO"])

        if args.ref_constant or needs_ref_model:
            if args.ref_model is None or args.ref_model == "":
                # Create ref_model as a frozen snapshot of the initial model
                # This ensures it's a copy of the initial state (with LoRA adapters), not a fresh load
                print("Creating ref_model as frozen snapshot of initial model...", flush=True)
                ref_offload = True  # Use CPU offloading to save memory
                ref_model = Actor(
                    pretrain_or_model=args.pretrain,  # Use same pretrain as main model
                    load_in_4bit=args.load_in_4bit, 
                    lora_rank=args.lora_rank,  # Same LoRA config as main model
                    lora_alpha=args.lora_alpha, 
                    target_modules=args.target_modules, 
                    lora_dropout=args.lora_dropout,
                    use_flash_attention_2=args.flash_attn,
                    bf16=args.bf16,
                    ds_config=strategy.get_ds_eval_config(offload=ref_offload),
                )
                # Set _offload flag for DeepSpeed to handle CPU offloading
                ref_model._offload = ref_offload
                get_tokenizer(args.pretrain, ref_model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
                
                # Don't copy state dict - both models are initialized the same way
                # Copying would load entire model into memory unnecessarily
                # Since both use same pretrain + same LoRA config, they'll have same initial state
                
                # Freeze all parameters - ensure no gradients are computed
                for param in ref_model.model.parameters():
                    param.requires_grad = False
                
                # Set ref_model to eval mode (no gradients needed)
                ref_model.eval()
                print("ref_model created as frozen snapshot of initial model", flush=True)
            else:
                # Load ref_model from specified checkpoint
                print(f"Loading ref_model from {args.ref_model}...", flush=True)
                ref_offload = True
                ref_model = Actor(
                    pretrain_or_model=args.ref_model, 
                    load_in_4bit=args.load_in_4bit, 
                    lora_rank=args.lora_rank, 
                    lora_alpha=args.lora_alpha, 
                    target_modules=args.target_modules, 
                    lora_dropout=args.lora_dropout,
                    use_flash_attention_2=args.flash_attn,
                    bf16=args.bf16,
                    ds_config=strategy.get_ds_eval_config(offload=ref_offload),
                )
                ref_model._offload = ref_offload
                get_tokenizer(args.ref_model, ref_model.model, "right", strategy, use_fast=not args.disable_fast_tokenizer)
                
                # Freeze all parameters
                for param in ref_model.model.parameters():
                    param.requires_grad = False
                
                ref_model.eval()
                print("ref_model loaded and frozen", flush=True)
        else:
            ref_model=None
            print("\n TODO: implement ref regularization. Right now there will be no reg.\n", flush=True)
        print("Reference model setup completed", flush=True)
        
        print("Setting up dataloaders...", flush=True)
        train_dataloader, eval_dataloader = get_dataloaders(strategy, 1, tokenizer)
        print("First dataloader setup completed", flush=True)
        
        print("Setting up second dataset...", flush=True)
        data_2, _ = blending_datasets(
            args.dataset_2,
            args.dataset_probs_2,
            strategy,
            args.seed,
            max_count=args.max_samples_2,
            train_split=args.train_split_2,
            eval_split=args.eval_split_2,
        )
        print("Second dataset blending completed", flush=True)
        
        print("Creating second dataset object...", flush=True)
        # Extract dataset from tuple if blending_datasets returns a tuple
        if isinstance(data_2, tuple):
            data_2 = data_2[0]  # Take first element
            print(f"Extracted dataset from tuple, shape: {data_2.shape}", flush=True)
        
        # Use SFTDatasetIndexed for both BMO and BDR modes to provide index field
        dataset_2 = SFTDatasetIndexed(
            data_2, tokenizer, args.max_len_2, strategy, input_template=args.input_template_2
        )
        print("Second dataset object created", flush=True)
        
        print("Setting up second dataloader...", flush=True)
        train_dataloader_2 = strategy.setup_dataloader(
            dataset_2, args.micro_train_batch_size, True, True, dataset_2.collate_fn
        )
        print("Second dataloader setup completed", flush=True)
        
    except Exception as e:
        print(f"Error in dataloader setup: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    try:
        print("Setting up scheduler...", flush=True)
        # Safety check for accumulated_gradient
        if not hasattr(strategy, 'accumulated_gradient') or strategy.accumulated_gradient <= 0:
            print(f"Warning: strategy.accumulated_gradient is {getattr(strategy, 'accumulated_gradient', 'undefined')}, setting to 1", flush=True)
            strategy.accumulated_gradient = 1
        
        num_update_steps_per_epoch = len(train_dataloader) // strategy.accumulated_gradient
        max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

        scheduler = get_scheduler(
            args.lr_scheduler,
            optim,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
        )
        print("Scheduler setup completed", flush=True)
        
        print("All setup completed successfully", flush=True)
        
    except Exception as e:
        print(f"Error during setup: {e}")
        import traceback
        traceback.print_exc()
        raise e

    # Initialize data selector based on mode
    if args.selector_mode == "BDR" or args.selector_mode == "BDR_Online":
        # Initialize data selector for BDR and BDR_Online (bilevel approach)
        p = TrainableTensorModule(size=dataset_2.__len__(),activation=args.selector_activation)
        p_opt=strategy.create_optimizer(p, lr=args.selector_learning_rate, betas=(0.9, 0.95), weight_decay=0.)
        p_scheduler = get_scheduler(
            args.selector_lr_scheduler,
            p_opt,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
        )
    else:
        # BMO mode doesn't need explicit weight parameters
        p = None
        p_opt = None
        p_scheduler = None

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # strategy prepare
    if ref_model is not None:
        ((model, optim, scheduler), ref_model) = strategy.prepare((model, optim, scheduler), ref_model)
    else:
        (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))
    
    # load checkpoint
    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    if args.selector_mode == "BDR":
        trainer = SelectorTrainer_BDR(
            model=model,
            ref_model=ref_model,
            ref_constant=args.ref_constant,
            strategy=strategy,
            optim=optim,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            new_dataloader=train_dataloader_2,
            p=p,
            p_opt=p_opt,
            p_scheduler=p_scheduler, 
            scheduler=scheduler,
            max_norm=args.max_norm,
            batch_size=args.train_batch_size,
            max_epochs=args.max_epochs,
            tokenizer=tokenizer,
        )
    elif args.selector_mode == "BDR_Online":
        trainer = SelectorTrainer_BDR_Online(
            model=model,
            ref_model=ref_model,
            ref_constant=args.ref_constant,
            strategy=strategy,
            optim=optim,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            new_dataloader=train_dataloader_2,
            p=p,
            p_opt=p_opt,
            p_scheduler=p_scheduler, 
            scheduler=scheduler,
            max_norm=args.max_norm,
            batch_size=args.train_batch_size,
            max_epochs=args.max_epochs,
            tokenizer=tokenizer,
            online_sample_ratio=args.online_sample_ratio,
            online_generation_freq=args.online_generation_freq,
            online_sample_size=args.online_sample_size,
            online_temperature=args.online_temperature,
            dynamic_mode=getattr(args, 'dynamic_mode', False),
        )
    else:
        raise ValueError(f"Unknown selector_mode: {args.selector_mode}. Choose from: BDR, BDR_Online")

    print("Starting training...")
    try:
        trainer.fit(args)
        print("Training completed successfully", flush=True)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    # save model checkpoint after fitting on only rank0
    if args.save_path:
        strategy.save_model(model, tokenizer, args.save_path)
    
    # Handle different trainer types for final weight saving
    if args.selector_mode == "BDR":
        p_tensor = trainer.p.logits
        # Save BDR selection weights
        if strategy.is_rank_0():
            weights_name = f"{args.save_path}/{args.selector_name}_{args.selector_activation}_final_weights.pt"
            os.makedirs(os.path.dirname(weights_name), exist_ok=True)
            torch.save(p_tensor, weights_name)
            print(f"BDR selection weights saved: {weights_name}")
    elif args.selector_mode == "BDR_Online":
        p_tensor = trainer.p.logits
        # Save BDR_Online selection weights
        if strategy.is_rank_0():
            weights_name = f"{args.save_path}/{args.selector_name}_{args.selector_activation}_final_weights.pt"
            os.makedirs(os.path.dirname(weights_name), exist_ok=True)
            torch.save(p_tensor, weights_name)
            print(f"BDR_Online selection weights saved: {weights_name}")
    elif args.selector_mode == "BMO":
        # BMO computes weights dynamically during training
        if strategy.is_rank_0():
            print("BMO training completed. Selection weights are computed dynamically during training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Logging and saving
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_biobj")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB

    ## Efficient training args
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size (effective batch size across all GPUs)")
    parser.add_argument("--micro_train_batch_size", type=int, default=1, help="Micro batch size per GPU per forward pass")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--ref_offload", action="store_true", default=False)
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)

    ## Model
    parser.add_argument("--pretrain", type=str, default=None, help="remote or local initial model path")
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    ## General optimization args
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--lambd", type=float, default=0.5, help="coefficient for mixing ojective 1 and objective 2")
    parser.add_argument("--eps", type=float, default=1e-4, help="stopping threshold tolerance")
    parser.add_argument("--lr_scheduler", type=str, default="constant") #cosine_with_min_lr
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="initial learning rate")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay coefficient")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    # Objective 1
    parser.add_argument("--obj_1", type=str, default="SFT", help="objective 1 type")
    parser.add_argument("--aux_loss_coef_1", type=float, default=0, help="MoE balancing loss coefficient for objective 1")
    parser.add_argument("--obj_opt_1", type=float, default=0.0, help="optimum value of objective 1")
    parser.add_argument("--ref_pareto_1", type=str, default="", help="set of reference pareto front values for objective 1")
    # Objective 2
    parser.add_argument("--obj_2", type=str, default="SFT", help="objective 2 type")
    parser.add_argument("--max_len_2", type=int, default=2048, help="Max tokens for the samples of dataset 2")
    parser.add_argument("--aux_loss_coef_2", type=float, default=0, help="MoE balancing loss coefficient for objective 2")
    parser.add_argument("--obj_opt_2", type=float, default=0.0, help="optimum value of objective 2")
    parser.add_argument("--ref_pareto_2", type=str, default="", help="set of reference pareto front values for objective 2")


    ## Dataset args
    # Dataset 1
    parser.add_argument("--dataset_1", type=str, default=None)
    parser.add_argument("--dataset_probs_1", type=str, default="1.0", help="sampling probs for dataset 1")
    parser.add_argument("--train_split_1", type=str, default="train", help="train split of datset 1")
    parser.add_argument("--eval_split_1", type=str, default="test", help="test split of the dataset 1")
    parser.add_argument("--input_template_1", type=str, default="User: {}\nAssistant: ")
    parser.add_argument("--max_samples_1", type=int, default=1e8, help="Max number of samples for dataset 1")
    parser.add_argument("--max_len_1", type=int, default=2048, help="Max tokens for the samples of dataset 1")
    parser.add_argument(
        "--apply_chat_template_1", action="store_true", default=False, help="Use HF tokenizer chat template for dataset 1" 
    ) # under development, recommend use input_template instead
    parser.add_argument("--tokenizer_chat_template_1", type=str, default=None)
    # Dataset 2
    parser.add_argument("--dataset_2", type=str, default=None)
    parser.add_argument("--dataset_probs_2", type=str, default="1.0", help="sampling probs for dataset 2")
    parser.add_argument("--train_split_2", type=str, default="train", help="train split of datset 2")
    parser.add_argument("--eval_split_2", type=str, default="test", help="test split of the dataset 2")
    parser.add_argument("--input_template_2", type=str, default="User: {}\nAssistant: ")
    parser.add_argument("--max_samples_2", type=int, default=1e8, help="Max number of samples for dataset 2")
    parser.add_argument(
        "--apply_chat_template_2", action="store_true", default=False, help="Use HF tokenizer chat template for dataset 2"
    ) # under development, recommend use input_template instead
    parser.add_argument("--tokenizer_chat_template_2", type=str, default=None)

    # (input,target output) data format
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default="output", help="JSON dataset key")
    # when both datasets are (input,target output) format, these will be used for dataset 2
    parser.add_argument("--input_key_2", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key_2", type=str, default="output", help="JSON dataset key")

    # preference dataset format (prompt,preferred,dispreferred)
    parser.add_argument("--pref_prompt_key", type=str, default="prompt")
    parser.add_argument("--pref_chosen_key", type=str, default="chosen")
    parser.add_argument("--pref_rejected_key", type=str, default="rejected")
     # when both datasets are preference dataset format (prompt,preferred,dispreferred), these will be used for dataset 2
    parser.add_argument("--pref_prompt_key_2", type=str, default="prompt")
    parser.add_argument("--pref_chosen_key_2", type=str, default="chosen")
    parser.add_argument("--pref_rejected_key_2", type=str, default="rejected")

    ## Obj unique args
    # DPO and KD
    parser.add_argument("--ref_model", type=str, default=None) # teacher model in KD loss, reference model in DPO loss
    # when obj1 is dpo or kd, obj2 is dpo or kd, ref_model_2 will be used for the obj_2
    parser.add_argument("--ref_model_2", type=str, default=None)

    # DPO args
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--dpo_ipo", action="store_true", default=False)  # IPO https://arxiv.org/pdf/2310.12036v2.pdf
    parser.add_argument("--dpo_label_smoothing", type=float, default=0.0)  # cDPO https://arxiv.org/pdf/2305.18290.pdf
    parser.add_argument(
        "--dpo_nll_loss_coef", type=float, default=0, help="Regularization with NLL loss, see LLama 3.1 tech report."
    )  
    # when obj_1 and obj_2 both dpo, these will be used for the obj_2
    parser.add_argument("--dpo_beta_2", type=float, default=0.1)
    parser.add_argument("--dpo_ipo_2", action="store_true", default=False)  # IPO https://arxiv.org/pdf/2310.12036v2.pdf
    parser.add_argument("--dpo_label_smoothing_2", type=float, default=0.0)  # cDPO https://arxiv.org/pdf/2305.18290.pdf\
    parser.add_argument(
        "--dpo_nll_loss_coef_2", type=float, default=0, help="Regularization with NLL loss, see LLama 3.1 tech report."
    )  



    # selector arguments (updated for softmax approach)
    parser.add_argument("--selector_learning_rate", type=float, default=1e-4, help="[DEPRECATED] No longer used with softmax approach")
    parser.add_argument("--selector_activation", type=str, default="softmax", help="[DEPRECATED] No longer used with softmax approach")
    parser.add_argument("--selector_name", type=str, default="selection_logits", help="[DEPRECATED] No longer used with softmax approach")
    parser.add_argument("--selector_lr_scheduler", type=str, default="constant", help="[DEPRECATED] No longer used with softmax approach")
    
    # New softmax selector arguments
    parser.add_argument("--selector_mode", type=str, default="BDR", help="Selector mode (BDR, BMO, or BDR_Online)")
    parser.add_argument("--selector_temperature", type=float, default=1.0, help="Temperature for softmax in selection weights")
    parser.add_argument("--selector_weight_decay", type=float, default=0.1, help="Weight decay for upper level weight")
    
    # Online sampling parameters
    parser.add_argument("--online_sample_ratio", type=float, default=0.5, help="Ratio of on-policy samples in dataset 2")
    parser.add_argument("--online_generation_freq", type=int, default=10, help="Generate new on-policy samples every N iterations")
    parser.add_argument("--online_sample_size", type=int, default=100, help="Number of on-policy samples to generate")
    parser.add_argument("--online_temperature", type=float, default=1.0, help="Temperature for on-policy generation")
    parser.add_argument("--generation_batch_size", type=int, default=32, help="Batch size for generation in BDR_Online")
    parser.add_argument("--dynamic_mode", action="store_true", help="Enable dynamic mode for BDR_Online (select questions based on current weights)")
    parser.add_argument("--filter_rate", type=float, default=0.1, help="Filter rate for gated loss (top-k selection for model updates)")
    parser.add_argument("--ref_constant", type=float, default=0.)
    parser.add_argument("--upperlevel_weight", type=float, default=0.9)
    parser.add_argument("--upperlevel_weight_decay", type=float, default=0.1)

    ## remote logging
    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="bipost_seal")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="exp_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    args = parser.parse_args()

    if args.obj_1 in ["DPO","KTO"] or args.obj_2 in ["DPO","KTO"]:
        if args.ref_model is None or args.ref_model == "":
            args.ref_model = args.pretrain
    args.upperlevel_weight = args.lambd

    train(args)
