import math
import time
from abc import ABC
import os 

from pathlib import Path
import json

import torch
from torch import nn
from torch.optim import Optimizer
from bipost.utils.distributed_sampler import DistributedSampler
from tqdm import tqdm
from transformers.trainer import get_scheduler
from bipost.models import GPTLMLoss
from bipost.models import batch_GPTLMLoss
from bipost.models import DPOLoss


class SelectorTrainer_BDR(ABC):
    """
    Bilevel Data Selection (BDR) trainer.
    
    Naming Convention:
    - train_effective_batch_size: The total effective batch size across all GPUs
    """

    def __init__(
        self,
        model,
        ref_model,
        ref_constant,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        new_dataloader,
        p,
        p_opt: Optimizer,
        p_scheduler, 
        scheduler,
        max_norm: float = 1,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.train_effective_batch_size = batch_size  # Store the effective batch size for consistency
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.new_dataloader = new_dataloader
        self.scheduler = scheduler
        self.model = model
        self.ref_model = ref_model
        self.ref_constant = ref_constant
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.ul_weight = self.args.upperlevel_weight
        self.ul_weight_decay = self.args.upperlevel_weight_decay
        
        # Print initialization info for consistency with BMO trainer
        print(f"BDR Trainer initialized - train_effective_batch_size: {self.train_effective_batch_size}", flush=True)

        self.loss_fn = GPTLMLoss()
        self.batch_loss_fn = batch_GPTLMLoss()
        
        # DPO loss for obj_1 if it's DPO
        obj_1_type = getattr(self.args, 'obj_1', 'SFT')
        if obj_1_type == "DPO":
            dpo_beta = getattr(self.args, 'dpo_beta', 0.1)
            dpo_label_smoothing = getattr(self.args, 'dpo_label_smoothing', 0.0)
            dpo_ipo = getattr(self.args, 'dpo_ipo', False)
            self.dpo_loss_fn = DPOLoss(dpo_beta, dpo_label_smoothing, dpo_ipo)
        else:
            self.dpo_loss_fn = None
        
        # data selection policy param
        self.p = p
        self.p_opt = p_opt
        self.p_scheduler = p_scheduler
        
        # Track batch losses for analysis
        self.batch_losses_history = []
        self.selection_weights_history = []
        self.timestamps_history = []
        self.start_time = None
        
        # Track best model for checkpoint saving
        self.best_upper_loss = float('inf')
        self.best_global_step = 0

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef_2 > 1e-8

        # wandb setting
        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1 
        
        # Log initial weights (epoch 0) before any training
        if self.strategy.is_rank_0():
            initial_weights = self.p().detach().cpu().numpy()
            initial_logs = {
                "epoch_0_selection_weights": initial_weights.tolist(),
                "epoch_0_weights_min": float(initial_weights.min()),
                "epoch_0_weights_max": float(initial_weights.max()),
                "epoch_0_weights_median": float(torch.median(torch.from_numpy(initial_weights)).item()),
                "epoch_0_weights_q25": float(torch.quantile(torch.from_numpy(initial_weights), 0.25).item()),
                "epoch_0_weights_q75": float(torch.quantile(torch.from_numpy(initial_weights), 0.75).item()),
                "epoch_0_weights_std": float(initial_weights.std()),
                "current_epoch": 0,
                "global_step": 0,
            }
            
            # Log to wandb if available
            if self._wandb is not None:
                wandb_logs = {"train/%s" % k: v for k, v in initial_logs.items()}
                self._wandb.log(wandb_logs)
            
            print(f"Initial weights stats - Mean: {initial_weights.mean():.6f}, Std: {initial_weights.std():.6f}")
        
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(self.epochs):
            # Initialize running averages once at the start of training
            if epoch == 0:
                loss_mean = 0
                gpt_loss_mean = 0
                weighted_gpt_loss_mean = 0
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)
                self.new_dataloader.sampler.set_epoch(epoch)
            if epoch>=1:
                self.ul_weight -= self.ul_weight_decay
                print("\n UL weight now", self.ul_weight, end="\n")
            
            # Initialize start time for this epoch
            if self.start_time is None:
                self.start_time = time.time()
            
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            self.p.train()

            epoch_step = 0  # Initialize epoch step counter
            
            # Check if obj_1 is DPO
            obj_1_type = getattr(self.args, 'obj_1', 'SFT')
            is_dpo = (obj_1_type == "DPO")
            
            for data_1, data_2 in zip(self.train_dataloader, self.new_dataloader):
                # Handle obj_1 (DPO or SFT)
                if is_dpo:
                    # DPO format: (chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens)
                    chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data_1
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                    
                    chosen_logps, rejected_logps, aux_loss, _ = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )
                    with torch.no_grad():
                        reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                            self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                        )
                    
                    # Use DPOLoss to compute mean loss and rewards
                    preference_loss, chosen_reward, reject_reward = self.dpo_loss_fn(
                        chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
                    )
                    
                    if not self.aux_loss:
                        aux_loss = 0
                    
                    gpt_loss = preference_loss + aux_loss * getattr(self.args, 'aux_loss_coef_1', 0.0)
                    # batch_gpt_loss will be computed from dataset 2 (lower-level) below
                else:
                    # SFT format: (prompts_id_len, inputs, attention_masks, _)
                    prompts_id_len, inputs, attention_masks, _ = data_1
                    inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                    attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                    output = self.model(inputs, attention_mask=attention_mask, return_output=True)
                    
                    labels = torch.where(
                        attention_mask.bool(),
                        inputs,
                        self.loss_fn.IGNORE_INDEX,
                    )
                    if self.aux_loss:
                        aux_loss = output.aux_loss
                    else:
                        aux_loss = 0
                    
                    for label, source_len in zip(labels, prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                    
                    gpt_loss = self.batch_loss_fn(output.logits, labels, sequence_reduce="mean").mean(0)
                    # batch_gpt_loss will be computed from dataset 2 (lower-level) below
                
                # Handle obj_2 (always SFT format)
                ide, new_prompts_id_len, new_input, new_attention_masks, _ = data_2
                new_inputs = new_input.squeeze(1).to(torch.cuda.current_device())
                new_attention_mask = new_attention_masks.squeeze(1).to(torch.cuda.current_device())
                new_output = self.model(new_inputs, attention_mask=new_attention_mask, return_output=True)
                with torch.no_grad():
                    batch_weights = self.p()[ide]
                
                new_labels = torch.where(
                    new_attention_mask.bool(),
                    new_inputs,
                    self.batch_loss_fn.IGNORE_INDEX,
                )
                for new_label, new_source_len in zip(new_labels, new_prompts_id_len):
                    new_label[:new_source_len] = self.batch_loss_fn.IGNORE_INDEX
                
                # Compute per-sample losses for dataset 2 (lower-level, always SFT)
                # This is what should be weighted with batch_weights from dataset 2
                batch_gpt_loss = self.batch_loss_fn(new_output.logits, new_labels, sequence_reduce="mean")
                
                # Track batch losses and selection weights for analysis
                with torch.no_grad():
                    self.batch_losses_history.append(batch_gpt_loss.detach().cpu())
                    self.selection_weights_history.append(batch_weights.detach().cpu())
                    # Track timestamp
                    current_time = time.time()
                    timestamp=current_time - self.start_time
                    self.timestamps_history.append(timestamp)
                
                device = batch_gpt_loss.device
                batch_weights = batch_weights.to(device, non_blocking=True)

                # Use full batch weights for model updates (no filtering)
                weighted_gpt_loss = (batch_weights * batch_gpt_loss).mean(0)
                loss = self.ul_weight * gpt_loss + (1-self.ul_weight) * weighted_gpt_loss + aux_loss * self.args.aux_loss_coef_2
                
                self.strategy.backward(loss, self.model, self.optimizer) 
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                p = self.p()                   
                p = p.to(device)
                
                # Use full weights (unfiltered) for data weight updates
                selector_loss = (p[ide]*batch_gpt_loss.detach()).mean()
                # self.strategy.backward(selector_loss, self.p, self.p_opt)
                
                # self.strategy.optimizer_step(self.p_opt, self.p, self.p_scheduler)

                selector_loss.backward()

                # Step optimizer and scheduler
                self.p_opt.step()
                self.p_scheduler.step()
                self.p_opt.zero_grad()

                # Increment epoch step
                epoch_step += 1

                gpt_loss_mean = gpt_loss_mean * 0.95 + 0.05 * gpt_loss.item()
                weighted_gpt_loss_mean = weighted_gpt_loss_mean * 0.95 + 0.05 * weighted_gpt_loss.item()
                loss_mean = loss_mean * 0.95 + 0.05 * loss.item()

                logs_dict = {
                    "upper_loss": gpt_loss.item(), 
                    "upper_loss_mean": gpt_loss_mean, 
                    "loss_mean": loss_mean,
                    "batch_loss": batch_gpt_loss.mean().item(),
                    "weighted_gpt_loss": weighted_gpt_loss.item(),
                    "weighted_gpt_loss_mean": weighted_gpt_loss_mean,
                    "selection_weight_mean": batch_weights.mean().item(),
                    "selection_weight_std": batch_weights.std().item(),
                    "iteration": global_step,
                    "time": timestamp
                }
                
                # Add full selection weights distribution at the end of each epoch
                is_last_step_of_epoch = (epoch_step == len(self.train_dataloader) - 1)
                
                if is_last_step_of_epoch:
                    # Get all selection weights for the entire dataset
                    all_weights = self.p().detach().cpu().numpy()
                    logs_dict.update({
                        f"epoch_{epoch+1}_selection_weights": all_weights.tolist(),
                        f"epoch_{epoch+1}_weights_min": float(all_weights.min()),
                        f"epoch_{epoch+1}_weights_max": float(all_weights.max()),
                        f"epoch_{epoch+1}_weights_median": float(torch.median(torch.from_numpy(all_weights)).item()),
                        f"epoch_{epoch+1}_weights_q25": float(torch.quantile(torch.from_numpy(all_weights), 0.25).item()),
                        f"epoch_{epoch+1}_weights_q75": float(torch.quantile(torch.from_numpy(all_weights), 0.75).item()),
                        f"epoch_{epoch+1}_weights_std": float(all_weights.std()),
                        "current_epoch": epoch + 1,
                    })

                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()

                # logs/checkpoints/evaluation
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, gpt_loss_mean)

                step_bar.update()
                global_step += self.strategy.world_size
            if self.strategy.is_rank_0():
                checkpoint_dir = args.save_path + "/"
                os.makedirs(checkpoint_dir, exist_ok=True)
                p_name = checkpoint_dir + args.selector_name+"_"+ args.selector_activation \
                    +"_ep"+str(epoch+1)+".pt"
                torch.save(self.p.logits, p_name)
                print(self.p.logits)
                
                # Save batch loss analysis data
                self.save_batch_loss_analysis(epoch, args)
                
            epoch_bar.update()

    def save_batch_loss_analysis(self, epoch, args):
        """Save batch loss and selection weight analysis data"""
        if self.batch_losses_history and self.selection_weights_history:
            # Get data for this epoch
            start_idx = epoch * len(self.train_dataloader)
            end_idx = (epoch + 1) * len(self.train_dataloader)
            
            epoch_batch_losses = self.batch_losses_history[start_idx:end_idx]
            epoch_selection_weights = self.selection_weights_history[start_idx:end_idx]
            
            if epoch_batch_losses and epoch_selection_weights:
                analysis_data = {
                    'epoch': epoch,
                    'batch_losses': torch.cat(epoch_batch_losses, dim=0),
                    'selection_weights': torch.cat(epoch_selection_weights, dim=0),
                    'timestamps': torch.tensor(self.timestamps_history[start_idx:end_idx]),
                    'method': 'BDR'
                }
                
                # Save analysis data
                # analysis_name = f"./checkpoint/bdr/bdr_selector_analysis_ep{epoch+1}.pt"
                # torch.save(analysis_data, analysis_name)

                analysis_name = Path(f"{args.save_path}/bdr_selector_analysis_ep{epoch+1}.pt")
                analysis_dir = analysis_name.parent
                analysis_dir.mkdir(parents=True, exist_ok=True)

                # optional: only save on rank 0 in distributed runs
                is_rank0 = int(os.environ.get("RANK", "0")) == 0
                if is_rank0:
                    torch.save(analysis_data, analysis_name.as_posix())

                # # Save final weights
                # weights_name = f"./checkpoint/bmo/temperature_{self.temperature}/bmo_selector_weights_ep{epoch+1}.pt"
                # os.makedirs(weights_name, exist_ok=True)
                # torch.save(analysis_data['selection_weights'], weights_name)
                
                # Log summary statistics
                batch_loss_mean = analysis_data['batch_losses'].mean().item()
                selection_weight_mean = analysis_data['selection_weights'].mean().item()
                selection_weight_std = analysis_data['selection_weights'].std().item()
                
                print(f"Epoch {epoch+1} BDR Analysis:")
                print(f"  Batch loss mean: {batch_loss_mean:.4f}")
                print(f"  Selection weight mean: {selection_weight_mean:.4f}")
                print(f"  Selection weight std: {selection_weight_std:.4f}")

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, gpt_loss_mean=None):
        if (global_step // self.strategy.world_size) % args.logging_steps == 0:
            # Separate large arrays from scalar metrics for distributed reduction
            large_arrays = {}
            scalar_metrics = {}
            
            for key, value in logs_dict.items():
                if isinstance(value, list) and len(value) > 100:  # Large arrays
                    large_arrays[key] = value
                else:  # Scalar metrics
                    scalar_metrics[key] = value
            
            # Only reduce scalar metrics across processes
            scalar_metrics = self.strategy.all_reduce(scalar_metrics)
            
            # Convert any remaining tensors to Python scalars for JSON serialization
            for key, value in scalar_metrics.items():
                if torch.is_tensor(value):
                    scalar_metrics[key] = value.item()
            
            # Combine back (large arrays only from rank 0)
            if self.strategy.is_rank_0():
                logs_dict = {**scalar_metrics, **large_arrays}
            else:
                logs_dict = scalar_metrics

            # log_path = os.path.join(args.ckpt_path, f"bdr/logs")
            # os.makedirs(log_path, exist_ok=True)            
            # log_file = os.path.join(log_path, "train_logs.jsonl")

            # with open(log_file, "a") as f:
            #     json.dump({"step": global_step, **logs_dict}, f)
            #     f.write("\n")


            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs, step=global_step)

        # eval
        if ((global_step-1) // self.strategy.world_size+1) % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # Only save when save_steps is reached (not for best models)

        if ((global_step-1) // self.strategy.world_size+1) % args.save_steps == 0:
            new_ckpt_path = os.path.join(args.ckpt_path, f"bdr")
            os.makedirs(new_ckpt_path, exist_ok=True)
            
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.model.model, new_ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)
            
            if self.strategy.is_rank_0():
                print(f"Checkpoint saved at step {global_step}")

    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        obj_1_type = getattr(self.args, 'obj_1', 'SFT')
        is_dpo = (obj_1_type == "DPO")
        
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for data in eval_dataloader:
                if is_dpo:
                    # DPO format
                    chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                    
                    chosen_logps, rejected_logps, _, _ = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )
                    reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                        self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )
                    
                    preference_loss, _, _ = self.dpo_loss_fn(
                        chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
                    )
                    loss = preference_loss
                else:
                    # SFT format
                    prompts_id_len, inputs, attention_masks, _ = data
                    inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                    attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                    logits = self.model(inputs, attention_mask=attention_mask, return_output=True)["logits"]

                    labels = torch.where(
                        attention_mask.bool(),
                        inputs,
                        self.loss_fn.IGNORE_INDEX,
                    )
                    for label, source_len in zip(labels, prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX
                    loss = self.batch_loss_fn(logits, labels).mean(0)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval gpt_loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs, step=steps)
        self.model.train()  # reset model state

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together."""
        input_ids, att_masks, prompt_id_lens = self.concatenated_inputs(
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
        )
        # Move inputs to CUDA - DeepSpeed will handle CPU offloading automatically
        # When using DeepSpeed CPU offloading, it automatically moves parameters to GPU when needed
        device = torch.cuda.current_device()
        input_ids = input_ids.to(device)
        att_masks = att_masks.to(device)
        output = model(input_ids, attention_mask=att_masks, return_output=True)
        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = self._get_batch_logps(
            all_logits, input_ids, att_masks, prompt_id_lens, average_log_prob=False
        )
        chosen_logps = all_logps_sum[: chosen_ids.shape[0]]
        rejected_logps = all_logps_sum[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: chosen_ids.shape[0]].mean()

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Concatenate the chosen and rejected inputs into a single tensor."""
        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat(
                    [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks, prompt_id_lens * 2

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits."""
        assert average_log_prob == False
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        loss_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(loss_masks, prompt_id_lens):
            mask[:source_len] = False
        loss_masks = loss_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[loss_masks == False] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        logprobs_sums = (per_token_logps * loss_masks).sum(-1)
        logprobs_means = (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
        return logprobs_sums, logprobs_means