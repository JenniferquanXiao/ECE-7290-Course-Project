import math
import time
from abc import ABC
import os 
import random
from pathlib import Path
import json
import gc 

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from bipost.utils.distributed_sampler import DistributedSampler
from tqdm import tqdm
from transformers.trainer import get_scheduler
from bipost.models import GPTLMLoss
from bipost.models import batch_GPTLMLoss
from bipost.models import BatchGPTLMLossIS
from bipost.models import DPOLoss


class SelectorTrainer_BDR_Online(ABC):
    """
    Bilevel Data Selection with Online Samples (BDR_Online) trainer.
    
    Key features:
    1. Mask out online_ratio of offline data (keep questions, remove responses)
    2. Generate multiple responses per masked question using current policy
    3. Apply data weights to each question-response pair
    4. Apply importance weights with stop gradients
    5. Use all generated responses for better learning
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
        # Online sampling parameters
        online_sample_ratio: float = 0.1,  # Ratio of data to mask out for online generation
        online_generation_freq: int = 1000,   # Generate new responses every N iterations
        online_sample_size: int = 1,      # Number of responses to generate per question
        online_temperature: float = 1.0,   # Temperature for generation
        dynamic_mode: bool = False,  # Enable dynamic mode: select questions based on current weights
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.train_effective_batch_size = batch_size
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
        # Resolve max sequence length from dataset if available; otherwise fall back to CLI args
        dataset_max_len = getattr(getattr(self.new_dataloader, "dataset", None), "max_length", None)
        if dataset_max_len is not None:
            self.max_length = int(dataset_max_len)
        else:
            self.max_length = int(getattr(self.args, "max_len_2", getattr(self.args, "max_len_1", 512)))
        
        # Online sampling parameters
        self.online_sample_ratio = online_sample_ratio
        self.online_generation_freq = online_generation_freq
        self.online_sample_size = online_sample_size
        self.online_temperature = online_temperature
        self.dynamic_mode = dynamic_mode
        
        # Online sample storage
        self.masked_indices = []  # Indices of masked samples in dataset
        self.masked_questions = []  # Questions for masked samples
        self.masked_responses = []  # List of lists: [[resp1, resp2, ...], [resp1, resp2, ...], ...]
        self.masked_log_probs = []  # List of lists: [[log1, log2, ...], [log1, log2, ...], ...]
        # self.masked_logits removed - we only need log_probs, not full logits
        self.masked_generated_labels = []  # List of lists: [[labels1, labels2, ...], [labels1, labels2, ...], ...]
        self.last_generation_step = -1  # Track when we last generated responses
        
        print(f"BDR_Online Trainer initialized - train_effective_batch_size: {self.train_effective_batch_size}", flush=True)
        print(f"Online sampling: ratio={online_sample_ratio}, freq={online_generation_freq}, size={online_sample_size}", flush=True)
        print(f"Dynamic mode: {dynamic_mode}", flush=True)

        self.loss_fn = GPTLMLoss()
        self.batch_loss_fn = batch_GPTLMLoss()
        self.is_loss_fn = BatchGPTLMLossIS(min_log_prob=-30.0, clip_max_weight=10.0, normalize=False)
        
        # DPO loss for obj_1 if it's DPO
        obj_1_type = getattr(self.args, 'obj_1', 'SFT')
        if obj_1_type == "DPO":
            dpo_beta = getattr(self.args, 'dpo_beta', 0.1)
            dpo_label_smoothing = getattr(self.args, 'dpo_label_smoothing', 0.0)
            dpo_ipo = getattr(self.args, 'dpo_ipo', False)
            self.dpo_loss_fn = DPOLoss(dpo_beta, dpo_label_smoothing, dpo_ipo)
        else:
            self.dpo_loss_fn = None
        
        # Initialize generation tracking
        self.initial_generation_done = False  # Track if initial generation is complete
        
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

    def initialize_masked_data(self):
        """
        OPTIMIZED: Initialize masked data with batch generation for speed.
        1. Selecting online_ratio of dataset samples to mask
        2. Generating multiple responses per question using batch processing (skipped in fixed mode)
        3. Storing masked questions, responses, and log probabilities
        """
        print("Initializing masked data for BDR_Online (optimized)...")
        
        total_samples = len(self.new_dataloader.dataset)
        num_masked = int(total_samples * self.online_sample_ratio)
        
        # Select random indices to mask (same for both fixed and dynamic modes)
        self.masked_indices = random.sample(range(total_samples), num_masked)
        self.masked_indices.sort()  # Keep sorted for easier lookup
        
        print(f"Masking {num_masked} out of {total_samples} samples ({self.online_sample_ratio:.2%})")
        
        # Extract all questions first
        self.masked_questions = []
        for idx in self.masked_indices:
            try:
                sample = self.new_dataloader.dataset[idx]
                question = sample[4]["input"]
                self.masked_questions.append(question)
            except Exception as e:
                print(f"Warning: Error processing masked sample {idx}: {e}")
                self.masked_questions.append("Please respond to this prompt.")
        
        # Generate responses immediately (for both fixed and dynamic modes when called from initialize_masked_data)
        # OPTIMIZATION: Batch generate all responses at once
        print(f"Generating {self.online_sample_size} responses for {len(self.masked_questions)} questions...")
        self.masked_responses, self.masked_log_probs, self.masked_generated_labels = self.batch_generate_all_responses(self.masked_questions)
        
        self.last_generation_step = 0
        
        print(f"Initialized {len(self.masked_questions)} masked samples")
        print(f"Each question has {self.online_sample_size} generated responses")
        
        # Calculate effective dataset size for proper p scaling
        num_non_masked = len(self.new_dataloader.dataset) - len(self.masked_indices)
        num_masked = len(self.masked_indices)
        self.effective_dataset_size = num_non_masked + num_masked * self.online_sample_size
        print(f"Effective dataset size: {self.effective_dataset_size} (non-masked: {num_non_masked}, masked: {num_masked} × {self.online_sample_size})")
        if len(self.masked_questions) > 0:
            print(f"Sample: Q: '{self.masked_questions[0][:50]}...' A: '{self.masked_responses[0][0][:50]}...' (1 of {len(self.masked_responses[0])})")

        # Update p scaling to use effective dataset size
        if hasattr(self.p, 'norma'):
            self.p.norma = len(self.new_dataloader.dataset)  # Use original dataset size
            print(f"Updated p scaling to original dataset size: {len(self.new_dataloader.dataset)}")
            print(f"Updated p scaling to effective dataset size: {self.effective_dataset_size}")

    def batch_generate_all_responses(self, questions, batch_size=None):
        """
        OPTIMIZATION: Generate all responses in batches for much better speed.
        """
        all_responses = []
        all_log_probs = []
        all_labels = []
        
        self.model.eval()
        
        # Process questions in flexible batches based on total generations
        if batch_size is None:
            # Use flexible batch size based on total generations with cap
            total_generations = len(questions) * self.online_sample_size
            max_generation_batch_size = int(getattr(self.args, "generation_batch_size", 32))
            flexible_batch_size = min(max(8, total_generations // 8), max_generation_batch_size)
        else:
            # Use provided batch_size with flexible adjustment and cap
            total_generations = len(questions) * self.online_sample_size
            max_generation_batch_size = int(getattr(self.args, "generation_batch_size", 32))
            flexible_batch_size = min(max(8, min(batch_size, total_generations // 8)), max_generation_batch_size)
        
        total_batches = (len(questions) + flexible_batch_size - 1) // flexible_batch_size
        
        print(f"Total generations: {total_generations}, Using flexible batch size: {flexible_batch_size}")
        
        for i in range(0, len(questions), flexible_batch_size):
            batch_questions = questions[i:i + flexible_batch_size]
            batch_num = i // flexible_batch_size + 1
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_questions)} questions)")
            
            # Generate responses for this batch of questions
            batch_responses, batch_log_probs, batch_labels = self.generate_responses_for_batch(batch_questions)
            all_responses.extend(batch_responses)
            all_log_probs.extend(batch_log_probs)
            all_labels.extend(batch_labels)
        
        self.model.train()
        return all_responses, all_log_probs, all_labels

    def generate_responses_for_batch(self, questions):
        """
        OPTIMIZED: Generate responses for multiple questions in a single model.generate() call.
        Implements improvements:
        1) Use encoder lengths directly for masking (no second tokenization of questions)
        2) Vectorized log-prob computation over the whole batch
        3) Avoid GPU sync by decoding from CPU
        4) Use batch loss IGNORE index
        5) Make padding policy explicit per phase and restore afterward
        6) Cast logits to float before log_softmax / CE for numerical stability
        7) Make max_new_tokens configurable via args (online_max_new_tokens)
        8) **Memory-optimized**: stream forward in chunks, compute log-probs immediately, do not store logits
        """


        # Save & set padding policy for generation
        prev_padding_side = getattr(self.tokenizer, 'padding_side', None)
        if hasattr(self.tokenizer, 'padding_side'):
            self.tokenizer.padding_side = 'left'  # required for decoder-only generation

        # (A) Tokenize prompts (LEFT padding) for generation
        enc = self.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        enc = {k: v.to(torch.cuda.current_device()) for k, v in enc.items()}

        B = len(questions)
        G = self.online_sample_size
        device = enc['input_ids'].device

        # Token IDs
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        IGNORE = self.batch_loss_fn.IGNORE_INDEX  # keep using your batch loss ignore index

        if pad_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            pad_id = self.tokenizer.pad_token_id

        # (B) Batched generation (B*G, L_out)
        max_new_tokens = int(getattr(self.args, "online_max_new_tokens", 256))  # Changed from 512 to 256
        with torch.no_grad():
            gen = self.model.model.generate(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=self.online_temperature,
                do_sample=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                use_cache=True,
                num_return_sequences=G,
            )  # (B*G, L_out)

            gc.collect()
            torch.cuda.empty_cache()

        ## ToDo: clear KV cache release, no grad!! release KV cache in with 

        # (C) Response-only decode in batch (mask out the prompt portion)
        enc_lens = enc["attention_mask"].sum(-1)                 # (B,)
        enc_lens_rep = enc_lens.repeat_interleave(G)             # (B*G,)
        L_out = gen.size(1)
        pos = torch.arange(L_out, device=device).unsqueeze(0)    # (1, L_out)
        cont_mask = pos >= enc_lens_rep.unsqueeze(1)             # (B*G, L_out)

        gen[~cont_mask] = pad_id

        decoded_all = self.tokenizer.batch_decode(
            gen.detach().cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        decoded_all = [s.strip() for s in decoded_all]

        # Group back to (B, G)
        grouped = [decoded_all[i * G : (i + 1) * G] for i in range(B)]

        # (D) Build full texts (prompt + response + EOS) batched
        full_texts = []
        for i, q in enumerate(questions):
            for r in grouped[i]:
                resp = r if r else "I apologize, but I couldn't generate a proper response."
                full_texts.append(q + resp + " " + self.tokenizer.eos_token)

        # Switch to RIGHT padding for forward pass
        if hasattr(self.tokenizer, 'padding_side'):
            self.tokenizer.padding_side = 'right'

        full_enc = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        full_enc = {k: v.to(device) for k, v in full_enc.items()}  # (B*G, Lmax)

        # (E) Build shifted labels, mask padding and prompt tokens
        Lmax = full_enc['input_ids'].size(1)
        labels = full_enc['input_ids'].clone()
        labels[:, :-1] = labels[:, 1:].clone()
        labels[:, -1] = IGNORE
        if 'attention_mask' in full_enc:
            labels[~full_enc['attention_mask'].bool()] = IGNORE

        # Keep loss starting from last prompt token so first response token is predicted from last prompt token
        start_pos = torch.clamp(enc_lens_rep - 1, min=0)         # (B*G,)
        posL = torch.arange(Lmax, device=device).unsqueeze(0)    # (1, Lmax)
        keep_mask = posL >= start_pos.unsqueeze(1)               # (B*G, Lmax)
        labels = torch.where(keep_mask, labels, torch.full_like(labels, IGNORE))

        # (F) Memory-optimized per-token log-probs:
        #     stream the forward in microbatches and immediately turn logits -> token log-probs,
        #     do NOT keep logits around.
        B_G = full_enc['input_ids'].size(0)
        V = len(self.tokenizer)  # vocab size from tokenizer

        # Preallocate CPU tensor for per-token log-probs (much smaller than logits)
        token_logps_all = torch.full((B_G, Lmax), float('-inf'), device='cpu')

        # CrossEntropyLoss (ignore_index) to get -log p(y|x) cheaply
        ce = nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE)

        # Choose a reasonable microbatch size to control peak memory
        mb = int(getattr(self.args, "online_forward_chunk_size", 8))  # tune as needed

        with torch.inference_mode():
            for s in range(0, B_G, mb):
                e = min(s + mb, B_G)
                inp_mb = full_enc['input_ids'][s:e]
                attn_mb = full_enc['attention_mask'][s:e] if 'attention_mask' in full_enc else None
                labels_mb = labels[s:e]

                # Forward this microbatch
                out_mb = self.model(
                    inp_mb,
                    attention_mask=attn_mb,
                    return_output=True,
                )
                # Immediately convert to float for numerical stability
                logits_mb = out_mb.logits.float()  # (mb, Lmax, V)
                
                # Ensure V matches the actual vocabulary size
                mb_actual, Lmax_actual, V_actual = logits_mb.shape
                if V_actual != V:
                    V = V_actual

                # Flatten for CE: (mb*Lmax, V) vs (mb*Lmax,)
                logits_flat = logits_mb.view(-1, V)
                labels_flat = labels_mb.view(-1)

                # CE = -log p(y|x); we want log p, so negate
                neg_logp_flat = ce(logits_flat, labels_flat)      # (mb*Lmax,)
                logp_mb = -neg_logp_flat.view(e - s, Lmax)        # (mb, Lmax)

                # Mask invalid positions to -inf (already ignored by CE, but make explicit in storage)
                invalid_mask = (labels_mb == IGNORE)
                logp_mb = logp_mb.masked_fill(invalid_mask, float('-inf'))

                # Move to CPU and store
                token_logps_all[s:e] = logp_mb.detach().cpu()

                # Free ASAP
                del out_mb, logits_mb, logits_flat, labels_flat, neg_logp_flat, logp_mb, invalid_mask
                torch.cuda.empty_cache()

        # (G) Pack outputs in your original structure
        batch_responses = grouped                                  # (B, G) strings
        batch_log_probs = []                                        # (B, G) tensors of per-token log-probs (masked -inf)
        batch_labels = []                                           # (B, G) tensors of labels

        for i in range(B):
            qi_logps = []
            qi_labels = []
            for j in range(G):
                r = i * G + j
                if 'attention_mask' in full_enc:
                    true_len = int(full_enc['attention_mask'][r].sum().item())
                else:
                    true_len = Lmax
                qi_logps.append(token_logps_all[r, :true_len].clone())           # CPU tensor
                qi_labels.append(labels[r, :true_len].detach().cpu())            # CPU tensor
            batch_log_probs.append(qi_logps)
            batch_labels.append(qi_labels)

        # Restore padding policy
        if prev_padding_side is not None and hasattr(self.tokenizer, 'padding_side'):
            self.tokenizer.padding_side = prev_padding_side

        return batch_responses, batch_log_probs, batch_labels

            
            

    def select_dynamic_questions(self, current_weights):
        """
        MEMORY-OPTIMIZED: Select bottom online_sample_ratio questions based on current selection weights for dynamic mode.
        Uses efficient selection without loading all weights into memory at once.
        Returns indices of questions to generate new responses for.
        """
        import numpy as np
        
        dataset = self.new_dataloader.dataset
        total_samples = len(dataset)
        num_to_select = int(total_samples * self.online_sample_ratio)
        
        # MEMORY OPTIMIZATION: Use torch.topk instead of full argsort
        with torch.no_grad():
            all_weights = self.p()  # Keep on GPU for efficiency
            
            # Use topk to get bottom k elements (most efficient)
            # We want the smallest weights, so we use topk with largest=False
            bottom_values, bottom_indices = torch.topk(all_weights, num_to_select, largest=False)
            
            # Move to CPU only for final result
            bottom_indices = bottom_indices.cpu().numpy()
            bottom_values = bottom_values.cpu().numpy()
        
        print(f"Dynamic selection: Selected {len(bottom_indices)} questions with lowest weights")
        print(f"Weight range: min={bottom_values.min():.6f}, max={bottom_values.max():.6f}")
        
        return bottom_indices.tolist()
    
    def generate_responses_for_dynamic_questions(self, question_indices):
        """
        Generate responses for dynamically selected questions.
        """
        dataset = self.new_dataloader.dataset
        
        # Extract questions for selected indices
        questions = []
        for idx in question_indices:
            try:
                sample = dataset[idx]
                question = sample[4]["input"]
                questions.append(question)
            except Exception as e:
                print(f"Warning: Error processing dynamic sample {idx}: {e}")
                questions.append("Please respond to this prompt.")
        
        # Generate responses for these questions
        print(f"Generating {self.online_sample_size} responses for {len(questions)} dynamically selected questions...")
        responses, log_probs, labels = self.batch_generate_all_responses(questions)
        
        return questions, responses, log_probs, labels

    def update_masked_responses(self, step):
        """
        Update masked responses using current policy (delayed generation).
        In fixed mode: Generate multiple responses per question for ALL masked questions.
        In dynamic mode: Select bottom online_sample_ratio questions based on current weights and generate new responses.
        """
        # DELAY: Wait until step 5000 before starting generation (let model train first)
        if step < 5000:
            return
        
        # Check if we need to generate new responses
        if step - self.last_generation_step >= self.online_generation_freq:
            if self.dynamic_mode:
                # Dynamic mode: select questions based on current weights
                print(f"[Step {step}] Dynamic mode: Selecting questions based on current weights...")
                
                # Select bottom online_sample_ratio questions based on current weights
                dynamic_indices = self.select_dynamic_questions(None)  # We'll get weights inside the method
                
                if len(dynamic_indices) == 0:
                    print(f"[Step {step}] No questions selected for dynamic generation")
                    self.last_generation_step = step
                    return
                
                # Generate responses for dynamically selected questions
                self.model.eval()
                with torch.no_grad():
                    questions, responses, log_probs, labels = self.generate_responses_for_dynamic_questions(dynamic_indices)
                
                self.model.train()
                
                # Update storage for dynamic mode
                self.masked_indices = dynamic_indices
                self.masked_questions = questions
                self.masked_responses = responses
                self.masked_log_probs = log_probs
                self.masked_generated_labels = labels
                
                print(f"[Step {step}] Dynamic mode: Updated {len(dynamic_indices)} questions")
                
            else:
                # Fixed mode: update all previously masked questions
                # Note: masked_questions should already be initialized in fit() for fixed mode
                if len(self.masked_questions) == 0:
                    print(f"[Step {step}] Warning: No masked questions found in fixed mode (online_sample_ratio may be 0)")
                    self.last_generation_step = step
                    return
                
                print(f"[Step {step}] Fixed mode: Updating ALL masked responses using current policy...")
                
                self.model.eval()
                with torch.no_grad():
                    # Generate refreshed responses/log_probs/labels for ALL masked questions using batched generator
                    all_responses, all_log_probs, all_labels = self.batch_generate_all_responses(self.masked_questions)
                    
                    # Update responses for ALL masked questions
                    for j in range(len(self.masked_questions)):
                        self.masked_responses[j] = all_responses[j]
                        self.masked_log_probs[j] = all_log_probs[j]
                        self.masked_generated_labels[j] = all_labels[j]
                
                self.model.train()
                print(f"[Step {step}] Fixed mode: Updated ALL {len(self.masked_questions)} masked questions")
            
            self.last_generation_step = step
            print(f"Each question now has {self.online_sample_size} responses")

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together."""
        input_ids, att_masks, prompt_id_lens = self.concatenated_inputs(
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
        )
        output = model(input_ids, attention_mask=att_masks, return_output=True)
        all_logits = output["logits"]
        all_logps = self._get_batch_logps(all_logits, input_ids, att_masks, prompt_id_lens, average_log_prob=False)
        chosen_logps = all_logps[: chosen_ids.shape[0]]
        rejected_logps = all_logps[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss

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

    def _get_batch_logps(self, logits, input_ids, att_masks, prompt_id_lens, average_log_prob=False):
        """Compute log probabilities for DPO."""
        labels = input_ids[:, 1:].clone()
        logits = logits[:, :-1, :]
        
        loss_masks = att_masks[:, 1:].clone().bool()
        # mask prompts
        for mask, source_len in zip(loss_masks, prompt_id_lens):
            mask[:source_len] = False
        
        labels[~loss_masks] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return (per_token_logps * loss_masks).sum(-1)

    def compute_mixed_batch_loss_with_stop_grad(self, batch_indices, original_logits, original_labels, original_questions):
        """
        OPTIMIZED (memory-friendly):
        - Unmasked questions: use provided original_logits/original_labels with self.batch_loss_fn
        - Masked questions: recompute current logits per response under torch.inference_mode(),
          reuse precomputed labels & old_log_probs, immediately reduce to a scalar, and discard tensors.
        """

        device = original_logits.device
        IGNORE = self.batch_loss_fn.IGNORE_INDEX

        # Build O(1) lookup once
        if not hasattr(self, "_masked_idx_map") or self._masked_idx_map is None:
            # maps global idx -> position in masked arrays
            self._masked_idx_map = {ix: pos for pos, ix in enumerate(self.masked_indices)}

        batch_losses = []

        # Save & set tokenizer padding side (RIGHT padding for these forwards is convenient)
        prev_padding_side = getattr(self.tokenizer, "padding_side", None)
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "right"

        # A tiny helper to pad/truncate a 1D tensor to length L with a fill value (on device)
        def _pad_to(t: torch.Tensor, L: int, fill) -> torch.Tensor:
            t = t.to(device)
            if t.numel() == L:
                return t
            if t.numel() > L:
                return t[:L]
            out = t.new_full((L,), fill)
            out[: t.numel()] = t
            return out

        for i, idx in enumerate(batch_indices):
            masked_pos = self._masked_idx_map.get(idx, None)

            if masked_pos is not None:
                # ===== MASKED QUESTION =====
                question = self.masked_questions[masked_pos]
                responses = self.masked_responses[masked_pos]            # List[str]
                labels_list = self.masked_generated_labels[masked_pos]   # List[Tensor], each (L_j,)
                old_logps_list = self.masked_log_probs[masked_pos]       # List[Tensor], each (L_j,)

                if not responses:
                    # No generated responses — return zero loss for this item
                    batch_losses.append(torch.zeros((), device=device, dtype=original_logits.dtype))
                    continue

                # Running (weighted) average without storing per-response tensors
                running_sum = torch.zeros((), device=device, dtype=original_logits.dtype)
                count = 0
                base_weight = 1.0 / max(1, self.online_sample_size)

                for j, response in enumerate(responses):
                    # Compose full text *just for the model forward* (labels/logprobs are reused)
                    full_text = question + response + " " + self.tokenizer.eos_token

                    # Tokenize once for this response (RIGHT padding; single sample -> no real pad)
                    tok = self.tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length,
                        add_special_tokens=True,
                        padding=False,
                    )
                    input_ids = tok["input_ids"].to(device)
                    attn_mask = tok.get("attention_mask", None)
                    if attn_mask is not None:
                        attn_mask = attn_mask.to(device)

                    # Forward current model; do NOT keep logits
                    out = self.model(input_ids, attention_mask=attn_mask, return_output=True)
                    logits = out.logits  # (1, L, V)
                    L = logits.size(1)

                    lbl = _pad_to(labels_list[j].squeeze(0) if labels_list[j].dim() == 2 else labels_list[j], L, IGNORE).unsqueeze(0)
                    old_lp = _pad_to(old_logps_list[j].squeeze(0) if old_logps_list[j].dim() == 2 else old_logps_list[j], L, 0.0).unsqueeze(0)

                    # Importance-sampled loss for this single response (scalar)
                    resp_loss = self.is_loss_fn(logits, lbl, old_lp, sequence_reduce="mean")

                    # Frequency compensation (keep your original semantics)
                    weighted = resp_loss * base_weight
                    running_sum = running_sum + weighted
                    count += 1

                    # Immediately drop large tensors
                    del out, logits, tok, input_ids, attn_mask, lbl, old_lp
                    torch.cuda.empty_cache()

                if count > 0:
                    # Your original code: mean over the list after multiplying by 1/G
                    masked_avg = running_sum / count
                    batch_losses.append(masked_avg)
                else:
                    batch_losses.append(torch.zeros((), device=device, dtype=original_logits.dtype))

            else:
                logits = original_logits[i:i+1]
                labels = original_labels[i:i+1]
                loss = self.batch_loss_fn(logits, labels, sequence_reduce="mean")
                batch_losses.append(loss)

        # Restore tokenizer padding_side
        if prev_padding_side is not None and hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = prev_padding_side

        # Final reduction over the batch (keep your robust checks)
        if not batch_losses:
            return torch.zeros((), device=device, dtype=original_logits.dtype)

        # Ensure scalars
        scalars = []
        for t in batch_losses:
            if t.numel() == 0:
                continue
            scalars.append(t if t.dim() == 0 else t.mean())

        if not scalars:
            return torch.zeros((), device=device, dtype=original_logits.dtype)

        return torch.stack(scalars).mean()


            
    def fit(self, args):
        # Initialize masked data based on mode
        if self.dynamic_mode:
            # Dynamic mode: Skip initial masking - will select questions based on weights after 5000 steps
            self.masked_indices = []
            self.masked_questions = []
            self.masked_responses = []
            self.masked_log_probs = []
            self.masked_generated_labels = []
            print("Dynamic mode: Will select questions based on current weights after 5000 steps")
        else:
            # Fixed mode: Initialize masked questions and generate responses immediately
            print("Fixed mode: Starting initialization of masked questions and generating responses...", flush=True)
            self.initialize_masked_data()
            print(f"Fixed mode: Initialization complete. {len(self.masked_questions)} questions selected and {self.online_sample_size} responses generated per question.", flush=True)
        
        self.effective_dataset_size = len(self.new_dataloader.dataset)
        
        # Update p scaling for effective dataset size
        if hasattr(self.p, 'norma'):
            self.p.norma = len(self.new_dataloader.dataset)  # Use original dataset size
            print(f"Updated p scaling to original dataset size: {len(self.new_dataloader.dataset)}")
            print(f"Updated p scaling to effective dataset size: {self.effective_dataset_size}")
        
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()
        if args.save_steps == -1:
            args.save_steps = float("inf")

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
                "masked_samples_count": len(self.masked_indices),
                "online_sample_ratio": self.online_sample_ratio,
            }
            
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
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)
                self.new_dataloader.sampler.set_epoch(epoch)
            if epoch>=1:
                self.ul_weight -= self.ul_weight_decay
                print("\n UL weight now", self.ul_weight, end="\n")
            
            if self.start_time is None:
                self.start_time = time.time()
            # Initialize running averages once at the start of training
            if epoch == 0:
                loss_mean = 0
                gpt_loss_mean = 0
                weighted_gpt_loss_mean = 0

            
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            self.p.train()
            epoch_step = 0
            
            # Check if obj_1 is DPO
            obj_1_type = getattr(self.args, 'obj_1', 'SFT')
            is_dpo = (obj_1_type == "DPO")
            
            for data_1, data_2 in zip(self.train_dataloader, self.new_dataloader):
                # Update masked responses with delayed generation
                self.update_masked_responses(global_step)
                
                # Handle obj_1 (DPO or SFT)
                if is_dpo:
                    # DPO format: (chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens)
                    chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data_1
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                    
                    chosen_logps, rejected_logps, aux_loss = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )
                    with torch.no_grad():
                        reference_chosen_logps, reference_rejected_logps, _ = self.concatenated_forward(
                            self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                        )
                    
                    preference_loss, _, _ = self.dpo_loss_fn(
                        chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
                    )
                    if not self.aux_loss:
                        aux_loss = 0
                    # Upper level loss (Dataset 1) - DPO
                    gpt_loss = preference_loss + aux_loss * getattr(self.args, 'aux_loss_coef_1', 0.0)
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
                    
                    # Upper level loss (Dataset 1) - pure offline SFT
                    gpt_loss = self.batch_loss_fn(output.logits, labels, sequence_reduce="mean").mean(0)
                
                # Forward pass for offline data (Dataset 2) - this will be mixed with online data
                ide, new_prompts_id_len, new_input, new_attention_masks, _ = data_2
                new_inputs = new_input.squeeze(1).to(torch.cuda.current_device())
                new_attention_mask = new_attention_masks.squeeze(1).to(torch.cuda.current_device())
                new_output = self.model(new_inputs, attention_mask=new_attention_mask, return_output=True)
                
                with torch.no_grad():
                    batch_weights = self.p()[ide]

                # Compute offline losses for dataset 2
                new_labels = torch.where(
                    new_attention_mask.bool(),
                    new_inputs,
                    self.batch_loss_fn.IGNORE_INDEX,
                )
                
                for new_label, new_source_len in zip(new_labels, new_prompts_id_len):
                    new_label[:new_source_len] = self.batch_loss_fn.IGNORE_INDEX
                
                # Lower level loss (Dataset 2) - mixed offline + online with proper weighting
                # CRITICAL: Compute loss on the actual mixed batch (original + generated responses)
                batch_gpt_loss = self.compute_mixed_batch_loss_with_stop_grad(
                    ide.cpu().numpy(),  # batch indices
                    new_output.logits,  # original logits
                    new_labels,         # original labels
                    [self.new_dataloader.dataset[idx][4]["input"] for idx in ide.cpu().numpy()],  # original questions
                )
                
                
                # Apply proper weighting: non-masked samples use only data weights,
                # masked samples use both data weights and importance weights (with stop grad)
                # Apply learnable data weights to the mixed batch loss
                # Ensure both tensors are on GPU before multiplication
                batch_weights = batch_weights.to(torch.cuda.current_device())
                batch_gpt_loss = batch_gpt_loss.to(torch.cuda.current_device())
                
                # Gated loss: use top-k selection for model updates, full weights for data weight updates
                filter_rate = getattr(self.args, 'filter_rate', 0.1)
                k = max(1, int(len(batch_weights) * (1 - filter_rate)))  # Keep top (1-filter_rate) samples
                
                # Get top-k indices based on weights (highest weights)
                _, top_k_indices = torch.topk(batch_weights, k, largest=True)
                
                # Create gated weights: 1 for top-k, 0 for others
                gated_weights = torch.zeros_like(batch_weights)
                gated_weights[top_k_indices] = 1.0
                
                # Use gated weights for model updates (discard bottom filter_rate)
                gated_weighted_gpt_loss = (gated_weights * batch_gpt_loss).mean(0)
                total_loss = self.ul_weight * gpt_loss + (1-self.ul_weight) * gated_weighted_gpt_loss + aux_loss * self.args.aux_loss_coef_2
                
                # Keep original weighted loss for logging
                weighted_gpt_loss = (batch_weights * batch_gpt_loss).mean(0)
                
                # Backward pass for model
                self.strategy.backward(total_loss, self.model, self.optimizer) 
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # Backward pass for data selection weights (offline)
                p = self.p()                   
                p = p.to(torch.cuda.current_device())
                
                # Lower-level loss: sum of batch_gpt_loss for unmasked + batch_gpt_loss for masked (generated)
                # Use full weights (unfiltered) for data weight updates
                selector_loss = (p[ide] * batch_gpt_loss.detach()).mean()
                selector_loss.backward()

                # Step optimizer and scheduler for offline data selection
                self.p_opt.step()
                self.p_scheduler.step()
                self.p_opt.zero_grad()
                
                # Track losses for analysis
                with torch.no_grad():
                    self.batch_losses_history.append(batch_gpt_loss.detach().cpu())
                    self.selection_weights_history.append(batch_weights.detach().cpu())
                    current_time = time.time()
                    timestamp = current_time - self.start_time
                    self.timestamps_history.append(timestamp)
                
                # Update moving averages
                gpt_loss_mean = gpt_loss_mean * 0.95 + 0.05 * gpt_loss.item()
                # Use gated weighted loss for the moving average (keep same log name)
                weighted_gpt_loss_val = gated_weighted_gpt_loss.item() if torch.is_tensor(gated_weighted_gpt_loss) else float(gated_weighted_gpt_loss)
                weighted_gpt_loss_mean = weighted_gpt_loss_mean * 0.95 + 0.05 * weighted_gpt_loss_val
                loss_mean = loss_mean * 0.95 + 0.05 * total_loss.item()

                # Logging
                logs_dict = {
                    "upper_loss": gpt_loss.item(), 
                    "upper_loss_mean": gpt_loss_mean, 
                    "weighted_gpt_loss": gated_weighted_gpt_loss.item(),  # Use gated loss but keep same name
                    "weighted_gpt_loss_mean": weighted_gpt_loss_mean,  # This now tracks gated loss mean
                    "filter_rate": filter_rate,
                    "k_selected": k,
                    "total_loss": total_loss.item(),
                    "loss_mean": loss_mean,
                    "batch_loss": batch_gpt_loss.mean().item(),
                    "selection_weight_mean": batch_weights.mean().item(),
                    "selection_weight_std": batch_weights.std().item(),
                    "iteration": global_step,
                    "time": timestamp,
                    "masked_samples_count": len(self.masked_indices),
                    "online_generation_step": self.last_generation_step,
                }
                
                # DEBUG: Print importance weight summary every 10 steps (disabled for cleaner output)
                # if global_step % 10 == 0:
                #     print(f"\n=== IMPORTANCE WEIGHT SUMMARY (Step {global_step}) ===")
                #     print(f"Masked samples: {len(self.masked_indices)}")
                #     print(f"Online sample size: {self.online_sample_size}")
                #     print(f"Frequency compensation factor: 1/{self.online_sample_size} = {1.0/self.online_sample_size:.4f}")
                #     print("Note: Check individual DEBUG lines above for actual importance weights")
                #     print("Look for patterns in current_log_prob vs old_log_prob differences")
                #     print("=== END SUMMARY ===\n")

                if self.aux_loss:
                    aux_loss_val = aux_loss.item() if hasattr(aux_loss, 'item') else aux_loss
                    logs_dict["aux_loss"] = aux_loss_val

                # logs/checkpoints/evaluation
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, gpt_loss_mean)

                step_bar.update()
                global_step += self.strategy.world_size
                epoch_step += 1
                
            if self.strategy.is_rank_0():
                p_name = args.save_path + "/" + args.selector_name+"_"+ args.selector_activation \
                    +"_ep"+str(epoch+1)+".pt"
                os.makedirs(os.path.dirname(p_name), exist_ok=True)
                torch.save(self.p.logits, p_name)
                print(self.p.logits)
                
            epoch_bar.update()

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

            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        if ((global_step+1) // self.strategy.world_size-1) % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        if ((global_step+1) // self.strategy.world_size-1) % args.save_steps == 0:
            new_ckpt_path = os.path.join(args.ckpt_path, f"bdr_online")
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
                    # DPO format: (chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens)
                    chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                    
                    chosen_logps, rejected_logps, _ = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )
                    reference_chosen_logps, reference_rejected_logps, _ = self.concatenated_forward(
                        self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )
                    
                    preference_loss, _, _ = self.dpo_loss_fn(
                        chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
                    )
                    loss = preference_loss
                else:
                    # SFT format: (prompts_id_len, inputs, attention_masks, _)
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
                self._wandb.log(logs)
        self.model.train()
