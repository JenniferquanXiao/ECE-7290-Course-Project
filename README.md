# BIPOST

A computationally efficient framework for LLM post-training involving bi-objective and bilevel fine-tuning.

## Example Scripts

We provide two ready-to-use training scripts for common use cases:

### 1. Bilevel Data Selection with Open-Orca and Alpaca (`run_llama3_8b_instruct_sv001_scratch.sh`)

This script demonstrates bilevel data selection training using:
- **Dataset 1**: Open-Orca/OpenOrca (HuggingFace dataset)
- **Dataset 2**: yahma/alpaca-cleaned (HuggingFace dataset)
- **Model**: meta-llama/Meta-Llama-3-8B-Instruct
- **Supported algorithms**: BDR, BDR_Online

#### Basic Usage

```bash
cd examples/scripts
bash run_llama3_8b_instruct_sv001_scratch.sh
```

#### Customization Examples

**Run with BDR_Online algorithm:**
```bash
bash run_llama3_8b_instruct_sv001_scratch.sh --selector_mode BDR_Online --online_sample_ratio 0.2
```

**Adjust training parameters:**
```bash
bash run_llama3_8b_instruct_sv001_scratch.sh \
    --train_batch_size 16 \
    --learning_rate 1e-5 \
    --max_epochs 5 \
    --max_samples_1 100000 \
    --max_samples_2 100000
```

**Disable Wandb logging:**
```bash
bash run_llama3_8b_instruct_sv001_scratch.sh --no_wandb
```

#### Available Options

Run `bash run_llama3_8b_instruct_sv001_scratch.sh --help` to see all available options including:
- `--selector_mode`: Choose algorithm (BDR, BDR_Online)
- `--online_sample_ratio`: For BDR_Online, ratio of online samples (default: 0.1)
- `--online_generation_freq`: Generation frequency for BDR_Online (default: 1000)
- `--train_batch_size`: Global training batch size (default: 32)
- `--learning_rate`: Learning rate (default: 5e-6)
- `--max_epochs`: Maximum training epochs (default: 3)
- And many more...

**Note**: This script is configured for a server with scratch space (`/scratch/qx232/.cache/`). You may need to modify the cache directory paths in the script for your environment.

---

### 2. Bilevel Data Selection with BlueOrca and RedOrca (`run_llama3_8b_blueorca_redorca.sh`)

This script demonstrates bilevel data selection training using local JSONL datasets:
- **Dataset 1**: `./examples/datasets/BlueOrca/train.jsonl`
- **Dataset 2**: `./examples/datasets/RedOrca/train.jsonl`
- **Model**: meta-llama/Meta-Llama-3-8B-Instruct
- **Supported algorithms**: BDR, BDR_Online

#### Basic Usage

```bash
cd examples/scripts
bash run_llama3_8b_blueorca_redorca.sh
```

#### Customization Examples

**Run with BDR algorithm:**
```bash
bash run_llama3_8b_blueorca_redorca.sh --selector_mode BDR --filter_rate 0.1
```

**Run with BDR_Online and dynamic mode:**
```bash
bash run_llama3_8b_blueorca_redorca.sh \
    --selector_mode BDR_Online \
    --dynamic_mode \
    --online_sample_ratio 0.1 \
    --filter_rate 0.1
```

#### Available Options

Run `bash run_llama3_8b_blueorca_redorca.sh --help` to see all available options including:
- `--selector_mode`: Choose algorithm (BDR, BDR_Online)
- `--filter_rate`: Filter rate for gated loss in BDR/BDR_Online (default: 0.1)
- `--dynamic_mode`: Enable dynamic mode for BDR_Online (selects questions based on current weights)
- `--online_sample_ratio`: For BDR_Online, ratio of online samples (default: 0.1)
- And many more...

#### Dataset Format

The script expects JSONL files with the following format:
```json
{"question": "Your question here", "response": "Your response here"}
```

Make sure your datasets are located at:
- `./examples/datasets/BlueOrca/train.jsonl`
- `./examples/datasets/RedOrca/train.jsonl`

Or modify the `DATASET_1` and `DATASET_2` variables in the script to point to your dataset paths.

---

### Common Use Cases

#### Comparing Different Algorithms

To compare BDR and BDR_Online on the same datasets:

```bash
# Run BDR
bash run_llama3_8b_blueorca_redorca.sh --selector_mode BDR --wandb_run_name comparison_bdr

# Run BDR_Online
bash run_llama3_8b_blueorca_redorca.sh --selector_mode BDR_Online --wandb_run_name comparison_bdr_online
```

#### Hyperparameter Tuning

For BDR_Online, you can tune online generation parameters:

```bash
# More frequent online generation
bash run_llama3_8b_blueorca_redorca.sh \
    --selector_mode BDR_Online \
    --online_generation_freq 250 \
    --online_sample_ratio 0.2

# Larger online sample size
bash run_llama3_8b_blueorca_redorca.sh \
    --selector_mode BDR_Online \
    --online_sample_size 5 \
    --online_sample_ratio 0.15
```

#### Using Your Own Datasets

1. Prepare your datasets in JSONL format:
   ```json
   {"question": "...", "response": "..."}
   ```

2. Modify the script to point to your datasets:
   ```bash
   DATASET_1="./path/to/your/dataset1.jsonl"
   DATASET_2="./path/to/your/dataset2.jsonl"
   ```

3. Update input/output keys if your dataset uses different field names:
   ```bash
   INPUT_KEY="your_question_field"
   OUTPUT_KEY="your_response_field"
   ```

