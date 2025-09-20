# Uncertainty-Aware Answer Selection for Improved Reasoning in Multi-LLM Systems

This repository contains the official code for the paper “Uncertainty-Aware Answer Selection for Improved Reasoning in Multi-LLM Systems.” It provides tools to:

- Generate multi-round, multi-model debate trajectories
- Score each completion with uncertainty metrics (log-likelihood, entropy, Gini, KL)
- Evaluate answer selection accuracy across rounds and baselines

Supported datasets: GSM8K, MMLU, ARC.

## Environment Setup

- Python 3.10+
- PyTorch (with CUDA if available)
- Hugging Face `transformers`

Set your Hugging Face token in the environment if models require authentication:

```bash
export HF_TOKEN=your_hf_token
```

Optional: set cache directory and CUDA device visibility as usual.

## Data

- GSM8K: a sample test file is provided at `gsm/data/test.json` (JSONL-style list of {question, answer}).
- MMLU: loaded via Hugging Face datasets (`cais/mmlu`), choose a subset (e.g., `abstract_algebra`).
- ARC: loaded via `allenai/ai2_arc` with split `ARC-Challenge`.

## 1) Generate Debate Trajectories

Script: `multi_llm_debate.py`

This script runs a multi-round debate across multiple LLMs and writes per-question JSON files named `{round}_{modelIdx}_{questionIdx}.json` that include the prompt, completions, token-level likelihoods for the generated sequences, and metadata. It also writes summary files for logging.

Key arguments:
- `--num_models` (int, default 3): number of LLMs (agents).
- `--rounds` (int, default 3): debate rounds.
- `--dataset_name` (str, default `gsm`): one of `gsm`, `mmlu`, `ARC`.
- `--dataset_path` (str): path to GSM8K JSONL or MMLU subset name when `dataset_name=mmlu`.
- `--device` (str, default `cuda`): `cuda` or `cpu`.
- `--num_samples` (int, default 1): number of samples per model per round.
- `--selfref` (bool, default False): allow self-reference in subsequent rounds.
- `--max_prob_element` (bool, default False): pass forward only the highest-probability agent context from the previous round.
- `--unique_response_only` (bool, default False): filter duplicate sampled responses per agent.

Notes:
- Model names are currently set to three instruction models: Qwen2.5 7B, Ministral 8B, Llama 3.1 8B. All are loaded on the specified device(s). Update the list in `multi_llm_debate.py` as needed.
- When `device=cuda`, the sample code maps all agents to `cuda:0`. Adjust to span multiple GPUs if desired.

Examples:
```bash
# GSM8K sample
python multi_llm_debate.py \
  --dataset_name gsm \
  --dataset_path gsm/data/test.json \
  --num_models 3 \
  --rounds 3 \
  --num_samples 1 \
  --device cuda

# MMLU (subset example)
python multi_llm_debate.py \
  --dataset_name mmlu \
  --dataset_path abstract_algebra \
  --num_models 3 \
  --rounds 1 \
  --num_samples 1

# ARC-Challenge
python multi_llm_debate.py \
  --dataset_name ARC \
  --num_models 3 \
  --rounds 1
```

Outputs (per question and per (round, model)) go into a folder named like:
- `{dataset}_{num_models}_{rounds}_{num_samples}_diff_LLM_allresults/` (e.g., `gsm_3_3_1_diff_LLM_allresults/`)
- Inside, files: `{round}_{modelIdx}_{questionIdx}.json`

Each JSON contains fields like: `prompt`, `agent_context`, `completions`, `gt`, `answer`, and a per-agent block with `log_likelihoods` and `tokens_list` for the agent’s generated samples.

## 2) Compute Uncertainty Metrics

Script: `metrics/main.py`

This step loads the JSON completions and computes uncertainty metrics per completion using one or more models (the same three by default), then writes metrics back to JSONs in-place or into a `calibrated` subfolder.

Key arguments:
- `--results_folder` (str): folder containing debate JSONs produced by step 1.
- `--device` (str, default `cuda`): `cuda` or `cpu`.
- `--num_models` (int, default 3): number of models used to score metrics.
- `--use_single_gpu` (flag): run models sequentially on a single GPU; otherwise, parallelize across devices.
- `--no_length_normalize_log_likelihood` (flag): by default we length-normalize token log-likelihoods; pass this flag to disable normalization and use sum.
- `--batch_size` (int, default 4): scoring batch size.
- `--num_workers` (int, default 24): dataloader workers (will be divided across models when parallelized).
- `--replace_in_place` (flag): write metrics back into `results_folder` instead of the default `calibrated/` subfolder.

What’s computed (per completion):
- `log_likelihood` (mean or sum over generated tokens)
- `entropy` (mean over tokens)
- `gini_impurity` (mean over tokens; note evaluation uses `1 - gini` so higher is better)
- `kl_score` (mean over tokens vs. uniform baseline)

Example:
```bash
python -m metrics.main \
  --results_folder gsm_3_3_1_diff_LLM_allresults \
  --device cuda \
  --num_models 3 \
  --batch_size 4 \
  --num_workers 24
```

Output:
- Writes `metrics` back into each JSON file. By default saved to `{results_folder}/calibrated/`. Use `--replace_in_place` to overwrite originals.

## 3) Evaluate Accuracy

Script: `eval.py`

This script aggregates ground-truth answers, completions (by round/model), and attached metrics to compute accuracy under different selection strategies and baselines.

Key arguments:
- `--json_folder_path` (str, default `gsm/results/`): folder with the debate JSONs (use the calibrated folder if you computed metrics there).
- `--eval_type` (str, default `debate`): one of:
  - `debate`: use the first completion per model per round; select answer per round using the provided metric.
  - `baseline-1`: use all completions from round 0 for each model.
  - `baseline-2`: flatten all completions from all models per round and evaluate jointly.
- `--calibration` (flag): average metrics across models for each completion before selection.
- `--num_completions` (int, default -1): limit the number of completions used per model; `-1` uses all.
- `--random_choice` (flag): enable random tie-breaking where applicable.
- `--metric_only` (flag): use metric scores directly for selection (ignore frequency heuristic).
- `--dataset_name` (str, default `gsm`): passed through for parsing answers (GSM vs MMLU/ARC differ in parsing).
- `--seed` (int, default 0): randomness control for tie-breaking.

Example (evaluate debate with calibrated metrics):
```bash
python eval.py \
  --json_folder_path gsm_3_3_1_diff_LLM_allresults/calibrated \
  --eval_type debate \
  --dataset_name gsm \
  --calibration \
  --num_completions -1
```

Outputs:
- `accuracy_question_level_{calibration}_{random_choice}_{metric_only}_{eval_type}_{num_completions}.csv`
- `accuracy_metric_level_{calibration}_{random_choice}_{metric_only}_{eval_type}_{num_completions}.csv`

The question-level file logs per-question, per-round correctness for each metric; the metric-level file aggregates mean accuracy per metric and round.

## File/Folder Conventions

- Debate JSONs are named `{round}_{modelIdx}_{questionIdx}.json`.
- Metrics are attached under the `metrics` field in each JSON as a list over models and completions: `metrics[model_idx][completion_idx] -> metric_dict`.
- When `--calibration` is used in evaluation, metrics are averaged across models for each completion.

## Tips / Troubleshooting

- Ensure `HF_TOKEN` is set if models require gated access.
- If using a single GPU, prefer `--use_single_gpu` in `metrics/main.py` to avoid OOM.
- Adjust `--batch_size`/`--num_workers` based on GPU/CPU memory.
- For multi-GPU scoring, ensure the device list in `metrics/main.py` maps models to distinct `cuda:{i}` devices.

## Citation

If you use this code, please cite the paper:

```
@inproceedings{uncertainty_multi_llm,
  title={Uncertainty-Aware Answer Selection for Improved Reasoning in Multi-LLM Systems},
  author={Agrawal, Aakriti and Aralikatti, Rohith and Satheesh, Anirudh and Chakraborty, Souradip and Bedi, Amrit Singh and Huang, Furong},
  booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
  year={2025}
}
```
