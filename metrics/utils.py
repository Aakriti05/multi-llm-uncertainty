import os
import torch
import orjson

from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from transformers import PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

huggingface_token = os.getenv("HF_TOKEN")

def transform_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Transform metrics such that a larger metric implies a better answer.
    """
    return {
        "log_likelihood": metrics["log_likelihood"],
        "gini_impurity": 1.0 - metrics["gini_impurity"],
        "entropy": -metrics["entropy"],
        "kl_score": metrics["kl_score"],
    }

def average_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Average metrics across multiple models.
    """
    keys = metrics[0].keys()
    return {key: sum([metric[key] for metric in metrics]) / len(metrics) for key in keys}

def load_model_and_tokenizer(model_name, device_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        # attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_cache=True,
        cache_dir="./cache",
        use_auth_token=huggingface_token,
    ).to(device_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_auth_token=huggingface_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def get_prompt_completion_pairs(
    results_folder: str,
    filter_jsons_with_metrics: bool = False
):
    """
    Collect prompt–completion pairs from JSON files.

    Faster than the original by:
      • reading bytes (no text decode) + orjson parsing
      • concurrent, batched I/O
    """
    paths = list(Path(results_folder).glob("*.json"))
    if not paths:
        return {"prompt": [], "completion": [], "file_key": [], "completion_idx": []}

    def _process(path: Path):
        try:
            data = orjson.loads(path.read_bytes())
        except Exception as e:            # malformed file? just skip
            print(f"[warn] {path.name}: {e}")
            return None

        file_key = path.stem
        prompt   = data["prompt"]
        chat_history = data["agent_context"]
        # build once, emit list of rows for this file
        if filter_jsons_with_metrics and "metrics" in data:
            return None
        return [
            (prompt, c, chat_history, file_key, i)
            for i, c in enumerate(data["completions"])
        ]

    # Gather rows in parallel (chunksize=32 cuts scheduler overhead)
    rows = []
    with ThreadPoolExecutor() as pool:
        for res in tqdm(pool.map(_process, paths, chunksize=32),
                        total=len(paths),
                        desc="Loading JSON files",
                        unit="file"):
            if res:
                rows.extend(res)

    # Transpose rows -> dict-of-lists
    if not rows:
        return {"prompt": [], "completion": [], "chat_history": [], "file_key": [], "completion_idx": []}

    prompt, completion, chat_history, file_key, completion_idx = zip(*rows)
    return {
        "prompt":          list(prompt),
        "completion":      list(completion),
        "chat_history":    list(chat_history),
        "file_key":        list(file_key),
        "completion_idx":  list(completion_idx),
    }

def collate_fn(batch: List[Dict[str, str]], tokenizer: PreTrainedTokenizer) -> Dict[str, torch.Tensor]:
    """
    Collate function to batch prompt/completion pairs.
    Produces `input_ids` by concatenating prompt + full completion,
    and keeps `completion_ids` for evaluation.
    Returns lengths for separation.
    """
    prompts = [item["prompt"] for item in batch]
    chat_histories = [item["chat_history"] for item in batch]
    completions = [item["completion"] for item in batch]
    file_keys = [item["file_key"] for item in batch]
    completion_idxs = [item["completion_idx"] for item in batch]
    # Tokenize prompts and completions separately
    prompts = [tokenizer.apply_chat_template(chat_history, tokenize=False, add_special_tokens=False, add_generation_prompt=True) for chat_history in chat_histories]
    prompt_encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
    comp_encodings = tokenizer(completions, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)

    prompt_ids = prompt_encodings["input_ids"]  # (batch, max_prompt_len)
    comp_ids = comp_encodings["input_ids"]      # (batch, max_comp_len)
    comp_mask = comp_encodings["attention_mask"]

    # Compute actual lengths from attention masks
    prompt_lengths = prompt_encodings["attention_mask"].sum(dim=1).tolist()
    comp_lengths = comp_mask.sum(dim=1).tolist()

    # Build combined inputs: [prompt_tokens, all completion tokens]
    input_seqs, attention_seqs = [], []
    for i in range(len(batch)):
        p_len = prompt_lengths[i]
        c_len = comp_lengths[i]
        seq = torch.cat([prompt_ids[i, :p_len], comp_ids[i, :c_len], torch.tensor([tokenizer.eos_token_id])], dim=0)
        mask = torch.ones_like(seq)
        input_seqs.append(seq)
        attention_seqs.append(mask)

    # Pad combined sequences to same length
    max_seq_len = max(seq.size(0) for seq in input_seqs)
    batch_input_ids = torch.full((len(batch), max_seq_len), tokenizer.pad_token_id, dtype=torch.long)
    batch_attention_mask = torch.zeros_like(batch_input_ids)
    for i, seq in enumerate(input_seqs):
        length = seq.size(0)
        batch_input_ids[i, :length] = seq
        batch_attention_mask[i, :length] = attention_seqs[i]

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "completion_ids": comp_ids,
        "prompt_lengths": prompt_lengths,
        "comp_lengths": comp_lengths,
        "file_keys": file_keys,
        "completion_idxs": completion_idxs
    }

def model_tokenizer_generator(model_names: List[str],
                              devices: str = "cuda"):
    for model_name, device in zip(model_names, devices):
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        yield model, tokenizer