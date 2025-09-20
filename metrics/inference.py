import torch
import math

from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

from metrics.dataset import PromptCompletionDataset
from metrics.utils import collate_fn

def compute_metrics(
    json_list: List[Dict[str, Any]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    device: torch.device = None,
    length_normalize_log_likelihood: bool = False,
    num_workers: int = 8
) -> List[Dict[str, float|List[float]]]:
    """
    For each prompt-completion pair, returns:
      - `tokens`: list of token ids
      - `log_likelihoods`: average log likelihood of each token
      - `gini_impurities`: average gini impurity of each token
      - `entropies`: average entropy of each token
      - `kl_scores`: average kl score of each token

    Now slices from (p_len-1) so that each logit's distribution corresponds exactly to the
    generation step of each completion token.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    dataset = PromptCompletionDataset(json_list)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=num_workers
    )

    metrics = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting logits"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            comp_ids = batch["completion_ids"].to(device)
            prompt_lengths = batch["prompt_lengths"]
            comp_lengths = batch["comp_lengths"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (batch, seq_len, vocab_size)

            for i in range(input_ids.size(0)):
                metric = {}
                p_len = prompt_lengths[i]
                c_len = comp_lengths[i]
                # Because HF causal-LM logits at position t predict token at input_ids[t+1],
                # the first completion token (at full sequence index p_len) is predicted by logits at index p_len-1
                start = p_len - 1
                end = start + c_len
                seq_logits = logits[i, start:end, :]
                token_ids = comp_ids[i, :c_len]
                token_probs = torch.nn.functional.softmax(seq_logits, dim=-1)
                token_log_probs = torch.log_softmax(seq_logits, dim=-1)
                if length_normalize_log_likelihood:
                    log_likelihood = token_log_probs[torch.arange(c_len), token_ids].mean().item()
                else:
                    log_likelihood = token_log_probs[torch.arange(c_len), token_ids].sum().item()
                gini_impurity = 1.0 - torch.sum(token_probs ** 2, dim=-1).mean().item()
                entropy = -torch.sum(token_probs * token_log_probs, dim=-1).mean().item()
                kl_score = -math.log(tokenizer.vocab_size) - 1.0/float(tokenizer.vocab_size) * torch.sum(token_log_probs, dim=-1).mean().item()
                metric = {
                    "log_likelihood": log_likelihood,
                    "gini_impurity": gini_impurity,
                    "entropy": entropy,
                    "kl_score": kl_score
                }
                metrics[(batch["file_keys"][i],batch["completion_idxs"][i])] = metric

    return metrics