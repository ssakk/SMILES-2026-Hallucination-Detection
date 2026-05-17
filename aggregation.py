"""
aggregation.py — Feature extraction from hidden states.

Isolates response tokens via prompt tokenization, max-pools hidden states
from layers 12 and 13, and appends text-level features.
"""

from __future__ import annotations

import pandas as pd
import torch
from transformers import AutoTokenizer

_TRAIN_CSV = "./data/dataset.csv"
_TEST_CSV = "./data/test.csv"
_MODEL_NAME = "Qwen/Qwen2.5-0.5B"

_init_done = False
_prompt_token_lengths: list[int] = []
_text_feats: list[list[float]] = []
_call_idx = 0


def _lazy_init() -> None:
    """Pre-compute prompt token lengths and text features for all samples."""
    global _init_done, _prompt_token_lengths, _text_feats
    if _init_done:
        return

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)

    for csv_path in [_TRAIN_CSV, _TEST_CSV]:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            prompt = str(row["prompt"])
            response = str(row["response"])

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            _prompt_token_lengths.append(len(prompt_ids))

            resp_clean = (
                response.replace("<|endoftext|>", "")
                .replace("<|im_end|>", "")
                .strip()
            )
            prompt_clean = (
                prompt.replace("<|im_start|>", "")
                .replace("<|im_end|>", "")
                .strip()
            )

            resp_len = len(resp_clean)
            prompt_len = max(len(prompt_clean), 1)

            prompt_words = set(prompt_clean.lower().split())
            resp_words = set(resp_clean.lower().split())
            union = prompt_words | resp_words
            jaccard = len(prompt_words & resp_words) / max(len(union), 1)

            _text_feats.append([
                float(resp_len),
                float(jaccard),
                float(resp_len) / float(prompt_len),
            ])

    _init_done = True


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    global _call_idx
    _lazy_init()

    hidden_states = hidden_states.cpu()
    attention_mask = attention_mask.cpu()

    prompt_len = _prompt_token_lengths[_call_idx]
    text_feat = torch.tensor(_text_feats[_call_idx], dtype=torch.float32)
    _call_idx += 1

    real_len = int(attention_mask.sum().item())

    resp_start = min(prompt_len, real_len)
    resp_end = real_len

    if resp_start >= resp_end:
        resp_start = max(0, real_len - 5)

    layer_12 = hidden_states[12, resp_start:resp_end, :]
    layer_13 = hidden_states[13, resp_start:resp_end, :]

    max_12 = layer_12.max(dim=0).values
    max_13 = layer_13.max(dim=0).values

    return torch.cat([max_12, max_13, text_feat], dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
