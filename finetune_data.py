#!/usr/bin/env python3
"""
Prepare LoRA SFT data for Gemma-3-4b-it from:
- dmis-lab/meerkat-instructions / ChatDoctor-cleaned (main single-turn medical consult set)
- stellalisy/MediQ_AskDocs           (follow-up question behavior)
- medalpaca/medical_meadow_mediqa    (small diverse-case supplement)
- dmis-lab/meerkat-instructions / MedQA-dialog (multi-turn supplement)

Outputs:
- train_unique.jsonl : deduplicated unique train pool before stage-specific resampling
- train_stage1.jsonl : main-stage mixture (domain adaptation / broad medical QA)
- train_stage2.jsonl : dialogue-refinement mixture (follow-up + multi-turn emphasis)
- val.jsonl / test.jsonl
- stats.json

Output example:
{
  "id": "...",
  "source": "chatdoctor_cleaned|mediq_askdocs|medical_meadow_mediqa|medqa_dialog",
  "task": "single_turn_consult|ask_followup|single_turn_reference|multi_turn_consult",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {...}
}

Recommended usage:
1) Train stage 1 on train_stage1.jsonl.
2) Continue the SAME LoRA adapter for stage 2 on train_stage2.jsonl.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import Dataset, load_dataset
from tqdm import tqdm


ASK_FOLLOWUP_SYSTEM = (
    "You are a careful medical dialogue assistant. Before making a clinical "
    "judgment, ask the single most useful follow-up question that would reduce "
    "uncertainty. Keep the question concise, clinically relevant, and focused on "
    "information that could change triage, differential diagnosis, or next steps."
)

CONSULT_SYSTEM = (
    "You are a careful medical dialogue assistant. Use the conversation context "
    "to respond helpfully and cautiously. When information is limited, avoid "
    "overconfidence, mention important red flags when appropriate, and encourage "
    "urgent in-person care for emergency symptoms."
)

DOCTORISH_USER_PATTERNS = [
    r"^hi[,!.\s]*may i answer your health queries",
    r"^may i answer your health queries",
    r"^please type your question",
    r"^thanks? for posting your query",
    r"^thank you for posting your query",
    r"^i have gone through your query",
    r"^i can understand your concern",
    r"^welcome to chat doctor",
    r"^hello[,!.\s]*i am dr\b",
]

MEDMEADOW_TAIL_PATTERNS = [
    r"updated by:.*$",
    r"review provided by.*$",
    r"also reviewed by.*$",
    r"doi:\s*10\.\S+.*$",
    r"n engl j med\..*$",
]


def parse_args() -> argparse.Namespace:
    """Parse dataset names, split sizes, mixture targets, and filtering thresholds."""
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--meerkat_name", type=str, default="dmis-lab/meerkat-instructions")
    p.add_argument("--chatdoctor_config", type=str, default="ChatDoctor-cleaned")
    p.add_argument("--medqa_dialog_config", type=str, default="MedQA-dialog")
    p.add_argument("--mediq_name", type=str, default="stellalisy/MediQ_AskDocs")
    p.add_argument("--medmeadow_name", type=str, default="medalpaca/medical_meadow_mediqa")

    p.add_argument("--chatdoctor_val_size", type=int, default=1500)
    p.add_argument("--chatdoctor_test_size", type=int, default=1500)
    p.add_argument("--medqa_dialog_val_size", type=int, default=500)
    p.add_argument("--medqa_dialog_test_size", type=int, default=500)
    p.add_argument("--medmeadow_val_size", type=int, default=200)
    p.add_argument("--medmeadow_test_size", type=int, default=200)

    # Stage 1: broad medical adaptation. Keep ChatDoctor dominant.
    p.add_argument("--stage1_chatdoctor_target", type=int, default=90000)
    p.add_argument("--stage1_mediq_target", type=int, default=18000)
    p.add_argument("--stage1_medmeadow_target", type=int, default=4000)
    p.add_argument("--stage1_medqa_dialog_target", type=int, default=12000)

    # Stage 2: shorter continuation focused on multi-turn & asking behavior.
    p.add_argument("--stage2_chatdoctor_target", type=int, default=6000)
    p.add_argument("--stage2_mediq_target", type=int, default=12000)
    p.add_argument("--stage2_medmeadow_target", type=int, default=600)
    p.add_argument("--stage2_medqa_dialog_target", type=int, default=10000)

    p.add_argument("--max_messages", type=int, default=12)
    p.add_argument("--min_chatdoctor_user_chars", type=int, default=20)
    p.add_argument("--min_assistant_chars", type=int, default=20)
    p.add_argument("--min_multi_turn_messages", type=int, default=4)
    p.add_argument("--max_medmeadow_output_chars", type=int, default=4500)
    p.add_argument("--max_medmeadow_input_chars", type=int, default=15000)
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    """Create an output directory and its parents if needed."""
    path.mkdir(parents=True, exist_ok=True)


def clean_text(text: Any) -> str:
    """Normalize whitespace and Unicode variants while preserving paragraph breaks."""
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", " ").replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_hash(text: str) -> str:
    """Create a loose normalized text form for duplicate detection."""
    text = clean_text(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\n ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stable_hash(parts: Sequence[str]) -> str:
    """Build a deterministic SHA-256 hash from multiple text parts."""
    joined = "\n<SEP>\n".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def merge_consecutive_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Merge adjacent messages from the same role after cleaning empty content."""
    merged: List[Dict[str, str]] = []
    for m in messages:
        role = m["role"]
        content = clean_text(m["content"])
        if not content:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] = clean_text(merged[-1]["content"] + "\n\n" + content)
        else:
            merged.append({"role": role, "content": content})
    return merged


def maybe_truncate_messages(messages: List[Dict[str, str]], max_messages: int) -> List[Dict[str, str]]:
    """Keep the most recent turns while ensuring the dialogue does not start with assistant."""
    if len(messages) <= max_messages:
        return messages
    messages = messages[-max_messages:]
    if messages and messages[0]["role"] == "assistant":
        messages = messages[1:]
    return messages


def valid_dialogue(messages: List[Dict[str, str]], min_user_chars: int, min_assistant_chars: int) -> bool:
    """Validate the basic role order and minimum content lengths for a dialogue."""
    if len(messages) < 2:
        return False
    if messages[0]["role"] != "user":
        return False
    if messages[-1]["role"] != "assistant":
        return False
    roles = {m["role"] for m in messages}
    if "user" not in roles or "assistant" not in roles:
        return False
    first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
    last_assistant = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")
    return len(first_user) >= min_user_chars and len(last_assistant) >= min_assistant_chars


def looks_like_provider_boilerplate(text: str) -> bool:
    """Detect provider-introduction boilerplate that should not be treated as a patient query."""
    text = clean_text(text).lower()
    return any(re.search(pat, text) for pat in DOCTORISH_USER_PATTERNS)


def strip_medmeadow_output(text: str, max_chars: int) -> str:
    """Remove citation/review tails from MedMeadow answers and cap answer length."""
    text = clean_text(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    kept: List[str] = []
    for line in lines:
        if any(re.search(pat, line, flags=re.IGNORECASE) for pat in MEDMEADOW_TAIL_PATTERNS):
            break
        kept.append(line)
    text = "\n".join(kept).strip()
    text = re.sub(r"\[[0-9,; ]+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def canonical_user_signature(messages: List[Dict[str, str]]) -> str:
    """Return the normalized first-user text used for cross-source duplicate checks."""
    users = [m["content"] for m in messages if m["role"] == "user"]
    return normalize_for_hash(users[0]) if users else ""


def canonical_full_signature(messages: List[Dict[str, str]]) -> str:
    """Return a role-aware hash for all non-system messages."""
    flat = [f"{m['role']}: {normalize_for_hash(m['content'])}" for m in messages]
    return stable_hash(flat)


def make_example(
    source: str,
    task: str,
    messages: List[Dict[str, str]],
    system_prompt: Optional[str],
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create one SFT example with optional system prompt and deterministic ID."""
    messages = [{"role": m["role"], "content": clean_text(m["content"])} for m in messages]
    if system_prompt:
        messages = [{"role": "system", "content": clean_text(system_prompt)}] + messages
    ex = {
        "source": source,
        "task": task,
        "messages": messages,
        "meta": meta or {},
    }
    ex["id"] = stable_hash([source, task, json.dumps(messages, ensure_ascii=False, sort_keys=True)])[:24]
    return ex


def load_dataset_with_config_fallback(name: str, config: Optional[str]) -> Dict[str, Dataset]:
    """Load a dataset config while trying common case and hyphen/underscore variants."""
    candidates: List[Optional[str]] = []
    if config is not None:
        candidates.append(config)
        for alt in [config.lower(), config.replace("-", "_"), config.lower().replace("-", "_")]:
            if alt not in candidates:
                candidates.append(alt)
    else:
        candidates.append(None)

    last_err: Optional[Exception] = None
    for cfg in candidates:
        try:
            return load_dataset(name, cfg) if cfg is not None else load_dataset(name)
        except Exception as e:  # pragma: no cover - fallback path only
            last_err = e
    assert last_err is not None
    raise last_err


def build_meerkat_messages(
    split_ds: Dataset,
    *,
    source: str,
    task: str,
    system_prompt: str,
    max_messages: int,
    min_user_chars: int,
    min_assistant_chars: int,
    min_messages: int = 2,
    filter_doctorish_first_user: bool = False,
) -> List[Dict[str, Any]]:
    """Convert Meerkat-style message rows into validated SFT dialogue examples."""
    items: List[Dict[str, Any]] = []
    for row in tqdm(split_ds, desc=source):
        msgs = row.get("messages")
        if not isinstance(msgs, list):
            continue
        out: List[Dict[str, str]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = clean_text(m.get("role", "")).lower()
            content = clean_text(m.get("content", ""))
            if role == "system":
                continue
            if role not in {"user", "assistant"} or not content:
                continue
            out.append({"role": role, "content": content})
        out = merge_consecutive_messages(out)
        out = maybe_truncate_messages(out, max_messages)
        if len(out) < min_messages:
            continue
        if not valid_dialogue(out, min_user_chars, min_assistant_chars):
            continue
        if filter_doctorish_first_user and looks_like_provider_boilerplate(out[0]["content"]):
            continue
        meta = {
            "orig_id": row.get("id"),
            "n_messages": len(out),
        }
        items.append(make_example(source, task, out, system_prompt, meta))
    return items


def build_mediq(split_ds: Dataset, args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Build follow-up-question examples from MediQ AskDocs rows."""
    items: List[Dict[str, Any]] = []
    for row in tqdm(split_ds, desc="mediq_askdocs"):
        msgs = row.get("messages")
        out: List[Dict[str, str]] = []
        if isinstance(msgs, list):
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = clean_text(m.get("role", "")).lower()
                content = clean_text(m.get("content", ""))
                if role not in {"user", "assistant"} or not content:
                    continue
                out.append({"role": role, "content": content})
        else:
            context = clean_text(row.get("context", ""))
            question = clean_text(row.get("question", ""))
            if context and question:
                out = [
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": question},
                ]

        out = merge_consecutive_messages(out)
        out = maybe_truncate_messages(out, args.max_messages)
        if not valid_dialogue(out, args.min_chatdoctor_user_chars, args.min_assistant_chars):
            continue
        if looks_like_provider_boilerplate(out[0]["content"]):
            continue
        meta = {
            "orig_id": row.get("id"),
            "n_messages": len(out),
        }
        items.append(make_example("mediq_askdocs", "ask_followup", out, ASK_FOLLOWUP_SYSTEM, meta))
    return items


def build_medmeadow(split_ds: Dataset, args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Build short single-turn reference QA examples from MedMeadow MediQA."""
    items: List[Dict[str, Any]] = []
    for row in tqdm(split_ds, desc="medical_meadow_mediqa"):
        instruction = clean_text(row.get("instruction", ""))
        user_input = clean_text(row.get("input", ""))
        assistant = strip_medmeadow_output(row.get("output", ""), args.max_medmeadow_output_chars)
        if not assistant:
            continue
        if len(user_input) > args.max_medmeadow_input_chars:
            user_input = user_input[: args.max_medmeadow_input_chars].rsplit(" ", 1)[0].strip()

        if instruction and user_input:
            norm_inst = normalize_for_hash(instruction)
            norm_input = normalize_for_hash(user_input)
            if norm_inst and norm_inst not in norm_input:
                user = f"Topic: {instruction}\n\n{user_input}"
            else:
                user = user_input
        else:
            user = user_input or instruction

        if len(user) < args.min_chatdoctor_user_chars or len(assistant) < args.min_assistant_chars:
            continue
        if looks_like_provider_boilerplate(user):
            continue
        msgs = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        if not valid_dialogue(msgs, args.min_chatdoctor_user_chars, args.min_assistant_chars):
            continue
        meta = {
            "instruction": instruction,
            "n_messages": 2,
        }
        items.append(make_example("medical_meadow_mediqa", "single_turn_reference", msgs, CONSULT_SYSTEM, meta))
    return items


def drop_overlaps_by_priority(
    items: List[Dict[str, Any]],
    seen_full: set,
    seen_first_user: set,
    *,
    drop_on_first_user_overlap: bool,
) -> List[Dict[str, Any]]:
    """Drop examples that overlap with higher-priority sources already seen."""
    kept: List[Dict[str, Any]] = []
    for ex in items:
        msgs_wo_system = [m for m in ex["messages"] if m["role"] != "system"]
        full_sig = canonical_full_signature(msgs_wo_system)
        user_sig = canonical_user_signature(msgs_wo_system)
        if full_sig in seen_full:
            continue
        if drop_on_first_user_overlap and user_sig in seen_first_user:
            continue
        seen_full.add(full_sig)
        seen_first_user.add(user_sig)
        kept.append(ex)
    return kept


def dedup_source_pools(source_to_items: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Deduplicate source pools in a fixed priority order for dialogue refinement."""
    seen_full: set = set()
    seen_first_user: set = set()
    ordered_sources = [
        ("mediq_askdocs", False),
        ("medqa_dialog", False),
        ("chatdoctor_cleaned", True),
        ("medical_meadow_mediqa", True),
    ]
    out: Dict[str, List[Dict[str, Any]]] = {}
    for source, strict_first_user in ordered_sources:
        items = source_to_items.get(source, [])
        out[source] = drop_overlaps_by_priority(
            items,
            seen_full,
            seen_first_user,
            drop_on_first_user_overlap=strict_first_user,
        )
    return out


def split_random(items: List[Dict[str, Any]], val_size: int, test_size: int, seed: int) -> Tuple[List, List, List]:
    """Shuffle once and split into train/validation/test lists."""
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    test = items[: min(test_size, len(items))]
    val = items[len(test) : len(test) + min(val_size, max(0, len(items) - len(test)))]
    train = items[len(test) + len(val) :]
    return train, val, test


def clone_with_repeat_idx(ex: Dict[str, Any], repeat_idx: int) -> Dict[str, Any]:
    """Clone repeated samples with a repeat index and a new deterministic ID."""
    if repeat_idx == 0:
        return ex
    new_ex = copy.deepcopy(ex)
    new_ex.setdefault("meta", {})
    new_ex["meta"]["repeat_idx"] = repeat_idx
    new_ex["id"] = stable_hash([new_ex["id"], str(repeat_idx)])[:24]
    return new_ex


def sample_or_repeat_to_target(items: List[Dict[str, Any]], target: int, seed: int) -> List[Dict[str, Any]]:
    """Sample without replacement when possible, otherwise repeat shuffled passes to hit target size."""
    if target <= 0 or not items:
        return []
    rng = random.Random(seed)
    items = list(items)
    if len(items) >= target:
        idxs = list(range(len(items)))
        rng.shuffle(idxs)
        return [items[i] for i in idxs[:target]]

    result: List[Dict[str, Any]] = []
    repeat_idx = 0
    while len(result) < target:
        idxs = list(range(len(items)))
        rng.shuffle(idxs)
        for i in idxs:
            result.append(clone_with_repeat_idx(items[i], repeat_idx))
            if len(result) >= target:
                break
        repeat_idx += 1
    return result[:target]


def write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    """Write examples as UTF-8 JSONL."""
    with path.open("w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def summarize(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize example counts, task/source distribution, and dialogue length."""
    by_source = Counter(ex["source"] for ex in items)
    by_task = Counter(ex["task"] for ex in items)
    turns = [len([m for m in ex["messages"] if m["role"] != "system"]) for ex in items]
    return {
        "n_examples": len(items),
        "by_source": dict(by_source),
        "by_task": dict(by_task),
        "avg_messages_wo_system": round(sum(turns) / len(turns), 3) if turns else 0.0,
        "max_messages_wo_system": max(turns) if turns else 0,
    }


def main() -> None:
    """Build all dialogue SFT splits and write JSONL plus stats outputs."""
    args = parse_args()
    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    # 1) Load source datasets.
    chatdoctor_ds = load_dataset_with_config_fallback(args.meerkat_name, args.chatdoctor_config)
    medqa_dialog_ds = load_dataset_with_config_fallback(args.meerkat_name, args.medqa_dialog_config)
    mediq_ds = load_dataset(args.mediq_name)
    medmeadow_ds = load_dataset(args.medmeadow_name)

    # 2) Convert each source into a shared messages-format schema.
    chatdoctor_all = build_meerkat_messages(
        chatdoctor_ds["train"],
        source="chatdoctor_cleaned",
        task="single_turn_consult",
        system_prompt=CONSULT_SYSTEM,
        max_messages=3,
        min_user_chars=args.min_chatdoctor_user_chars,
        min_assistant_chars=args.min_assistant_chars,
        min_messages=2,
        filter_doctorish_first_user=True,
    )
    medqa_dialog_all = build_meerkat_messages(
        medqa_dialog_ds["train"],
        source="medqa_dialog",
        task="multi_turn_consult",
        system_prompt=CONSULT_SYSTEM,
        max_messages=args.max_messages,
        min_user_chars=args.min_chatdoctor_user_chars,
        min_assistant_chars=args.min_assistant_chars,
        min_messages=args.min_multi_turn_messages,
        filter_doctorish_first_user=False,
    )
    mediq_train = build_mediq(mediq_ds["train"], args)
    mediq_val = build_mediq(mediq_ds.get("validation", []), args) if "validation" in mediq_ds else []
    mediq_test = build_mediq(mediq_ds.get("test", []), args) if "test" in mediq_ds else []
    medmeadow_all = build_medmeadow(medmeadow_ds["train"], args)

    # 3) Create random splits for sources without official validation/test data.
    chatdoctor_train, chatdoctor_val, chatdoctor_test = split_random(
        chatdoctor_all, args.chatdoctor_val_size, args.chatdoctor_test_size, args.seed
    )
    medqa_dialog_train, medqa_dialog_val, medqa_dialog_test = split_random(
        medqa_dialog_all, args.medqa_dialog_val_size, args.medqa_dialog_test_size, args.seed + 1
    )
    medmeadow_train, medmeadow_val, medmeadow_test = split_random(
        medmeadow_all, args.medmeadow_val_size, args.medmeadow_test_size, args.seed + 2
    )

    # 4) Deduplicate pools with priority: MediQ > MedQA-dialog > ChatDoctor > MedMeadow.
    train_pools = dedup_source_pools(
        {
            "mediq_askdocs": mediq_train,
            "medqa_dialog": medqa_dialog_train,
            "chatdoctor_cleaned": chatdoctor_train,
            "medical_meadow_mediqa": medmeadow_train,
        }
    )
    val_pools = dedup_source_pools(
        {
            "mediq_askdocs": mediq_val,
            "medqa_dialog": medqa_dialog_val,
            "chatdoctor_cleaned": chatdoctor_val,
            "medical_meadow_mediqa": medmeadow_val,
        }
    )
    test_pools = dedup_source_pools(
        {
            "mediq_askdocs": mediq_test,
            "medqa_dialog": medqa_dialog_test,
            "chatdoctor_cleaned": chatdoctor_test,
            "medical_meadow_mediqa": medmeadow_test,
        }
    )

    # 5) Build the unique train pool and stage-specific resampled mixtures.
    train_unique = (
        train_pools["mediq_askdocs"]
        + train_pools["medqa_dialog"]
        + train_pools["chatdoctor_cleaned"]
        + train_pools["medical_meadow_mediqa"]
    )

    train_stage1 = []
    train_stage1 += sample_or_repeat_to_target(train_pools["chatdoctor_cleaned"], args.stage1_chatdoctor_target, args.seed + 10)
    train_stage1 += sample_or_repeat_to_target(train_pools["mediq_askdocs"], args.stage1_mediq_target, args.seed + 11)
    train_stage1 += sample_or_repeat_to_target(train_pools["medical_meadow_mediqa"], args.stage1_medmeadow_target, args.seed + 12)
    train_stage1 += sample_or_repeat_to_target(train_pools["medqa_dialog"], args.stage1_medqa_dialog_target, args.seed + 13)

    train_stage2 = []
    train_stage2 += sample_or_repeat_to_target(train_pools["chatdoctor_cleaned"], args.stage2_chatdoctor_target, args.seed + 20)
    train_stage2 += sample_or_repeat_to_target(train_pools["mediq_askdocs"], args.stage2_mediq_target, args.seed + 21)
    train_stage2 += sample_or_repeat_to_target(train_pools["medical_meadow_mediqa"], args.stage2_medmeadow_target, args.seed + 22)
    train_stage2 += sample_or_repeat_to_target(train_pools["medqa_dialog"], args.stage2_medqa_dialog_target, args.seed + 23)

    val = val_pools["mediq_askdocs"] + val_pools["medqa_dialog"] + val_pools["chatdoctor_cleaned"] + val_pools["medical_meadow_mediqa"]
    test = test_pools["mediq_askdocs"] + test_pools["medqa_dialog"] + test_pools["chatdoctor_cleaned"] + test_pools["medical_meadow_mediqa"]

    rng = random.Random(args.seed)
    rng.shuffle(train_unique)
    rng.shuffle(train_stage1)
    rng.shuffle(train_stage2)
    rng.shuffle(val)
    rng.shuffle(test)

    # 6) Write JSONL files and a machine-readable stats summary.
    write_jsonl(outdir / "train_unique.jsonl", train_unique)
    write_jsonl(outdir / "train_stage1.jsonl", train_stage1)
    write_jsonl(outdir / "train_stage2.jsonl", train_stage2)
    write_jsonl(outdir / "val.jsonl", val)
    write_jsonl(outdir / "test.jsonl", test)

    stats = {
        "config": vars(args),
        "unique_train_pools": {k: len(v) for k, v in train_pools.items()},
        "unique_val_pools": {k: len(v) for k, v in val_pools.items()},
        "unique_test_pools": {k: len(v) for k, v in test_pools.items()},
        "train_unique": summarize(train_unique),
        "train_stage1": summarize(train_stage1),
        "train_stage2": summarize(train_stage2),
        "val": summarize(val),
        "test": summarize(test),
        "pre_dedup": {
            "chatdoctor_all": len(chatdoctor_all),
            "chatdoctor_train": len(chatdoctor_train),
            "chatdoctor_val": len(chatdoctor_val),
            "chatdoctor_test": len(chatdoctor_test),
            "medqa_dialog_all": len(medqa_dialog_all),
            "medqa_dialog_train": len(medqa_dialog_train),
            "medqa_dialog_val": len(medqa_dialog_val),
            "medqa_dialog_test": len(medqa_dialog_test),
            "mediq_train": len(mediq_train),
            "mediq_val": len(mediq_val),
            "mediq_test": len(mediq_test),
            "medmeadow_all": len(medmeadow_all),
            "medmeadow_train": len(medmeadow_train),
            "medmeadow_val": len(medmeadow_val),
            "medmeadow_test": len(medmeadow_test),
        },
    }
    with (outdir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
