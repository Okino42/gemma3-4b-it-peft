#!/usr/bin/env python3
"""
Prepare Stage-1 LoRA SFT data for Gemma-3-4b-it using the dataset recipe we
settled on for broad medical knowledge adaptation:

Medical core:
- openlifescienceai/medmcqa              (MCQ -> options + explanation format)
- lavita/MedQuAD                         (reference-style medical QA)
- medalpaca/medical_meadow_mediqa       (small diverse supplement)
- bio-nlp-umass/bioinstruct             (small biomedical instruction supplement)

General instruction supplement:
- databricks/databricks-dolly-15k       (filtered non-code general instruction)

Outputs:
- train_unique.jsonl   : deduplicated unique train pool before resampling
- train_stage1.jsonl   : final 65k stage-1 mixture
- val.jsonl            : validation split
- test.jsonl           : test split
- stats.json

Format example:
{
  "id": "...",
  "source": "medmcqa|medquad|medical_meadow_mediqa|bioinstruct|databricks_dolly_15k",
  "task": "medical_mcq_explained|medical_reference_qa|biomedical_instruction|general_instruction",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {...}
}
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

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm


MEDICAL_SYSTEM = (
    "You are a careful medical assistant. Provide accurate, cautious, and "
    "clinically grounded answers. Do not overstate certainty. When appropriate, "
    "mention important red flags and recommend in-person evaluation for urgent symptoms."
)

GENERAL_SYSTEM = (
    "You are a helpful, concise, and honest assistant. Follow the user's request "
    "clearly and avoid unnecessary verbosity."
)

MEDMEADOW_TAIL_PATTERNS = [
    r"updated by:.*$",
    r"review provided by.*$",
    r"also reviewed by.*$",
    r"doi:\s*10\.\S+.*$",
    r"n engl j med\..*$",
]

GENERAL_DROP_PATTERNS = [
    r"\bpython\b",
    r"\bjavascript\b",
    r"\bc\+\+\b",
    r"\bjava\b",
    r"\bregex\b",
    r"\bsql\b",
    r"\bhtml\b",
    r"\bcss\b",
    r"\bcode\b",
    r"\bprogram\b",
    r"\balgorithm\b",
    r"\bdebug\b",
    r"\bimplement\b",
    r"\bwrite a function\b",
    r"\bscript\b",
]


def parse_args() -> argparse.Namespace:
    """Parse dataset names, split sizes, target mixture sizes, and cleaning thresholds."""
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--medmcqa_name", type=str, default="openlifescienceai/medmcqa")
    p.add_argument("--medquad_name", type=str, default="lavita/MedQuAD")
    p.add_argument("--medmeadow_name", type=str, default="medalpaca/medical_meadow_mediqa")
    p.add_argument("--bioinstruct_name", type=str, default="bio-nlp-umass/bioinstruct")
    p.add_argument("--general_name", type=str, default="databricks/databricks-dolly-15k")

    # Random splits for datasets without official val/test.
    p.add_argument("--medquad_val_size", type=int, default=1200)
    p.add_argument("--medquad_test_size", type=int, default=1200)
    p.add_argument("--medmeadow_val_size", type=int, default=200)
    p.add_argument("--medmeadow_test_size", type=int, default=200)
    p.add_argument("--bioinstruct_val_size", type=int, default=500)
    p.add_argument("--bioinstruct_test_size", type=int, default=500)
    p.add_argument("--general_val_size", type=int, default=200)
    p.add_argument("--general_test_size", type=int, default=200)

    # Final stage-1 train recipe from the agreed dataset mixture (total = 64,708).
    p.add_argument("--stage1_medmcqa_target", type=int, default=38000)
    p.add_argument("--stage1_medquad_target", type=int, default=18000)
    p.add_argument("--stage1_bioinstruct_target", type=int, default=3500)
    p.add_argument("--stage1_medmeadow_target", type=int, default=2208)
    p.add_argument("--stage1_general_target", type=int, default=3000)

    p.add_argument("--min_user_chars", type=int, default=20)
    p.add_argument("--min_assistant_chars", type=int, default=20)
    p.add_argument("--max_medmcqa_question_chars", type=int, default=2000)
    p.add_argument("--max_medmcqa_explanation_chars", type=int, default=2800)
    p.add_argument("--max_medquad_answer_chars", type=int, default=4000)
    p.add_argument("--max_medmeadow_input_chars", type=int, default=12000)
    p.add_argument("--max_medmeadow_output_chars", type=int, default=3500)
    p.add_argument("--max_bioinstruct_input_chars", type=int, default=8000)
    p.add_argument("--max_bioinstruct_output_chars", type=int, default=3500)
    p.add_argument("--max_general_input_chars", type=int, default=6000)
    p.add_argument("--max_general_output_chars", type=int, default=2200)
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
    return hashlib.sha256("\n<SEP>\n".join(parts).encode("utf-8")).hexdigest()


def truncate_at_word(text: str, max_chars: int) -> str:
    """Cap text length without cutting the final word in half when possible."""
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def valid_pair(user: str, assistant: str, min_user_chars: int, min_assistant_chars: int) -> bool:
    """Check minimum cleaned lengths for a single-turn user/assistant pair."""
    return len(clean_text(user)) >= min_user_chars and len(clean_text(assistant)) >= min_assistant_chars


def canonical_user_signature(messages: List[Dict[str, str]]) -> str:
    """Return the normalized first-user text used for cross-source duplicate checks."""
    users = [m["content"] for m in messages if m["role"] == "user"]
    return normalize_for_hash(users[0]) if users else ""


def canonical_full_signature(messages: List[Dict[str, str]]) -> str:
    """Return a role-aware hash for all messages in an example."""
    flat = [f"{m['role']}: {normalize_for_hash(m['content'])}" for m in messages]
    return stable_hash(flat)


def make_example(
    source: str,
    task: str,
    user: str,
    assistant: str,
    system_prompt: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create one messages-format SFT example with deterministic ID."""
    messages = [
        {"role": "system", "content": clean_text(system_prompt)},
        {"role": "user", "content": clean_text(user)},
        {"role": "assistant", "content": clean_text(assistant)},
    ]
    ex = {
        "source": source,
        "task": task,
        "messages": messages,
        "meta": meta or {},
    }
    ex["id"] = stable_hash([source, task, json.dumps(messages, ensure_ascii=False, sort_keys=True)])[:24]
    return ex


def split_random(items: List[Dict[str, Any]], val_size: int, test_size: int, seed: int) -> Tuple[List, List, List]:
    """Shuffle once and split into train/validation/test lists."""
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    test = items[: min(test_size, len(items))]
    val = items[len(test): len(test) + min(val_size, max(0, len(items) - len(test)))]
    train = items[len(test) + len(val):]
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
    """Summarize example counts and task/source distributions."""
    by_source = Counter(ex["source"] for ex in items)
    by_task = Counter(ex["task"] for ex in items)
    return {
        "n_examples": len(items),
        "by_source": dict(by_source),
        "by_task": dict(by_task),
    }


def dedup_pool(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove exact dialogue duplicates and repeated first-user prompts within one pool."""
    seen_full = set()
    seen_first_user = set()
    kept: List[Dict[str, Any]] = []
    for ex in items:
        msgs_wo_system = [m for m in ex["messages"] if m["role"] != "system"]
        full_sig = canonical_full_signature(msgs_wo_system)
        user_sig = canonical_user_signature(msgs_wo_system)
        if full_sig in seen_full or user_sig in seen_first_user:
            continue
        seen_full.add(full_sig)
        seen_first_user.add(user_sig)
        kept.append(ex)
    return kept


def clean_medmcqa_explanation(text: str, max_chars: int) -> str:
    """Clean MedMCQA explanation text and remove embedded answer suffixes."""
    text = clean_text(text)
    text = re.sub(r"\bAns(?:wer)?\s*[:\-].*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return truncate_at_word(text, max_chars)


def medmcqa_correct_index(raw: Any) -> Optional[int]:
    """Convert MedMCQA answer labels such as 1-4 or A-D into a zero-based index."""
    if raw is None:
        return None
    text = clean_text(raw).lower()
    m = re.search(r"([1-4])", text)
    if m:
        return int(m.group(1)) - 1
    m = re.search(r"\b([abcd])\b", text)
    if m:
        return ord(m.group(1)) - ord("a")
    return None


def build_medmcqa(ds: DatasetDict, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert MedMCQA splits into explained multiple-choice SFT examples."""
    label_names = ["A", "B", "C", "D"]

    def convert(split_ds: Dataset, split_name: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for row in tqdm(split_ds, desc=f"medmcqa_{split_name}"):
            question = truncate_at_word(row.get("question", ""), args.max_medmcqa_question_chars)
            explanation = clean_medmcqa_explanation(row.get("exp", ""), args.max_medmcqa_explanation_chars)
            choice_type = clean_text(row.get("choice_type", "")).lower()
            if not question or not explanation:
                continue
            if choice_type and choice_type != "single":
                continue

            options = [
                clean_text(row.get("opa", "")),
                clean_text(row.get("opb", "")),
                clean_text(row.get("opc", "")),
                clean_text(row.get("opd", "")),
            ]
            if any(not x for x in options):
                continue
            answer_idx = medmcqa_correct_index(row.get("cop"))
            if answer_idx is None or not (0 <= answer_idx < 4):
                continue

            user = (
                "Answer the following medical multiple-choice question. "
                "First explain briefly, then give the final answer.\n\n"
                f"Question: {question}\n"
                f"A. {options[0]}\n"
                f"B. {options[1]}\n"
                f"C. {options[2]}\n"
                f"D. {options[3]}"
            )
            assistant = (
                f"Explanation: {explanation}\n\n"
                f"Final answer: {label_names[answer_idx]}. {options[answer_idx]}"
            )
            if not valid_pair(user, assistant, args.min_user_chars, args.min_assistant_chars):
                continue
            items.append(
                make_example(
                    "medmcqa",
                    "medical_mcq_explained",
                    user,
                    assistant,
                    MEDICAL_SYSTEM,
                    {
                        "orig_id": row.get("id"),
                        "subject_name": clean_text(row.get("subject_name", "")),
                        "topic_name": clean_text(row.get("topic_name", "")),
                        "answer_idx": answer_idx,
                    },
                )
            )
        return dedup_pool(items)

    train = convert(ds["train"], "train")
    val_split_name = "validation" if "validation" in ds else ("valid" if "valid" in ds else None)
    test_split_name = "test" if "test" in ds else None
    val = convert(ds[val_split_name], val_split_name) if val_split_name else []
    test = convert(ds[test_split_name], test_split_name) if test_split_name else []
    return train, val, test


MEDQUAD_EXCLUDED_SOURCES = {"MPlusDrugs", "MPlusHerbsSupplements", "ADAM", "GARD"}
MEDQUAD_EXCLUDED_QTYPES = {
    "how can i learn more",
    "where to find support groups",
    "research",
    "clinical trial",
}


def build_medquad(split_ds: Dataset, args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Convert MedQuAD rows into reference-style medical QA examples."""
    items: List[Dict[str, Any]] = []
    for row in tqdm(split_ds, desc="medquad"):
        source = clean_text(row.get("document_source", ""))
        qtype = clean_text(row.get("question_type", "")).lower()
        question = clean_text(row.get("question", ""))
        answer = truncate_at_word(row.get("answer", ""), args.max_medquad_answer_chars)
        if source in MEDQUAD_EXCLUDED_SOURCES:
            continue
        if qtype in MEDQUAD_EXCLUDED_QTYPES:
            continue
        if not valid_pair(question, answer, args.min_user_chars, args.min_assistant_chars):
            continue
        items.append(
            make_example(
                "medquad",
                "medical_reference_qa",
                question,
                answer,
                MEDICAL_SYSTEM,
                {
                    "document_source": source,
                    "question_type": qtype,
                    "question_focus": clean_text(row.get("question_focus", "")),
                    "document_id": clean_text(row.get("document_id", "")),
                },
            )
        )
    return dedup_pool(items)


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
    return truncate_at_word(text, max_chars)


def build_medmeadow(split_ds: Dataset, args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Build medical reference QA examples from MedMeadow MediQA rows."""
    items: List[Dict[str, Any]] = []
    for row in tqdm(split_ds, desc="medical_meadow_mediqa"):
        instruction = clean_text(row.get("instruction", ""))
        user_input = truncate_at_word(row.get("input", ""), args.max_medmeadow_input_chars)
        assistant = strip_medmeadow_output(row.get("output", ""), args.max_medmeadow_output_chars)
        if not assistant:
            continue
        if instruction and user_input:
            norm_inst = normalize_for_hash(instruction)
            norm_input = normalize_for_hash(user_input)
            user = user_input if norm_inst and norm_inst in norm_input else f"Topic: {instruction}\n\n{user_input}"
        else:
            user = user_input or instruction
        if not valid_pair(user, assistant, args.min_user_chars, args.min_assistant_chars):
            continue
        items.append(
            make_example(
                "medical_meadow_mediqa",
                "medical_reference_qa",
                user,
                assistant,
                MEDICAL_SYSTEM,
                {"instruction": instruction},
            )
        )
    return dedup_pool(items)


BIOINSTRUCT_DROP_PATTERNS = [
    r"\bextract\b.*\bjson\b",
    r"\btag\b.*\bentity\b",
    r"\bnormalize\b.*\bbiomedical\b",
    r"\bwrite python\b",
    r"\btable\b",
    r"\bxml\b",
]


def should_drop_bioinstruct(instruction: str, output: str) -> bool:
    """Filter BioInstruct tasks that are too extraction/table/code oriented for this mixture."""
    text = f"{instruction}\n{output}".lower()
    return any(re.search(pat, text) for pat in BIOINSTRUCT_DROP_PATTERNS)


def build_bioinstruct(split_ds: Dataset, args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Build biomedical instruction-following examples from BioInstruct."""
    items: List[Dict[str, Any]] = []
    for row in tqdm(split_ds, desc="bioinstruct"):
        instruction = clean_text(row.get("instruction", ""))
        input_text = truncate_at_word(row.get("input", ""), args.max_bioinstruct_input_chars)
        output = truncate_at_word(row.get("output", ""), args.max_bioinstruct_output_chars)
        if should_drop_bioinstruct(instruction, output):
            continue
        if instruction and input_text:
            user = f"Instruction: {instruction}\n\nInput: {input_text}"
        else:
            user = instruction or input_text
        if not valid_pair(user, output, args.min_user_chars, args.min_assistant_chars):
            continue
        items.append(
            make_example(
                "bioinstruct",
                "biomedical_instruction",
                user,
                output,
                MEDICAL_SYSTEM,
                {},
            )
        )
    return dedup_pool(items)


UNSAFE_GENERAL_PATTERNS = [
    r"\bweapon\b",
    r"\bbomb\b",
    r"\bmalware\b",
    r"\bexploit\b",
    r"\bdrugs?\b",
    r"\bsuicide\b",
]


def should_keep_general(instruction: str, output: str) -> bool:
    """Keep only non-code, non-unsafe general instruction examples."""
    text = f"{instruction}\n{output}".lower()
    if "```" in text:
        return False
    if any(re.search(pat, text) for pat in GENERAL_DROP_PATTERNS):
        return False
    if any(re.search(pat, text) for pat in UNSAFE_GENERAL_PATTERNS):
        return False
    return True


def build_general(split_ds: Dataset, args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Build filtered general instruction examples from Databricks Dolly."""
    items: List[Dict[str, Any]] = []
    for row in tqdm(split_ds, desc="databricks_dolly_15k"):
        instruction = clean_text(row.get("instruction", ""))
        context = truncate_at_word(row.get("context", ""), args.max_general_input_chars)
        output = truncate_at_word(row.get("response", ""), args.max_general_output_chars)
        category = clean_text(row.get("category", ""))
        if not should_keep_general(instruction + "\n" + context, output):
            continue
        if instruction and context:
            user = f"Instruction: {instruction}\n\nInput: {context}"
        else:
            user = instruction or context
        if not valid_pair(user, output, args.min_user_chars, args.min_assistant_chars):
            continue
        items.append(
            make_example(
                "databricks_dolly_15k",
                "general_instruction",
                user,
                output,
                GENERAL_SYSTEM,
                {"category": category},
            )
        )
    return dedup_pool(items)


def dedup_source_pools(source_to_items: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    # Keep higher-priority medical knowledge first.
    priority = [
        "medmcqa",
        "medquad",
        "medical_meadow_mediqa",
        "bioinstruct",
        "databricks_dolly_15k",
    ]
    seen_full = set()
    seen_first_user = set()
    out: Dict[str, List[Dict[str, Any]]] = {}
    for source in priority:
        kept: List[Dict[str, Any]] = []
        for ex in source_to_items.get(source, []):
            msgs_wo_system = [m for m in ex["messages"] if m["role"] != "system"]
            full_sig = canonical_full_signature(msgs_wo_system)
            user_sig = canonical_user_signature(msgs_wo_system)
            if full_sig in seen_full or user_sig in seen_first_user:
                continue
            seen_full.add(full_sig)
            seen_first_user.add(user_sig)
            kept.append(ex)
        out[source] = kept
    return out


def main() -> None:
    """Build the stage-1 knowledge-expansion SFT mixture and write stats."""
    args = parse_args()
    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    medmcqa_ds = load_dataset(args.medmcqa_name)
    medquad_ds = load_dataset(args.medquad_name)
    medmeadow_ds = load_dataset(args.medmeadow_name)
    bioinstruct_ds = load_dataset(args.bioinstruct_name)
    general_ds = load_dataset(args.general_name)

    medmcqa_train, medmcqa_val, medmcqa_test = build_medmcqa(medmcqa_ds, args)
    medquad_all = build_medquad(medquad_ds["train"], args)
    medmeadow_all = build_medmeadow(medmeadow_ds["train"], args)
    bioinstruct_all = build_bioinstruct(bioinstruct_ds["train"], args)
    general_all = build_general(general_ds["train"], args)

    medquad_train, medquad_val, medquad_test = split_random(
        medquad_all, args.medquad_val_size, args.medquad_test_size, args.seed + 10
    )
    medmeadow_train, medmeadow_val, medmeadow_test = split_random(
        medmeadow_all, args.medmeadow_val_size, args.medmeadow_test_size, args.seed + 11
    )
    bioinstruct_train, bioinstruct_val, bioinstruct_test = split_random(
        bioinstruct_all, args.bioinstruct_val_size, args.bioinstruct_test_size, args.seed + 12
    )
    general_train, general_val, general_test = split_random(
        general_all, args.general_val_size, args.general_test_size, args.seed + 13
    )

    train_pools = dedup_source_pools(
        {
            "medmcqa": medmcqa_train,
            "medquad": medquad_train,
            "medical_meadow_mediqa": medmeadow_train,
            "bioinstruct": bioinstruct_train,
            "databricks_dolly_15k": general_train,
        }
    )
    val_pools = dedup_source_pools(
        {
            "medmcqa": medmcqa_val,
            "medquad": medquad_val,
            "medical_meadow_mediqa": medmeadow_val,
            "bioinstruct": bioinstruct_val,
            "databricks_dolly_15k": general_val,
        }
    )
    test_pools = dedup_source_pools(
        {
            "medmcqa": medmcqa_test,
            "medquad": medquad_test,
            "medical_meadow_mediqa": medmeadow_test,
            "bioinstruct": bioinstruct_test,
            "databricks_dolly_15k": general_test,
        }
    )

    train_unique = (
        train_pools["medmcqa"]
        + train_pools["medquad"]
        + train_pools["medical_meadow_mediqa"]
        + train_pools["bioinstruct"]
        + train_pools["databricks_dolly_15k"]
    )

    train_stage1 = []
    train_stage1 += sample_or_repeat_to_target(train_pools["medmcqa"], args.stage1_medmcqa_target, args.seed + 21)
    train_stage1 += sample_or_repeat_to_target(train_pools["medquad"], args.stage1_medquad_target, args.seed + 22)
    train_stage1 += sample_or_repeat_to_target(train_pools["bioinstruct"], args.stage1_bioinstruct_target, args.seed + 23)
    train_stage1 += sample_or_repeat_to_target(train_pools["medical_meadow_mediqa"], args.stage1_medmeadow_target, args.seed + 24)
    train_stage1 += sample_or_repeat_to_target(train_pools["databricks_dolly_15k"], args.stage1_general_target, args.seed + 25)

    val = (
        val_pools["medmcqa"]
        + val_pools["medquad"]
        + val_pools["medical_meadow_mediqa"]
        + val_pools["bioinstruct"]
        + val_pools["databricks_dolly_15k"]
    )
    test = (
        test_pools["medmcqa"]
        + test_pools["medquad"]
        + test_pools["medical_meadow_mediqa"]
        + test_pools["bioinstruct"]
        + test_pools["databricks_dolly_15k"]
    )

    rng = random.Random(args.seed)
    rng.shuffle(train_unique)
    rng.shuffle(train_stage1)
    rng.shuffle(val)
    rng.shuffle(test)

    write_jsonl(outdir / "train_unique.jsonl", train_unique)
    write_jsonl(outdir / "train_stage1.jsonl", train_stage1)
    write_jsonl(outdir / "val.jsonl", val)
    write_jsonl(outdir / "test.jsonl", test)

    stats = {
        "config": vars(args),
        "unique_train_pools": {k: len(v) for k, v in train_pools.items()},
        "unique_val_pools": {k: len(v) for k, v in val_pools.items()},
        "unique_test_pools": {k: len(v) for k, v in test_pools.items()},
        "train_unique": summarize(train_unique),
        "train_stage1": summarize(train_stage1),
        "val": summarize(val),
        "test": summarize(test),
        "pre_dedup": {
            "medmcqa_train": len(medmcqa_train),
            "medmcqa_val": len(medmcqa_val),
            "medmcqa_test": len(medmcqa_test),
            "medquad_all": len(medquad_all),
            "medquad_train": len(medquad_train),
            "medquad_val": len(medquad_val),
            "medquad_test": len(medquad_test),
            "medmeadow_all": len(medmeadow_all),
            "medmeadow_train": len(medmeadow_train),
            "medmeadow_val": len(medmeadow_val),
            "medmeadow_test": len(medmeadow_test),
            "bioinstruct_all": len(bioinstruct_all),
            "bioinstruct_train": len(bioinstruct_train),
            "bioinstruct_val": len(bioinstruct_val),
            "bioinstruct_test": len(bioinstruct_test),
            "general_all": len(general_all),
            "general_train": len(general_train),
            "general_val": len(general_val),
            "general_test": len(general_test),
        },
    }
    with (outdir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
