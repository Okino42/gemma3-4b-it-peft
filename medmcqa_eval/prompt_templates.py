from __future__ import annotations

from typing import Any, Mapping


OPTION_LETTERS = ("A", "B", "C", "D")


def build_medmcqa_messages(row: Mapping[str, Any], fields: Mapping[str, Any], prompt_cfg: Mapping[str, Any]) -> list[dict[str, str]]:
    question = str(row[fields["question"]]).strip()
    option_fields = fields["options"]

    option_lines = []
    for letter in OPTION_LETTERS:
        value = row[option_fields[letter]]
        option_lines.append(f"{letter}. {str(value).strip()}")

    user = "\n".join(
        [
            "Question:",
            question,
            "",
            "Options:",
            *option_lines,
            "",
            str(prompt_cfg["answer_format"]).strip(),
        ]
    )

    return [
        {"role": "system", "content": str(prompt_cfg["system"]).strip()},
        {"role": "user", "content": user},
    ]
