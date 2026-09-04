"""Tokenizer + chat templating.

The GGUF carries its own vocabulary and chat template, but the HF
`tokenizer.json` is used for encoding because it also carries the pre-tokenizer
regex and the byte-level decoder. The two id spaces were verified to agree for
every real token; the GGUF simply pads the vocabulary out to 248320 with
`[PADnnnnn]` placeholders that never appear in text.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer


class Qwen4ExpTokenizer:
    def __init__(self, tokenizer_path: str | Path, chat_template: str | None = None,
                 eos_token_ids: list[int] | None = None):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.chat_template = chat_template
        self.eos_token_ids = eos_token_ids or []
        self._template = None

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self, messages: list[dict], add_generation_prompt: bool = True, **kwargs
    ) -> str:
        if self.chat_template is None:
            raise ValueError("this checkpoint carries no chat template")
        if self._template is None:
            from jinja2 import Environment
            from jinja2.exceptions import TemplateError

            def raise_exception(msg: str):
                raise TemplateError(msg)

            env = Environment(trim_blocks=True, lstrip_blocks=True)
            env.globals["raise_exception"] = raise_exception
            self._template = env.from_string(self.chat_template)
        return self._template.render(
            messages=messages, add_generation_prompt=add_generation_prompt, **kwargs
        )
