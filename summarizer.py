"""Local Burmese summarization model and preprocessing logic."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
)

BREAK_PATTERN = re.compile(
    r"((?<!္)[က-အ](?![်္])|[a-zA-Z0-9ဣဤဥဦဧဩဪဿ၌၍၏၀-၉၊ ။!-/:-@[-` {-~\s])"
)


class BurmeseSummarizer:
    """Load and run the local Burmese sequence-to-sequence checkpoint."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        missing_files = [
            name for name in MODEL_FILES if not (self.model_dir / name).is_file()
        ]
        if missing_files:
            missing = ", ".join(missing_files)
            raise FileNotFoundError(
                f"Missing local model files in {self.model_dir}: {missing}. "
                "Place the exported checkpoint in the model directory or set "
                "BURMESE_MODEL_DIR."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_dir,
            local_files_only=True,
        ).to(self.device)
        self._model.eval()

    @staticmethod
    def segment_syllables(text: str) -> str:
        """Apply the same Burmese syllable segmentation used during training."""
        normalized = re.sub(r"\s+", " ", str(text).strip())
        segmented = BREAK_PATTERN.sub(r" \1", normalized)
        return segmented.strip().replace("  ", " ")

    def summarize(
        self,
        article: str,
        max_length: int,
        min_length: int,
        num_beams: int,
        length_penalty: float,
        no_repeat_ngram_size: int,
    ) -> tuple[str, str]:
        article = (article or "").strip()
        if not article:
            return "", "Please enter an article first."

        max_length = int(max_length)
        min_length = int(min_length)
        num_beams = int(num_beams)
        no_repeat_ngram_size = int(no_repeat_ngram_size)
        if min_length >= max_length:
            return "", "Minimum output length must be lower than maximum output length."

        prompt = "summarize: " + self.segment_syllables(article)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        generation_kwargs: dict[str, Any] = {
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "length_penalty": float(length_penalty),
            "early_stopping": True,
        }
        if no_repeat_ngram_size > 0:
            generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary.replace(" ", "").strip(), f"{len(article):,} characters processed"
