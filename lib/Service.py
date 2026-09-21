#
# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: MIT
#
"""Translation service"""

import json
import logging
import os
import re
from copy import deepcopy
from time import perf_counter
from typing import TypedDict

import ctranslate2
from nc_py_api.ex_app import setup_nextcloud_logging
from sentencepiece import SentencePieceProcessor
from util import clean_text

logger = logging.getLogger(os.environ["APP_ID"] + __name__)

# Languages that do not use spaces between words — join chunks without a space separator
_NO_SPACE_LANGUAGES = {"zh", "yue", "ja", "th", "my", "km", "lo", "bo", "dz", "shn"}


class ServiceException(Exception):
    pass


class TranslateRequest(TypedDict):
    origin_language: str
    input: str
    target_language: str


if os.getenv("CI") is not None:
    ctranslate2.set_random_seed(420)


class Service:
    def __init__(self, config: dict):
        global logger
        try:
            self.load_config(config)
            ctranslate2.set_log_level(config["log_level"])
            logger.setLevel(config["log_level"])
            setup_nextcloud_logging(os.environ["APP_ID"] + "_" + __name__, config["log_level"])

            with open("languages.json") as f:
                self.languages = json.loads(f.read())
        except Exception as e:
            raise ServiceException(
                "Error reading languages list, ensure languages.json is present in the project root"
            ) from e

    def get_languages(self) -> dict[str, str]:
        return self.languages

    def load_config(self, config: dict):
        config_copy = deepcopy(config)
        config_copy["loader"].pop("model_name", None)

        if "hf_model_path" in config_copy["loader"]:
            config_copy["loader"]["model_path"] = config_copy["loader"].pop("hf_model_path")

        self.config = config_copy

    def load_model(self):
        try:
            self.tokenizer = SentencePieceProcessor()
            self.tokenizer.Load(os.path.join(self.config["loader"]["model_path"], self.config["tokenizer_file"]))

            self.translator = ctranslate2.Translator(
                **{
                    "device": "cuda" if os.getenv("COMPUTE_DEVICE") == "CUDA" else "cpu",
                    **self.config["loader"],
                }
            )
        except KeyError as e:
            raise ServiceException(
                "Incorrect config file, ensure all required keys are present from the default config"
            ) from e
        except Exception as e:
            raise ServiceException("Error loading the translation model") from e

    def _chunk_text(self, text: str, max_words: int, source_language: str = "") -> list[str]:
        """Split text into sentence-boundary chunks of at most max_words words.

        For no-space languages (Chinese, Japanese, Thai, etc.) character count is used
        instead of word count.

        Uses a simple sentence-boundary regex that handles:
        - Period / exclamation / question mark followed by whitespace (Latin scripts)
        - CJK (Chinese, Japanese and Korean-alike languages) sentence-ending punctuation (U+3002, U+FF01, U+FF1F) without requiring
          trailing whitespace, since CJK sentences run together.

        For no-space languages split() always returns a single token regardless of
        length, so character count is used as the unit instead of word count.
        """
        source_base = source_language.split("_")[0].lower() if source_language else ""
        is_no_space = source_base in _NO_SPACE_LANGUAGES

        # Sentence-boundary split: keep the delimiter attached to the preceding sentence.
        # For no-space languages (CJK etc.) use \s* because sentences run together without
        # whitespace. For all other languages use \s+ to avoid splitting on abbreviations,
        # decimals, URLs, and other mid-word periods (e.g. "Dr.", "3.14", "U.S.A").
        if is_no_space:
            sentences = re.split(r"(?<=[。！？\u3002\uff01\uff1f])\s*", text)  # noqa: RUF001
        else:
            sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: list[str] = []
        current_parts: list[str] = []
        current_count = 0
        sep = "" if is_no_space else " "

        for sentence in sentences:
            # Count characters for no-space languages, words for everything else
            unit_count = len(sentence) if is_no_space else len(sentence.split())
            if unit_count == 0:
                continue

            # If adding this sentence would overflow the chunk, flush first
            if current_count + unit_count > max_words and current_parts:
                chunks.append(sep.join(current_parts))
                current_parts = []
                current_count = 0

            # If a single sentence is longer than max_words on its own, hard-split it
            if unit_count > max_words:
                if is_no_space:
                    for i in range(0, len(sentence), max_words):
                        chunks.append(sentence[i:i + max_words])
                else:
                    words = sentence.split()
                    for i in range(0, len(words), max_words):
                        chunks.append(" ".join(words[i:i + max_words]))
                continue

            current_parts.append(sentence)
            current_count += unit_count

        if current_parts:
            chunks.append(sep.join(current_parts))

        return chunks if chunks else [text]

    def _join_chunks(self, chunks: list[str], target_language: str) -> str:
        """Join translated chunks respecting language-specific rules.

        - No-space languages (zh, ja, th, …): join with empty string.
        - All other languages: join in forward order.

        Chunks are always kept in their original document order regardless of
        source/target writing direction. Each chunk is translated independently
        by the model, which already produces output in the correct reading order
        for the target language. Reversing the chunk list would scramble the
        logical sequence of the document.
        """
        # Strip leading/trailing whitespace from each chunk before joining
        chunks = [c.strip() for c in chunks if c.strip()]
        if not chunks:
            return ""

        target_base = target_language.split("_")[0].lower()

        separator = "" if target_base in _NO_SPACE_LANGUAGES else " "
        return separator.join(chunks)

    def translate(self, data: TranslateRequest) -> str:
        logger.debug(f"translating text to: {data['target_language']}")

        try:
            start = perf_counter()
            cleaned = clean_text(data["input"])

            chunking = self.config.get("chunking", {})
            chunk_threshold = chunking.get("chunk_threshold", 250)
            chunk_size = chunking.get("chunk_size", 80)
            min_repetition_penalty = chunking.get("min_repetition_penalty", 1.5)
            max_decoding_multiplier = chunking.get("max_decoding_multiplier", 3)

            source_base = data.get("origin_language", "").split("_")[0].lower()
            is_no_space_source = source_base in _NO_SPACE_LANGUAGES

            # For no-space languages (CJK, Thai, etc.) use character count as the unit;
            # split() always returns 1 for these scripts regardless of actual length.
            text_size = len(cleaned) if is_no_space_source else len(cleaned.split())
            chunks = (
                self._chunk_text(cleaned, chunk_size, data.get("origin_language", ""))
                if text_size > chunk_threshold
                else [cleaned]
            )

            # Encode all chunks up-front so we can submit them in a single batch.
            all_input_tokens = [
                self.tokenizer.Encode(
                    f"<2{data['target_language']}> {chunk}",
                    out_type=str,
                )
                for chunk in chunks
            ]

            # Cap max_decoding_length using the longest chunk in the batch.
            # This prevents runaway repetition loops (e.g. Hindi/Devanagari producing
            # endless '=' characters) while still covering all chunks in one pass.
            # The multiplier is configurable via chunking.max_decoding_multiplier
            # (default 3). Languages that expand significantly (e.g. French ~1.3x,
            # Devanagari ~1.5x) may need a higher value. Floor of 64 handles very
            # short inputs.
            max_input_tokens = max(len(t) for t in all_input_tokens)
            batch_max_decoding = max(max_input_tokens * max_decoding_multiplier, 64)
            inference_config = {
                **self.config["inference"],
                "max_decoding_length": batch_max_decoding,
                "repetition_penalty": max(
                    self.config["inference"].get("repetition_penalty", 1.0), min_repetition_penalty
                ),
            }

            results = self.translator.translate_batch(
                all_input_tokens,
                batch_type="tokens",
                **inference_config,
            )

            if len(results) != len(chunks) or any(len(r.hypotheses) == 0 for r in results):
                raise ServiceException("Empty result returned from translator")

            # todo: handle multiple hypotheses
            translated_chunks = [self.tokenizer.Decode(r.hypotheses[0]) for r in results]

            translation = self._join_chunks(translated_chunks, data["target_language"])
            elapsed = perf_counter() - start
            logger.info(f"time taken: {elapsed:.2f}s")
        except Exception as e:
            raise ServiceException("Error translating the input text") from e

        logger.debug(f"Translated string: {translation}")
        return translation
