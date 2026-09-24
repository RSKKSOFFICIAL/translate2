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


def _is_no_space_text(text: str) -> bool:
    """Return True when the source text appears to use a no-space writing system.

    Instead of relying on the origin_language (which is always "detect_language"),
    we inspect the text itself. Languages like Chinese, Japanese, Thai,
    Burmese, and Khmer have very few or no ASCII/Unicode space characters, giving a
    space-to-total-character ratio close to zero. Space-delimited languages (English,
    German, Arabic, Persian, Hindi, etc.) consistently produce a ratio above 10%.

    A threshold of 5% is conservative enough to avoid false positives on short
    punctuation-heavy snippets while correctly identifying dense scripts.

    Empty or whitespace-only strings return False (they will produce an empty
    translation regardless of counting method).
    """
    stripped = text.strip()
    if not stripped:
        return False
    space_count = stripped.count(" ") + stripped.count("\t") + stripped.count("\n")
    return (space_count / len(stripped)) < 0.05


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

    def _chunk_text(self, text: str, max_words: int, is_no_space: bool = False) -> list[str]:
        """Split text into sentence-boundary chunks of at most max_words words (or chars).

        Args:
            text: The text to split.
            max_words: Maximum words per chunk for space-delimited text, or maximum
                characters per chunk for no-space writing systems.
            is_no_space: Whether the source text uses a no-space writing system
                (Chinese, Japanese, Thai, etc.). When True, character count is used
                as the unit instead of word count. This should be derived from the
                actual source text, not from a language code.

        Uses a simple sentence-boundary regex that handles:
        - Period / exclamation / question mark followed by whitespace (Latin scripts)
        - CJK (Chinese, Japanese and Korean-alike languages) sentence-ending
          punctuation (U+3002, U+FF01, U+FF1F) without requiring trailing whitespace, since CJK
          sentences run together.

        """
        # Sentence-boundary split: keep the delimiter attached to the preceding sentence.
        # For no-space text (CJK etc.) use `\s*` because sentences run together without
        # whitespace. For all other text use `\s+`.
        if is_no_space:
            # split on special sentence boundaries and spaces if present
            sentences = re.split(r"(?<=[\u3002\uff01\uff1f])\s*", text)
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

            # Detect whether the source text uses a no-space writing system (e.g. Chinese,
            # Japanese, Thai) by examining the actual text rather than origin_language.
            # origin_language may be "Detect Language" or otherwise unavailable, so it is
            # not a reliable signal. A space ratio below 5% indicates a no-space script;
            # for space-delimited languages (English, German, Arabic, etc.) the ratio is
            # typically 15-20%.
            is_no_space_source = _is_no_space_text(cleaned)

            # For no-space text use character count as the threshold unit;
            # for space-delimited text use word count.
            text_size = len(cleaned) if is_no_space_source else len(cleaned.split())
            chunks = (
                self._chunk_text(cleaned, chunk_size, is_no_space=is_no_space_source)
                if text_size > chunk_threshold
                else [cleaned]
            )

            # Tokenise every chunk prefixed with the target-language tag expected by
            # the MADLAD-400 model (e.g. "<2de> ").
            all_input_tokens = [
                self.tokenizer.Encode(
                    f"<2{data['target_language']}> {chunk}",
                    out_type=str,
                )
                for chunk in chunks
            ]

            # translate_iterable streams all chunk token sequences through a single
            # coordinated set of translate_batch calls, enabling asynchronous prefetching
            # and (where inter_threads > 1) parallel translation. It preserves input
            # order: results are yielded in the same order as the source iterable.
            inference_config = {k: v for k, v in self.config["inference"].items()
                                if k != "max_batch_size"}
            max_batch_size = self.config["inference"].get("max_batch_size", 32)

            results = list(self.translator.translate_iterable(
                all_input_tokens,
                max_batch_size=max_batch_size,
                batch_type="tokens",
                **inference_config,
            ))

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
