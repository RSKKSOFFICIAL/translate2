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
_NO_SPACE_LANGUAGES = {"zh", "ja", "th", "my", "km", "lo", "bo"}

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

    def _chunk_text(self, text: str, max_words: int) -> list[str]:
        """Split text into sentence-boundary chunks of at most max_words words.

        Uses a simple sentence-boundary regex that handles:
        - Period / exclamation / question mark followed by whitespace or end-of-string
        - Newlines (already collapsed to spaces by clean_text, so this is a safety net)

        For no-space languages the concept of "word" doesn't apply the same way,
        but the sentence-boundary split still works because those languages use
        punctuation (。！？) as sentence terminators.
        """
        # Sentence-boundary split: keep the delimiter attached to the preceding sentence
        sentences = re.split(r"(?<=[.!?。！？])\s+", text)

        chunks: list[str] = []
        current_words: list[str] = []
        current_count = 0

        for sentence in sentences:
            word_count = len(sentence.split())
            if word_count == 0:
                continue

            # If adding this sentence would overflow the chunk, flush first
            if current_count + word_count > max_words and current_words:
                chunks.append(" ".join(current_words))
                current_words = []
                current_count = 0

            # If a single sentence is longer than max_words on its own, hard-split it
            if word_count > max_words:
                words = sentence.split()
                for i in range(0, len(words), max_words):
                    chunks.append(" ".join(words[i:i + max_words]))
                continue

            current_words.append(sentence)
            current_count += word_count

        if current_words:
            chunks.append(" ".join(current_words))

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

            # Only chunk if the input exceeds the threshold
            if len(cleaned.split()) > chunk_threshold:
                chunks = self._chunk_text(cleaned, chunk_size)
            else:
                chunks = [cleaned]

            translated_chunks: list[str] = []
            for chunk in chunks:
                input_tokens = self.tokenizer.Encode(
                    f"<2{data['target_language']}> {chunk}",
                    out_type=str,
                )
                # Cap max_decoding_length proportionally to the input token count.
                # This applies to both chunked and non-chunked inputs to prevent
                # runaway repetition loops (e.g. Hindi/Devanagari producing endless
                # '=' characters). The multiplier is configurable via
                # chunking.max_decoding_multiplier (default 3). Languages that
                # expand significantly (e.g. French ~1.3x, Devanagari ~1.5x) may
                # need a higher value. Floor of 64 handles very short inputs.
                chunk_max_decoding = max(len(input_tokens) * max_decoding_multiplier, 64)
                inference_config = {
                    **self.config["inference"],
                    "max_decoding_length": chunk_max_decoding,
                    "repetition_penalty": max(
                        self.config["inference"].get("repetition_penalty", 1.0), min_repetition_penalty
                    ),
                }
                results = self.translator.translate_batch(
                    [input_tokens],
                    batch_type="tokens",
                    **inference_config,
                )

                if len(results) == 0 or len(results[0].hypotheses) == 0:
                    raise ServiceException("Empty result returned from translator")

                # todo: handle multiple hypotheses
                translated_chunks.append(self.tokenizer.Decode(results[0].hypotheses[0]))

            translation = self._join_chunks(translated_chunks, data["target_language"])
            elapsed = perf_counter() - start
            logger.info(f"time taken: {elapsed:.2f}s")
        except Exception as e:
            raise ServiceException("Error translating the input text") from e

        logger.debug(f"Translated string: {translation}")
        return translation
