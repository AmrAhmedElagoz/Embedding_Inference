from dataclasses import fields, dataclass
from pydantic import Field
from typing import Dict, Any
import argparse
import json


@dataclass
class EmbbederArgs:
    model_name: str
    show_progress: bool = False
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--model_name",
            type=str,
            default="sentence-transformers/all-mpnet-base-v2",
            help="Model name to use.",
        )

        parser.add_argument(
            "--show_progress",
            type=bool,
            default=False,
            help="Whether to show a progress bar",
        )

        parser.add_argument(
            "--model_kwargs",
            type=json.loads,
            default={},
            help="""Keyword arguments to pass to the Sentence Transformer model, such as `device`,
    `prompts`, `default_prompt_name`, `revision`, `trust_remote_code`, or `token`.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer""",
        )

        parser.add_argument(
            "--encode_kwargs",
            type=json.loads,
            default={},
            help="""Keyword arguments to pass when calling the `encode` method of the Sentence
    Transformer model, such as `prompt_name`, `prompt`, `batch_size`, `precision`,
    `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode""",
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EmbbederArgs":
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args
