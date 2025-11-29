import argparse
import json
import pytest
from types import SimpleNamespace

# Replace this with the actual import if the class is in a module.
from localEmb.config import EmbbederArgs


@pytest.fixture
def default_values():
    return {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "show_progress": False,
        "model_kwargs": {},
        "encode_kwargs": {}
    }


def test_add_cli_args_defaults(default_values):
    parser = argparse.ArgumentParser()
    parser = EmbbederArgs.add_cli_args(parser)
    args = parser.parse_args([])

    assert args.model_name == default_values["model_name"]
    assert args.show_progress == default_values["show_progress"]
    assert args.model_kwargs == default_values["model_kwargs"]
    assert args.encode_kwargs == default_values["encode_kwargs"]


def test_add_cli_args_custom_values():
    parser = argparse.ArgumentParser()
    parser = EmbbederArgs.add_cli_args(parser)

    model_name = "custom-model"
    show_progress = True
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {'normalize_embeddings': True}

    cli_args = [
        "--model_name", model_name,
        "--show_progress", str(show_progress),
        "--model_kwargs", json.dumps(model_kwargs),
        "--encode_kwargs", json.dumps(encode_kwargs)
    ]

    args = parser.parse_args(cli_args)

    assert args.model_name == model_name
    assert args.show_progress is True
    assert args.model_kwargs == model_kwargs
    assert args.encode_kwargs == encode_kwargs


def test_from_cli_args_custom_values():
    model_name = "custom-model"
    show_progress = True
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {'normalize_embeddings': True}

    args = SimpleNamespace(
        model_name=model_name,
        show_progress=show_progress,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    emb_args = EmbbederArgs.from_cli_args(args)

    assert emb_args.model_name == model_name
    assert emb_args.show_progress is True
    assert emb_args.model_kwargs == model_kwargs
    assert emb_args.encode_kwargs == encode_kwargs


def test_from_cli_args_defaults(default_values):
    args = SimpleNamespace(
        model_name=default_values["model_name"],
        show_progress=default_values["show_progress"],
        model_kwargs=default_values["model_kwargs"],
        encode_kwargs=default_values["encode_kwargs"]
    )

    emb_args = EmbbederArgs.from_cli_args(args)

    assert emb_args.model_name == default_values["model_name"]
    assert emb_args.show_progress == default_values["show_progress"]
    assert emb_args.model_kwargs == default_values["model_kwargs"]
    assert emb_args.encode_kwargs == default_values["encode_kwargs"]