import sys
import pathlib
import pytest

# Ensure src/ is on sys.path for tests (repo layout)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from utils import get_embedding_text_from_response


def test_mc_choice_mapping_simple():
    question = {"question_type": "mc", "choices": ["Paris", "London", "Rome", "Berlin"]}
    raw = "A"
    out = get_embedding_text_from_response(raw, question=question, mc_embed_choices=True)
    assert out == "Paris"


def test_mc_choice_mapping_letter_in_text():
    question = {"question_type": "mc", "choices": ["Red", "Blue", "Green", "Yellow"]}
    raw = "Answer: b"
    out = get_embedding_text_from_response(raw, question=question, mc_embed_choices=True)
    assert out == "Blue"


def test_mc_choice_fallback_to_final():
    # When letter cannot be extracted, fallback to final answer extraction
    question = {"question_type": "mc", "choices": ["X", "Y", "Z", "W"]}
    raw = "<think>calculating...</think> Y"
    out = get_embedding_text_from_response(raw, question=question, mc_embed_choices=True)
    assert out.strip().lower() == "y"
