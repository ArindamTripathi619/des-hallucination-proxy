import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from utils import get_embedding_text_from_response, is_correct


def test_deepseek_unclosed_think():
    raw = "<think>working through reasoning that was cut off\nmore reasoning\nFinal: 42"
    out = get_embedding_text_from_response(raw, question=None, mc_embed_choices=False, model_alias='deepseek-r1')
    assert out.strip().endswith('42') or '42' in out


def test_deepseek_truncation():
    long = '<think>' + ('word ' * 2000) + '</think> Answer: final'
    out = get_embedding_text_from_response(long, question=None, mc_embed_choices=False, model_alias='deepseek-r1')
    # Should not be extremely long (we truncate post-think result)
    assert len(out) <= 500


def test_is_correct_open_exact():
    assert is_correct('paris', ['Paris'], 'open')

def test_is_correct_open_overlap():
    assert is_correct('the capital of France, Paris', ['Paris'], 'open')

def test_is_correct_open_levenshtein():
    assert is_correct('pariss', ['paris'], 'open')

def test_is_correct_mc_letter():
    choices = ['Paris', 'London', 'Rome', 'Berlin']
    assert is_correct('A', ['A'], 'mc', choices=choices)

def test_is_correct_mc_text():
    choices = ['Paris', 'London', 'Rome', 'Berlin']
    assert is_correct('Paris', ['Paris'], 'mc', choices=choices)
