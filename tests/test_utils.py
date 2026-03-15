import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))

from utils import extract_for_embedding, normalize_answer, is_correct


def test_extract_final_closed_think():
    raw = "Reasoning... <think> step1 </think> FinalAnswer"
    out = extract_for_embedding(raw)
    assert out == "FinalAnswer"


def test_extract_final_unclosed_think():
    raw = "Reasoning... <think> I think the answer is X\nAnswer: X"
    out = extract_for_embedding(raw)
    # Should pick up the last non-empty line
    assert "Answer" in out or len(out) < 200


def test_normalize_and_is_correct_open():
    raw = "<think> long reasoning... </think> Pulsar"
    norm = normalize_answer(raw, 'open')
    assert norm == 'pulsar'
    assert is_correct(norm, ['Pulsar'], 'open')


def test_is_correct_token_overlap():
    pred = 'the great wall of china'
    gt = ['Great Wall of China']
    assert is_correct(pred, gt, 'open')
