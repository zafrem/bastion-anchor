"""Lexical bias detection in the Phase 2 verifier: stereotypes are flagged, neutral mentions are not."""
import pytest

from anchor.verifier import Verifier


@pytest.fixture
def v():
    return Verifier()


def bias(v, text):
    return v._detect_bias(text, 0.15)


@pytest.mark.parametrize("text", [
    "Women are too emotional to lead, and men are better leaders.",
    "Older workers cannot learn new technology and are too slow.",
    "People from that country are lazy and untrustworthy.",
    "Immigrants are dangerous and don't understand our values.",
    "He is too old to learn this.",
    "여자는 감정적이라서 리더가 될 수 없다",
])
def test_stereotypes_are_flagged(v, text):
    assert bias(v, text).exceeds_threshold, text


@pytest.mark.parametrize("text", [
    "The engineer, a woman named Alex, led the project and shipped on time.",
    "The course is open to staff of any age and covers the basics of data analysis.",
    "The report covers revenue, costs and next quarter's plan.",
    "She presented the results. Her team agreed.",
    "Older workers were offered a training course, and many took it.",
    "Immigrants contributed 12% of new businesses last year.",
    "The senior engineer reviewed the change.",
    "한국인 고객과 외국인 고객 모두 환불이 가능합니다.",
])
def test_neutral_text_is_clean(v, text):
    r = bias(v, text)
    assert not r.exceeds_threshold, (text, r)


def test_punctuation_does_not_hide_a_group(v):
    t = "He, him. He. His. Him! She."
    assert bias(v, t).overall_score > 0  # many mentions, lopsided


def test_korean_particles_still_count(v):
    t = "남성은 남성이 남성을 남성의 여성은"
    assert bias(v, t).overall_score > 0
