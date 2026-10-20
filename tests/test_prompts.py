import pytest

from medbench.prompts import build_prompt, parse, parse_mcq, parse_yesno

MCQ = {"kind": "mcq", "lang": "en", "question": "Q?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}}
TW = {"kind": "mcq", "lang": "zh-TW", "question": "問?", "options": {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}}
YN = {"kind": "yesno", "lang": "en", "question": "Does X?", "context": "BACKGROUND: ...", "options": None}


@pytest.mark.parametrize("text,want", [
    ("The best choice is clearly ...\nAnswer: C", "C"),
    ("**Answer: B**", "B"),
    ("Answer: (D) Lisinopril", "D"),
    ("Final answer is A.", "A"),
    ("答案：C", "C"),
    ("最終答案為 B", "B"),
    ("Answer：Ｃ", "C"),
    ("I first thought Answer: A, but on reflection\nAnswer: D", "D"),
    ("The answer is \\boxed{B}", "B"),
    ("Option C is correct.", "C"),
    ("C", "C"),
    ("Answer: E", None),
    ("The answer is a beta blocker such as metoprolol.", None),
    ("Answer: b", "B"),
    ("Answer: c.\n", "C"),
    ("I would choose a different approach", None),
    ("A good answer needs more information.", None),
    ("", None),
    (None, None),
])
def test_parse_mcq(text, want):
    assert parse_mcq(text) == want


@pytest.mark.parametrize("text,want", [
    ("...\nAnswer: yes", "yes"),
    ("**Answer: No**", "no"),
    ("Answer: maybe", "maybe"),
    ("The evidence suggests\nyes", "yes"),
    ("Yes or no? It depends", None),
    ("", None),
])
def test_parse_yesno(text, want):
    assert parse_yesno(text) == want


def test_prompts():
    p = build_prompt(MCQ)
    assert "A. a\nB. b\nC. c\nD. d" in p and 'A, B, C or D' in p and p.rstrip().endswith("C or D.")
    t = build_prompt(TW)
    assert "A. 甲" in t and "A、B、C、D" in t and "Answer: X" in t
    y = build_prompt(YN)
    assert "BACKGROUND: ..." in y and "Does X?" in y and parse(YN, "Answer: Yes") == "yes"
    assert parse(MCQ, "Answer: b") == "B"
