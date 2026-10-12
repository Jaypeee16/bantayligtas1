import pandas as pd
import pytest

from bantayligtas import config
from bantayligtas.language import detect_language

MIXED = [
    "Wala na kaming malinis na tubig dito.",
    "May pamilya na na-trap sa second floor. Need rescue po.",
    "Baha na sa barangay namin, hindi na makalabas ang mga tao.",
    "Need namin ng pagkain at drinking water sa evacuation center.",
    "Our team is offering free rescue transport sa flooded areas.",
]
ENGLISH = [
    "Bridge collapsed after the earthquake, roads are closed.",
    "Please send food and water to the evacuation center.",
    "The barangay hall is now an evacuation site",
    "Is anyone at the shelter? We may need help soon",
    "RT @user: Hurricane Harvey flooding in Houston http://t.co/abc",
    "",
]


@pytest.mark.parametrize("text", MIXED)
def test_taglish_detected(text):
    assert detect_language(text).language == "taglish"


@pytest.mark.parametrize("text", ENGLISH)
def test_english_detected(text):
    assert detect_language(text).language == "english"


def test_markers_reported():
    r = detect_language("Need rescue po, na-trap kami")
    assert "po" in r.markers and "kami" in r.markers
    assert 0 < r.filipino_ratio <= 1


def test_all_project_examples_with_tagalog_words_route_to_taglish():
    frame = pd.concat(pd.read_csv(config.DATA_DIR / f) for f in
                      ("taglish_adaptation_v1.csv", "taglish_heldout_v1.csv", "taglish_diagnostic_v1.csv"))
    results = frame["tweet_text"].map(detect_language)
    taglish_share = results.map(lambda r: r.is_taglish).mean()
    assert taglish_share > 0.9          # 159 / 170; the rest are pure English sentences
