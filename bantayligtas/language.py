"""Automatic English vs. Filipino/Taglish detection.

A lightweight lexicon heuristic: count distinctive Tagalog function words and
common disaster vocabulary. It needs no extra downloads and is transparent
(the app shows which words triggered the decision). Users can always override
it with the manual language selector.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Distinctive Tagalog words. Short words that are also common in English
# ("at", "may", "pa", "si", "ka", "mo", "ba", "ay", "ni", "para") and loanwords
# used in English reports ("barangay") are deliberately excluded.
TAGALOG_MARKERS = frozenset(
    """
    ang ng mga sa na nang po opo ho
    ako ko akin kami namin naming kaming tayo natin atin sila nila kayo niyo ninyo
    siya niya ito iyan iyon yan yun yung dito diyan doon
    wala walang wla meron mayroon hindi di huwag wag rin din lang lamang naman
    nga kasi pero dahil kung kapag pag sana baka ngayon kahapon bukas mamaya
    kailangan kailangang tulong tulungan tumulong saklolo
    tubig baha binaha bumaha bagyo lindol ulan hangin agos ilog lupa
    lumikas inilikas ilikas nasira sira gumuho bumagsak naputol wasak
    nawawala nawala nahanap hinahanap mahanap sugatan nasugatan namatay patay
    bahay tirahan pamilya kalsada tulay tao mga tao residente lugar bayan
    agad malakas mabilis pagkain gamot bata batang matanda lolo lola kapatid
    nanay tatay anak tumataas pagtaas makalabas makain nilalamig
    salamat dasal ligtas nasalanta naapektuhan nakikiramay kapit
    iwasan iwasang magdala maghanda pakuluan makinig huminga hirap
    naman talaga sobrang grabe ganun ganito paano bakit saan sino ano
    marami maraming lahat isa dalawa tatlo dalawang tatlong
    ngayong habang matapos pagkatapos hangga ilang kabilang malawakang pagbaha
    gabi umaga hapon maaga naglilinis kwarto
    """.split()
)

# Words that never appear in English text: a single one is enough.
STRONG_MARKERS = frozenset(
    "ang ng mga sa nang po opo hindi ngayong habang matapos pagkatapos yung naman kasi".split()
)

_TOKEN = re.compile(r"[a-zñ]+")


@dataclass
class LanguageResult:
    language: str               # "english" or "taglish"
    filipino_ratio: float       # share of tokens that are Tagalog markers
    marker_count: int
    markers: list[str] = field(default_factory=list)

    @property
    def is_taglish(self) -> bool:
        return self.language == "taglish"


def detect_language(text: str) -> LanguageResult:
    tokens = _TOKEN.findall(str(text).lower())
    if not tokens:
        return LanguageResult("english", 0.0, 0, [])

    hits = [t for t in tokens if t in TAGALOG_MARKERS]
    ratio = len(hits) / len(tokens)
    strong = any(t in STRONG_MARKERS for t in hits)
    is_taglish = len(hits) >= 2 or strong or (len(hits) >= 1 and ratio >= 0.2)

    unique_hits = list(dict.fromkeys(hits))
    return LanguageResult(
        language="taglish" if is_taglish else "english",
        filipino_ratio=round(ratio, 3),
        marker_count=len(hits),
        markers=unique_hits,
    )
