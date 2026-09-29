"""Turning whatever a caller passes into something a literature API can match.

Callers do not pass search queries. The log shows entire research objectives
arriving at these clients -- thousands of characters of data description,
lists of prior findings, and in one case the model's own reasoning text ("We
need answer one focused testable research question ... Let's"). OpenAlex
answered those with 400s, arXiv with HTTP 500s, and the whole federated search
timed out at 90 seconds waiting for them.

This lives beside the clients rather than inside one of them because the
reduction was first written into the OpenAlex client alone -- which fixed
OpenAlex and left arXiv and PubMed receiving the same unreduced text. A query
policy that only one of three clients obeys is not a policy.
"""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)


# Vocabulary that describes the TASK rather than the science. It dominates the
# objectives Kosmos passes here -- "determine whether", "report effect size",
# "the served datasets" -- and matching on it returns methodology reviews.
#
# The line is drawn at "describes doing science" versus "is the content of
# some science". An earlier version of this list was tuned against Mendelian
# randomization text and swept up `instrument`, `product`, `sign`, `effect`
# and `direction` -- scaffolding in THAT domain and the subject itself in
# others. It reduced "instrument calibration drift in mass spectrometry" to
# "calibration drift spectrometry" and "product inhibition kinetics of
# hexokinase" to "inhibition kinetics hexokinase", deleting the noun the query
# was about. A blocklist tuned on one field is a bug in every other field, so
# a word stays out of here unless it is meaningless as a research subject.
_TASK_WORDS = frozenset("""
the a an and or of to in on for with by from as at is are was were be been being this that these those
it its their they we you them us our your which what who whose when where why how not no nor but if
then than so such both each few more most other some any all can could may might must shall should
will would do does did done have has had having need needs needed use used using uses single sentence
preamble real column columns name names frame relationships among available analyses propose one
question questions likely testable via association between let given provided dataset datasets data
table tables row rows value values variable variables analysis analyses determine whether tested
compared control controls experiment experiments result results report reports significant
significantly test tests testing hypothesis hypotheses objective task find finding findings perform
consider appropriate suitable steps step make sure keep separate after before following below above
here there now study studies paper papers described description instructions verbatim used check
state every candidate rank explicit only also across their about into over under
answer answers focused research researcher available propose single overall
genetically predicted causally circulating levels level estimate estimates estimated
higher lower larger smaller increase increased decrease decreased magnitude
observed expected shared across within between
chance often more same
""".split())

# ALL-CAPS tokens that are STATISTICS COLUMNS, not subjects. `BETA` and
# `LOG10P` appear in every summary-statistics table ever written, so treating
# them as identifiers spends the small term budget on words that match
# nothing: one real hypothesis reduced to "T1 BETA effect direction", where
# only `T1` carried meaning.
#
# Every entry is at least four characters, and that limit is the point. The
# first version of this list held `N`, `P`, `F`, `T` and `NA` -- which are
# nitrogen, phosphorus, fluorine, T cells and sodium. It reduced "NA K ATPase
# pump activity" to "atpase activity" and "P and N limitation in
# phytoplankton" to "phytoplankton limitation", deleting the element the study
# was about. A short token is somebody's identifier; only a long one is safely
# a column name.
_COLUMN_NAMES = frozenset({
    "BETA", "SEBETA", "STDERR", "LOG10P", "PVAL", "PVALUE", "CHISQ",
    "A1FREQ", "A2FREQ", "ZSCORE", "TSTAT", "CHROM", "NOBS", "NCASE",
})

# Deliberately small. Measured against the live API on a real question:
# nine terms returned "Cytochrome P450 enzymes in drug metabolism" and
# "Nitric Oxide and Peroxynitrite in Health and Disease"; five returned
# "A Mendelian randomization study of IL6 signaling in cardiovascular
# diseases" and "Endoplasmic Reticulum Protein TXNDC5 Augments Myocardial
# Fibrosis". OpenAlex ranks a long multi-concept phrase toward highly-cited
# general reviews, so adding terms makes the result WORSE, not broader.
_MAX_SEARCH_TERMS = 6

# ...and fewer still once an identifier is in hand. Measured on a real
# hypothesis statement: "SOD2 myocardial genetically predicted circulating
# protein" (6) returned "Cytochrome P450 enzymes in drug metabolism", while
# "SOD2 myocardial fibrosis T1" (4) returned "Genetics of myocardial
# interstitial fibrosis in the human heart". A gene symbol is a near-unique
# key, and every generic word added to it pulls the match back toward the
# broad reviews it was supposed to cut through.
_MAX_SEARCH_TERMS_WITH_ID = 4


def _keywords(text: str, max_terms: int = _MAX_SEARCH_TERMS) -> str:
    """The few terms worth searching for, out of whatever the caller passed.

    Callers do not pass search queries. The log shows entire research
    objectives arriving here -- thousands of characters of data description,
    lists of prior findings, and in one case the model's own reasoning text
    ("We need answer one focused testable research question ... Let's"). None
    of that is a query, and OpenAlex answered every one of them with a 400 or
    with methodology reviews.

    Identifiers first (SOD2, HLA-DRA, CD14): an ALL-CAPS token is the highest
    signal available and the thing a literature search should hinge on. Then
    ordinary words by frequency, which in a long objective is what the text is
    actually about.
    """
    if not text:
        return ""
    from collections import Counter

    caps: List[str] = []
    words: List[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text):
        if token.isupper() and 2 <= len(token) <= 12:
            if token not in _COLUMN_NAMES:
                caps.append(token)
        elif len(token) >= 5 and token.lower() not in _TASK_WORDS:
            words.append(token.lower())

    def _key(term: str) -> str:
        """`cis-pqtl` and `cis_pqtl` are one term, not two.

        Kosmos text carries both the prose spelling and the column spelling of
        the same thing, and counting them separately spent two slots of a
        four-slot budget on one concept.
        """
        return re.sub(r"[^a-z0-9]", "", term.lower())

    chosen: List[str] = []
    seen = set()
    cap_counts = Counter(caps)
    for term, count in cap_counts.most_common(3):
        # A two-character token like `F1` is usually an enumeration label
        # ("F1. Depletion of innate immune subsets"), not an identifier --
        # but `T1` in a T1-mapping objective is exactly the identifier that
        # matters. Repetition is what separates them: a real subject recurs,
        # a list label appears once.
        if len(term) < 3 and count < 2:
            continue
        if _key(term) not in seen:
            seen.add(_key(term))
            chosen.append(term)
    # An identifier is worth more than any number of generic words beside it.
    limit = min(max_terms, _MAX_SEARCH_TERMS_WITH_ID) if chosen else max_terms
    for term, _ in Counter(words).most_common():
        if len(chosen) >= limit:
            break
        if _key(term) not in seen:
            seen.add(_key(term))
            chosen.append(term)
    return " ".join(chosen[:limit])


def _sanitise_search(query: str) -> str:
    """Strip the two characters OpenAlex reads as wildcards.

    `?` and `*` are wildcard metacharacters in OpenAlex's search parser, and
    using them without an exact/no-stem search is rejected outright:

        400 Bad Request -- "Wildcards (* or ?) require exact (no-stem) search."

    Every Kosmos research question is phrased as a question and ends in `?`,
    so the literature agent was failing on essentially EVERY run and returning
    an empty list. Nothing downstream distinguishes "no papers found" from
    "the search never happened", so runs silently proceeded with no literature
    context at all.

    Length was never the problem -- the same 123-character question succeeds
    once the `?` is removed. So this removes the two characters and nothing
    else; truncating would throw away terms that work.
    """
    if not query:
        return ""
    return re.sub(r"\s+", " ", query.replace("?", " ").replace("*", " ")).strip()


def reduce_query(query: str) -> str:
    """The query every literature client should receive.

    One decision, made once, upstream of the fan-out: strip the characters
    that break a query parser, and reduce anything longer than a sentence to
    the terms worth matching on. Applying it per client meant arXiv and PubMed
    kept receiving the raw objective after OpenAlex stopped.
    """
    text = _sanitise_search(query)
    if not text:
        return ""
    if len(text) > 120 or len(text.split()) > _MAX_SEARCH_TERMS + 4:
        reduced = _keywords(text)
        if reduced:
            logger.info("Literature query reduced to %r (from %d chars)",
                        reduced, len(text))
            return reduced
    return text
