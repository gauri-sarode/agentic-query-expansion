"""One prompt per expansion operator. Each is corpus-grounded (conditioned
on retrieved evidence, not query-only) and asks for a single compact
search expression -- the LLM generates a search action, not free-form
prose, keeping output within the 32-64 token budget.
"""
from __future__ import annotations

_COMMON_TAIL = """Respond with ONLY the rewritten search query on a single line. \
No explanation, no quotes, no punctuation beyond what a search query would use. \
Keep it short and search-style -- do not change the core intent."""

VOCABULARY_PROMPT = """The query below may use informal or non-corpus terminology. \
Rewrite it using terminology that actually appears in the retrieved evidence, \
while preserving the original intent.

Original query: {query}

Retrieved evidence (top passages from the corpus):
{evidence}

""" + _COMMON_TAIL

CONTEXT_PROMPT = """The query below may be underspecified. Using the retrieved \
evidence, add the most important missing concept or qualifying facet the query \
leaves implicit -- do not introduce facts unsupported by the evidence.

Original query: {query}

Retrieved evidence (top passages from the corpus):
{evidence}

""" + _COMMON_TAIL

AMBIGUITY_PROMPT = """The query below may have more than one plausible interpretation \
given the retrieved evidence. Pick the interpretation best supported by the evidence \
and rewrite the query to disambiguate toward it.

Original query: {query}

Retrieved evidence (top passages from the corpus):
{evidence}

""" + _COMMON_TAIL

ENTITY_NORMALIZE_PROMPT = """The query below may reference an entity (name, \
abbreviation, alias) that differs from how the corpus refers to it. Using the \
retrieved evidence, rewrite the query using the canonical entity form found in \
the corpus.

Original query: {query}

Retrieved evidence (top passages from the corpus):
{evidence}

""" + _COMMON_TAIL
