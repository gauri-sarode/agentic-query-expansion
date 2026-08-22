# Failure taxonomy

Natural and injected episodes are labeled into this compact taxonomy so
the observability contribution has something concrete to predict and
recover from, rather than treating "observability" as dashboards around a
search experiment.

| Failure | Observable signature | Preferred recovery |
|---|---|---|
| Vocabulary mismatch | Low lexical support; semantic evidence stronger | `VOCABULARY` |
| Underspecification | Weak separation, dispersed results, very short query | `CONTEXT` |
| Ambiguity | Multiple coherent but divergent result clusters | `AMBIGUITY` |
| Entity mismatch | Alias/acronym/canonical-name disagreement | `ENTITY_NORMALIZE` |
| Intent drift | Expansion diverges from original; ranking shifts sharply | `ROLLBACK` |
| Confirmation bias | Expansion amplifies evidence from a weak initial ranking | `ROLLBACK`/`REPLAN` |
| Retriever disagreement | Sparse and semantic rankings strongly disagree | `INSPECT`/`REPLAN` |
| Tool degradation | Timeout, truncated candidates, missing reranking | `FALLBACK`/`STOP` |
| Budget exhaustion | Additional action yields little state improvement | `STOP` |
| Healthy retrieval | Stable, supported, high-confidence state | `NO_EXPAND` |

## Fault injection protocol

Seeded variants that deliberately: remove a high-IDF query term; append
an irrelevant corpus term; replace a grounding passage with a topically
similar non-supporting passage; inject an unsupported entity into a
generated expansion; truncate the retrieval pool; disable the reranker;
or simulate reranker/LLM timeout.

For each fault, record **detection -> diagnosis -> corrective action ->
recovered quality**, and keep a clean paired episode for the same query
so recovery is measured relative to the uncorrupted trajectory.

Strongest test:

```
Full observable agent  >  same agent without observability-driven recovery
```

on recovery rate and harm, at the same action budget.
