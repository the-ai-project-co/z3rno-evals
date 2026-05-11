# Z3rno evals — `golden_v1` v1

Server: `http://127.0.0.1:8000` · Items: **10**

## Summary

| Metric | Value | Δ vs baseline |
|---|---|---|
| recall@k | 0.300 | — |
| MRR | — | — |
| entity coverage | 0.300 | — |
| faithfulness | 0.375 | — |
| p50 latency (ms) | 16.6 | — |
| p95 latency (ms) | 19154.8 | — |
| p99 latency (ms) | 19154.8 | — |
| items within budget | 5/10 | — |

## Per-item results

| ID | Strategy | Latency (ms) | recall@k | MRR | Entity cov. | Faithfulness | Within budget | Error |
|---|---|---|---|---|---|---|---|---|
| q-001 | VECTOR | 15775.1 | 0.000 | — | 0.000 | 0.286 | ❌ | — |
| q-002 | LEXICAL | 16.6 | 0.000 | — | 0.000 | 0.000 | ✅ | — |
| q-003 | GRAPH | 9.6 | 0.000 | — | 0.000 | 0.000 | ✅ | — |
| q-004 | TRIPLET | 10.8 | 0.000 | — | 0.000 | 0.000 | ❌ | Validation error: strategy 'TRIPLET' requires an LLM gateway, which is not configured on this server |
| q-005 | TEMPORAL | 13.3 | 1.000 | — | 1.000 | 1.000 | ✅ | — |
| q-006 | ASK | 7.4 | 0.000 | — | 0.000 | — | ❌ | Validation error: strategy 'ASK' requires an LLM gateway, which is not configured on this server |
| q-007 | CODE | 57356.6 | 0.000 | — | 0.000 | 0.000 | ❌ | Server error: Internal Server Error |
| q-008 | VECTOR | 19154.8 | 1.000 | — | 1.000 | 1.000 | ❌ | — |
| q-009 | TRACE | 25.7 | 1.000 | — | 1.000 | 0.714 | ✅ | — |
| q-010 | LEXICAL | 10.4 | 0.000 | — | 0.000 | — | ✅ | — |
