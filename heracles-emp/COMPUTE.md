# COMPUTE ledger

Budget: USD 40.77, including idle time. Every run must finish within 2 hours. Pods were created and deleted by the orchestrator only until 17:25 IST; from then on the EMPIRICAL LEAD agent owns pod lifecycle (NORMS F21 amendment).

SECURE CLOUD ONLY (author, 17:00 IST). The default is an RTX 5090 (secure); the fallback is an RTX 4090 (secure).

Prices at 2026-09-25 16:50 IST (RunPod list prices, USD per hour):

| GPU | community | secure |
|---|---|---|
| RTX 4090 | 0.34 | 0.74 |
| RTX 5090 | 0.69 | 0.99 |
| RTX A5000 | 0.16 | 0.27 |

| pod id | GPU | cloud | $/h | created (IST) | deleted (IST) | hours | cost | experiment |
|---|---|---|---|---|---|---|---|---|
| pu7uto001uy9o5 | CPU cpu5g 8 vCPU / 32 GB | SECURE (EU-RO-1) | 0.368 | 2026-09-25 17:17 | 2026-09-25 18:52 | 1.58 | $0.58 | Lean formalization (paper proofs; not heracles-emp) |
| grbou4dlnvjo6f | RTX 4090 (5090 secure out of stock: flagged fallback) | SECURE | 0.74 | 2026-09-25 17:22 | 2026-09-25 19:00 | 1.63 | $1.21 | empirical sweep (handed to the empirical lead); deleted after all runs + predictions + controls pulled and sha256-verified (E0, E1, E6) |
| 0o915f2qtu4sou | RTX 4090 (5090 secure out of stock: flagged fallback) | SECURE | 0.74 | 2026-09-25 17:22 | 2026-09-25 19:00 | 1.63 | $1.21 | empirical sweep (handed to the empirical lead); deleted after all runs pulled + sha256-verified (E0 R3P2/R4P2, E1) |
| aywi71v0htdqs3 | RTX 4090 (5090 secure out of stock: flagged fallback) | SECURE | 0.74 | 2026-09-25 17:22 | 2026-09-25 18:46 | 1.40 | $1.04 | empirical sweep (handed to the empirical lead); deleted after all runs pulled + sha256-verified (E0 R3P3/R4P3, E1 nu5) |
| mceuqia93fkz9s | RTX 4090 (5090 secure out of stock: flagged fallback) | SECURE | 0.74 | 2026-09-25 17:22 | 2026-09-25 18:46 | 1.40 | $1.04 | empirical sweep (handed to the empirical lead); deleted by mistake while the E1 pull (48 runs) was still running -> those runs lost and rerun on P1/P2 (see E1/LOG.md); E0 R3P4/R4P4 and E7 had been pulled + verified |
| ll88fbo16sqxm7 | RTX 4090 (5090 secure out of stock at 17:31, re-checked with get-capacity: flagged fallback) | SECURE | 0.74 | 2026-09-25 17:31 | 2026-09-25 18:32 | 1.02 | $0.75 | empirical sweep (created by the empirical lead, P5); deleted after all runs pulled + sha256-verified (E0 R4P5, E1, E3) |
| bstipcgnttyiea | RTX 4090 (5090 secure out of stock at 17:31, re-checked with get-capacity: flagged fallback) | SECURE | 0.74 | 2026-09-25 17:31 | 2026-09-25 18:49 | 1.30 | $0.96 | empirical sweep (created by the empirical lead, P6); deleted after all runs pulled + sha256-verified (E1 nu4 p23/p31, E3 nu3 product/power) |
| ft2atnmgm4leq0 | RTX 4090 (flagged fallback; 5090 secure out of stock) | SECURE | 0.74 | 2026-09-25 17:47 | 2026-09-25 17:50 | 0.05 | ~$0.04 | P7: never used (user: cut idle pods; host CUDA 12.8 < cu130 torch) |
| ey12exkk03b6bs | RTX 4090 (flagged fallback; 5090 secure out of stock) | SECURE | 0.74 | 2026-09-25 17:47 | 2026-09-25 18:49 | 1.03 | $0.76 | empirical sweep (P8); deleted after all runs pulled + sha256-verified (E3 nu5, E8) |
| r0p6rkuwwan07u | RTX PRO 4500 Blackwell (flagged fallback: 5090 and 4090 secure out of stock at 17:47) | SECURE | 0.72 | 2026-09-25 17:47 | 2026-09-25 17:50 | 0.05 | ~$0.04 | P9: never used (user: cut idle pods) |
| 6gq79dq15sg18x | RTX PRO 4500 Blackwell (flagged fallback: 5090 and 4090 secure out of stock at 17:47) | SECURE | 0.72 | 2026-09-25 17:47 | 2026-09-25 17:50 | 0.05 | ~$0.04 | P10: never used (user: cut idle pods) |

Running total (heracles-emp sweep pods, from the rows above): $7.67 (excludes the Lean pod pu7uto001uy9o5, which the empirical lead never touched). All sweep pods deleted by 19:00 IST (list-pods at 19:00 returned no pods).
| kypd49tv1vs09h | RTX 4090 (5090 secure out of stock at every CUDA version: flagged fallback) | SECURE | 0.74 | 2026-09-26 00:00 | 2026-09-26 00:31 | 0.52 | $0.38 | AUDIT retrain of 6 sampled runs (created and deleted by the AUDIT agent); results pulled and sha256-verified (pod list + queue manifests, kept in audit/retrain/) before deletion; list-pods at 00:31 returned no pods |

AUDIT addendum (2026-09-26 00:31 IST): one audit pod, $0.38; heracles-emp total incl. audit $8.05 (Lean pod excluded).
