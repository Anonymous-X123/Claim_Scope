# ClaimScope matched-control analysis

Analyzer `0.1.0`; 72 tasks; 48 unique templates.

## Overall results and inference cost

| Condition | JSON | Verdict | Witness | Safe | Strict | Generations | Output tokens | GPU seconds | Safe / 1k output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | 72/72 | 43/72 | 21/72 | 18/72 | 15/72 | 72 | 8719 | 853.9 | 2.064 |
| self_reflect | 70/72 | 41/72 | 18/72 | 15/72 | 13/72 | 144 | 17457 | 1757.5 | 0.859 |
| counterexample_guided | 72/72 | 43/72 | 20/72 | 17/72 | 14/72 | 144 | 17418 | 1744.9 | 0.976 |
| generic_feedback | 71/72 | 44/72 | 23/72 | 20/72 | 17/72 | 144 | 17367 | 1772.3 | 1.152 |
| verifier_feedback | 71/72 | 43/72 | 30/72 | 25/72 | 17/72 | 144 | 17436 | 1672.4 | 1.434 |

## Template-weighted results

Each unique mathematical source specification receives equal weight; `surface_variant` is excluded from the template key.

| Condition | Templates | Verdict | Witness | Safe | Strict |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct | 48 | 0.569 | 0.247 | 0.184 | 0.122 |
| self_reflect | 48 | 0.528 | 0.212 | 0.149 | 0.108 |
| counterexample_guided | 48 | 0.569 | 0.240 | 0.177 | 0.115 |
| generic_feedback | 48 | 0.580 | 0.267 | 0.205 | 0.142 |
| verifier_feedback | 48 | 0.569 | 0.403 | 0.299 | 0.153 |

## Paired changes

| Comparison | Witness delta | Safe delta | Strict delta | Safe gains | Safe losses | Template-safe improved / unchanged / worsened | Template-safe mean delta (95% cluster bootstrap interval) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct_to_self_reflect | -3 | -3 | -2 | 0 | 3 | 0 / 46 / 2 | -0.035 ([-0.090, +0.000]) |
| direct_to_counterexample_guided | -1 | -1 | -1 | 0 | 1 | 0 / 47 / 1 | -0.007 ([-0.021, +0.000]) |
| direct_to_generic_feedback | +2 | +2 | +2 | 2 | 0 | 1 / 47 / 0 | +0.021 ([+0.000, +0.062]) |
| direct_to_verifier_feedback | +9 | +7 | +2 | 7 | 0 | 6 / 42 / 0 | +0.115 ([+0.031, +0.208]) |
| counterexample_guided_to_verifier_feedback | +10 | +8 | +3 | 8 | 0 | 7 / 41 / 0 | +0.122 ([+0.042, +0.215]) |
| generic_feedback_to_verifier_feedback | +7 | +5 | +0 | 7 | 2 | 6 / 41 / 1 | +0.094 ([+0.000, +0.198]) |

## Provenance

- `direct` run SHA-256: `9d6c3cb8ffb4701494e855d7eb80fa46b0fd059604a73dac64b211a1e4b6bf33`
- `direct` score SHA-256: `c991959d445a6b9a513cfa5ce2d50651a0dc6b14bc02a1efc2a58649f74ba9b9`
- `self_reflect` run SHA-256: `9ab047339e79243b96e9646351650c66b4056b666cd15f5053cbaa2d7c65fb0b`
- `self_reflect` score SHA-256: `330077d741965d081e3ba77ab52a6b6df685380ee0cebbe91135f502cb3ec097`
- `counterexample_guided` run SHA-256: `9ab047339e79243b96e9646351650c66b4056b666cd15f5053cbaa2d7c65fb0b`
- `counterexample_guided` score SHA-256: `330077d741965d081e3ba77ab52a6b6df685380ee0cebbe91135f502cb3ec097`
- `generic_feedback` run SHA-256: `cddd01c473c51777ec687a9f9fc79b5f9d0e86f143e580ebe5453f3340ea995c`
- `generic_feedback` score SHA-256: `3cc4eb7cc69aabe06cedb4db3a813134931fa71848fb1684c672be445dd7194e`
- `verifier_feedback` run SHA-256: `4fb3a4cf7f2b6c38094703682f8c292efd711e204fc808b3fc338694bdb51276`
- `verifier_feedback` score SHA-256: `f5568f15673fa598e31b1790ac1d2260727cb0e7c06fc244bc2cc5a4141d18b0`
- Gold SHA-256: `071fa625f6e12648d438e8a2afc65475c27f6c5224db848a24955bc0c681d121`
