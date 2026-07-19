# LoRA dataset and model audit

Audit date: 2026-07-19

## Controlled smoke

The active smoke is synthetic float64 linear algebra. It downloads no model or dataset, contains no human-subject data, performs no model training, and has no external-data license dependency. It is suitable for the first stage because the question is exact representation invariance under a planted `GL(r)` action.

Expected compute is under one CPU minute for four rank-3 adapters, four gauge families, and 20 scrambles per well-conditioned family. The output is a fixed-rank synthetic merged delta plus diagnostic CSVs; it is not a pretrained-model result.

## Previously used real-adapter setup

The repository's existing language path pins:

| Item | Revision | Published license metadata | Audit decision |
|---|---|---|---|
| `google/bert_uncased_L-2_H-128_A-2` | `30b0a37ccaaa32f332884b96992754e246e48c5f` | Apache-2.0 | model usable after protocol review |
| `stanfordnlp/sst2` | `8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb` | unknown | blocked for a new confirmatory run |
| `stanfordnlp/imdb` | `e6281661ce1c48d982bc483cf8a173c1bbeb5d31` | other | blocked pending explicit terms |
| `fancyzhx/yelp_polarity` | `bbf1c97a1f0cf005e5aded43839fd814654a1557` | no license metadata returned | blocked pending explicit terms |
| `fancyzhx/amazon_polarity` | `9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289` | Apache-2.0 | dataset usable after protocol review |

Primary metadata endpoints:

- <https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/main>
- <https://huggingface.co/datasets/stanfordnlp/sst2>
- <https://huggingface.co/datasets/stanfordnlp/imdb>
- <https://huggingface.co/api/datasets/fancyzhx/yelp_polarity>
- <https://huggingface.co/api/datasets/fancyzhx/amazon_polarity>

The Hugging Face dataset-script license, where present, is not treated as a license for the underlying dataset. A cached artifact also does not resolve an unclear license.

## Split and leakage status

The prior BERT experiment used bounded task subsets and saved outputs, but it is not adopted as the new confirmatory protocol. Before any real pilot, the replacement or resolved suite must document each source's official train/validation/test split, construct validation only from training data when necessary, freeze all choices before test evaluation, and record the validation-evaluation count for every method.

## Compute and suitability gate

The local environment has the required Python packages and a cached small model, so dependencies are not the active blocker. The active blocker is dataset licensing plus the need to freeze a new, genuine rank-space synchronization protocol. No real adapter training is authorized until both are resolved and the controlled smoke passes.
