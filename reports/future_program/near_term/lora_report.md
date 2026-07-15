# Real adapter benchmark

Four rank-4 adapters were trained from pinned `google/bert_uncased_L-2_H-128_A-2` revision `30b0a37ccaaa32f332884b96992754e246e48c5f` on bounded SST-2, IMDb, Yelp, and Amazon sentiment subsets. All predictions were executed and saved-logit permutation checks passed. The persistent-residual-and-gain gate was **not passed**; no positive claim is made when cycle maps close or correction fails to improve held-out accuracy.
