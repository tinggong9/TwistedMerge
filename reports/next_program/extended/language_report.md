# Language checkpoint transition geometry

Execution commit: `74b1d5e9779324615eaf21fe53e7a8f8639190d2`. Four checkpoints from pinned `google/bert_uncased_L-2_H-128_A-2` revision `30b0a37ccaaa32f332884b96992754e246e48c5f` were partially fine-tuned on real SST-2, IMDb, Yelp, and Amazon sentiment subsets. Attention-output, MLP-intermediate, and final-hidden subspaces were measured with five resamples and 200 edge-shuffle nulls. No adapter was present, so adapter-subspace results are marked unavailable. The aligned correction changed accuracy by `-0.106250`; the complete gate did not pass.
