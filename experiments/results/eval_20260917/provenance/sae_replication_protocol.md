# SAE check on additional WikiText articles

This follow-up is specified after inspecting the initial transfer and its mechanism
controls, before evaluating the additional articles. It is an additional sample
within WikiText-2, not a new benchmark or a preregistered experiment.

- Data: 64 nonoverlapping 1,024-token windows, packed with EOS between documents.
  All 31 test documents used by the first 128-window run are excluded in full.
  The new cache uses 14 other documents, shuffled with seed 20260922; the dataset
  fingerprint is unchanged. The last included document is truncated at the cache
  boundary. Document IDs are in [the cache manifest](sae_replication_cache_metadata.json).
- Checkpoint: the same 300M DAGFormer step 9000 and exported SAE directions.
- Fixed features: 7326, 1986, 3583, 3560, 7019, 7068, 452 and 4222. No direction
  or target-token set is reselected.
- Arms: each feature, the same five whole-head controls, R only, and Q/K/V only,
  each at alpha -4 and +4: 128 edited arms plus one shared reference. Control
  generation uses seed 20260917, preserving the earlier head permutations.
- Outcomes: target-token and other-token NLL changes, signed dose spans, component
  effects, and feature-minus-control span contrasts. Target-token pieces and their
  occurrence counts remain visible; missing targets are unassessable.
- Uncertainty: paired, token-weighted bootstrap with 10,000 draws over windows;
  sensitivity uses adjacent-window blocks of 4, 8 and 16. All eight features and
  all controls are retained, including failures. Opposite-sign dose effects are
  assessed with each dose's interval; a positive span alone is not sufficient.

The purpose is to check which fixed loss effects and component interpretations
survive a change of articles. No free-generation or clean semantic-control claim
follows from this endpoint alone.
