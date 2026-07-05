# Roadmap — from this repo to the full EoWorld pipeline

This drop delivers the **perception + state-extraction** foundation. The rest of
the proposal builds on the query-token sequences produced by
`scripts/06_extract_query_tokens.py` (per-video `.npz`, each holding
`query_tokens (T, num_q, D)`, `class_conf`, `pred_class`, and optional
`class_logits`).

### Done (Week-1 fast-track, §11.1)
- [x] CholecSeg8k data pipeline into EoMT (video-level splits, watershed decoding)
- [x] Dataset understanding + journal-ready figures
- [x] EoMT-S/B/L fine-tuning configs (Exp. 8.1 — segmentation fidelity)
- [x] Query-token state extraction (the world-model latent z_t)
- [x] Quick-setup smoke gate + ungated checkpoints

### Next
- [ ] **Latent dynamics model (Exp. 8.2).** Small causal-Transformer / GRU over
      `z_1..z_t` predicting `z_{t+1..t+k}` (k ∈ {1,5,15}). Decode predicted tokens
      back to masks with EoMT's own frozen `mask_head` (`models/eomt.py::_predict`)
      and score IoU@horizon. Baselines: optical-flow mask propagation, ConvLSTM/
      PredRNN over rendered masks, decoder-then-forecast.
- [ ] **Uncertainty head (Exp. 8.4).** Propagate per-query `class_conf` through the
      rollout + MC-dropout / deep-ensemble over the dynamics model; ECE + reliability
      diagrams. Hooks for `class_conf` are already emitted per frame.
- [ ] **Runtime feasibility (Exp. 8.5).** Perception + dynamics latency on the 5090;
      EoMT-S config is the intended measurement point.
- [ ] **Cross-dataset (EndoVis17/18) + planning (SAR-RARP50, Exp. 8.3).** New data
      modules mirroring `eoworld/data/cholecseg8k.py`; a planning/action head on the
      forecast rollout.

### Decoding predicted tokens back to masks (implementation note)
The dynamics model predicts tokens in the same space as EoMT's final query
embeddings. To render a predicted mask, reuse the frozen upstream heads:
`mask_logits = einsum("bqc,bchw->bqhw", mask_head(z_pred), upscale(patch_tokens))`
— i.e. `EoMT._predict` with `z_pred` substituted for the live queries. Cache the
patch-token feature map alongside `z_t` when you need pixel-space forecasts.
