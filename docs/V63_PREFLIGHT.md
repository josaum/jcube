# V6.3 Pre-flight — audit findings and the gate for the next paid run

**Date:** 2026-07-25
**Context:** Full review of `scale_pipeline_v6.py` before launching another GPU
run. V5 spent ~$35 of its ~$78 on failed runs; every item here maps to a way
that happens again.

## Findings → fixes (all landed in this branch)

| # | Finding | Fix |
|---|---------|-----|
| 1 | DDP never synced encoder/predictor grads (no `DistributedDataParallel` wrapper existed despite the docstring) — a 4-GPU run trained 4 divergent encoders and kept only rank 0's | Encoder wrapped in DDP (`find_unused_parameters=True` for the unused `rwse_proj`); predictor + TGN grads all-reduced manually (the legacy lookahead loop runs multiple predictor forwards per backward, which DDP's reducer rejects) |
| 2 | TGN `msg_fn`/GRU only ever ran under `no_grad` — never trained, frozen at random init, injected into node features for 3 of 5 epochs | `TGNMemory.integrate_batch()` integrates batch edges inside the gradient path (edge features detached so encoder submodules stay inside the DDP forward); buffer committed post-step via `commit_batch()` |
| 3 | Dense lookahead is cross-entity: chronologically sorted *random* seeds predict each other — not any entity's own future | New `lookahead_mode="same_entity"`: per-batch time cutoff `t_c`, online encoder sees only edges ≤ `t_c`, each seed predicts its OWN full-history EMA representation at `dt = t_last − t_c`. Default stays `cross_entity` (checkpoint continuity); the smoke A/B decides |
| 4 | ~90GB of checkpoint writes per epoch (full state saved twice + embeddings twice) | Latest is a JSON pointer; per-epoch `node_embeddings.pt` duplicate dropped (epoch-numbered files kept for probes) |
| 5 | Dense 18GB embedding gradient per step (`nn.Embedding sparse=False`) — OOM risk on the default `H100:4` (80GB) spec | `sparse=True` (guarded: requires `emb_momentum=0`, `emb_weight_decay=0`) |
| 6 | On a cold cache all DDP ranks encoded ontology concurrently and raced `jepa_cache.commit()` | Rank 0 encodes; other ranks wait at a barrier and read the cache |
| 7 | `_build_csr_cache` saved a marker file that cached nothing and skipped the (equally useless) warm build | Removed; sampler speed depends on pyg_lib, which the smoke action verifies |
| 8 | `pyg_lib` wheel for torch 2.12+cu130 never verified in isolation (V5's 10x slowdown was exactly a missing pyg_lib) | `--action smoke` on a cheap L4: imports, `WITH_PYG_LIB`, one real NeighborLoader batch |
| 9 | Numeric column pick iterated a Python **set** — hash randomization could silently switch a table between `VL_TOTAL`/`VL_GLOSA` across materializations | `NUMERIC_PRIORITY` tuple + `_pick_numeric_column()` (regression-tested) |
| 10 | Entrypoint ignored `--parquet-path`/`--ontology-path` for the materialize step | Paths passed through to `materialize_remote` |
| 11 | "Strict causality" mask is a no-op (`node_time` = max incident edge time ⇒ `delta_t ≥ 0` always); stale prints claimed A100 and a 3+7-epoch curriculum | Comment now states what the mask actually does; prints derive from `CURRICULUM_PHASES_V6` |

Legacy V6.2 checkpoints still resume: `_clean_state_dict()` strips the old
`_orig_mod.` (compiled-module) key prefixes, and the legacy
`checkpoint_latest.pt` is used when no JSON pointer exists.

## Known-open (deliberately not fixed here)

- **Probe leakage:** embeddings see each admission's full history including
  outcomes, so LOS/Glosa probes partially read the label (V5's 0.690 AUC
  included). Honest measurement needs time-split probes (embed with edges
  truncated at admission time). Applies to the verify script, not the trainer.
- **Per-rank graph loading:** each DDP rank still loads the 165M-edge table
  independently (~memory-bounded, works in the 200GB container).
- `onto_projection` is a fixed random projection (never trained; used once at
  init). Harmless; left as-is.

## The gate — run in this order, no skipping

1. `modal run event_jepa_cube/scale_pipeline_v6.py --action smoke`
   (~cents). Must print `with_pyg_lib: True` and `sampler_ok: True`.
2. Single-hospital smoke train (~$5):
   `--action train --config '{"hospital_filter":"GHO-BRADESCO","epochs":2}'`
   — verify loss decreases, TGN grads flow (watch the `[DIAG]` lines), no OOM.
3. Same-hospital A/B: repeat step 2 with
   `'{"hospital_filter":"GHO-BRADESCO","epochs":2,"lookahead_mode":"same_entity"}'`
   and compare probe results. The winner's mode goes into the full run.
4. Full run only after 1–3 pass, then the `verify_v62_epoch1.py` protocol
   (delta check → structural health → probes) before promoting anything.
