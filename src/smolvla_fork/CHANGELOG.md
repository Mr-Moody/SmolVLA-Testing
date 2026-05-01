# SmolVLA Fork CHANGELOG

## 2026-05-01 — Initial fork from lerobot==0.4.4

### Files vendored
- `modeling_smolvla.py` — forked from `lerobot/policies/smolvla/modeling_smolvla.py`
- `configuration_smolvla.py` — forked from `lerobot/policies/smolvla/configuration_smolvla.py`

### Files NOT vendored (imported directly from upstream)
- `smolvlm_with_expert.py`
- `processor_smolvla.py`

### Changes

#### `configuration_smolvla.py`
- Renamed class to `SmolVLAForkedConfig`, registered under `"smolvla_forked"`.
- Added `use_phase_conditioning: bool = False`.
- Added `phase_dropout_prob: float = 0.15`.

#### `modeling_smolvla.py`
- Renamed policy class to `SmolVLAPhasedPolicy`.
- `VLAFlowMatching` → `VLAFlowMatchingPhased`.
- In `VLAFlowMatchingPhased.__init__`: added `self.phase_embedding = nn.Embedding(6, hidden_dim)` when `use_phase_conditioning=True`. All weights zero-initialized with `nn.init.zeros_`.
- In `embed_prefix`: calls `self._phase_embed(state_emb, phase_id, self.training)` to add the phase embedding to the state token. No-op when `use_phase_conditioning=False`.
- `_phase_embed`: looks up the embedding, applies training-time dropout (replace with unknown token index 5), adds to state embedding.
- `forward` and `sample_actions` pass `phase_id` through to `embed_prefix`.
- `SmolVLAPhasedPolicy.forward` reads `batch.get("phase_id")` and passes it to the model.

### Zero-init invariant
With `use_phase_conditioning=True`, a freshly initialized `VLAFlowMatchingPhased` and a freshly initialized `VLAFlowMatching` (upstream) produce identical logits for the same input because all phase embedding weights are zero. This is verified in `tests/test_smolvla_fork.py`.
