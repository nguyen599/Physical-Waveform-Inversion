## mHC (Manifold-Constrained Hyper-Connections)

Research implementation of **mHC** (DeepSeek; https://arxiv.org/abs/2512.24880) as a drop-in variant of **Hyper-Connections** (https://arxiv.org/abs/2409.19606).

### What we're building

A runnable PyTorch implementation of the mHC layer update

`x_{l+1} = H_l^{res} x_l + H_l^{post,T} F(H_l^{pre} x_l, W_l)`

with the key constraints:

- `H_res`: **doubly stochastic** (Birkhoff polytope; entries ≥ 0, rows sum to 1, cols sum to 1), via **Sinkhorn-Knopp**.
- `H_pre`, `H_post`: **non-negative** mixing maps.

### Implementation direction

Static per-layer matrices:
- learn `H_res_logits ∈ R^{s×s}` and project to `H_res` with Sinkhorn
- learn `H_pre_logits`, `H_post_logits` and map to non-negative weights (e.g. softmax)

This is a research prototype aimed at correctness + clarity, not the paper's systems optimizations.

### Running (nanoGPT on FineWeb10B)

Run from `examples/nanogpt/`. Adjust `--nproc_per_node` to match your GPU count.

**6-layer configs (~20M params):**
```bash
python train.py config/train_fineweb10B.py
python train.py config/train_fineweb10B_hc.py
python train.py config/train_fineweb10B_mhc.py
python train.py config/train_fineweb10B_vres.py
```

**48-layer configs (~20M params):**
```bash
python train.py config/train_fineweb10B_48l.py
python train.py config/train_fineweb10B_hc_48l.py
python train.py config/train_fineweb10B_mhc_48l.py
python train.py config/train_fineweb10B_vres_48l.py
```

**Multi-GPU example:**
```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B_mhc_48l.py
```

#### Orthostochastic mHC option
mHC supports an orthostochastic H_res projection via Newton-Schulz. Set `mhc_h_res_proj = "orthostochastic"` in your config and keep `ns_steps`, `ns_eps`, `ns_coeffs` as provided in the mHC configs.

### Next steps planned
- [x] Value residual ablations with baseline/HC/mHC
- [ ] AltUP ablation
- [x] H^res = `(1−α)*I + α*S` instead of full doubly stochastic (branch: `feat/mhc-residual-identity-mix`)
- [ ] Replace sinkhorn-knopp w/ Muon's orthogonalization op
- [ ] U-net-based variants + value embeddings


### Acknowledgements

Built using code snippets from `nanogpt`, `lucidrains/hyper-connections` and my own mHC implementation.

### License

Apache 2.0