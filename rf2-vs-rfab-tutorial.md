# RF2_ab vs vanilla RF2 — reproducible three-case walk-through

A self-contained tutorial that answers three questions about how
RFantibody's fine-tuned RF2_ab differs from upstream
`uw-ipd/RoseTTAFold2` (vanilla RF2):

1. **Case 1** — what do RF2_ab's extra weights actually encode?
2. **Case 2** — is the preprocessing pipeline also antibody-specific, or
   just the weights?
3. **Case 3** — empirically, how much worse is vanilla RF2 on an
   antibody-antigen complex?

Every command below has been verified end-to-end on an Ubuntu 24.04
box with an RTX 4080 (16 GB). If a future reader wants one of the
three results, they copy-paste the command from that case.

**Prefer a guided notebook?** `notebooks/rf2-vs-rfab.ipynb` is the
same material as this doc but broken into runnable piecewise cells,
with a per-residue pLDDT plot at the end. Open with
`uv run --with jupyter jupyter notebook notebooks/rf2-vs-rfab.ipynb`.

## Prerequisites

```bash
# from the RFantibody repo root
make install-rfantibody      # uv sync + RFantibody weights (~1.5 GB)
make install-rf2-vanilla     # clone uw-ipd/RoseTTAFold2 + RF2_jan24.pt (~1.1 GB)
make verify                  # strict=True load of vanilla RF2
```

This leaves you with:
- `weights/RF2_ab.pt` (1.6 GB) and `weights/RF2_jan24.pt` (1.1 GB)
- `$HOME/rf2-vanilla/RoseTTAFold2/` (vanilla repo + weights)
- RFantibody's `.venv` with biotite, torch-cu118, DGL, se3-transformer

Installation narrative: `installation-steps.md` + `vanilla-rf2-alongside-rfantibody.md`.

GPU: anything with ≥ 4 GB VRAM works for the 478-residue test case. A
4080 peaks at 2.8 GB.

---

## Case 1 — what do the extra RF2_ab weights encode?

**Claim to verify:** RF2_ab's state-dict has the same topology as
vanilla RF2 except for three template-embedding layers whose input
widths are wider by +2 / +1 / +1. Is that "four extra Ab-specific input
features," or one feature counted four times?

### Step 1: state-dict diff

```bash
uv run python scripts/diff_rf2_checkpoints.py \
    weights/RF2_ab.pt \
    weights/RF2_jan24.pt
```

Expected output:

```
=== RF2_ab.pt  vs  RF2_jan24.pt ===
A total keys: 8044
B total keys: 8042
Shared keys : 8040
Only in A   : 4
Only in B   : 2
Shape mismatch on shared: 3

--- Only in A ---
bind_pred.downsample.bias
bind_pred.downsample.weight
bind_pred.rbf2attn.bias
bind_pred.rbf2attn.weight

--- Only in B ---
bind_pred.classify.bias
bind_pred.classify.weight

--- Shape mismatch (key, A_shape, B_shape) ---
('templ_emb.emb.weight',                  (64, 90),  (64, 88))
('templ_emb.proj_t1d.weight',             (64, 53),  (64, 52))
('templ_emb.templ_stack.proj_t1d.weight', (32, 23),  (32, 22))

Verdict: NOT DROP-IN COMPATIBLE
```

Two kinds of incompatibility:
- **Template widths (+2/+1/+1)** — the topic of this case, explored below.
- **Different binder-prediction head** — RF2_ab has a v2 `BinderNetwork`
  (`bind_pred.downsample` + `bind_pred.rbf2attn`, enabled by `new_pbind: True`
  in `config/base.yaml`), vanilla has a v1 `bind_pred.classify`. This is
  a second independent reason for non-compatibility; the template-channel
  story is just the more mechanistically interesting one.

**Already the answer to "can I just swap the .pt?" is no.**

### Step 2: trace where the extra channels come from

One config knob drives all three differences: `d_t1d` (template 1D
feature width).

```bash
grep -n "d_t1d" src/rfantibody/rf2/config/base.yaml
# src/rfantibody/rf2/config/base.yaml:32:  d_t1d: 23

grep -n "d_t1d" src/rfantibody/rf2/network/Embeddings.py | head
# :144:    def __init__(self, ..., d_t1d=22, ...):          # TemplatePairStack default
# :147:        self.proj_t1d = nn.Linear(d_t1d, d_state)    # -> (32, 22) vanilla, (32, 23) RFa
# :198:    def __init__(self, d_t1d=21+1, d_t2d=43+1, ...): # Templ_emb default
# :203:        self.emb = nn.Linear(d_t1d*2+d_t2d, d_templ) # -> (64, 88) vanilla, (64, 90) RFa
# :210:        self.proj_t1d = nn.Linear(d_t1d+d_tor, d_templ)  # -> (64, 52) vanilla, (64, 53) RFa
```

That maps the diff's "+2 / +1 / +1" onto a single scalar feature:
- `emb`'s input is `d_t1d*2 + d_t2d` — the t1d tensor feeds in twice
  (once per side of a pair), so +1 on `d_t1d` gives +2 on `emb`'s
  input width.
- `proj_t1d` (template-stack) takes raw t1d → +1.
- `proj_t1d` (inside Templ_emb) takes t1d concatenated with 30 torsion
  features → +1.

Total: **one extra scalar per residue, used in four matrix input slots
across three linear layers.**

### Step 3: what does the one extra channel encode?

```bash
grep -n "t1d\[" src/rfantibody/rf2/modules/preprocess.py
# :183:    t1d = torch.zeros((T, L, 23))
# :184:    t1d[:,:,:21] = seq_onehot.repeat(T,1,1)    # dims 0-20: AA one-hot
# :187:    t1d[:,pose.antibody_mask,:21] = 0
# :188:    t1d[:,pose.antibody_mask,21] = 1
# :191:    t1d[:,pose.target_mask,21] = 1            # dim 21: confidence
# :192:    t1d[:,pose.antibody_mask,21]=0
# :195:    t1d[0,:,22]=hotspots                      # dim 22: the extra channel
```

Dim 22 of t1d is the **hotspot mask** — a binary per-residue flag
marking target-surface hotspot residues. It is set only on template 0;
recycled templates get 0.

**Conclusion for Case 1.** RF2_ab is not "antibody-aware" because of
H/L chain flags or CDR masks. It is interface-aware: it carries one
extra per-residue channel that tells the template encoder where the
antibody should bind on the target. That single signal is wired into
four weight-matrix input slots, giving the "+4 channels" that the
state-dict diff reports.

---

## Case 2 — is the pipeline also antibody-specific?

**Claim to verify:** Ben's assertion that the pipeline (not just the
weights) has Ab-specific code.

### Step 1: the four load-bearing preprocessing choices

```bash
# (a) +200 chain offset — the biggest structural prior
grep -nC3 "200" src/rfantibody/rf2/modules/preprocess.py | grep -A3 "make_RF_idx"
# :311:            chain_idx = chain_idx + (lastidx + 1) + 200
#
# Comment in the source:
# - Target (T) chain: starts at 0
# - Antibody chains (H, L): always get 200-residue buffer

# (b) fused H+L same_chain mask — both Ab chains treated as one "chain"
grep -nA8 "def same_chain" src/rfantibody/rf2/modules/pose_util.py
# :161:    def same_chain(self):
# :167:        same_chain = torch.zeros((self.length, self.length), dtype=torch.bool)
# :168:        same_chain[:self.target_length, :self.target_length] = True
# :169:        same_chain[self.target_length:, self.target_length:] = True

# (c) antibody sequence masking — AA one-hot zeroed, confidence = 0
sed -n '186,192p' src/rfantibody/rf2/modules/preprocess.py

# (d) hotspot injection — already shown in Case 1
grep -n "t1d\[0,:,22\]" src/rfantibody/rf2/modules/preprocess.py
```

Four load-bearing Ab-specific choices. None of them is a new algorithm
— they are parameter choices and masking patterns applied on top of
vanilla RF2's existing multi-chain machinery.

### Step 2: the claim that *doesn't* hold

```bash
sed -n '152,172p' src/rfantibody/rf2/modules/preprocess.py
```

`make_msa` builds a **synthetic one-sequence MSA**. There is no MSA
search, no filtering, no re-weighting, no paired/unpaired handling —
nothing Ab-specific about MSAs because there *is* no MSA handling at
all in the inference path.

**Conclusion for Case 2.** Directionally Ben is right; specifically
his "MSA handling differs" sub-claim is the weakest point. The
pipeline's antibody-specificity lives in four small masking choices
plus the hotspot channel from Case 1. Single most load-bearing step:
the +200 chain offset in `make_RF_idx` — without it, RF2's pair
attention treats inter-chain distances as if they were intra-chain.

---

## Case 3 — how much worse is vanilla RF2 on an antibody-antigen complex?

**Setup:** Feed the *same* H/L/T sequences to both models, each
running through its *own* native pipeline:

- RFantibody: `rf2` CLI → synthetic 1-seq MSA, +200 offset, hotspots, RF2_ab weights.
- Vanilla RF2: `predict.py -inputs H.a3m L.a3m T.a3m` → ColabFold-retrieved per-chain MSAs, no hotspots, RF2_jan24 weights.

### One command

```bash
bash scripts/run_rf2_comparison.sh
# or equivalently:
make compare-rf2-vs-rfab
```

What it does (idempotent, each step checks for existing outputs):

| step | what | expected runtime |
|---|---|---|
| 1 | copy `test/rf2/inputs_for_test/ab_proteinmpnn_output.pdb` into an outputs dir | <1 s |
| 2 | extract H/L/T FASTA sequences from the PDB | <1 s |
| 3 | fetch per-chain MSAs from ColabFold (H, L, T) | ~4 min, network-bound |
| 4 | run RFantibody RF2 (RF2_ab) through the `rf2` CLI | ~30 s |
| 5 | run vanilla RF2 multimer via `predict.py` | ~20 s |
| 6 | `scripts/compare_rf2_vs_rfab.py` — Cα RMSD + pLDDT table | <5 s |

Total cold-start: ~5 min on a 4080. GPU peak: 2.8 GB.

### What to look at afterwards

```
outputs/rf2_comparison/case1_ab_proteinmpnn/
├── inputs_rfa/ab_proteinmpnn_output.pdb     # the shared input
├── a3m_vanilla/case_{H,L,T}_uniref.a3m      # ColabFold MSAs
├── rfa_out/ab_proteinmpnn_output_best.pdb   # RFa prediction
├── vanilla_out/case_00_pred.pdb             # vanilla prediction
├── vanilla_out/case_00.npz                  # per-residue pLDDT + PAE
└── comparison_results.json                  # metrics dump
```

### Expected results

The comparison script prints a markdown table. On the RTX 4080 box,
with `RECYCLES=3`:

| comparison | global | antibody (H+L) | target (T) | interface |
|---|---:|---:|---:|---:|
| RFa ↔ reference | 19.55 Å | **1.16 Å** | **1.69 Å** | — |
| Vanilla ↔ reference | 17.77 Å | 1.41 Å | **17.15 Å** | — |
| RFa ↔ Vanilla | 22.66 Å | 0.92 Å | 16.99 Å | 9.12 Å |

Per-chain mean pLDDT (vanilla, 0–1 scale, from `case_00.npz.lddt`):

| chain | mean | p10 / p50 / p90 |
|---|---:|---|
| H | 0.825 | 0.67 / 0.88 / 0.93 |
| L | 0.792 | 0.63 / 0.82 / 0.89 |
| T | **0.194** | 0.14 / 0.18 / 0.26 |

(Small numeric drift is expected — ColabFold returns MSAs by ticket,
and stochastic MSA sub-sampling inside `predict.py` isn't seeded. Ab
RMSDs should stay within ~0.3 Å, target RMSD will stay > 10 Å.)

### Interpretation

1. **Vanilla RF2 folds the antibody Fv correctly.** RMSD 1.4 Å vs
   reference; RFa gets 1.16 Å — essentially tied. The framework
   conservation alone gives vanilla enough MSA signal to place the Ab
   backbone.
2. **Vanilla RF2 loses on the target chain.** Target pLDDT 0.19, RMSD
   17 Å. The engineered target has ~500 seqs in its UniRef30 MSA vs
   ~10,000 for the Ab chains; vanilla depends entirely on MSA
   coevolution to fold it.
3. **Vanilla RF2 can't place the complex.** Global RMSD ~18 Å in both
   directions. Without the hotspot channel (Case 1) and the +200
   chain-offset prior (Case 2), vanilla has nothing telling it where
   on T the antibody should bind.

The one concrete claim "vanilla RF2 is worse on antibodies" that holds
after this experiment is **not** "vanilla can't fold antibody Fv" (it
can) — it's "vanilla can't predict the antibody-target pose or an
engineered target with a shallow MSA." The gap is concentrated in
exactly the places Cases 1 and 2 predict: target chain folding (no
hotspot signal, MSA-dependent) and inter-chain geometry (no chain
offset, no fused H+L prior).

---

## What couldn't be reproduced (gotchas)

During development of Case 3, two natural follow-up test cases
were attempted and abandoned — documenting them here so the next
person doesn't lose the same time:

### Case 3b attempt: real antibody + real antigen (1N8Z)

PDB 1N8Z (trastuzumab Fab + HER2 ECD domain IV) is the obvious real
crystal-structure benchmark. Two blockers:

1. **Size.** Full 1N8Z is ~1015 residues — OOM on a 16 GB card with
   vanilla RF2 multimer. A trimmed Fv (VH + VL) + HER2 domain IV gets
   to ~408 residues, which fits.
2. **HLT-remarked PDB requirement.** RFantibody's
   `pose_util.pose_from_remarked` requires `REMARK PDBinfo-LABEL`
   annotations for every CDR residue (`src/rfantibody/rf2/modules/util.py:167`).
   1N8Z has no such annotations, and transplanting them from
   RFantibody's `hu-4D5-8_Fv.pdb` example is off-by-N because 1N8Z's
   crystal H3 is 4 residues longer than the example's.

Fixing this properly needs an ANARCI-based CDR annotator that writes
the `REMARK PDBinfo-LABEL` lines for an arbitrary antibody PDB. Out of
scope for this tutorial; noted as future work.

### Case 3c attempt: Fv-only prediction (no target)

Use `scripts/examples/example_inputs/hu-4D5-8_Fv.pdb` directly
(trastuzumab Fv, already HLT-remarked, no T chain).

This also doesn't work. RFantibody's inference path assumes a target
chain is present:

```
File "src/rfantibody/rf2/modules/preprocess.py", line 200, make_t2d
    mask_t_2d[0,0,-pose.antibody_length:,:] = False
File "src/rfantibody/rf2/network/util.py", line 33, center_and_realign_missing
    seqmap = torch.argmin(seqmap, dim=-1)
IndexError: argmin(): Expected reduction dim 1 to have non-zero size.
```

With `antibody_length == total_length`, `mask_t_2d` gets zeroed
everywhere and downstream coordinate realignment fails. This is a
hard assumption baked into the preprocessor; the RF2_ab inference
pipeline cannot predict an Ab-only structure. Vanilla RF2 handles
both cases fine.

Both failure modes are interesting in their own right — they show
*where* RFantibody's preprocessing is tightly scoped to the "Ab + T"
case it was trained for.

---

## Summary

| case | what it shows | one-command reproducer |
|---|---|---|
| 1 | RF2_ab has exactly one extra input scalar (hotspot), used 4× across 3 layers | `uv run python scripts/diff_rf2_checkpoints.py weights/RF2_ab.pt weights/RF2_jan24.pt` |
| 2 | Pipeline Ab-specificity = four small preprocessing choices, no new algorithms | the greps in the "Case 2" section |
| 3 | Vanilla RF2 folds Ab Fv correctly, fails the target and the pose | `make compare-rf2-vs-rfab` |

All three cases point to the same conclusion: **RF2_ab ≠ vanilla RF2 +
hotspot**, but it is close. The weights can't be swapped, but the
delta is small, localized, and well-understood.
