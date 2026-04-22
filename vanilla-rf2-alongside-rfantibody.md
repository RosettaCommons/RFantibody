# Running Vanilla RF2 Alongside RFantibody

Reproducible recipe for running the standard, non-antibody-finetuned
RoseTTAFold2 in the same machine / same Python environment as RFantibody.
Verified end-to-end on 2026-04-21 with an Ubuntu 24.04 / RTX 4080 box.

## Why a separate install

The RF2 inside RFantibody is an **antibody-finetuned** variant. It uses
`d_t1d = 23` (an extra hotspot channel) and a v2 BinderNetwork, and its
preprocessor always masks an antibody chain and injects hotspots. You
cannot drop a vanilla `RF2_jan24.pt` into RFantibody's RF2 module —
`load_state_dict(strict=True)` errors with three shape mismatches:

```
templ_emb.emb.weight:                   ckpt (64, 88) vs model (64, 90)
templ_emb.templ_stack.proj_t1d.weight:  ckpt (32, 22) vs model (32, 23)
templ_emb.proj_t1d.weight:              ckpt (64, 52) vs model (64, 53)
```

(0 missing / 0 unexpected keys — the topology is identical; only the
three input-embedding widths differ because of the hotspot channel.)

The clean fix is **don't fight the antibody fine-tune** — install vanilla
RF2 side-by-side and call it directly when you want general-purpose
folding. The good news: vanilla RF2's deps are a strict subset of what
RFantibody already installs, so you don't need a second Python env.

## Prerequisites

- RFantibody already installed and working (see `installation-steps.md`).
  You have `uv` on `PATH` and a functioning `.venv` with PyTorch 2.3+cu118,
  DGL 2.4, `se3_transformer`, biotite, etc.
- ~3 GB of disk: ~1 GB for `RF2_jan24.pt`, plus the repo (small).
- Internet access to the ColabFold MMseqs2 endpoint
  (`https://api.colabfold.com`) for real MSAs. No local
  UniRef30/BFD download required.

## Step 1 — Clone vanilla RoseTTAFold2

Put it outside the RFantibody repo so the two don't entangle. Pick a
space-free path (the upstream `run_RF2.sh` has the same unquoted-path
fragility as the RFantibody download script; skipping it avoids the
issue).

```bash
mkdir -p ~/rf2-vanilla
cd ~/rf2-vanilla
git clone --depth 1 https://github.com/uw-ipd/RoseTTAFold2.git
```

## Step 2 — Download the vanilla weights

```bash
cd ~/rf2-vanilla/RoseTTAFold2
mkdir -p network/weights
cd network/weights
wget -c https://files.ipd.uw.edu/dimaio/RF2_jan24.tgz
tar xzf RF2_jan24.tgz
rm RF2_jan24.tgz           # optional, to reclaim ~1 GB
ls -lh RF2_jan24.pt        # 1.1G
```

## Step 3 — Verify the install (no prediction yet)

Vanilla RF2's module should import and its weights should load cleanly
inside RFantibody's venv. Run from the RFantibody repo root so `uv run`
resolves its `.venv`:

```bash
cd /path/to/RFantibody
uv run python - << 'PY'
import sys
sys.path.insert(0, '/home/YOU/rf2-vanilla/RoseTTAFold2/network')

import torch
from RoseTTAFoldModel import RoseTTAFoldModule

MODEL_PARAM = dict(
    n_extra_block=4, n_main_block=36, n_ref_block=4,
    d_msa=256, d_pair=128, d_templ=64,
    n_head_msa=8, n_head_pair=4, n_head_templ=4,
    d_hidden=32, d_hidden_templ=32, p_drop=0.0,
    SE3_param_full=dict(num_layers=1, num_channels=48, num_degrees=2,
                        l0_in_features=32, l0_out_features=32,
                        l1_in_features=2, l1_out_features=2,
                        num_edge_features=32, div=4, n_heads=4),
    SE3_param_topk=dict(num_layers=1, num_channels=128, num_degrees=2,
                        l0_in_features=64, l0_out_features=64,
                        l1_in_features=2, l1_out_features=2,
                        num_edge_features=64, div=4, n_heads=4),
)
model = RoseTTAFoldModule(**MODEL_PARAM)
ck = torch.load(
    '/home/YOU/rf2-vanilla/RoseTTAFold2/network/weights/RF2_jan24.pt',
    map_location='cpu', weights_only=False)
model.load_state_dict(ck['model_state_dict'], strict=True)
print("OK: vanilla RF2 model + weights load cleanly in RFantibody's venv")
PY
```

Expected: `OK: vanilla RF2 model + weights load cleanly in RFantibody's venv`.
Remember to replace `/home/YOU/...` with your actual path (don't put it
under a space-containing path unless you're prepared to quote every
shell invocation).

## Step 4 — Get a real MSA without installing HHblits

Vanilla RF2's `predict.py` wants an `.a3m` MSA. The official path is
HHblits + UniRef30 (46 GB) + optionally BFD (272 GB). Skip that for
everyday use: hit ColabFold's public MMseqs2 endpoint instead. The
returned MSA is real biological data, searched against UniRef30 and the
ColabFoldDB — same data ColabFold itself uses for production
predictions.

Save this as `~/rf2-vanilla/fetch_msa.py`:

```python
#!/usr/bin/env python
"""
Fetch a real MSA for a single query sequence via ColabFold's MMseqs2 API.
Writes two .a3m files:
  <name>_uniref.a3m   — UniRef30 hits
  <name>_envbfd.a3m   — BFD / MGnify / metaeuk / smag30 hits
Use either one (or concatenate) as input to predict.py.
"""
import io
import pathlib
import sys
import tarfile
import time

import requests

API = "https://api.colabfold.com"


def fetch_msa(sequence: str, name: str, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[submit] {len(sequence)} aa -> {API}/ticket/msa (mode=env)")
    r = requests.post(
        f"{API}/ticket/msa",
        data={"q": f">{name}\n{sequence}", "mode": "env"},
    )
    r.raise_for_status()
    tid = r.json()["id"]

    for _ in range(240):      # up to 20 min at 5 s/poll
        time.sleep(5)
        s = requests.get(f"{API}/ticket/{tid}").json()
        if s.get("status") in ("COMPLETE", "ERROR"):
            break
    else:
        raise RuntimeError("ColabFold MSA timed out")

    if s["status"] == "ERROR":
        raise RuntimeError(f"ColabFold returned ERROR: {s}")

    print("[download] fetching tarball")
    tb = requests.get(f"{API}/result/download/{tid}").content

    with tarfile.open(fileobj=io.BytesIO(tb)) as t:
        for member in t.getmembers():
            if not member.name.endswith(".a3m"):
                continue
            base = pathlib.Path(member.name).stem
            # standardise naming
            out_name = {
                "uniref": f"{name}_uniref.a3m",
                "bfd.mgnify30.metaeuk30.smag30": f"{name}_envbfd.a3m",
            }.get(base, f"{name}_{base}.a3m")
            out_path = out_dir / out_name
            with t.extractfile(member) as src:
                out_path.write_bytes(src.read())
            n = sum(1 for line in out_path.open() if line.startswith(">"))
            print(f"[a3m] {out_path.name}: {n} sequences")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: fetch_msa.py <name> <sequence> <out_dir>")
        sys.exit(1)
    fetch_msa(sys.argv[2], sys.argv[1], pathlib.Path(sys.argv[3]))
```

Run it (uses `requests` from RFantibody's venv — no extra install):

```bash
cd /path/to/RFantibody
uv run python ~/rf2-vanilla/fetch_msa.py ubiquitin \
    MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG \
    ~/rf2-vanilla/RoseTTAFold2/example_inputs
```

Expected output (~30 s):

```
[submit] 76 aa -> https://api.colabfold.com/ticket/msa (mode=env)
[download] fetching tarball
[a3m] ubiquitin_uniref.a3m: 9655 sequences
[a3m] ubiquitin_envbfd.a3m: 11922 sequences
```

## Step 5 — Run vanilla RF2 with the real MSA

`predict.py` uses relative imports (`from parsers import ...`), so you
must run it from the `network/` directory. Borrow RFantibody's venv by
calling its Python interpreter directly:

```bash
cd /path/to/RFantibody
VENV=$(uv run python -c "import sys; print(sys.prefix)")

mkdir -p ~/rf2-vanilla/RoseTTAFold2/test_out
cd ~/rf2-vanilla/RoseTTAFold2/network

"$VENV/bin/python" predict.py \
    -inputs ../example_inputs/ubiquitin_uniref.a3m \
    -prefix ../test_out/ubi \
    -n_recycles 3
```

Expected output (~2 s on an RTX 4080 plus the one-time model load):

```
Running on GPU
N=2048 L=76
recycle 0 plddt 0.871 pae 2.850 rmsd 11.410
recycle 1 plddt 0.883 pae 2.602 rmsd 0.288
recycle 2 plddt 0.882 pae 2.604 rmsd 0.180
recycle 3 plddt 0.883 pae 2.598 rmsd 0.156
runtime=2.26 vram=0.55
```

Outputs:

```
~/rf2-vanilla/RoseTTAFold2/test_out/ubi_00_pred.pdb    # structure, pLDDT in B-factor column
~/rf2-vanilla/RoseTTAFold2/test_out/ubi_00.npz         # PAE, per-residue pLDDT
```

## Step 6 — Sanity-check against the experimental structure

For a protein with a known crystal structure, confirm RF2 reproduces it:

```bash
curl -sSL -o /tmp/1UBQ.pdb https://files.rcsb.org/download/1UBQ.pdb

uv run python - << 'PY'
import biotite.structure.io.pdb as pdb
from biotite.structure import filter_amino_acids, superimpose, rmsd

def ca(path, chain=None):
    a = pdb.PDBFile.read(path).get_structure(model=1)
    a = a[filter_amino_acids(a)]
    if chain is not None:
        a = a[a.chain_id == chain]
    return a[a.atom_name == "CA"]

exp = ca("/tmp/1UBQ.pdb", chain="A")
pred = ca("/home/YOU/rf2-vanilla/RoseTTAFold2/test_out/ubi_00_pred.pdb")
fitted, _ = superimpose(exp, pred)
print(f"Ca RMSD vs 1UBQ: {rmsd(exp, fitted):.3f} A")
PY
```

Measured on the verification run: **1.04 Å Cα RMSD** vs 1UBQ — a
publication-quality fit, on par with expected RF2 performance for
small well-conserved domains.

## Using it for longer sequences / harder targets

- **Memory**: `-low_vram` offloads some ops to CPU so bigger systems
  fit on a 16 GB card. Recycles dominate cost; drop `-n_recycles` first
  if you run out of VRAM.
- **Subcropping**: `-topk 512` (default 1536) reduces quadratic work
  in the structure module.
- **Symmetry / multimers**: `predict.py` accepts `-symm Cn|Dn|T|O|I`
  and a space-separated list of per-chain a3m paths. See
  `RoseTTAFold2/README.md` examples 2–5.
- **Single-sequence mode**: if you don't want to call the ColabFold
  API, a one-line a3m (just the query) works. Quality drops (pLDDT ~0.05
  lower on ubiquitin in testing) but the pipeline is unchanged.

## Cross-pipeline handoff (RFantibody → vanilla RF2)

If you want to use RFantibody's RFdiffusion → ProteinMPNN to design a
sequence and then fold the design with vanilla RF2 instead of RF2_ab:

1. Run RFdiffusion + ProteinMPNN as usual (see `scripts/examples/`).
2. Extract the designed chain(s) as FASTA from the Quiver output.
3. `fetch_msa.py` the designed sequence.
4. Run `predict.py` with the MSA.

Caveat: vanilla RF2 on an antibody-antigen complex is unlikely to beat
the antibody-finetuned RF2_ab — that's precisely the niche RF2_ab was
trained for. Vanilla RF2 is the right tool if the target is *not* a
typical antibody-antigen interaction, or if you want a single-model
prediction with no hotspot/framework priors.

## What this install does *not* get you

- **Local HHblits / UniRef30**: not installed. If you need fully offline
  MSAs, follow the upstream RF2 README for hhsuite + UniRef30 + BFD
  (~320 GB + hhsuite install). ColabFold's API gives comparable MSA
  depth for everyday use.
- **Structure templates (`-db` / `-hhpred`)**: not installed. Templates
  are optional in RF2; the ColabFold-MSA pipeline here runs template-free.
- **Training / fine-tuning**: vanilla RF2's `network/train_multi_deep.py`
  additionally needs `pyg` and the sequence/structure databases. Out of
  scope for this tutorial.

## Summary

| Step | What it does | One-time / per-query |
|---|---|---|
| Clone + `RF2_jan24.pt` | Put vanilla code + weights on disk | One-time, ~1 GB |
| Venv sharing | Reuse RFantibody's `.venv` | One-time, zero cost |
| `fetch_msa.py` + ColabFold API | Real MSA, 9000+ sequences | Per-query, ~30 s |
| `predict.py` via RFantibody venv Python | Run RF2 | Per-query, ~2 s/76 aa on RTX 4080 |
| Cα RMSD vs PDB | Optional validation | Per-query, ~1 s |

Nothing in the RFantibody source was modified to make this work.
Nothing in vanilla RoseTTAFold2's source was modified either.
