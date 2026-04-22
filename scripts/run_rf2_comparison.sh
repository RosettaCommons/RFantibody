#!/usr/bin/env bash
# scripts/run_rf2_comparison.sh
#
# End-to-end reproducer for "Case 3" in rf2-vs-rfab-tutorial.md:
# runs RFantibody's RF2_ab and vanilla RF2 on the same antibody-antigen
# complex, then prints a side-by-side Cα-RMSD + pLDDT comparison.
#
# Idempotent: each step checks for existing outputs and skips if present.
# Delete the OUT_DIR to force a clean re-run.
#
# Prerequisites:
#   - make install-rfantibody   (uv sync + RFantibody weights)
#   - make install-rf2-vanilla  (vanilla RoseTTAFold2 + RF2_jan24.pt)
#   - Internet access to https://api.colabfold.com (ColabFold MSA API)
#   - GPU with >= 4 GB VRAM (tested on RTX 4080, 16 GB)
#
# Runtime on RTX 4080: ~4 min (dominated by 3 ColabFold MSA fetches)
#
# Usage:
#   bash scripts/run_rf2_comparison.sh
#   # or
#   make compare-rf2-vs-rfab

set -euo pipefail

# ---------- paths ----------
RFA_ROOT="${RFA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RF2_VANILLA_DIR="${RF2_VANILLA_DIR:-$HOME/rf2-vanilla/RoseTTAFold2}"
CASE_NAME="${CASE_NAME:-case1_ab_proteinmpnn}"
OUT_DIR="${OUT_DIR:-$RFA_ROOT/outputs/rf2_comparison/$CASE_NAME}"
SRC_PDB="${SRC_PDB:-$RFA_ROOT/test/rf2/inputs_for_test/ab_proteinmpnn_output.pdb}"
RECYCLES="${RECYCLES:-3}"

# ---------- preflight ----------
echo "=== preflight ==="
cd "$RFA_ROOT"

[[ -f "$SRC_PDB" ]] || { echo "ERROR: input PDB not found: $SRC_PDB"; exit 1; }
[[ -f "$RFA_ROOT/weights/RF2_ab.pt" ]] || { echo "ERROR: missing weights/RF2_ab.pt — run 'make install-rfantibody'"; exit 1; }
[[ -f "$RF2_VANILLA_DIR/network/weights/RF2_jan24.pt" ]] || { echo "ERROR: vanilla RF2 not installed — run 'make install-rf2-vanilla'"; exit 1; }

VENV_PY="$(uv run python -c 'import sys; print(sys.prefix)')/bin/python"
[[ -x "$VENV_PY" ]] || { echo "ERROR: cannot resolve .venv python via uv"; exit 1; }

echo "  RFA_ROOT         = $RFA_ROOT"
echo "  RF2_VANILLA_DIR  = $RF2_VANILLA_DIR"
echo "  OUT_DIR          = $OUT_DIR"
echo "  SRC_PDB          = $SRC_PDB"
echo "  RECYCLES         = $RECYCLES"

mkdir -p "$OUT_DIR"/{inputs_rfa,a3m_vanilla,rfa_out,vanilla_out}

# ---------- step 1: copy input PDB ----------
echo
echo "=== step 1: stage input PDB ==="
INPUT_PDB="$OUT_DIR/inputs_rfa/$(basename "$SRC_PDB")"
if [[ -f "$INPUT_PDB" ]]; then
    echo "  [skip] already staged: $INPUT_PDB"
else
    cp "$SRC_PDB" "$INPUT_PDB"
    echo "  copied -> $INPUT_PDB"
fi

# ---------- step 2: extract H/L/T sequences ----------
echo
echo "=== step 2: extract per-chain sequences ==="
if [[ -f "$OUT_DIR/a3m_vanilla/chain_H.fasta" && -f "$OUT_DIR/a3m_vanilla/chain_L.fasta" && -f "$OUT_DIR/a3m_vanilla/chain_T.fasta" ]]; then
    echo "  [skip] fastas already exist"
else
    INPUT_PDB="$INPUT_PDB" A3M_DIR="$OUT_DIR/a3m_vanilla" uv run python - <<'PY'
import os
from pathlib import Path
import biotite.structure.io.pdb as pdbio
from biotite.structure import filter_amino_acids
from biotite.structure.info import one_letter_code

pdb = Path(os.environ["INPUT_PDB"])
out = Path(os.environ["A3M_DIR"])
atoms = pdbio.PDBFile.read(str(pdb)).get_structure(model=1)
atoms = atoms[filter_amino_acids(atoms)]
ca = atoms[atoms.atom_name == "CA"]
for c in "HLT":
    sub = ca[ca.chain_id == c]
    if len(sub) == 0:
        raise SystemExit(f"chain {c} missing from {pdb}")
    seq = "".join(one_letter_code(r) for r in sub.res_name).replace("?", "X")
    (out / f"chain_{c}.fasta").write_text(f">case_{c}\n{seq}\n")
    print(f"  chain {c}: {len(sub)} residues")
PY
fi

# ---------- step 3: fetch MSAs via ColabFold ----------
echo
echo "=== step 3: fetch ColabFold MSAs (H, L, T) ==="
for C in H L T; do
    if [[ -f "$OUT_DIR/a3m_vanilla/case_${C}_uniref.a3m" ]]; then
        echo "  [skip] chain $C MSA already fetched"
        continue
    fi
    SEQ=$(tail -1 "$OUT_DIR/a3m_vanilla/chain_${C}.fasta")
    echo "  fetching chain $C (${#SEQ} aa)..."
    uv run python "$RFA_ROOT/vanilla_rf2_helper.py" msa "case_${C}" "$SEQ" "$OUT_DIR/a3m_vanilla"
done

# ---------- step 4: run RFantibody RF2 ----------
echo
echo "=== step 4: run RFantibody RF2 (RF2_ab) ==="
if compgen -G "$OUT_DIR/rfa_out/*_best.pdb" > /dev/null; then
    echo "  [skip] RFa prediction already present"
else
    export PYTHONHASHSEED=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export TORCH_DETERMINISTIC=1
    export TORCH_USE_CUDA_DSA=0
    uv run rf2 \
        --input-dir "$OUT_DIR/inputs_rfa" \
        --output-dir "$OUT_DIR/rfa_out" \
        --num-recycles "$RECYCLES" \
        --no-cautious \
        --seed 42 \
        --extra "inference.hotspot_show_proportion=0" \
        --extra "hydra.run.dir=$OUT_DIR/rfa_hydra"
fi

# ---------- step 5: run vanilla RF2 multimer ----------
echo
echo "=== step 5: run vanilla RF2 (RF2_jan24 via predict.py) ==="
if [[ -f "$OUT_DIR/vanilla_out/case_00_pred.pdb" ]]; then
    echo "  [skip] vanilla prediction already present"
else
    (
        cd "$RF2_VANILLA_DIR/network"
        "$VENV_PY" predict.py \
            -inputs "$OUT_DIR/a3m_vanilla/case_H_uniref.a3m" \
                    "$OUT_DIR/a3m_vanilla/case_L_uniref.a3m" \
                    "$OUT_DIR/a3m_vanilla/case_T_uniref.a3m" \
            -prefix "$OUT_DIR/vanilla_out/case" \
            -n_recycles "$RECYCLES"
    )
fi

# ---------- step 6: comparison ----------
echo
echo "=== step 6: compare predictions ==="
RFA_PDB=$(ls "$OUT_DIR"/rfa_out/*_best.pdb | head -1)
VAN_PDB="$OUT_DIR/vanilla_out/case_00_pred.pdb"
REF_PDB="$INPUT_PDB"

uv run python "$RFA_ROOT/scripts/compare_rf2_vs_rfab.py" \
    --case "$CASE_NAME" "$RFA_PDB" "$VAN_PDB" "$REF_PDB" \
    --out "$OUT_DIR/comparison_results.json"

echo
echo "=== done ==="
echo "  RFa prediction   : $RFA_PDB"
echo "  vanilla pred     : $VAN_PDB"
echo "  comparison JSON  : $OUT_DIR/comparison_results.json"
