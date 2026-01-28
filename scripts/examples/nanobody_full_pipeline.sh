#!/bin/bash

# ============================================================================
# Full Nanobody Design Pipeline
# ============================================================================
# This script runs the complete nanobody design workflow:
#   1. RFdiffusion  - Design nanobody backbone structures
#   2. ProteinMPNN  - Design sequences for the backbones
#   3. RF2          - Predict/refine final structures
#
# Usage: bash scripts/examples/nanobody_full_pipeline.sh
# ============================================================================

set -e  # Exit on error

# ============================================================================
# EDITABLE PARAMETERS
# ============================================================================

# Input files
TARGET_PDB="scripts/examples/example_inputs/flu_HA.pdb"       # Target antigen
FRAMEWORK_PDB="scripts/examples/example_inputs/h-NbBCII10.pdb" # Nanobody framework

# Output directory (will be created if it doesn't exist)
OUTPUT_DIR="scripts/examples/example_outputs/nb_ha_pipeline"

# RFdiffusion parameters
NUM_DESIGNS=1000                        # Number of backbone designs to generate
DESIGN_LOOPS="H1:7,H2:6,H3:5-13"        # CDR loop lengths (nanobody uses H chain naming)
HOTSPOTS="B146,B170,B177"               # Target residues to focus binding (T=target chain)
DIFFUSER_T=50                           # Number of diffusion timesteps

# ProteinMPNN parameters
NUM_SEQS=4                              # Number of sequences per backbone
SAMPLING_TEMP=0.2                       # Sampling temperature (lower=more conservative)

# RF2 parameters
HOTSPOT_SHOW_PROP=0.0                   # Proportion of hotspots to show (1.0=all)
NUM_RECYCLES=10                         # Number of recycling iterations

# ============================================================================
# SETUP
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Intermediate files (all quiver format)
DIFFUSION_OUTPUT="$OUTPUT_DIR/1_rfdiffusion.qv"
MPNN_OUTPUT="$OUTPUT_DIR/2_proteinmpnn.qv"
RF2_OUTPUT="$OUTPUT_DIR/3_rf2.qv"

echo "=============================================="
echo "Nanobody Design Pipeline for HA"
echo "=============================================="
echo "Target:    $TARGET_PDB"
echo "Framework: $FRAMEWORK_PDB"
echo "Output:    $OUTPUT_DIR"
echo "=============================================="

# ============================================================================
# STEP 1: RFdiffusion - Design backbone structures
# ============================================================================

echo ""
echo "[Step 1/3] Running RFdiffusion..."
echo "  - Designing $NUM_DESIGNS backbones"
echo "  - Loop lengths: $DESIGN_LOOPS"
echo "  - Hotspots: $HOTSPOTS"

uv run rfdiffusion \
    --target "$TARGET_PDB" \
    --framework "$FRAMEWORK_PDB" \
    --output-quiver "$DIFFUSION_OUTPUT" \
    --num-designs "$NUM_DESIGNS" \
    --design-loops "$DESIGN_LOOPS" \
    --hotspots "$HOTSPOTS" \
    --diffuser-t "$DIFFUSER_T" \
    --deterministic

echo "[Step 1/3] RFdiffusion complete."

# ============================================================================
# STEP 2: ProteinMPNN - Design sequences
# ============================================================================

echo ""
echo "[Step 2/3] Running ProteinMPNN..."
echo "  - Generating $NUM_SEQS sequences per backbone"
echo "  - Sampling temperature: $SAMPLING_TEMP"

uv run proteinmpnn \
    --input-quiver "$DIFFUSION_OUTPUT" \
    --output-quiver "$MPNN_OUTPUT" \
    --seqs-per-struct "$NUM_SEQS" \
    --temperature "$SAMPLING_TEMP"

echo "[Step 2/3] ProteinMPNN complete."

# ============================================================================
# STEP 3: RF2 - Predict/refine structures
# ============================================================================

echo ""
echo "[Step 3/3] Running RF2..."
echo "  - Refining structures with $NUM_RECYCLES recycles"

uv run rf2 \
    --input-quiver "$MPNN_OUTPUT" \
    --output-quiver "$RF2_OUTPUT" \
    --hotspot-show-prop "$HOTSPOT_SHOW_PROP" \
    --num-recycles "$NUM_RECYCLES"

echo "[Step 3/3] RF2 complete."

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "Outputs:"
echo "  1. RFdiffusion backbones: $DIFFUSION_OUTPUT"
echo "  2. ProteinMPNN sequences: $MPNN_OUTPUT"
echo "  3. RF2 refined structures: $RF2_OUTPUT"
echo ""
echo "View results with: uv run qvls $RF2_OUTPUT"
echo "Extract PDBs with: uv run qvextract $RF2_OUTPUT <output_dir>"
echo "=============================================="
