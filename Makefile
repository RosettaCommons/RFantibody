# RFantibody + vanilla RF2 — unified install, test, and demo
#
# Quick start:
#   make install          # RFantibody venv + weights + vanilla RF2 + weights
#   make test             # RFantibody pytest suite (18 tests; needs GPU)
#   make demo             # ubiquitin end-to-end: MSA -> vanilla RF2 -> RMSD
#   make help             # show all targets
#
# Per-query (override on CLI):
#   make predict NAME=mytarget SEQ=MQIFVK...
#
# Companion docs:
#   installation-steps.md                  — RFantibody install + test fixes
#   vanilla-rf2-alongside-rfantibody.md    — vanilla RF2 narrative tutorial
#
# Note on Make + spaces: Make cannot escape spaces in target names, so
# every target here is .PHONY and idempotency is enforced inside each
# recipe by shell-level existence checks. This keeps the Makefile
# working even when the repo lives under a path like
# "/home/you/Scripps Research Dropbox/..." .

SHELL       := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# ---- paths (override on CLI) ----
RFA_ROOT        := $(CURDIR)
VENV_PY         := $(RFA_ROOT)/.venv/bin/python
RFA_WEIGHTS_DIR := $(RFA_ROOT)/weights
HELPER          := $(RFA_ROOT)/vanilla_rf2_helper.py

RF2_DIR         ?= $(HOME)/rf2-vanilla/RoseTTAFold2
RF2_WEIGHTS     := $(RF2_DIR)/network/weights/RF2_jan24.pt

# ---- per-query parameters ----
NAME      ?= ubiquitin
SEQ       ?= MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
REF_PDB   ?= 1UBQ
REF_CHAIN ?= A
RECYCLES  ?= 3

MSA_DIR   := $(RF2_DIR)/example_inputs
MSA_FILE  := $(MSA_DIR)/$(NAME)_uniref.a3m
OUT_DIR   := $(RF2_DIR)/test_out
PRED_PDB  := $(OUT_DIR)/$(NAME)_00_pred.pdb

.PHONY: help install install-rfantibody install-rf2-vanilla \
        clone-rf2-vanilla download-rf2-vanilla-weights \
        download-rfantibody-weights \
        verify verify-rf2-vanilla test \
        demo msa predict rmsd clean distclean

# ===========================================================
# help
# ===========================================================
help:
	@echo "RFantibody + vanilla RF2 — unified pipeline"
	@echo
	@echo "Setup (one-time):"
	@echo "  make install                  RFantibody + vanilla RF2 end-to-end"
	@echo "  make install-rfantibody       uv sync + download RFantibody weights"
	@echo "  make install-rf2-vanilla      clone RoseTTAFold2 + download RF2_jan24.pt"
	@echo
	@echo "Verify:"
	@echo "  make verify                   all load-tests"
	@echo "  make verify-rf2-vanilla       strict=True load of vanilla RF2"
	@echo "  make test                     RFantibody pytest suite (18 tests)"
	@echo
	@echo "Predict (defaults: NAME=$(NAME), REF_PDB=$(REF_PDB)):"
	@echo "  make demo                     full ubiquitin pipeline -> 1UBQ RMSD"
	@echo "  make msa                      ColabFold MSA for \$$NAME / \$$SEQ"
	@echo "  make predict                  run vanilla RF2 on the MSA"
	@echo "  make rmsd                     Ca RMSD of prediction vs \$$REF_PDB"
	@echo
	@echo "Cleanup:"
	@echo "  make clean                    remove outputs for \$$NAME"
	@echo "  make distclean                remove vanilla RF2 clone entirely"
	@echo
	@echo "Config:"
	@echo "  RFA_ROOT  = $(RFA_ROOT)"
	@echo "  VENV_PY   = $(VENV_PY)"
	@echo "  RF2_DIR   = $(RF2_DIR)"
	@echo
	@echo "Prereq: 'uv' must be on PATH. Install with:"
	@echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"

# ===========================================================
# install: RFantibody
# ===========================================================
install-rfantibody: download-rfantibody-weights

$(VENV_PY):
	@true   # placeholder so recipes can depend on venv existence via shell check

.PHONY: _ensure_venv
_ensure_venv:
	@if [ ! -x "$(VENV_PY)" ]; then \
	  echo "=== uv sync --all-extras ==="; \
	  cd "$(RFA_ROOT)" && uv sync --all-extras; \
	else \
	  echo "[skip] venv already present"; \
	fi

download-rfantibody-weights: _ensure_venv
	@mkdir -p "$(RFA_WEIGHTS_DIR)"
	@cd "$(RFA_WEIGHTS_DIR)" && \
	for entry in \
	  "RFdiffusion_Ab.pt|https://files.ipd.uw.edu/pub/RFantibody/RFdiffusion_Ab.pt" \
	  "ProteinMPNN_v48_noise_0.2.pt|https://files.ipd.uw.edu/pub/RFantibody/ProteinMPNN_v48_noise_0.2.pt" \
	  "RF2_ab.pt|https://files.ipd.uw.edu/pub/RFantibody/RF2_ab.pt" \
	  "RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt|https://zenodo.org/records/17488258/files/RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt?download=1" ; do \
	    name="$${entry%%|*}" ; url="$${entry#*|}" ; \
	    if [ -s "$$name" ]; then \
	      echo "[skip] $$name already downloaded" ; \
	    else \
	      echo "=== Downloading $$name ===" ; \
	      wget -nv -c -O "$$name" "$$url" ; \
	    fi ; \
	done

# ===========================================================
# install: vanilla RF2
# ===========================================================
install-rf2-vanilla: clone-rf2-vanilla download-rf2-vanilla-weights

clone-rf2-vanilla:
	@if [ -d "$(RF2_DIR)/.git" ]; then \
	  echo "[skip] vanilla RF2 already cloned at $(RF2_DIR)" ; \
	else \
	  echo "=== Cloning vanilla RF2 into $(RF2_DIR) ===" ; \
	  mkdir -p "$(dir $(RF2_DIR))" ; \
	  git clone --depth 1 https://github.com/uw-ipd/RoseTTAFold2.git "$(RF2_DIR)" ; \
	fi

download-rf2-vanilla-weights: clone-rf2-vanilla
	@if [ -s "$(RF2_WEIGHTS)" ]; then \
	  echo "[skip] $(RF2_WEIGHTS) already present" ; \
	else \
	  echo "=== Downloading vanilla RF2 weights ===" ; \
	  mkdir -p "$(dir $(RF2_WEIGHTS))" ; \
	  cd "$(dir $(RF2_WEIGHTS))" && \
	    wget -nv -c https://files.ipd.uw.edu/dimaio/RF2_jan24.tgz && \
	    tar xzf RF2_jan24.tgz && \
	    rm -f RF2_jan24.tgz ; \
	fi

install: install-rfantibody install-rf2-vanilla

# ===========================================================
# verify
# ===========================================================
verify-rf2-vanilla: _ensure_venv clone-rf2-vanilla download-rf2-vanilla-weights
	@echo "=== strict=True load test: vanilla RF2 model + weights ==="
	@"$(VENV_PY)" "$(HELPER)" verify --rf2-dir "$(RF2_DIR)"

verify: verify-rf2-vanilla

# ===========================================================
# test (RFantibody pytest suite)
# ===========================================================
test: _ensure_venv download-rfantibody-weights
	@echo "=== RFantibody full test suite ==="
	@cd "$(RFA_ROOT)" && uv run python -m test.run_tests

# ===========================================================
# MSA
# ===========================================================
msa: _ensure_venv clone-rf2-vanilla
	@if [ -s "$(MSA_FILE)" ]; then \
	  echo "[skip] MSA already exists: $(MSA_FILE)" ; \
	else \
	  echo "=== ColabFold MSA for $(NAME) (L=$$(echo -n '$(SEQ)' | wc -c)) ===" ; \
	  mkdir -p "$(MSA_DIR)" ; \
	  "$(VENV_PY)" "$(HELPER)" msa "$(NAME)" "$(SEQ)" "$(MSA_DIR)" ; \
	fi

# ===========================================================
# predict (vanilla RF2)
# ===========================================================
predict: msa download-rf2-vanilla-weights
	@if [ -s "$(PRED_PDB)" ]; then \
	  echo "[skip] prediction already exists: $(PRED_PDB)" ; \
	  echo "       (delete it or 'make clean' to re-run)" ; \
	else \
	  echo "=== Running vanilla RF2 on $(NAME) (recycles=$(RECYCLES)) ===" ; \
	  mkdir -p "$(OUT_DIR)" ; \
	  cd "$(RF2_DIR)/network" && "$(VENV_PY)" predict.py \
	    -inputs "../example_inputs/$(NAME)_uniref.a3m" \
	    -prefix "../test_out/$(NAME)" \
	    -n_recycles $(RECYCLES) ; \
	fi

# ===========================================================
# RMSD check vs reference PDB
# ===========================================================
rmsd: _ensure_venv
	@if [ ! -s "$(PRED_PDB)" ]; then \
	  echo "ERROR: prediction not found: $(PRED_PDB)" ; \
	  echo "       run 'make predict' first" ; \
	  exit 1 ; \
	fi
	@echo "=== Ca RMSD: $(NAME) vs $(REF_PDB) chain $(REF_CHAIN) ==="
	@curl -sSL -o "/tmp/$(REF_PDB).pdb" "https://files.rcsb.org/download/$(REF_PDB).pdb"
	@"$(VENV_PY)" "$(HELPER)" rmsd "$(PRED_PDB)" "/tmp/$(REF_PDB).pdb" --chain "$(REF_CHAIN)"

# ===========================================================
# demo: one command for the full vanilla-RF2 story
# ===========================================================
demo: install verify msa predict rmsd
	@echo
	@echo "=== demo complete ==="
	@echo "Prediction: $(PRED_PDB)"

# ===========================================================
# cleanup
# ===========================================================
clean:
	@rm -f "$(MSA_FILE)" "$(PRED_PDB)" "$(OUT_DIR)/$(NAME)_00.npz"
	@echo "cleaned outputs for NAME=$(NAME)"

distclean: clean
	@rm -rf "$(RF2_DIR)"
	@echo "removed $(RF2_DIR)"
