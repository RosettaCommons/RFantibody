# Docker Plan for RFantibody

Planning document for making RFantibody reproducibly runnable on
Windows (and any Linux host) via Docker, specifically to support the
workflow Thomas is using on a Windows server.

Not a tutorial. Decisions still open — this doc is here to align on
approach before we touch the Dockerfile or do a public fork.

## Goal

One command takes Thomas from a freshly cloned repo to a running
RFantibody container with GPU access, verified tests, and a
working pipeline — with **no host-side Python, uv, or CUDA install**.

Specifically:

```powershell
# Thomas on Windows
git clone https://github.com/<fork>/RFantibody.git
cd RFantibody
docker build -t rfantibody .
docker run --rm --gpus all -v ${PWD}:/home rfantibody make test
```

The final line must print the 18-tests-passing summary we verified
on the host.

## Current state (upstream as of 2026-04-21)

`Dockerfile` at the repo root is 15 lines:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3.10 python3-pip vim make wget curl
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
WORKDIR /home
ENTRYPOINT /bin/bash
```

What it does: base Ubuntu 22.04 + CUDA 11.8, installs Python 3.10 +
`uv`, drops you into bash at `/home`.

What it does **not** do:
- Copy the repo into the image (expects `-v .:/home` bind mount).
- Run `uv sync` — you must run it inside the container on first start.
- Fetch model weights.
- Run or provide a test target.
- Handle the `test/*/conftest.py` GPU allowlist or the
  `include/download_weights.sh` path-quoting bug (covered in
  `installation-steps.md`).

The `README.md` install section describes this flow for Linux, and
there's a Singularity (`rfantibody.sif`) build path too.

## Gaps to close for Thomas

| # | Gap | Proposed fix |
|---|---|---|
| 1 | `include/download_weights.sh` breaks on any path with spaces (Dropbox on Windows, `Program Files`, etc.) | Ship the quoted version (already applied locally) |
| 2 | Host `.venv/` is bind-mounted into the container and poisons `uv sync` (absolute host paths baked in) | Move container venv to `/root/.venv` or `/opt/venv`, not under `/home` |
| 3 | `test/*/conftest.py` skip all GPU tests on non-A4000/H100 hardware | Ship the loosened version (already applied locally) + surface a `RFA_STRICT_GPU=1` env var so upstream behaviour stays available |
| 4 | No single-command entrypoint for "run the test suite" | Add `make test` target (already done) and consider a CMD that runs it when no override given |
| 5 | Windows CRLF line endings break `.sh` scripts | Ship `.gitattributes` forcing LF on `*.sh`, `Makefile`, `*.py` |
| 6 | `docker run -v .:/home` on Windows PowerShell uses `${PWD}`, not `.` | Document both, example in README |
| 7 | `--memory 10g` in `CLAUDE.md` caps the container but Docker Desktop on Windows has its own VM-wide memory cap in settings | Note in README: raise Docker Desktop VM memory first |
| 8 | Base image is ~4 GB; build is not cached layer-by-layer well | Split `apt-get` and `curl uv` into separate `RUN`s so uv reinstall doesn't invalidate apt layer (already OK — minor) |
| 9 | No test that verifies the container itself works (the repo's own tests need to run inside it) | Add a `make docker-test` target that builds and runs the image, then runs `make test` inside it |

## Proposed changes — concrete diff plan

### A. Dockerfile — minimal improvements

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    python3.10 python3-pip vim make wget curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv — installed to /root/.local, added to PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Force container venv off the bind-mounted /home
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /home
CMD ["/bin/bash"]
```

Key change: `UV_PROJECT_ENVIRONMENT=/opt/venv` moves the venv outside
the bind mount, so the container's sync doesn't clobber the host
`.venv` and vice versa.

Risk: the user's `Makefile` currently expects `.venv/bin/python` at
`$(RFA_ROOT)/.venv/bin/python`. If we relocate, the Makefile needs
a container-aware `VENV_PY` (detect `/opt/venv/bin/python` if present,
else `$(RFA_ROOT)/.venv/bin/python`).

### B. `.gitattributes` — prevent Windows CRLF breakage

```
*.sh        text eol=lf
*.py        text eol=lf
Makefile    text eol=lf
*.yaml      text eol=lf
*.yml       text eol=lf
```

### C. `Makefile` — add docker targets

```makefile
# ---- docker ----
DOCKER_IMAGE ?= rfantibody:local

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-run: docker-build
	docker run --rm --gpus all -v "$(RFA_ROOT)":/home -it $(DOCKER_IMAGE)

docker-test: docker-build
	docker run --rm --gpus all -v "$(RFA_ROOT)":/home $(DOCKER_IMAGE) \
	  bash -lc "cd /home && uv sync --all-extras && uv run python -m test.run_tests"
```

`docker-test` is the reproducibility hammer: build image, sync deps
inside, run the 18-test suite. Should print `18 passed` (or the same
distribution we see on the host).

### D. Don't commit host `.venv` into the image layer

Add `.venv/` to `.dockerignore` (also `weights/` if we don't want
to ship weights inside the image — keep them as a host-mounted
volume).

## Test plan

On **this Linux host** (as a proxy for Thomas's Windows-host
correctness):

1. `make docker-build` — expect ~5 min download + small apt layer.
2. `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi` — confirm GPU passthrough before blaming the image.
3. `make docker-test` — expect `18 passed` matching host run.
4. `make docker-run` → drop into container → manual `make demo` — expect same 1.04 Å Cα RMSD result from the vanilla-RF2 tutorial.

For **Thomas on Windows**, we can't test directly but can
preemptively cover:
- `.gitattributes` prevents the CRLF breakage before he clones.
- Document `-v ${PWD}:/home` (PowerShell) and `-v %cd%:/home` (cmd).
- Note Docker Desktop VM memory must be ≥ 10 GB in settings.
- Note: `--memory 10g` caps the container; doesn't raise it.

## Windows-specific caveats (for documentation, not Dockerfile)

Things I won't fix in the Dockerfile but will document:

1. **Line endings**: Thomas must clone with `core.autocrlf=input` or
   the `.gitattributes` I add will handle it.
2. **Docker Desktop memory**: default is 2 GB on Windows. RFantibody
   needs more. Settings → Resources → Memory → 16 GB recommended.
3. **WSL2 path performance**: bind-mounting from `C:\Users\...` into
   the container is slow on Windows. Cloning into `\\wsl$\...` is
   ~10× faster.
4. **GPU passthrough on WSL2**: requires recent NVIDIA driver on the
   Windows host + Docker Desktop with WSL2 backend + NVIDIA Container
   Toolkit enabled. Not a change to RFantibody; worth linking to
   nvidia docs in README.

## Open questions

- **Repo scope**: should the `docker-test`, `.gitattributes`, and
  Dockerfile edits go in the RFantibody fork, or in a separate
  "bootstrap" repo so upstream stays minimal? My preference: **fork
  RFantibody and put everything there** — Thomas gets one `git clone`.
- **Upstream strategy**: the path-quoting fix and the
  test-reference fallback are upstream-worthy bugs (not
  hardware-specific). Worth a separate PR split from the "teach
  others" material.
- **Vanilla RF2 in Docker**: if Thomas also wants vanilla RF2, we'd
  need a second Dockerfile or a multi-stage build. Out of scope for
  round one. `make install-rf2-vanilla` on the host works; inside a
  container is a future concern.
- **Baking weights vs bind-mount**: weights are 1.6 GB (RFantibody)
  + 1.1 GB (vanilla RF2). Baking them into the image adds ~3 GB and
  forces rebuild on weight updates. Keep them bind-mounted.

## Rough sizing

| Item | Size/time |
|---|---|
| Docker CE + NVIDIA container toolkit install | ~200 MB, 5 min |
| RFantibody image build | ~4 GB, 5 min |
| `uv sync` inside container (first run) | ~5 GB written to `/opt/venv`, 5 min |
| `make test` inside container (GPU) | ~3 min |
| Total cold-start for Thomas | ~15 min from fresh Windows Docker Desktop |

## Not doing

- Singularity / Apptainer — Thomas is on Docker.
- Docker Compose — overkill for a single service.
- Kubernetes / Helm — way out of scope.
- Multi-arch (ARM) — RFantibody requires CUDA; x86_64 only.

## Decision checkpoint

Before I start implementing:

1. Fork target: `troysincomb/RFantibody` or a branch on
   `RosettaCommons/RFantibody`?
2. Scope: do I implement the full list above, or only Dockerfile
   changes + `.gitattributes` + `make docker-test`?
3. Upstream PR: separate PR for the path-quoting + reference-fallback
   bug fixes (they don't depend on anything Docker-related)?
