# RFantibody (hegelab fork)

**Structure-Based _de novo_ Antibody Design with Fixes and Docker Support**

This is a modified and functional fork of [RosettaCommons/RFantibody](https://github.com/RosettaCommons/RFantibody), enhanced for easier use, clean installation via Docker, and script compatibility.

## Key Fixes and Additions

- Working `Dockerfile` with active Python environment
- Patched Python scripts for RFdiffusion, ProteinMPNN, RF2
- Added working examples and corrected file paths
- Quiver-based workflows are not tuned
- Container user write permissions fixed with `--user $(id -u)`
- Preserved original README as `README.orig.md`

## Installation (Docker-based)

```bash
git clone --branch main https://github.com/hegelab/RFantibody.git
cd RFantibody
docker build -t rfantibody .
sudo usermod -aG docker $USER  # Only once, then logout/login (restart terminal)
docker run --name rfantibody --gpus all -v $(pwd):/home --memory 10g --user $(id -u) -it rfantibody
```

**Optional: Change bash prompt inside container**

```bash
source /home/setenv.sh
```

## Python Environment Setup (inside container)

```bash
bash /home/include/setup.sh
```

This script:
- Installs dependencies via Poetry
- Installs DGL and Biotite
- Builds the USalign executable
- Downloads weights to `/home/weights/`

## Usage Overview

### Input File Preparation: Convert PDB → HLT format

```bash
poetry run python /home/scripts/util/chothia2HLT.py input.pdb -o output_HLT.pdb -H H -L L
```

Parameters:
- `-H` Heavy chain ID
- `-L` Light chain ID (omit for nanobodies)
- `-T` Optional Target chain(s)

### RFdiffusion (backbone design)

```bash
poetry run python /home/src/rfantibody/scripts/rfdiffusion_inference.py \
    --config-name antibody \
    antibody.target_pdb=/home/scripts/examples/example_inputs/rsv_site3.pdb \
    antibody.framework_pdb=/home/scripts/examples/example_inputs/hu-4D5-8_Fv.pdb \
    inference.ckpt_override_path=/home/weights/RFdiffusion_Ab.pt \
    'ppi.hotspot_res=[T305,T456]' \
    'antibody.design_loops=[L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13]' \
    inference.num_designs=10 \
    inference.output_prefix=/home/scripts/examples/example_outputs/ab_des
```

### ProteinMPNN (sequence design)

```bash
poetry run python /home/scripts/proteinmpnn_interface_design.py \
    -pdbdir /path/to/HLT_inputs \
    -outpdbdir /path/to/output_dir
```

Other useful flags:
- `--seqs_per_struct 5`
- `--omit_AAs C` to exclude cysteine

### RF2 (structure prediction and filtering)

```bash
poetry run python /home/scripts/rf2_predict.py \
    input.pdb_dir=/path/to/HLT_inputs \
    output.pdb_dir=/path/to/output_dir
```

## Modified Scripts Summary

| Script | Changes |
|--------|---------|
| `scripts/util/chothia2HLT.py` | Fixed CLI, added cropping, ID handling |
| `scripts/rfdiffusion_inference.py` | Patched example paths and model loading |
| `scripts/proteinmpnn_interface_design.py` | Checkpoint fixes, added runlist support |
| `src/rfantibody/proteinmpnn/struct_manager.py` | Quiver support fix for `seqs_per_struct` |
| `src/src/rfantibody/util/pose.py` | Fixed last-loop skipping bug |
| `setenv.sh` | Adds readable bash prompt in container |
| `Dockerfile` | Activates venv using `ENTRYPOINT` fix |

## Saving Your Installed Environment

To save a fully installed image:

```bash
docker commit rfantibody rfantibody2
```

Then later run:

```bash
docker start -i rfantibody2
```

## Known Issues

| Component | Issue |
|----------|-------|
| ProteinMPNN | Cannot process Quiver input (internal StructManager bug); partially patched |
| RF2 | Will crash if `output.pdb_dir` does not exist |
| AF3 | Seems to work better for filtering than RF2 based on [1] |
| Prompt | Appears as `I have no name...` inside container unless `setenv.sh` is sourced |
| pAE/pBind | Not parsed automatically; post-processing needed from output PDB REMARKs |
| File format | Ensure chain IDs and input numbering match expectations |


## References

1. RFantibody: https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1  
2. RFdiffusion: https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1  
3. ProteinMPNN: https://www.science.org/doi/10.1126/science.add2187

## License

Modifications licensed under MIT. Original code remains under RosettaCommons terms.
