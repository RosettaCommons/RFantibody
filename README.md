# RFantibody (hegelab fork)

**Structure-Based _de novo_ Antibody Design with Fixes and Docker + Singularity Support**

This is a modified and functional fork of [RosettaCommons/RFantibody](https://github.com/RosettaCommons/RFantibody), enhanced for easier use, clean installation via Docker or Singularity, and script compatibility. 
Done by **Orsolya Gereben @ hegelab.org**

## Key Fixes and Additions

- Working `Dockerfile` with active Python environment
- Patched Python scripts for RFdiffusion, ProteinMPNN, RF2
- Added working examples and corrected file paths
- Quiver-based workflows are not tuned
- Container user write permissions fixed with `--user $(id -u)`
- Preserved original README as `README.orig.md`
- Singularity definition file (`singularity.def`) added for HPC-compatible container builds

## Installation (Docker-based)

```bash
git clone --branch main https://github.com/hegelab/RFantibody.git
cd RFantibody
bash include/download_weights.sh

docker build -t rfantibody .
sudo usermod -aG docker $USER  # Only once, then logout/login (restart terminal), but if you do not have sudo permission then leave this out, it worked for me without it
docker run --name rfantibody --gpus all -v $(pwd):/home --memory 10g -it rfantibody
```

## Python Environment Setup (inside container)

```bash
bash /opt/RFantibody/include/setup.sh
```

This script:
- Installs dependencies via Poetry
- Installs DGL and Biotite
- Builds the USalign executable

Now you will have a container fully equipped to run the RFantibody pipeline.  


## Saving Your Installed Environment

To save a fully installed image, while the above created rfantibody container runs start a new terminal and create a new image (rfantibody2) from the fully installed container: 

```bash
docker commit rfantibody rfantibody2
```

Then you can run a container from this image:

```bash
docker run --name rfantibody2 --gpus all -v $(pwd):/home --memory 10g --user $(id -u) -it rfantibody2
```

**Optional: Change bash prompt inside container**

```bash
source /opt/RFantibody/setenv.sh
```

## Singularity-based Installation and Usage

```
singularity build --fakeroot RFantibody_base.sif singularity.def

singularity shell --nv --no-home -B $(pwd):/opt/RFantibody RFantibody_base.sif
(Singularity)$ cd $HOME
(Singularity)$ bash include/setup_sif.sh
(Singularity)$ exit

singularity run --nv --no-home -B $(pwd):/opt/RFantibody RFantibody_base.sif poetry run python /opt/RFantibody/scripts/rfdiffusion_inference.py ... (please see detailed usage overview/examples below)
```

## Usage Overview

### Input File Preparation: Convert PDB → HLT format
to create the framework, input should be a Chothia-formatted pdb, for example:
```bash
poetry run python /opt/RFantibody/scripts/util/chothia2HLT.py /opt/RFantibody/scripts/examples/example_inputs/9dpe_chothia.pdb -o /opt/RFantibody/scripts/examples/example_inputs/9dpe_ab_HLT.pdb -H H -L L
```
to create the target:
```bash
poetry run python /opt/RFantibody/scripts/util/chothia2HLT.py /opt/RFantibody/scripts/examples/example_inputs/9dpe_chothia.pdb -o /opt/RFantibody/scripts/examples/example_inputs/9dpe_target_HLT.pdb -T A
```
there are additional options related to cropping the input, see it with 
```bash
poetry run python /opt/RFantibody/scripts/util/chothia2HLT.py -h
``` 

### RFdiffusion (backbone design)
In the container you can run first the backbone design. There are several parameters, you can see them with
```bash
poetry run python /opt/RFantibody/scripts/rfdiffusion_inference.py -h
```
and you can override them. **ONLY OVERRIDE, if you know, what you are doing!**

Example:

```bash
poetry run python /opt/RFantibody/scripts/rfdiffusion_inference.py --config-name antibody     antibody.target_pdb=/opt/RFantibody/scripts/examples/example_inputs/rsv_site3.pdb antibody.framework_pdb=/opt/RFantibody/scripts/examples/example_inputs/hu-4D5-8_Fv.pdb inference.ckpt_override_path=/opt/RFantibody/weights/RFdiffusion_Ab.pt 'ppi.hotspot_res=[T305,T456]'     'antibody.design_loops=[L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13]' inference.num_designs=10   inference.output_prefix=/opt/RFantibody/scripts/examples/example_outputs/ab_des
```

### ProteinMPNN (sequence design)
After the backbone design you can design sequence for the loops with ProteinMPNN.
Example:

```bash
poetry run python /opt/RFantibody/scripts/proteinmpnn_interface_design.py -pdbdir /opt/RFantibody/scripts/examples/example_outputs/ -outpdbdir /opt/RFantibody/scripts/examples/example_outputs/pmpnn
```

Other useful flags:
- `-seqs_per_struct 5` Number of sequiences to design for an input structure
- `-omit_AAs CR` to exclude cysteine and arginine
- `-loop_string H1,H3` The list of loops which you wish to design

See all the parameters with
```bash
poetry run python /opt/RFantibody/scripts/proteinmpnn_interface_design.py -h
```

### RF2 (structure prediction and filtering)
After the sequence design the structures can be filtered with RF2. The scores can be found at the end of the pdb files.

```bash
poetry run python /opt/RFantibody/scripts/rf2_predict.py input.pdb_dir=/opt/RFantibody/scripts/examples/example_outputs/pmpnn output.pdb_dir=/opt/RFantibody/scripts/examples/example_outputs/rf2
```

There are additional parameters, see it with

```bash
poetry run python /opt/RFantibody/scripts/rf2_predict.py -h
```

## Modified Scripts Summary

| Script | Changes |
|--------|---------|
| `scripts/util/chothia2HLT.py` | Fixed CLI, added cropping, target handling |
| `scripts/rfdiffusion_inference.py` |  Patched config paths and model loading, empty lines between the meaningful lines was removed from the output pdb |
| `scripts/proteinmpnn_interface_design.py` | Detection of existing output files (check.point fix), adding new arguments for name_tag and seed|
| `src/rfantibody/proteinmpnn/struct_manager.py` | Detection of existing output files (check.point fix) |
| `src/rfantibody/util/pose.py` |  Fixing shifted sequence prediction windows bug for ProteinMPNN|
| `src/rfantibody/rf2/config/base.yaml` | changing model.model_weights to the supplied model: /opt/RFantibody/weights/RF2_ab.pt |
| `src/rfantibody/rf2/modules/model_runner.py ` | output directory created if not exists |
| `setenv.sh` | Adds readable bash prompt in container |
| `Dockerfile` | Activates venv using `ENTRYPOINT` fix |
| `Fixed pyproject.toml` | include path fixed, biotite installation fixed |



## Known Issues

| Component | Issue |
|----------|-------|
| ProteinMPNN | Cannot process Quiver input (internal StructManager bug) |
| AF3 | Seems to work better for filtering than RF2 based on [1] |
| RFantibody | The parameters of the model provided does not match the loading script, a lot of warnings are produced. T=200 is in the loaded model, but T=50 is in the config base.yaml file. No information in the paper what was the actual T used in training and what should be used at inference | 
| Prompt | Appears as `I have no name...` inside container unless `setenv.sh` is sourced |
| RF2 Scores | Appear at the end of each pdb file, additional processing required |
| RF2 pBind | Not found at the end of the pdb files at allpost-processing needed from output PDB REMARKs |


## References

1. RFantibody: https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1  
2. RFdiffusion: https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1  
3. ProteinMPNN: https://www.science.org/doi/10.1126/science.add2187

## License

Modifications licensed under MIT. Original code remains under RosettaCommons terms.
