# RFantibody
### Structure-Based _de novo_ Antibody Design

![](https://github.com/RosettaCommons/RFantibody/blob/main/github_image.png)

# Description
RFantibody is a pipeline for the structure-based design of _de novo_ antibodies and nanobodies. RFantibody consists of three separate methods:
- Protein backbone design with an antibody-finetuned version of [RFdiffusion](https://www.nature.com/articles/s41586-023-06415-8)
- Protein sequence design with [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187)
- _In silico_ filtering of designs using an antibody-finetuned version of [RoseTTAFold2](https://www.biorxiv.org/content/10.1101/2023.05.24.542179v1)

The RFantibody pipeline is described in detail in [this preprint](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1)

# Table of Contents
- [Requirements](#requirements)
- [Downloading Weights](#downloading-weights)
- [Installation](#installation)
  - [Local Installation (Without Docker)](#local-installation-without-docker)
  - [Docker Installation (Alternative)](#docker-installation-alternative)
  - [Apptainer Installation (HPC)](#apptainer-installation-hpc)
- [Command Line Interface](#command-line-interface)
  - [Inference Commands](#inference-commands)
  - [Quiver Utilities](#quiver-utilities)
  - [Full Pipeline Example](#full-pipeline-example)
- [Usage](#usage)
  - [HLT File Format](#hlt-file-format)
  - [Input Preparation](#input-preparation)
  - [RFdiffusion](#rfdiffusion)
  - [ProteinMPNN](#proteinmpnn)
  - [RF2](#rf2)
- [Practical Considerations for Antibody Design](#practical-considerations-for-antibody-design)
  - [Selecting a Target Site](#selecting-a-target-site)
  - [Nanobody Docks](#nanobody-docks)
  - [Truncating your Target Protein](#truncating-your-target-protein)
  - [Picking Hotspots](#picking-hotspots)
  - [Antibody Design Scale](#antibody-design-scale)
  - [Choosing CDR Lengths](#choosing-cdr-lengths)
  - [Filtering Strategies](#filtering-strategies)
- [Quiver Files](#quiver-files)
- [Conclusion](#conclusion)


# Requirements

### GPU Acceleration

RFantibody requires an NVIDIA GPU with CUDA support to run. You can check whether you have a compatible NVIDIA GPU available by running:
```
nvidia-smi
```
If this command runs successfully then you have a compatible GPU and RFantibody will be able to run on it.

### System Requirements

- **GPU**: NVIDIA GPU with CUDA 11.8+ support
- **OS**: Linux (Ubuntu 22.04 recommended)


# Installation

## Local Installation (Without Docker)

This is the recommended installation method for most users. It installs RFantibody directly on your system.

### 1. Install uv (Package Manager)

RFantibody uses [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management. uv will automatically download the required Python version (3.10) if needed.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
After installation, restart your terminal or run:
```bash
source ~/.bashrc  # or ~/.zshrc if using zsh
```

### 2. Clone the Repository

```bash
git clone https://github.com/RosettaCommons/RFantibody.git
cd RFantibody
```

### 3. Download Model Weights

```bash
bash include/download_weights.sh
```

### 4. Set Up the Python Environment

From the RFantibody directory, run:
```bash
uv sync
```
This uses uv to:
- Download Python 3.10 if not already installed
- Create a virtual environment in `.venv/`
- Install all dependencies including PyTorch with CUDA 11.8 support
- Install [Deep Graph Library](https://www.dgl.ai) from CUDA-enabled wheels

### 5. Activate the Environment

After setup, you can either:

**Option A**: Activate the virtual environment (recommended):
```bash
source .venv/bin/activate
rfdiffusion --help
```

**Option B**: Prefix commands with `uv run`:
```bash
uv run rfdiffusion --help
```

### Verifying the Installation

Run the following to verify that RFantibody is installed correctly:
```bash
uv run rfdiffusion --help
```
This should display the RFdiffusion help message with available options.

---

## Docker Installation (Alternative)

Docker provides a containerized environment that isolates RFantibody from your host system. This can be useful for:
- Simplified dependency management
- Reproducible environments across different machines
- Avoiding conflicts with existing system packages

### Prerequisites

Install Docker from [here](https://docs.docker.com/engine/install/). If you are running RFantibody on cloud compute, Docker may already be installed. Check by running:
```
which docker
```

You will need to add yourself to the docker group to run containers without sudo:
```
sudo usermod -aG docker $USER
```
After running this command, restart your terminal session for this change to take effect.

### Build the Docker Image

Navigate to the directory where RFantibody is downloaded, then build the Docker image:
```
docker build -t rfantibody .
```

### Start the Docker Container

Run the following command to start the container:
```
docker run --name rfantibody --gpus all -v .:/home --memory 10g -it rfantibody
```
This will put you into the RFantibody container at the `/home` directory which mirrors the directory that you ran the command from.

### Set Up the Python Environment (Inside Container)

From inside the RFantibody container, navigate to the project root and run:
```bash
cd /home
uv sync
```

---

## Apptainer Installation (HPC)

[Apptainer](https://apptainer.org/) (formerly Singularity) is a container platform designed for HPC environments. Unlike Docker, it runs without root privileges and integrates with job schedulers like SLURM.

### Prerequisites

Install Apptainer on Ubuntu:
```bash
sudo apt update
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer
```

### Build the Apptainer Image

Navigate to the RFantibody directory and build the image:
```bash
cd RFantibody
sudo apptainer build rfantibody.sif rfantibody.def
```

This creates a self-contained `rfantibody.sif` file (~8GB) with all dependencies, model weights, and the Python environment pre-installed.

### Running Commands

Always use the `--nv` flag (for GPU support) and `--writable-tmpfs` flag (for temporary file writes):

```bash
# Get help for any command
apptainer exec --nv --writable-tmpfs rfantibody.sif rfdiffusion --help
apptainer exec --nv --writable-tmpfs rfantibody.sif proteinmpnn --help
apptainer exec --nv --writable-tmpfs rfantibody.sif rf2 --help
```

### Bind Mounting Data

Use `-B` to mount directories from your host system:

```bash
# Mount a data directory
apptainer exec --nv --writable-tmpfs \
    -B /path/to/data:/data \
    rfantibody.sif rfdiffusion \
    -t /data/target.pdb \
    -f /data/framework.pdb \
    -o /data/output \
    -n 10
```

### Full Pipeline Example (Apptainer)

```bash
# Set up bind mount
DATA_DIR=scripts/examples
APPTAINER_OPTS="--nv --writable-tmpfs -B $DATA_DIR:/data"

# 1. Design backbones with RFdiffusion
apptainer exec $APPTAINER_OPTS rfantibody.sif rfdiffusion \
    -t /data/example_inputs/flu_HA.pdb \
    -f /data/example_inputs/h-NbBCII10.pdb \
    -q /data/example_outputs/1_app_rfdiffusion.qv \
    -n 2 \
    -l "H1:7,H2:6,H3:5-13" \
    -h "B146,B170,B177"

# 2. Design sequences with ProteinMPNN
apptainer exec $APPTAINER_OPTS rfantibody.sif proteinmpnn \
    -q /data/example_outputs/1_app_rfdiffusion.qv \
    --output-quiver /data/example_outputs/2_app_proteinmpnn.qv \
    -n 4 \
    -t 0.2

# 3. Predict structures with RF2
apptainer exec $APPTAINER_OPTS rfantibody.sif rf2 \
    -q /data/example_outputs/2_app_proteinmpnn.qv \
    --output-quiver /data/example_outputs/3_app_rf2.qv \
    -r 10

# 4. Extract scores to scripts/examples/example_outputs/3_app_rf2.sc
apptainer exec $APPTAINER_OPTS rfantibody.sif qvscorefile /data/example_outputs/3_app_rf2.qv
```

### Interactive Shell

To enter an interactive shell inside the container:
```bash
apptainer shell --nv --writable-tmpfs rfantibody.sif
```

Type `exit` or press `Ctrl+D` to exit.

### SLURM Integration

Example SLURM batch script:
```bash
#!/bin/bash
#SBATCH --job-name=rfantibody
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00

apptainer exec --nv --writable-tmpfs \
    -B /scratch/$USER:/data \
    /path/to/rfantibody.sif rfdiffusion \
    -t /data/target.pdb \
    -f /data/framework.pdb \
    -o /data/designs \
    -n 100
```

# Command Line Interface

RFantibody provides a set of command-line tools for running the design pipeline and working with Quiver files. After setting up the environment, these commands are available directly in your terminal.

## Inference Commands

### RFdiffusion (Backbone Design)

```bash
rfdiffusion -t antigen.pdb -f framework.pdb -o designs/ab -n 10
```

Key options:
- `-t, --target`: Target PDB file (antigen)
- `-f, --framework`: Framework PDB file (antibody scaffold)
- `-o, --output`: Output prefix for designs
- `-q, --output-quiver`: Output to Quiver file instead of PDBs
- `-n, --num-designs`: Number of designs to generate
- `-l, --design-loops`: Loop lengths, e.g., `"H1:7,H2:6,H3:5-13,L1:8-13,L2:7,L3:9-11"`
- `-h, --hotspots`: Hotspot residues, e.g., `"A305,A456"`
- `--deterministic`: Enable reproducible results

Example with all options:
```bash
rfdiffusion \
    -t target.pdb \
    -f framework.pdb \
    -q designs.qv \
    -n 100 \
    -l "H1:7,H2:6,H3:5-13,L1:8-13,L2:7,L3:9-11" \
    -h "T305,T456" \
    --deterministic
```

### ProteinMPNN (Sequence Design)

```bash
proteinmpnn -i structures/ -o sequences/ -n 5
```

Key options:
- `-i, --input-dir`: Input directory of PDB files
- `-q, --input-quiver`: Input Quiver file
- `-o, --output-dir`: Output directory for PDB files
- `--output-quiver`: Output Quiver file
- `-l, --loops`: Loops to design (default: `H1,H2,H3,L1,L2,L3`)
- `-n, --seqs-per-struct`: Sequences per structure
- `-t, --temperature`: Sampling temperature (default: 0.1)
- `--deterministic`: Enable reproducible results

Example with Quiver files:
```bash
proteinmpnn -q backbones.qv --output-quiver sequences.qv -n 5 -t 0.2
```

### RF2 (Structure Prediction)

```bash
rf2 -i structures/ -o predictions/
```

Key options:
- `-p, --input-pdb`: Single input PDB file
- `-i, --input-dir`: Input directory of PDB files
- `-q, --input-quiver`: Input Quiver file
- `-o, --output-dir`: Output directory for PDB files
- `--output-quiver`: Output Quiver file
- `-r, --num-recycles`: Recycling iterations (default: 10)
- `-s, --seed`: Random seed for reproducibility
- `--hotspot-show-prop`: Proportion of hotspot residues to show to model (default: 0.1)

Example with Quiver files:
```bash
rf2 -q sequences.qv --output-quiver predictions.qv -r 10
```

## Quiver Utilities

Commands for working with Quiver files (protein design databases):

| Command | Description |
|---------|-------------|
| `qvls` | List all tags in a Quiver file |
| `qvextract` | Extract all PDB files from a Quiver |
| `qvextractspecific` | Extract specific PDBs by tag name |
| `qvscorefile` | Extract scores to a TSV file |
| `qvsplit` | Split into multiple files |
| `qvslice` | Extract specific tags to new Quiver |
| `qvrename` | Rename tags in a Quiver file |
| `qvfrompdbs` | Create a Quiver from PDB files |

All commands support `--help` for detailed usage information.

## Full Pipeline Example

We provide a complete pipeline script at `scripts/examples/nanobody_full_pipeline.sh` that runs all three steps of the RFantibody workflow. Here's a walkthrough:

### Configuration

The script defines editable parameters at the top:

```bash
# Input files
TARGET_PDB="scripts/examples/example_inputs/flu_HA.pdb"       # Target antigen
FRAMEWORK_PDB="scripts/examples/example_inputs/h-NbBCII10.pdb" # Nanobody framework

# Output directory
OUTPUT_DIR="scripts/examples/example_outputs/nb_ha_pipeline"

# RFdiffusion parameters
NUM_DESIGNS=1000                        # Number of backbone designs
DESIGN_LOOPS="H1:7,H2:6,H3:5-13"        # CDR loop lengths
HOTSPOTS="B146,B170,B177"               # Target residues to focus binding

# ProteinMPNN parameters
NUM_SEQS=4                              # Sequences per backbone
SAMPLING_TEMP=0.2                       # Sampling temperature

# RF2 parameters
NUM_RECYCLES=10                         # Recycling iterations
```

### Step 1: RFdiffusion (Backbone Design)

```bash
uv run rfdiffusion \
    --target "$TARGET_PDB" \
    --framework "$FRAMEWORK_PDB" \
    --output-quiver "$OUTPUT_DIR/1_rfdiffusion.qv" \
    --num-designs "$NUM_DESIGNS" \
    --design-loops "$DESIGN_LOOPS" \
    --hotspots "$HOTSPOTS"
```

This generates nanobody backbone structures docked to the target, with CDR loops designed to contact the specified hotspot residues.

### Step 2: ProteinMPNN (Sequence Design)

```bash
uv run proteinmpnn \
    --input-quiver "$OUTPUT_DIR/1_rfdiffusion.qv" \
    --output-quiver "$OUTPUT_DIR/2_proteinmpnn.qv" \
    --seqs-per-struct "$NUM_SEQS" \
    --temperature "$SAMPLING_TEMP"
```

Takes the backbone designs and generates amino acid sequences for the CDR loops. Multiple sequences per backbone increases diversity.

### Step 3: RF2 (Structure Prediction)

```bash
uv run rf2 \
    --input-quiver "$OUTPUT_DIR/2_proteinmpnn.qv" \
    --output-quiver "$OUTPUT_DIR/3_rf2.qv" \
    --num-recycles "$NUM_RECYCLES"
```

Predicts the final structures and provides confidence scores (pLDDT, PAE) for filtering.

### Working with Results

After the pipeline completes, use Quiver utilities to analyze results:

```bash
# List all designs
qvls $OUTPUT_DIR/3_rf2.qv

# Extract information on how well RFdiffusion targeted the hotspots for each design
qvscorefile $OUTPUT_DIR/1_rfdiffusion.qv

# Extract RF2 scores to a tab-separated scorefile (3_rf2.sc)
qvscorefile $OUTPUT_DIR/3_rf2.qv

# Extract all PDBs for visual inspection
qvextract $OUTPUT_DIR/3_rf2.qv -o final_designs/
```

### Running the Pipeline

```bash
bash scripts/examples/nanobody_full_pipeline.sh
```

The script outputs three Quiver files representing each stage, allowing you to inspect intermediate results or restart from any step.

## Production Docker Image

process described above is well suited for development.  However, if you are deploying the RFAntibody pipeline to a production environment, it is better to have a docker image with all dependencies pre-installed.  The `production.Dockerfile` Dockerfile accomplishes this. It builds an image with all dependencies installed.  The RFAntibody source and scripts are installed in `/opt/rfantibody`.

### Building production Docker Image

```
docker build -t rfantibody-production -f production.Dockerfile .
```


# Usage

> **Note:** The examples below assume you have the RFantibody environment active. Either activate it with `source .venv/bin/activate`, or prefix each command with `uv run` (e.g., `uv run rfdiffusion ...`).

## HLT File Format
We must pass structures between the different steps of the RFantibody pipeline. Each step of the pipeline must know:
- The antibody-target complex structure we are currently designing for
- Which chain is the Heavy chain, Light chain, and Target chain
- Which residues are in which of the CDR loops

To enable the passing of this information between steps of the pipeline, we define a file format that we call an HLT file. An HLT file is simply a .pdb file but with the following modifications:
- The Heavy chain is denoted as chain id 'H'
- The Light chain is denoted as chain id 'L'
- The Target chain(s) are denoted as chain id 'T' (even if there are multiple target chains)
- The order of the chains in the file is Heavy then Light then Target
- At the end of the file are PDB Remarks indicating the 1-indexed absolute (not per-chain) residue index of each of the CDR loops. For example:
  ```
  REMARK PDBinfo-LABEL:   32 H1
  REMARK PDBinfo-LABEL:   52 H2
  ```

## Input Preparation

The antibody-finetuned version of RFdiffusion in RFantibody requires an HLT-remarked framework structure as input. We provide a script to perform this conversion:
```bash
python scripts/util/chothia_to_HLT.py -inpdb mychothia.pdb -outpdb myHLT.pdb
```

This script expects a Chothia annotated .pdb file. A great source for these files is [SabDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab), which provides Chothia annotated structures of all antibodies and nanobodies in the PDB and is updated every few months.

We provide the HLT-formatted antibody and nanobody frameworks that were used in the design campaigns from the RFantibody preprint here:
```
Nanobody Framework: scripts/examples/example_inputs/h-NbBCII10.pdb
ScFv Framework: scripts/examples/example_inputs/hu-4D5-8_Fv.pdb
```

## RFdiffusion

The first step in RFantibody is to generate antibody-target docks using an antibody-finetuned version of RFdiffusion. Here is an example command that will run RFdiffusion:
```bash
rfdiffusion \
    -t scripts/examples/example_inputs/rsv_site3.pdb \
    -f scripts/examples/example_inputs/hu-4D5-8_Fv.pdb \
    -o scripts/examples/example_outputs/ab_des \
    -n 20 \
    -l "H1:7,H2:6,H3:5-13,L1:8-13,L2:7,L3:9-11" \
    -h "T305,T456"
```

Let's go through this command in more detail to understand what these options are doing:
- `-t, --target`: A path to the target structure that we wish to design antibodies against. This is commonly a cropped target structure to reduce the computational expense of running the pipeline. Cropping strategies are explained in more depth [here](#truncating-your-target-protein).
- `-f, --framework`: A path to the HLT-formatted antibody framework that we wish to use for our design. RFdiffusion will only design the structure and sequence of regions of the framework which are annotated as loops, this allows us to design the dock and loops of already optimized frameworks.
- `-o, --output`: The prefix of the .pdb file outputs that we will generate.
- `-n, --num-designs`: The number of designs we should generate.
- `-l, --design-loops`: A mapping of each CDR loop to a range of allowed loop lengths. The length of each loop is sampled uniformly from this range and is sampled independently of the lengths sampled for other loops. If a CDR loop exists in the framework but is not specified, this CDR loop will have its sequence and structure fixed during design. If a CDR loop is included but no range of lengths is provided (e.g., `H1:7`), this CDR loop will have its sequence and structure designed but only with the specified length.
- `-h, --hotspots`: A list of hotspot residues that define our epitope. We discuss selecting hotspots in more detail [here](#selecting-a-target-site).

For Quiver file output, use `-q` instead of `-o`:
```bash
rfdiffusion \
    -t target.pdb \
    -f framework.pdb \
    -q designs.qv \
    -n 100 \
    -l "H1:7,H2:6,H3:5-13,L1:8-13,L2:7,L3:9-11" \
    -h "T305,T456"
```

Run `rfdiffusion --help` to see all available options.

## ProteinMPNN

The second step in RFantibody is to take the docks generated by RFdiffusion and assign sequences to the CDR loops. We do this using the base version of ProteinMPNN, ie. not an antibody-finetuned model. For convenience, we package the necessary ProteinMPNN scripts in this repo and provide a wrapper that enables the design of just the CDR loops using ProteinMPNN.

At its simplest, ProteinMPNN may be run on a directory of HLT-formatted .pdb files using the following command:
```bash
proteinmpnn -i /path/to/inputdir -o /path/to/outputdir
```

This will design all CDR loops and will provide one sequence per input structure. To generate multiple sequences per structure or use Quiver files:
```bash
# Generate 5 sequences per structure with Quiver I/O
proteinmpnn -q backbones.qv --output-quiver sequences.qv -n 5

# Specify which loops to design and adjust temperature
proteinmpnn -i structures/ -o sequences/ -l "H1,H2,H3" -t 0.2
```

Run `proteinmpnn --help` to see all available options.

## RF2

The final step of the RFantibody pipeline is to use our antibody-finetuned RF2 to predict the structure of the sequences we just designed. We then assess whether RF2 is confident that the sequence will bind as we designed.

At its simplest, RF2 may be run on a directory of HLT-formatted .pdb files using the following command:
```bash
rf2 -i /path/to/inputdir -o /path/to/outputdir
```

By default this will run with 10 recycling iterations and with 10% of hotspots provided to the model. We don't yet know what combination of these hyperparameters will be most predictive of design success but it should be possible to tune these values once we have data on more antibody and nanobody campaigns.

For Quiver file I/O or to adjust recycling iterations:
```bash
# Use Quiver files with 10 recycles
rf2 -q sequences.qv --output-quiver predictions.qv -r 10

# Run on a single PDB file
rf2 -p design.pdb -o predictions/
```

Run `rf2 --help` to see all available options.

# Practical Considerations for Antibody Design
Designing antibodies is similar to designing _de novo_ binders but is in an earlier stage of development. Here we share advice and learnings on how best to use this pipeline to design antibodies which will work experimentally. We expect some of this advice to change as more antibody design campaigns are performed and best-practices crystallize. Several of these sections are adapted from the analogous section of the RFdiffusion README as these two methods share many similarities and the advice applies to both.

## Selecting a Target Site

Not every site on a target protein is a good candidate for antibody design. For a site to be an attractive candidate for binding it should have >~3 hydrophobic residues for the binder to interact with. Binding to charged polar sites is still quite hard. Binding to sites with glycans close to them is also hard since they often become ordered upon binding and you will take an energetic hit for that. Binding to unstructured loops has historically been hard but [this paper](https://www.nature.com/articles/s41586-023-06953-1) outlines a strategy to use RFdiffusion to bind unstructured peptides which share much in common unstructured loops, using this strategy should work with antibodies but depending on the flexibility of the loop, you will pay an energetic price for ordering the loop during binding.

## Nanobody Docks

When you begin looking at your nanobody outputs, you may notice that many are binding in a side-on dock. This is not a bug and is a result of the model being trained on natural nanobody docks which often bind in this side-on docking style and make some framework-mediated contacts. You may be able to tune your hotspots and CDR lengths to get a more antibody-like dock, but we recommend that if you desire an antibody-like dock, then you should design with an antibody framework.

## Truncating your Target Protein

RFdiffusion and RF2 scale in runtime as O(N^2) where N is the number of residues in your system. As such, it is a very good idea to truncate large targets so that your computations are not unnecessarily expensive. All steps in the RFantibody pipeline are designed to allow for a truncated target. Truncating a target is an art. For some targets, such as multidomain extracellular membranes, a natural truncation point is where two domains are joined by a flexible linker. For other proteins, such as virus spike proteins, this truncation point is less obvious. Generally you want to preserve secondary structure and introduce as few chain breaks as possible. You should also try to leave ~10A of target protein on each side of your intended target site. We recommend using PyMol to truncate your target protein.

## Picking Hotspots

Hotspots are a feature that we integrated into the model to allow for the control of the site on the target which the antibody will interact with. During training, we classify a target residue as a hotspot if it has an average Cβ distance to the closest 5 antibody CDR residues of less than 8 Angstroms. Of all of the hotspots which are identified on the target 0-100% of these hotspots are actually provided to the model and the rest are masked. We find that RFantibody is more sensitive to exactly which hotspots are selected than vanilla RFdiffusion is. Where RFdiffusion tends to generative long helices when given a bad set of hotspots, RFantibody will generally just generate an undocked antibody if a bad set of hotspots is given. It is a very good idea to run a few pilot runs before generating thousands of designs to make sure the number of hotspots you are providing will give results you like.

## Antibody Design Scale

For some of the target campaigns that we report on in our manuscript, we were able to identify VHH binders from a set of 95 designs. In the more general case, however, we expect that design campaigns in the 10k range will be required to identify hits. This is in large part due to the lack of a reliable filtering metric (discussed further in the [Filtering Strategies](#filtering-strategies) section). All data, both positive and negative, is useful for tuning and evaluating filters so if you run a design campaign and wish to share your data with the broader community that would be extremely helpful for moving toward a more reliable filter, higher success rates, and cheaper design campaigns.

## Choosing CDR Lengths

The loop ranges that we used for our design campaigns are provided in the RFdiffusion example files. We determined these ranges by looking at the frequency of naturally occuring lengths for each loop and trying to cover most of the density with our range. We also tried to choose relatively short H3 loops, as we figured these would be easier to design and predict while still giving us enough length to bind effectively. There are some targets where having a long H3 may be useful, for instance when targeting a hydrophobic pocket in a protein. In these cases, the H3 range should be increased beyond what we provide in the examples.

## Filtering Strategies

We recommend the following minimal filtering critieria: <br />
<br />
RF2 pAE < 10 <br />
RMSD (design versus RF2 predicted) < 2&#197; <br />
It may also be helpful to filter by Rosetta ddG < -20 <br />
<br />
The lack of an effective filter is the main limitation of the RFantibody pipeline at the moment. The version of RF2 that we provide may show weak enrichment of binders over non-binders in some cases but more data is needed to make this conclusion convincingly. Newly available structure prediction models such as AF3 present a promising alternative to RF2 and we are in the process of evaluating these models for predictivity on our design campaigns.

# Quiver Files

When running large-scale design campaigns it is often useful to have a single file which holds many designs and the scores associated with those designs. This is gentler on file systems than storing and accessing thousands of individual .pdb files. We offer the ability to use [Quiver files](https://github.com/nrbennet/quiver) in the RFantibody pipeline. These files are simply one large file with the contents of many smaller files inside of them. Each entry has a unique name and can store meta_data about the entry.

RFantibody provides command line tools for working with Quiver files. These are composable (pipe-able) commands inspired by Brian Coventry's [silent_tools](https://github.com/bcov77/silent_tools) project. Use `--help` with any command for detailed options.

```bash
# make a quiver file
qvfrompdbs *.pdb > my.qv

# ask what's in a quiver file
qvls my.qv

# ask how many things are in a quiver file
qvls my.qv | wc -l

# extract all pdbs from a quiver file
qvextract my.qv

# extract to a specific directory
qvextract my.qv -o output_dir/

# extract the first 10 pdbs from a quiver file
qvls my.qv | head -n 10 | qvextractspecific my.qv

# extract a random 10 pdbs from a quiver file
qvls my.qv | shuf | head -n 10 | qvextractspecific my.qv

# extract a specific pdb from a quiver file
qvextractspecific my.qv name_of_pdb_0001

# produce a scorefile from a quiver file
qvscorefile my.qv > scores.tsv

# combine qv files
cat 1.qv 2.qv 3.qv > my.qv

# rename tags in a quiver file
qvls my.qv | sed 's/$/_v2/' | qvrename my.qv > renamed.qv

# slice specific tags into a new quiver file
qvls | shuf | head -n 10 | qvslice > subset.qv

# split a quiver file into groups of 100
qvsplit my.qv 100 -o split_dir/
```

## Reading and Writing Quiver Files
All steps of RFantibody allow for the use of Quiver files. The syntax is summarized here:

RFdiffusion takes only a .pdb file target and framework as input. To output the designed backbones at quiver files append this argument to your input command:
```
inference.quiver=/path/to/myoutput.qv
```

For ProteinMPNN, to input and output a Quiver file, use the following two arguments:
```
-inquiver /path/to/myinput.qv -outquiver /path/to/myoutput.qv
```

RFantibody takes the following two configs to work with Quiver file input and output
```
input.quiver=/path/to/myinput.qv output.quiver=/path/to/myoutput.qv
```

# Conclusion
We are really excited to release RFantibody open-source! We can't wait to see what kinds of designs the broader community comes up with. We have worked hard to make this codebase as easy to setup and run as possible but please open a GitHub issue if you run into any problems.

\- Nate, Joe, and the RFantibody Team

---

RFantibody builds directly off of the architecture and weights of several methods which we acknowledge here. We thank Minkyung Baek and Frank DiMaio for developing RoseTTAFold and RoseTTAFold2 which the original RFdiffusion and our antibody-fine tuned RoseTTAFold2 model are based off of. We thank Justas Dauparas for developing ProteinMPNN which we provide an antibody-specific wrapper for in this repo. As the antibody-finetuned RFdiffusion we provide here is directly based off of the original version of RFdiffusion, we also thank David Juergens, Brian Trippe, and Jason Yim who co-developed the original RFdiffusion with us. RFantibody is released under an MIT License (see LICENSE file). It is free for both non-profit and for-profit use.
