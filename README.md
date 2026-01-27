# RFantibody
### Structure-Based _de novo_ Antibody Design

![g1](https://github.com/user-attachments/assets/6457cd75-3950-4be2-8536-9a564d2013c3)

# Description
RFantibody is a pipeline for the structure-based design of _de novo_ antibodies and nanobodies. RFantibody consists of three separate methods:
- Protein backbone design with an antibody-fintuned version of [RFdiffusion](https://www.nature.com/articles/s41586-023-06415-8)
- Protein sequence design with [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187)
- _In silico_ filtering of designs using an antibody-finetuned version of [RoseTTAFold2](https://www.biorxiv.org/content/10.1101/2023.05.24.542179v1)

The RFantibody pipeline is described in detail in [this preprint](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1)

# Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
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

### Docker

RFantibody is designed to run in a Docker container. Containers run a seperate operating system on top of the host operating system. This offers the following advantages:
- Simplified installation: all you need is the Docker software suite
- Host system-invariance: because this is run inside a container it behaves essentially the same wherever you run it from

All you need to install on your host system is Docker which is free to install [here](https://docs.docker.com/engine/install/).
If you are running RFantibody on cloud compute, Docker will often be preinstalled. You can check this by running:
```
which docker
```
If this returns a path then you have docker available and are good to go.

### GPU Acceleration

RFantibody requires an NVIDIA GPU to run. You can check whether you have an NVIDIA GPU available by running:
```
nvidia-smi
```
If this command runs successfully then you have an compatible GPU and RFantibody will be able to run on it.

# Downloading Weights

Navigate to the directory where RFantibody is downloaded. Then run the following command to download the pipeline weights to the RFantibody/weights directory.
```
bash include/download_weights.sh
```

# Installation

## Building and Running the RFantibody Docker Container

You will need to run this to make sure that you have the correct privileges to start your docker container:

```
sudo usermod -aG docker $USER
```

After running this command, you will need to restart your terminal session for this change to take effect.


### Build docker image
Navigate to the directory where RFantibody is downloaded. Then run the following command to build the Docker image for RFantibody:
```
docker build -t rfantibody .
```

### Start the Docker image
Run the following command to start the docker container based on the image you just built:
```
docker run --name rfantibody --gpus all -v .:/home --memory 10g -it rfantibody
```
This will put you into the RFantibody container at the /home directory which mirrors the directory that you ran the last command from.



## Setup the Python Environment
From the RFantibody container, navigate to the project root and run:
```bash
cd /home
uv sync --all-extras
```
This uses [uv](https://docs.astral.sh/uv/) to:
- Create a virtual environment in `.venv/`
- Install all dependencies including PyTorch with CUDA 11.8 support
- Install [Deep Graph Library](https://www.dgl.ai) from CUDA-enabled wheels
- Install test dependencies (`--all-extras`)

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

```bash
# 1. Design backbones with RFdiffusion
rfdiffusion \
    -t antigen.pdb \
    -f framework.pdb \
    -q backbones.qv \
    -n 100 \
    -l "H1:7,H2:6,H3:5-13,L1:8-13,L2:7,L3:9-11" \
    -h "T305,T456"

# 2. Design sequences with ProteinMPNN
proteinmpnn -q backbones.qv --output-quiver sequences.qv -n 5

# 3. Predict structures with RF2
rf2 -q sequences.qv --output-quiver predictions.qv -r 10

# 4. Extract scores and top designs
qvscorefile predictions.qv > scores.tsv
qvextract predictions.qv -o final_designs/
```

# Usage

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

The antibody-finetuned version of RFdiffusion in RFantibody requires an HLT-remarked framework structure as input. We provide a script to perform this conversion that can be run as follows:
```
# From inside of the rfantibody container

uv run python /home/scripts/util/chothia_to_HLT.py -inpdb mychothia.pdb -outpdb myHLT.pdb
```

This script expects a Chothia annotated .pdb file. A great source for these files is [SabDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab), which provides Chothia annotated structures of all antibodies and nanobodies in the PDB and is updated every few months.

We provide the HLT-formatted antibody and nanobody frameworks that were used in the design campaigns from the RFantibody preprint here:
```
Nanobody Framework: RFantibody/scripts/examples/example_inputs/h-NbBCII10.pdb
ScFv Framework: RFantibody/scripts/examples/example_inputs/hu-4D5-8_Fv.pdb
```

## RFdiffusion

The first step in RFantibody is to generate antibody-target docks using an antibody-finetuned version of RFdiffusion. Here is an example command that will run RFdiffusion:
```
# From inside of the rfantibody container

uv run python  /home/scripts/rfdiffusion_inference.py \
    --config-name antibody \
    antibody.target_pdb=/home/scripts/examples/example_inputs/rsv_site3.pdb \
    antibody.framework_pdb=/home/scripts/examples/example_inputs/hu-4D5-8_Fv.pdb \
    inference.ckpt_override_path=/home/weights/RFdiffusion_Ab.pt \
    'ppi.hotspot_res=[T305,T456]' \
    'antibody.design_loops=[L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13]' \
    inference.num_designs=20 \
    inference.output_prefix=/home/scripts/examples/example_outputs/ab_des
```

Let's go through this command in more detail to understand what these configs are doing:
- antibody.target_pdb: A path to the target structure that we wish to design antibodies against. This is commonly a cropped target structure to reduce the computational expense of running the pipeline. Cropping strategies are explained in more depth [here](#truncating-your-target-protein).
- antibody.framework_pdb: A path to the HLT-formatted antibody framework that we wish to use for our design. RFdiffusion will only design the structure and sequence of regions of the framework which are annotated as loops, this allows us to design the dock and loops of already optimized frameworks.
- inference.ckpt_override_path: The path to the set of RFdiffusion model weights we will use for inference
- ppi.hotspot_res: A list of hotspot residues that define our epitope. These are provided in the same format as in vanilla RFdiffusion. We discuss selecting hotspots in more detail [here](#selecting-a-target-site).
- antibody.design_loops: A dictionary that maps each CDR loop to a range of allowed loop lengths. The length of each loop is sampled uniformly from this range and is sampled independently of the lengths sampled for other loops. If a CDR loop exists in the framework but is not in the dict, this CDR loop will have its sequence and structure fixed during design. If a CDR loop is included in the dict but no range of lengths is provided, this CDR loop will have its sequence and structure designed but only with the length of the loop that is provided in the framework structure.
- inference.num_designs: The number of designs we should generate.
- inference.output_prefix: The prefix of the .pdb file outputs that we will generate.

We provide an example command with example inputs which can be run as follows:
```
# From inside of the rfantibody container

bash /home/scripts/examples/rfdiffusion/antibody_pdbdesign.sh
```

## ProteinMPNN

The second step in RFantibody is to take the docks generated by RFdiffusion and assign sequences to the CDR loops. We do this using the base version of ProteinMPNN, ie. not an antibody-finetuned model. For convenience, we package the necessary ProteinMPNN scripts in this repo and provide a wrapper script that enables the design of just the CDR loops using ProteinMPNN.

At its simplest, ProteinMPNN may be run on a directory of HLT-formatted .pdb files using the following command:
```
# From inside of the rfantibody container

uv run python /home/scripts/proteinmpnn_interface_design.py \
    -pdbdir /path/to/inputdir \
    -outpdbdir /path/to/outputdir
```

This will design all CDR loops and will provide one sequence per input structure. There are many more arguments that may be experimented with and are explained by running:
```
uv run python /home/scripts/proteinmpnn_interface_design.py --help
```

We provide an example command with example inputs which can be run as follows:
```
# From inside of the rfantibody container

bash /home/scripts/examples/proteinmpnn/ab_pdb_example.sh
```

## RF2

The final step of the RFantibody pipeline is to use our antibody-finetuned RF2 to predict the structure of the sequences we just designed. We then assess whether RF2 is confident that the sequence will bind as we designed.

At it's simplest, RF2 may be run on a directory of HLT-formatted .pdb files using the following command:
```
# From inside of the rfantibody container

uv run python /home/scripts/rf2_predict.py \
    input.pdb_dir=/path/to/inputdir \
    output.pdb_dir=/path/to/outputdir
```

By default this will run with 10 recycling iterations and with 10% of hotspots provided to the model. We don't yet know what combination of these hyperparameters will be most predictive of design success but it should be possible to tune these values once we have data on more antibody and nanobody campaigns.

We provide an example with example inputs which can be run as follows:
```
# From inside of the rfantibody container

bash /home/scripts/examples/rf2/ab_pdb_example.sh
```

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
