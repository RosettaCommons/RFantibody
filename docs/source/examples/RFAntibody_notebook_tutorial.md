# Running the RFAntibody Pipeline Using the Notebook

## Introduction
In this tutorial, we're going to walk through how to run the RFantibody pipeline using the dedicated [Google Colab notebook included in the repository](RFAntibody_v1.ipynb). Running the pipeline via the notebook on Google CoLab means that no local installation of RFAntibody is necessary to follow this tutorial. 

The pipeline follows three main steps: first, we use [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion?tab=readme-ov-file) to generate antibody backbone structures, then we use [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) to design sequences for those backbones, and finally we use [AlphaFold3](https://github.com/google-deepmind/alphafold3) to predict the structure of our designs and evaluate them. 

```{Note}
This tutorial uses multiple models, and the code and model weights for each are distributed under separate licenses. Some, including AlphaFold3, restrict use of the weights to non-commercial research and require you to request them directly from the developer; they cannot be redistributed. Review the license terms for every model before using it in your own work, particularly in commercial settings.
```

Before getting started: For your own projects, we recommend running RFantibody on GPU nodes within a high-performance computing (HPC) cluster. See the RFantibody README for instructions for installing RFantibody. This notebook allows you to run small calculations to either get a feel for the tool or if you do not otherwise have access to HPC resources.

## Table of Contents
- [Introduction](#introduction)
- [Limitations](#limitations)
- [Background: The Structure of antibodies](#background-the-structure-of-antibodies)
- [Running the Colab Notebook](#running-the-colab-notebook)
  - [Setup](#setup)
  - [Input preparation](#input-preparation)
    - [Antibody framework](#antibody-framework)
      - [Case 1: Not sure what framework to use?](#case-1-not-sure-what-framework-to-use)
    - [Case 2: Known therapeutic antibody target, but unknown structure](#case-2-known-therapeutic-antibody-target-but-unknown-structure)
  - [Thetarget structure](#thetarget-structure)
  - [Hotspot Selection](#hotspot-selection)
- [Running RFdiffusion](#running-rfdiffusion)
- [Running MPNN](#running-mpnn)
- [Structure prediction (AlphaFold3)](#structure-prediction-alphafold3)
- [Analyzing your results](#analyzing-your-results)
- [References](#references)

## Limitations
- The model is fine-tuned specifically to design the **complementarity-determining region (CDRs)** of antibodies. It's not trained to design frameworks or other constant regions.
- It was trained on **Chothia-defined CDRs**: if you define your CDRs by a different numbering scheme, results may not be optimal.
- This version **only supports amino acid targets**. All-atom design isn't supported here. 
- If you're using the AlphaFold3 web server for the prediction step, DeepMind specifies a 5,000 amino acid limit for the combined length of your antibody and antigen.

## Background: The Structure of antibodies
Conventional antibodies contain 4 chains–2 heavy chains and 2 light chains. Each heavy chain (teal) forms a dimer with a light chain (orange), as shown below:
```{figure} ../.assets/antibody_structure.png
:width: 100%
:alt: Example structure of an antibody with a close up view of the VH and VL domains of one Fab. 

Example structure of an antibody. The light chain is shown in orange and the heavy chain is shown in teal.
```

Each chain contains multiple domains. The light chain has one constant domain and one variable domain, V<sub>L</sub>. The heavy chain contains 3-4 constant domains (depending on the antibody class) and one variable domain, V<sub>H</sub>. Each variable domain contains three complementarity-determining regions, or CDRs. The CDRs are primarily responsible for antibody binding, and these are the regions that RFantibody is trained to design. 

```{figure} ../.assets/antibody_structure_sequence.png
:width: 100%
:alt: The VL and VH domains from the image above with the sequences for the full sequences. 

V<sub>L</sub> (orange) and V<sub>H</sub> (teal) domains with their full sequences. Variable domain sequences are bolded.
```

```{figure} ../.assets/cdr_sequence_highlight.png
:width: 100%
:alt: CDR regions in the VL and VH domains highlighted with their corresponding sequences underlined and labeled.

CDR regions in the V<sub>L</sub> (green) and V<sub>H</sub> (yellow) domains highlighted with their corresponding sequences underlined and labeled.
```

There are several CDR definitions in the literature, and sources describing the different definitions and associated numbering conventions can be found in the [References section](#references). 
RFantibody was trained on CDRs following the Chothia definition.


```{Note}
Nanobodies are the single variable domains derived from heavy chain-only antibodies naturally produced by camelids and cartilaginous fish. RFantibody is validated for the design of [camelid-like nanobodies, or VHHs](https://www.mdpi.com/1422-0067/24/6/5994). These, like conventional antibody VHs, contain three CDRs. The improved solubility, ease of expression, and modularity of VHHs may make them a more appropriate format than a VH/VL pair, depending on the application. All of the instructions and tools in this tutorial apply to both conventional antibodies and VHHs.  
```

## Running the Colab Notebook

### Setup
To get started: 
1. Log into a google account and set up a Google Drive folder for your run. This is where all of your input and output files will be stored. 
1. Make a copy of the [RFAntibody_v1.ipynb](RFAntibody_v1.ipynb) notebook and store it in your own Google Drive. You will need your own copy in order to save any of the changes you make to the notebook. 
1. If you have a CoLab Pro subscription, make sure to go to `Runtime > Change runtime type` and select a more powerful system such as H100, A100, or L4 GPU. 
1. Run the setup cells (sections A-C) to seup your Google Drive connection and install RFAntibody in the notebook. 

The notebook is fairly long. You canse use the Table of Contents button in the left panel to easily switch between sections. Any section with **- TO EDIT** in the heading will need information about your inputs or file system in order to run. 

### Input preparation
We will need two inputs: 
1. Antibody frameworks
1. A target structure

#### Antibody framework
It is recommended to have multiple antibody frameworks as starting points since different frameworks can support different types of CDR loops. This allows for the production of more diverse designs leading to more chances of meeting your design goals. 

However, what if you don't know what framework to use, what structure to use, or both? We will cover these situations in the following cases:

##### Case 1: Not sure what framework to use? 
If you want to find some validated heavy-light chain pairs with known structures, [Thera-SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/therasabdab/search/) is a good place to start. On that page scroll down until you see **Search therapeutics by attribute** and restrict to search to 'known structures':
```{figure} ../.assets/thera_sabdab.png
:width: 100%

Attribute search menu from Thera-SAbDab.
```
For your design projects you may want to add additional restrictions.

You should see that your search has retrieved a few hundred structures. In this tutorial we will explore the abciximab entry, but feel free to explore others.

```{figure} ../.assets/thera_sabdab_search.png
:width: 100%
First few results from the search of the database with the abciximab entry highlighted.
```

When you click on the abciximab entry you are brought to a summary page with further information for that particular structure. In the dropdown labeled **SAbDab Links** click on **6v4p [FVs:CD]** to see the full structure

```{figure} ../.assets/structral_summary.png
width:100%

Thera-SAbDab summary page for abciximab.
```

On the new page that opens you should see an interactive window with the structure of abciximab. Scroll to the bottom of this page to find the download options and download the **full experimental structure with IMGT-numbered antibody chains**. This will give you the input information you need to run RFAntibody. 

```{important}
Note that the outputs from the Thera-SAbDab database are IMGT-numbered. The RFAntibody is expecting **Chothia-numbered** PDB inputs. You will need to convert the numbering before using RFAntibody. 
```

If you are starting with a full IgG or Fab, we strongly recommend truncating to the variable domains for quicker RFdiffusion and structure prediction runs. The heavy chain variable domain will likely begin with “EV” and end with “TVSS” and the light chain will usually begin with “DIQ” or “DIV” and end in “LEIK” or “VEIK” or something similar (see [Background](#background-the-structure-of-antibodies)). 

#### Case 2: Known therapeutic antibody target, but unknown structure
For this example, let's say that we know we want to work on trastuzumab but we don't have a PDB structure. 

Start by finding the sequence for this antibody in the [KEGG drug database](https://www.genome.jp/kegg/drug/):

```{figure} ../.assets/KEGG_search.png
:widtth: 100%

KEGG Drug Database homepage with search for trastuzumab hilighted.
```

There are a handful of entries that come up, each with information about the disease it is associated with, various brand names, etc. You can pick any of these entries, for this tutorial we have picked **D03257**:

```{figure} ../.assets/kegg_search_trastuzumab.png
:width: 100%

Search results for trastuzumab. 
```

Once you click on an entry you will see a new page with more information including the recorded sequence for trastuzumab. Take note of the variable regions in the sequence, as a reminder, the heavy chain variable domain will likely begin with “EV” and end with “TVSS” and the light chain will usually begin with “DIQ” or “DIV” and end in “LEIK” or “VEIK” or something similar (see [Background](#background-the-structure-of-antibodies)). 

```{figure} ../.assets/trastuzumab_sequence.png

Sequence cell for trastuzumab with variable regions highlighted. 
```

Copy the sequences of the variable regions and search for them using the [RCSB PDB sequence search](https://www.rcsb.org/search/advanced/sequence). Set the sequence identity to 100% to find results that are exact matches.

```{figure} ../.assets/PDB_advanced_search.png
:width: 100%

RCSB PDB Advanced Search Query Builder with the sequence for trastuzumab and the sequence identity setting set to 100%. 
```

This search will come up with a variety of structures, which one you choose will depend on your design goals. However, we do recommend that you check to ensure that all of the variable domain framework (non-CDR) residues are solved. You can check this by looking at the sequence of the structure you select and confirming that there are no grayed-out residues the variable domain framework:

```{figure} ../.assets/good_heavy_chain.png
:width:100%

1N8Z heavy chain sequence showing residues that are all solved. 
```

```{figure} ../.assets/bad_heavy_chain.png
:width: 100%

Example of a structure whose residues are not all solved. This particular sequence comes from the trastuzumab FAB heavy chain of entry [22GE](https://www.rcsb.org/3d-view/22GE).
```

For this tutorial we'll select [1N8Z](https://www.rcsb.org/structure/1N8Z#entity-2). Once you have identified your PDB entry, search for it in the [SAbDab2 database](https://sabdab.opig.stats.ox.ac.uk/pdb/pdb_00001n8z) and download the structure files. Keep in mind that the SAbDab2 structure files are IMGT-numbered while RFantibody is expecting Chothia numbered antbibodies. Once you have your correctly numbered chains, open your structure in [PyMOL](https://www.pymol.org/) and copy the antibody chains into a new object. Save that new object as its own PDB file and add it to your google drive folder. It is this cropped PDB file that you want to use as input to RFantibody. 

### Thetarget structure
Make sure to crop your target protein to just the region around the epitope you want to bind, this will significantly reduce the amount of time and resources it takes for the RFantibody pipeline to run to completion. You can crop your structures using [PyMOL](https://www.pymol.org/).

Once you have your cropped structre, upload it to your google drive folder. You now have the input files necessary to run RFantibody. 

### Hotspot Selection
Hotspots are solvent-accessible residues on your target that you want the designed antibody to contact. You can imagine them as tethers for a parade balloon. For the RFantibody pipeline, the antibody has to position itself so that a CDR Cα atom is within about 10 Å of each hotspot. 

When choosing hotspots we recommend beginning with 3 and trying to arrange them in a triangle across the epitope. If they're too close together they become redundant, but if they are too far apart it might be impossible for a single antibody to satisfy all three at once. 

Once you've identified your hotspots, fill them in in section A of the 'Prepare input files' section of the notebook. Make sure you have filled out the necessary information in section A and section B and then run these cells. They will clean up your input structures and format them in a way that is compatible with the pipeline.

## Running RFdiffusion
This section refers to part A in the 'Run pipeline' section of the notebook. 

To run this section:
1. Copy the path to your output folder in Google Drive and paste it into the appropriate location in the cell. 
1. Fill in the desired CDR loop lengths. A typical approach is to keep a smaller length range for CDR H1 and H2, and leave more diversity (a wider range) for CDR H3, since CDR H3 tends to be the most important loop for binding specificity.
1. Input the VH and VL chain IDs into the cell. Then run this cell and the next one.

Once it finishes, you'll see your output files. At this point, you can download them and take a look in PyMOL. **Just keep in mind that there will be no sidechains at this stage, since we haven't run ProteinMPNN yet.** ([RFdiffusion](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1) outputs backbone structures only, the sequence of any designed regions will be composed only of glycine residues.) But you can look at the overall dock and the CDR loop conformations to get a sense of whether the designs look reasonable. Alternatively, you can use the inline visualizer right in the notebook: the target is colored teal, the heavy chain is pink, and the light chain is wheat.

## Running MPNN
This section refers to section B of the pipeline.

Now that we have our backbones, we're going to use [ProteinMPNN](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1) to design sequences for them. Essentially, what ProteinMPNN does is take a protein backbone (just the 3D coordinates, no amino acid identities) and predicts sequences that would fold into that structure.  

We have a few suggestions for the options: 
- Generate **more sequences** per backbone, this gives you more candidates to evaluate downstream
- Increasing the temperature gives you more sequence diversity, but keep it **at or below 0.3**. Any higher and the quality of the sequences it creates starts to dimish. 
- We strongly recommend omitting **cysteines** in antibody and VHH design to avoid potential sites of oxidation and inadvertant disulfide bonds.
    - Methionine can also be omitted from CDRs since methionine is also prone to oxidation. 


## Structure prediction (AlphaFold3)
This section refers to section C in the notebook. 

To complete the RFantibody design pipeline, we will use the ipTM metric from [AlphaFold3](https://www.nature.com/articles/s41586-024-07487-w) to verify that the sequences generated by ProteinMPNN fold into reasonable structures and interact in ways that are expected. 

However, there are a few situations in which you will not be able to complete this step with AlphaFold3: 
1. Your target is too large (over 5,000 amino acids) and cannot be reasonably cropped. 
1. Your target is a viral target or otherwise restricted by the AlphaFold server. (You can run AlphaFold locally instead, but instructions for how to set up a local installation are outside the scope of this tutorial.)
1. You have hit the prediction limit on the AlphaFold3 server, the server caps you at 30 predictions per day per email address. 

```{important}
The AlphaFold3 weights and server are covered by a license that restricts commercial use. Make sure you have the appropriate licenses if you are using this pipeline to create commercial products or try alternative weights.
```

```{note}
If you are unable to run AlphaFold3 for any reason, you can run section D (RoseTTAFold2) instead.
```

To run the AF3 web server: 
1. Run the first few cells in section C of the notebook. 
1. Fill in your Google Drive path to your ProteinMPNN outputs in the appropriate place. 
1. Paste the sequence of your target chain(s) into the appropriate location in that same cell. Make sure to give each chain a unique ID, we recommend starting at "A" and skipping "H" and "L" since these will be reserved for the antibody chains. 
1. Add the path where you want the JSON file created by the web server to be saved.
1. Run the cell. 
1. Download the JSON file that is created to your local computer. 
1. Go to the [AlphaFold3 web server](https://alphafoldserver.com/) and upload the JSON file. 
1. Submit your predictions. 

## Analyzing your results
Download your results from the AlphaFold3 web server. (Or from Google Drive if you ran RoseTTAFold2 instead.) You can visually inspect the structures using [PyMOL](https://pymol.org/). Check to see if the dock and the CDR loops look reasonable. 

Also check the ipTM and pAE metrics, these tell you: 
- ipTM (interface predicted TM-score): confidence metric for the interface between your antibody and the target. Values closer to 1 indicate a high confidence that the chains will actually interact the way they are shown in the predicted structure. 
- pAE (predicted aligned error): measure of positional uncertainty. Lower pAE values indicate a higher confidence in the relative positions of the two chains. 

You can also calculate the RMSD between your RFdiffusion backbone and the predicted structure. A lower RMSD means that the backbone and structure are similar, leading to a higher confidence that the structure is real and stable. 

## References
- [Manuscript describing the RFantibody pipelinehttps](https://www.nature.com/articles/s41586-025-09721-5)
- [ProteinMPNN manuscript](https://www.science.org/doi/full/10.1126/science.add2187)
- [AlphaFold3 manuscript](https://www.nature.com/articles/s41586-024-07487-w)
- [ANARCI, for antibody structure renumbering](https://academic.oup.com/bioinformatics/article/32/2/298/1743894)
- [Antibody numbering schemes and CDR definitions](http://www.bioinf.org.uk/abs/info.html)
- [Review on antibody numbering schemes (Zhu et al, 2025)](https://academic.oup.com/peds/article/doi/10.1093/protein/gzaf005/8102903?guestAccessKey=#524740074)
