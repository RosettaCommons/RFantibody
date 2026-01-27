#!/usr/bin/env python3
"""
Model inference CLI commands.

Commands for running RFdiffusion, ProteinMPNN, and RF2.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import click

from rfantibody.config import PathConfig


# =============================================================================
# RFdiffusion CLI
# =============================================================================

@click.command()
@click.option('--target', '-t', type=click.Path(exists=True, path_type=Path), required=True,
              help='Target PDB file (antigen)')
@click.option('--framework', '-f', type=click.Path(exists=True, path_type=Path), required=True,
              help='Framework PDB file (antibody scaffold)')
@click.option('--output', '-o', type=click.Path(path_type=Path), default='designs/ab_des',
              help='Output prefix for designs (default: designs/ab_des)')
@click.option('--output-quiver', '-q', type=click.Path(path_type=Path), default=None,
              help='Output to Quiver file instead of PDB files')
@click.option('--num-designs', '-n', type=int, default=10,
              help='Number of designs to generate (default: 10)')
@click.option('--design-loops', '-l', type=str, default='H1:,H2:,H3:,L1:,L2:,L3:',
              help='Loops to design with optional length ranges (default: H1:,H2:,H3:,L1:,L2:,L3:)')
@click.option('--hotspots', '-h', type=str, default=None,
              help='Hotspot residues on target, e.g., "A100,A105,A110"')
@click.option('--weights', '-w', type=click.Path(exists=True, path_type=Path), default=None,
              help='Model weights path (default: auto-detect)')
@click.option('--diffuser-t', type=int, default=50,
              help='Number of diffusion timesteps (default: 50)')
@click.option('--final-step', type=int, default=1,
              help='Final diffusion step (default: 1)')
@click.option('--deterministic', is_flag=True,
              help='Enable deterministic mode for reproducibility')
@click.option('--extra', '-e', type=str, multiple=True,
              help='Extra Hydra overrides (can be specified multiple times)')
def rfdiffusion(
    target: Path,
    framework: Path,
    output: Path,
    output_quiver: Optional[Path],
    num_designs: int,
    design_loops: str,
    hotspots: Optional[str],
    weights: Optional[Path],
    diffuser_t: int,
    final_step: int,
    deterministic: bool,
    extra: tuple
):
    """Run RFdiffusion antibody design.

    Generates antibody structures targeting a specific antigen using diffusion-based
    design. The designed loops can be specified with length ranges.

    \b
    Examples:
        # Basic antibody design
        rfdiffusion -t antigen.pdb -f framework.pdb -o my_designs/ab

        # Design with specific loop lengths and hotspots
        rfdiffusion -t antigen.pdb -f framework.pdb \\
            -l "H1:7,H2:6,H3:5-13,L1:8-13,L2:7,L3:9-11" \\
            -h "A305,A456" -n 5

        # Output to Quiver file
        rfdiffusion -t antigen.pdb -f framework.pdb -q designs.qv -n 100
    """
    # Find the inference script
    script_path = PathConfig.SCRIPTS_DIR / 'rfdiffusion_inference.py'
    if not script_path.exists():
        click.echo(f'Error: RFdiffusion script not found at {script_path}', err=True)
        sys.exit(1)

    # Build command
    cmd = ['python', str(script_path), '--config-name', 'antibody']

    # Required parameters
    cmd.append(f'antibody.target_pdb={target}')
    cmd.append(f'antibody.framework_pdb={framework}')

    # Output
    if output_quiver:
        cmd.append(f'inference.quiver={output_quiver}')
    else:
        # Ensure output directory exists
        output_dir = Path(output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd.append(f'inference.output_prefix={output}')

    # Design parameters
    cmd.append(f'inference.num_designs={num_designs}')

    # Parse design loops into Hydra format
    loops_list = [l.strip() for l in design_loops.split(',')]
    cmd.append(f"antibody.design_loops=[{','.join(loops_list)}]")

    # Hotspots
    if hotspots:
        hotspot_list = [h.strip() for h in hotspots.split(',')]
        cmd.append(f"ppi.hotspot_res=[{','.join(hotspot_list)}]")

    # Model weights
    if weights:
        cmd.append(f'inference.ckpt_override_path={weights}')
    else:
        default_weights = PathConfig.get_weight_path('rfdiffusion')
        if default_weights.exists():
            cmd.append(f'inference.ckpt_override_path={default_weights}')

    # Diffuser settings
    cmd.append(f'diffuser.T={diffuser_t}')
    cmd.append(f'inference.final_step={final_step}')

    # Deterministic mode
    if deterministic:
        cmd.append('inference.deterministic=True')

    # Extra overrides
    for override in extra:
        cmd.append(override)

    click.echo(f'Running RFdiffusion with {num_designs} designs...')
    click.echo(f'Target: {target}')
    click.echo(f'Framework: {framework}')
    click.echo(f'Design loops: {design_loops}')

    result = subprocess.run(cmd, cwd=str(PathConfig.PROJECT_ROOT))
    sys.exit(result.returncode)


# =============================================================================
# ProteinMPNN CLI
# =============================================================================

@click.command()
@click.option('--input-dir', '-i', type=click.Path(exists=True, path_type=Path), default=None,
              help='Input directory containing PDB files')
@click.option('--input-quiver', '-q', type=click.Path(exists=True, path_type=Path), default=None,
              help='Input Quiver file')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), default='outputs',
              help='Output directory for PDB files (default: outputs)')
@click.option('--output-quiver', type=click.Path(path_type=Path), default=None,
              help='Output Quiver file')
@click.option('--loops', '-l', type=str, default='H1,H2,H3,L1,L2,L3',
              help='Loops to design (default: H1,H2,H3,L1,L2,L3)')
@click.option('--seqs-per-struct', '-n', type=int, default=1,
              help='Number of sequences per structure (default: 1)')
@click.option('--temperature', '-t', type=float, default=0.1,
              help='Sampling temperature (default: 0.1)')
@click.option('--weights', '-w', type=click.Path(exists=True, path_type=Path), default=None,
              help='Model weights path (default: auto-detect)')
@click.option('--omit-aas', type=str, default='CX',
              help='Amino acids to omit from design (default: CX)')
@click.option('--deterministic', is_flag=True,
              help='Enable deterministic mode for reproducibility')
@click.option('--debug', is_flag=True,
              help='Enable debug mode (crash on errors)')
def proteinmpnn(
    input_dir: Optional[Path],
    input_quiver: Optional[Path],
    output_dir: Path,
    output_quiver: Optional[Path],
    loops: str,
    seqs_per_struct: int,
    temperature: float,
    weights: Optional[Path],
    omit_aas: str,
    deterministic: bool,
    debug: bool
):
    """Run ProteinMPNN sequence design for antibodies.

    Designs sequences for antibody CDR loops using ProteinMPNN.
    Input can be a directory of PDB files or a Quiver file.

    \b
    Examples:
        # Design sequences for PDBs in a directory
        proteinmpnn -i structures/ -o designed/ -n 5

        # Design from Quiver file
        proteinmpnn -q designs.qv --output-quiver designed.qv

        # Design only specific loops with higher temperature
        proteinmpnn -i structures/ -l "H3,L3" -t 0.2
    """
    # Validate input
    if input_dir is None and input_quiver is None:
        click.echo('Error: Must specify either --input-dir or --input-quiver', err=True)
        sys.exit(1)
    if input_dir is not None and input_quiver is not None:
        click.echo('Error: Cannot specify both --input-dir and --input-quiver', err=True)
        sys.exit(1)

    # Find the inference script
    script_path = PathConfig.SCRIPTS_DIR / 'proteinmpnn_interface_design.py'
    if not script_path.exists():
        click.echo(f'Error: ProteinMPNN script not found at {script_path}', err=True)
        sys.exit(1)

    # Build command
    cmd = ['python', str(script_path)]

    # Input
    if input_dir:
        cmd.extend(['-pdbdir', str(input_dir)])
    if input_quiver:
        cmd.extend(['-quiver', str(input_quiver)])

    # Output
    if output_quiver:
        cmd.extend(['-outquiver', str(output_quiver)])
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cmd.extend(['-outpdbdir', str(output_dir)])

    # Design parameters
    cmd.extend(['-loop_string', loops])
    cmd.extend(['-seqs_per_struct', str(seqs_per_struct)])
    cmd.extend(['-temperature', str(temperature)])
    cmd.extend(['-omit_AAs', omit_aas])

    # Model weights
    if weights:
        cmd.extend(['-checkpoint_path', str(weights)])
    else:
        default_weights = PathConfig.get_weight_path('proteinmpnn')
        if default_weights.exists():
            cmd.extend(['-checkpoint_path', str(default_weights)])

    # Flags
    if deterministic:
        cmd.append('-deterministic')
    if debug:
        cmd.append('-debug')

    input_source = input_dir or input_quiver
    click.echo(f'Running ProteinMPNN sequence design...')
    click.echo(f'Input: {input_source}')
    click.echo(f'Loops: {loops}')
    click.echo(f'Sequences per structure: {seqs_per_struct}')

    result = subprocess.run(cmd, cwd=str(PathConfig.PROJECT_ROOT))
    sys.exit(result.returncode)


# =============================================================================
# RF2 CLI
# =============================================================================

@click.command()
@click.option('--input-pdb', '-p', type=click.Path(exists=True, path_type=Path), default=None,
              help='Input PDB file')
@click.option('--input-dir', '-i', type=click.Path(exists=True, path_type=Path), default=None,
              help='Input directory containing PDB files')
@click.option('--input-quiver', '-q', type=click.Path(exists=True, path_type=Path), default=None,
              help='Input Quiver file')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), default=None,
              help='Output directory for PDB files')
@click.option('--output-quiver', type=click.Path(path_type=Path), default=None,
              help='Output Quiver file')
@click.option('--num-recycles', '-r', type=int, default=10,
              help='Number of recycling iterations (default: 10)')
@click.option('--weights', '-w', type=click.Path(exists=True, path_type=Path), default=None,
              help='Model weights path (default: auto-detect)')
@click.option('--seed', '-s', type=int, default=None,
              help='Random seed for reproducibility')
@click.option('--cautious/--no-cautious', default=True,
              help='Skip existing outputs (default: True)')
@click.option('--extra', '-e', type=str, multiple=True,
              help='Extra Hydra overrides (can be specified multiple times)')
def rf2(
    input_pdb: Optional[Path],
    input_dir: Optional[Path],
    input_quiver: Optional[Path],
    output_dir: Optional[Path],
    output_quiver: Optional[Path],
    num_recycles: int,
    weights: Optional[Path],
    seed: Optional[int],
    cautious: bool,
    extra: tuple
):
    """Run RF2 antibody structure prediction.

    Predicts/refines antibody structures using RoseTTAFold2.
    Input can be a single PDB, a directory of PDBs, or a Quiver file.

    \b
    Examples:
        # Predict from single PDB
        rf2 -p antibody.pdb -o predictions/

        # Predict from directory
        rf2 -i structures/ -o predictions/ -r 5

        # Predict from Quiver to Quiver
        rf2 -q designs.qv --output-quiver predictions.qv
    """
    # Validate input
    inputs_provided = sum([input_pdb is not None, input_dir is not None, input_quiver is not None])
    if inputs_provided == 0:
        click.echo('Error: Must specify one of --input-pdb, --input-dir, or --input-quiver', err=True)
        sys.exit(1)
    if inputs_provided > 1:
        click.echo('Error: Can only specify one input type', err=True)
        sys.exit(1)

    # Validate output
    if output_dir is None and output_quiver is None:
        click.echo('Error: Must specify --output-dir or --output-quiver', err=True)
        sys.exit(1)

    # Find the inference script
    script_path = PathConfig.SCRIPTS_DIR / 'rf2_predict.py'
    if not script_path.exists():
        click.echo(f'Error: RF2 script not found at {script_path}', err=True)
        sys.exit(1)

    # Build command
    cmd = ['python', str(script_path)]

    # Input
    if input_pdb:
        cmd.append(f'input.pdb={input_pdb}')
    elif input_dir:
        cmd.append(f'input.pdb_dir={input_dir}')
    elif input_quiver:
        cmd.append(f'input.quiver={input_quiver}')

    # Output
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cmd.append(f'output.pdb_dir={output_dir}')
    if output_quiver:
        cmd.append(f'output.quiver={output_quiver}')

    # Inference parameters
    cmd.append(f'inference.num_recycles={num_recycles}')
    cmd.append(f'inference.cautious={cautious}')

    # Model weights
    if weights:
        cmd.append(f'model.model_weights={weights}')

    # Seed for reproducibility
    if seed is not None:
        cmd.append(f'+inference.seed={seed}')

    # Extra overrides
    for override in extra:
        cmd.append(override)

    input_source = input_pdb or input_dir or input_quiver
    click.echo(f'Running RF2 structure prediction...')
    click.echo(f'Input: {input_source}')
    click.echo(f'Recycles: {num_recycles}')

    result = subprocess.run(cmd, cwd=str(PathConfig.PROJECT_ROOT))
    sys.exit(result.returncode)
