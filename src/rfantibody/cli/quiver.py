#!/usr/bin/env python3
"""
Quiver file CLI utilities.

Commands for working with Quiver files (protein design databases).
"""

import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from rfantibody.util.quiver import Quiver


@click.command()
@click.argument('quiver_file', type=click.Path(exists=True, path_type=Path))
def qvls(quiver_file: Path):
    """List all tags in a Quiver file."""
    qv = Quiver(str(quiver_file), 'r')
    for tag in qv.get_tags():
        click.echo(tag)


@click.command()
@click.argument('quiver_file', type=click.Path(exists=True, path_type=Path))
@click.option('-o', '--output-dir', type=click.Path(path_type=Path),
              default=None, help='Output directory (default: current directory)')
@click.option('--prefix', default='', help='Prefix for output PDB files')
@click.option('--force', is_flag=True, help='Overwrite existing files')
def qvextract(quiver_file: Path, output_dir: Optional[Path], prefix: str, force: bool):
    """Extract all PDB files from a Quiver file."""
    if output_dir is None:
        output_dir = Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)

    qv = Quiver(str(quiver_file), 'r')
    extracted = 0
    skipped = 0

    for tag in qv.get_tags():
        outfn = output_dir / f'{prefix}{tag}.pdb'

        if outfn.exists() and not force:
            click.echo(f'File {outfn} already exists, skipping', err=True)
            skipped += 1
            continue

        lines = qv.get_pdblines(tag)
        with open(outfn, 'w') as f:
            f.writelines(lines)
        extracted += 1

    click.echo(f'Successfully extracted {extracted} PDB files from {quiver_file}')
    if skipped > 0:
        click.echo(f'Skipped {skipped} existing files (use --force to overwrite)', err=True)


@click.command()
@click.argument('quiver_file', type=click.Path(exists=True, path_type=Path))
@click.argument('tags', nargs=-1)
@click.option('-o', '--output-dir', type=click.Path(path_type=Path),
              default=None, help='Output directory (default: current directory)')
@click.option('--force', is_flag=True, help='Overwrite existing files')
def qvextractspecific(quiver_file: Path, tags: tuple, output_dir: Optional[Path], force: bool):
    """Extract specific PDB files from a Quiver file by tag name.

    Tags can be provided as arguments or piped via stdin.

    \b
    Examples:
        qvextractspecific my.qv tag1 tag2 tag3
        qvls my.qv | head -n 10 | qvextractspecific my.qv
        qvls my.qv | grep "good" | qvextractspecific my.qv -o selected/
    """
    # Read tags from stdin if none provided as arguments
    if not tags:
        if sys.stdin.isatty():
            click.echo('Error: No tags provided. Provide tags as arguments or pipe them via stdin.', err=True)
            sys.exit(1)
        tags = tuple(line.strip() for line in sys.stdin if line.strip())

    if not tags:
        click.echo('Error: No tags provided', err=True)
        sys.exit(1)

    if output_dir is None:
        output_dir = Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)

    qv = Quiver(str(quiver_file), 'r')
    available_tags = qv.get_tags()

    extracted = 0
    for tag in tags:
        if tag not in available_tags:
            click.echo(f'Warning: Tag {tag} not found in Quiver file', err=True)
            continue

        outfn = output_dir / f'{tag}.pdb'

        if outfn.exists() and not force:
            click.echo(f'File {outfn} already exists, skipping', err=True)
            continue

        lines = qv.get_pdblines(tag)
        with open(outfn, 'w') as f:
            f.writelines(lines)
        extracted += 1

    click.echo(f'Successfully extracted {extracted} PDB files', err=True)


@click.command()
@click.argument('quiver_file', type=click.Path(exists=True, path_type=Path))
def qvscorefile(quiver_file: Path):
    """Extract scores from a Quiver file to a tab-separated scorefile.

    Output is written to stdout. Use shell redirection to save to a file.

    \b
    Examples:
        qvscorefile designs.qv > scores.tsv
        qvscorefile designs.qv | head -20
    """
    records = []

    with open(quiver_file, 'r') as qvfile:
        for line in qvfile:
            if line.startswith('QV_SCORE'):
                splits = line.split()
                tag = splits[1]

                scores = {
                    entry[0]: float(entry[1])
                    for entry in [score.split('=') for score in splits[2].split('|')]
                }
                scores['tag'] = tag

                for key, value in scores.items():
                    if value == '':
                        scores[key] = None

                records.append(scores)

    if not records:
        click.echo('Error: No scorelines found in Quiver file', err=True)
        sys.exit(1)

    df = pd.DataFrame.from_records(records)
    df.to_csv(sys.stdout, sep='\t', na_rep='NaN', index=False)


@click.command()
@click.argument('quiver_file', type=click.Path(exists=True, path_type=Path))
@click.argument('ntags', type=int)
@click.option('-o', '--output-dir', type=click.Path(path_type=Path),
              default=None, help='Output directory (default: current directory)')
@click.option('--prefix', default='split', help='Prefix for output files')
def qvsplit(quiver_file: Path, ntags: int, output_dir: Optional[Path], prefix: str):
    """Split a Quiver file into multiple files with N tags per file."""
    if ntags <= 0:
        click.echo('Error: ntags must be positive', err=True)
        sys.exit(1)

    if output_dir is None:
        output_dir = Path.cwd()

    qv = Quiver(str(quiver_file), 'r')
    qv.split(ntags, str(output_dir), prefix)

    total_tags = qv.size()
    num_files = (total_tags + ntags - 1) // ntags
    click.echo(f'Split {total_tags} structures into {num_files} files in {output_dir}')


@click.command()
@click.argument('quiver_file', type=click.Path(exists=True, path_type=Path))
@click.argument('tags', nargs=-1)
def qvslice(quiver_file: Path, tags: tuple):
    """Extract specific tags from a Quiver file into a new Quiver file.

    Tags can be provided as arguments or piped via stdin.
    Output is written to stdout.

    \b
    Examples:
        qvslice my.qv tag1 tag2 tag3 > subset.qv
        qvls my.qv | head -n 10 | qvslice my.qv > first10.qv
        qvls my.qv | grep "good" | qvslice my.qv > good.qv
    """
    # Read tags from stdin if none provided as arguments
    if not tags:
        if sys.stdin.isatty():
            click.echo('Error: No tags provided. Provide tags as arguments or pipe them via stdin.', err=True)
            sys.exit(1)
        tags = tuple(line.strip() for line in sys.stdin if line.strip())

    if not tags:
        click.echo('Error: No tags provided', err=True)
        sys.exit(1)

    qv = Quiver(str(quiver_file), 'r')
    tag_list = list(tags)

    qv_string, found_tags = qv.get_struct_list(tag_list)

    if not found_tags:
        click.echo('Error: None of the specified tags were found', err=True)
        sys.exit(1)

    click.echo(qv_string, nl=False)

    missing = set(tag_list) - set(found_tags)
    if missing:
        click.echo(f'Warning: {len(missing)} tags not found: {", ".join(sorted(missing))}', err=True)


@click.command()
@click.argument('quiver_file', type=click.Path(exists=True, path_type=Path))
@click.argument('new_tags', nargs=-1)
def qvrename(quiver_file: Path, new_tags: tuple):
    """Rename tags in a Quiver file.

    New tag names can be provided as arguments or piped via stdin.
    The number of new tags must match the number of tags in the file.
    Output is written to stdout.

    \b
    Examples:
        qvrename old.qv newtag1 newtag2 > new.qv
        qvls old.qv | sed 's/$/_v2/' | qvrename old.qv > new.qv
        cat newtags.txt | qvrename old.qv > new.qv
    """
    # Read tags from stdin if none provided as arguments
    if not new_tags:
        if sys.stdin.isatty():
            click.echo('Error: No new tag names provided. Provide as arguments or pipe via stdin.', err=True)
            sys.exit(1)
        new_tags = tuple(line.strip() for line in sys.stdin if line.strip())

    if not new_tags:
        click.echo('Error: No new tag names provided', err=True)
        sys.exit(1)

    tag_list = list(new_tags)

    qv = Quiver(str(quiver_file), 'r')
    present_tags = qv.get_tags()

    if len(present_tags) != len(tag_list):
        click.echo(
            f'Error: Number of tags in file ({len(present_tags)}) does not match '
            f'number of new tags provided ({len(tag_list)})',
            err=True
        )
        sys.exit(1)

    tag_idx = 0
    output_lines = []

    with open(quiver_file, 'r') as f:
        for line in f:
            if line.startswith('QV_TAG'):
                line = f'QV_TAG {tag_list[tag_idx]}\n'

                next_line = f.readline()

                if next_line.startswith('QV_TAG'):
                    click.echo('Error: Found two QV_TAG lines in a row', err=True)
                    sys.exit(1)

                if next_line.startswith('QV_SCORE'):
                    splits = next_line.split(' ')
                    splits[1] = tag_list[tag_idx]
                    next_line = ' '.join(splits)

                line += next_line
                tag_idx += 1

            output_lines.append(line)

    result = ''.join(output_lines)
    click.echo(result, nl=False)


@click.command()
@click.argument('pdb_files', nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
def qvfrompdbs(pdb_files: tuple):
    """Create a Quiver file from PDB files.

    Output is written to stdout. Use shell redirection to save to a file.

    \b
    Examples:
        qvfrompdbs *.pdb > designs.qv
        qvfrompdbs design1.pdb design2.pdb > designs.qv
    """
    for pdb_file in pdb_files:
        with open(pdb_file, 'r') as f:
            pdb_lines = f.read()

        tag = pdb_file.stem
        click.echo(f'QV_TAG {tag}')
        click.echo(pdb_lines, nl=False)
        if not pdb_lines.endswith('\n'):
            click.echo()
