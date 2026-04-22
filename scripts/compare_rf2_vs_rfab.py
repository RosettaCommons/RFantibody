"""Compare RFantibody RF2 vs vanilla RF2 predictions of antibody-antigen complexes.

Inputs: one or more test cases, each with
  --rfa   PDB from RFantibody's rf2 CLI
  --van   PDB from vanilla RF2 (uw-ipd/RoseTTAFold2 predict.py)
  --ref   optional reference PDB (crystal structure or design input)
  --name  case label

For each pair it reports:
  - global Cα RMSD after Kabsch superposition
  - per-chain Cα RMSD (on the antibody chains and target chain separately)
  - interface Cα RMSD (residues within 8 Å of a different chain)
  - pLDDT stats (mean, median, min) from the B-factor column of each prediction
Prints a markdown table and writes a JSON dump of all metrics.

Usage:
    python scripts/compare_rf2_vs_rfab.py \\
        --case NAME1 RFA1.pdb VAN1.pdb [REF1.pdb] \\
        --case NAME2 RFA2.pdb VAN2.pdb [REF2.pdb] \\
        --out results.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import biotite.structure.io.pdb as pdbio
import numpy as np
from biotite.structure import (
    AtomArray,
    filter_amino_acids,
    rmsd,
    superimpose,
)


@dataclass
class ChainMetrics:
    n_res: int
    rmsd_vs_ref: Optional[float]
    plddt_mean: Optional[float]
    plddt_median: Optional[float]
    plddt_min: Optional[float]


@dataclass
class CaseMetrics:
    name: str
    rfa_path: str
    van_path: str
    ref_path: Optional[str]
    rfa_len: int
    van_len: int
    ref_len: Optional[int]
    # Pred-vs-pred
    rmsd_rfa_vs_van_global: float
    rmsd_rfa_vs_van_ab: Optional[float]
    rmsd_rfa_vs_van_target: Optional[float]
    rmsd_rfa_vs_van_interface: Optional[float]
    # Pred-vs-ref
    rmsd_rfa_vs_ref_global: Optional[float]
    rmsd_van_vs_ref_global: Optional[float]
    rmsd_rfa_vs_ref_ab: Optional[float]
    rmsd_van_vs_ref_ab: Optional[float]
    rmsd_rfa_vs_ref_target: Optional[float]
    rmsd_van_vs_ref_target: Optional[float]
    # pLDDT
    rfa_plddt_mean: float
    van_plddt_mean: float
    rfa_plddt_ab_mean: Optional[float]
    van_plddt_ab_mean: Optional[float]
    rfa_plddt_target_mean: Optional[float]
    van_plddt_target_mean: Optional[float]


def load_ca(path: Path) -> AtomArray:
    atoms = pdbio.PDBFile.read(str(path)).get_structure(
        model=1, extra_fields=["b_factor", "occupancy"]
    )
    atoms = atoms[filter_amino_acids(atoms)]
    return atoms[atoms.atom_name == "CA"]


def find_chain_split(ca: AtomArray, target_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (antibody, target) for the CA array.

    Strategy: if chain IDs include H, L, T, use them. Otherwise split by order,
    assuming chains are concatenated in (H, L, T) order with target at the end
    and the target length given.
    """
    chain_ids = set(ca.chain_id)
    if {"H", "L", "T"}.issubset(chain_ids):
        ab = (ca.chain_id == "H") | (ca.chain_id == "L")
        tg = ca.chain_id == "T"
        return ab, tg
    # Fallback: assume H-L-T order by position
    n = len(ca)
    tg = np.zeros(n, dtype=bool)
    tg[-target_len:] = True
    ab = ~tg
    return ab, tg


def superimpose_rmsd(ref_xyz: np.ndarray, mob_xyz: np.ndarray) -> float:
    """Kabsch superposition of mob onto ref by Cα, return RMSD in Å.

    Operates on raw coordinate arrays; lengths must match.
    """
    assert ref_xyz.shape == mob_xyz.shape, (ref_xyz.shape, mob_xyz.shape)
    # biotite.structure.superimpose expects AtomArray; build minimal fake arrays
    # by fitting manually with a small Kabsch implementation.
    ref_c = ref_xyz - ref_xyz.mean(0)
    mob_c = mob_xyz - mob_xyz.mean(0)
    H = mob_c.T @ ref_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    fitted = mob_c @ R.T
    return float(np.sqrt(np.mean(np.sum((fitted - ref_c) ** 2, axis=1))))


def paired_rmsd(a: AtomArray, b: AtomArray, mask_a: Optional[np.ndarray] = None,
                mask_b: Optional[np.ndarray] = None) -> Optional[float]:
    """Align the two CA arrays by truncating to the shorter length, then Kabsch RMSD."""
    xyz_a = a.coord if mask_a is None else a.coord[mask_a]
    xyz_b = b.coord if mask_b is None else b.coord[mask_b]
    n = min(len(xyz_a), len(xyz_b))
    if n < 3:
        return None
    return superimpose_rmsd(xyz_a[:n], xyz_b[:n])


def interface_mask(ca: AtomArray, ab_mask: np.ndarray, tg_mask: np.ndarray,
                   cutoff: float = 8.0) -> np.ndarray:
    """Boolean mask: Cα residues within `cutoff` Å of a residue on the other side."""
    ab_xyz = ca.coord[ab_mask]
    tg_xyz = ca.coord[tg_mask]
    if len(ab_xyz) == 0 or len(tg_xyz) == 0:
        return np.zeros(len(ca), dtype=bool)
    # all-pairs distance, [n_ab, n_tg]
    d = np.linalg.norm(ab_xyz[:, None, :] - tg_xyz[None, :, :], axis=-1)
    ab_at_iface = d.min(axis=1) < cutoff
    tg_at_iface = d.min(axis=0) < cutoff
    out = np.zeros(len(ca), dtype=bool)
    ab_idx = np.nonzero(ab_mask)[0]
    tg_idx = np.nonzero(tg_mask)[0]
    out[ab_idx[ab_at_iface]] = True
    out[tg_idx[tg_at_iface]] = True
    return out


def plddt_stats(ca: AtomArray) -> tuple[float, float, float]:
    """Return (mean, median, min) of the B-factor column (treated as pLDDT)."""
    if not hasattr(ca, "b_factor"):
        return float("nan"), float("nan"), float("nan")
    b = ca.b_factor.astype(float)
    return float(np.mean(b)), float(np.median(b)), float(np.min(b))


def compute_case(name: str, rfa: Path, van: Path, ref: Optional[Path]) -> CaseMetrics:
    rfa_ca = load_ca(rfa)
    van_ca = load_ca(van)
    ref_ca = load_ca(ref) if ref is not None else None

    rfa_ab, rfa_tg = find_chain_split(rfa_ca, target_len=0)
    van_ab, van_tg = find_chain_split(van_ca, target_len=rfa_tg.sum())

    # Global pred-vs-pred
    rmsd_global = paired_rmsd(rfa_ca, van_ca)
    rmsd_ab = paired_rmsd(rfa_ca, van_ca, rfa_ab, van_ab)
    rmsd_target = paired_rmsd(rfa_ca, van_ca, rfa_tg, van_tg)

    # Interface residues (defined from rfa prediction)
    rfa_iface_mask = interface_mask(rfa_ca, rfa_ab, rfa_tg)
    # For cross-model interface RMSD we need same indices on both; align by matching
    # the first N_iface residues of the ab/target union on both sides.
    rmsd_iface = paired_rmsd(rfa_ca, van_ca, rfa_iface_mask,
                             interface_mask(van_ca, van_ab, van_tg))

    # pLDDT
    rfa_pmean, _, _ = plddt_stats(rfa_ca)
    van_pmean, _, _ = plddt_stats(van_ca)
    rfa_pmean_ab = float(np.mean(rfa_ca.b_factor[rfa_ab])) if rfa_ab.any() else None
    van_pmean_ab = float(np.mean(van_ca.b_factor[van_ab])) if van_ab.any() else None
    rfa_pmean_t = float(np.mean(rfa_ca.b_factor[rfa_tg])) if rfa_tg.any() else None
    van_pmean_t = float(np.mean(van_ca.b_factor[van_tg])) if van_tg.any() else None

    # Pred-vs-ref
    rfa_vs_ref = van_vs_ref = None
    rfa_vs_ref_ab = van_vs_ref_ab = None
    rfa_vs_ref_t = van_vs_ref_t = None
    if ref_ca is not None:
        ref_ab, ref_tg = find_chain_split(ref_ca, target_len=rfa_tg.sum())
        rfa_vs_ref = paired_rmsd(rfa_ca, ref_ca)
        van_vs_ref = paired_rmsd(van_ca, ref_ca)
        rfa_vs_ref_ab = paired_rmsd(rfa_ca, ref_ca, rfa_ab, ref_ab)
        van_vs_ref_ab = paired_rmsd(van_ca, ref_ca, van_ab, ref_ab)
        rfa_vs_ref_t = paired_rmsd(rfa_ca, ref_ca, rfa_tg, ref_tg)
        van_vs_ref_t = paired_rmsd(van_ca, ref_ca, van_tg, ref_tg)

    return CaseMetrics(
        name=name,
        rfa_path=str(rfa),
        van_path=str(van),
        ref_path=str(ref) if ref else None,
        rfa_len=len(rfa_ca),
        van_len=len(van_ca),
        ref_len=len(ref_ca) if ref_ca is not None else None,
        rmsd_rfa_vs_van_global=rmsd_global,
        rmsd_rfa_vs_van_ab=rmsd_ab,
        rmsd_rfa_vs_van_target=rmsd_target,
        rmsd_rfa_vs_van_interface=rmsd_iface,
        rmsd_rfa_vs_ref_global=rfa_vs_ref,
        rmsd_van_vs_ref_global=van_vs_ref,
        rmsd_rfa_vs_ref_ab=rfa_vs_ref_ab,
        rmsd_van_vs_ref_ab=van_vs_ref_ab,
        rmsd_rfa_vs_ref_target=rfa_vs_ref_t,
        rmsd_van_vs_ref_target=van_vs_ref_t,
        rfa_plddt_mean=rfa_pmean,
        van_plddt_mean=van_pmean,
        rfa_plddt_ab_mean=rfa_pmean_ab,
        van_plddt_ab_mean=van_pmean_ab,
        rfa_plddt_target_mean=rfa_pmean_t,
        van_plddt_target_mean=van_pmean_t,
    )


def fmt(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def format_table(results: list[CaseMetrics]) -> str:
    lines = [
        "| case | len | RMSD RFa↔Van (global / Ab / Tgt / iface) | RMSD RFa↔Ref (global / Ab / Tgt) | RMSD Van↔Ref (global / Ab / Tgt) | pLDDT RFa (all / Ab / Tgt) | pLDDT Van (all / Ab / Tgt) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ref_rfa = f"{fmt(r.rmsd_rfa_vs_ref_global)} / {fmt(r.rmsd_rfa_vs_ref_ab)} / {fmt(r.rmsd_rfa_vs_ref_target)}"
        ref_van = f"{fmt(r.rmsd_van_vs_ref_global)} / {fmt(r.rmsd_van_vs_ref_ab)} / {fmt(r.rmsd_van_vs_ref_target)}"
        rfa_van = f"{fmt(r.rmsd_rfa_vs_van_global)} / {fmt(r.rmsd_rfa_vs_van_ab)} / {fmt(r.rmsd_rfa_vs_van_target)} / {fmt(r.rmsd_rfa_vs_van_interface)}"
        pl_rfa = f"{fmt(r.rfa_plddt_mean)} / {fmt(r.rfa_plddt_ab_mean)} / {fmt(r.rfa_plddt_target_mean)}"
        pl_van = f"{fmt(r.van_plddt_mean)} / {fmt(r.van_plddt_ab_mean)} / {fmt(r.van_plddt_target_mean)}"
        lines.append(f"| {r.name} | {r.rfa_len}/{r.van_len} | {rfa_van} | {ref_rfa} | {ref_van} | {pl_rfa} | {pl_van} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", nargs="+", action="append", required=True,
                   metavar=("NAME RFA VAN", "REF"),
                   help="Case: NAME RFA.pdb VAN.pdb [REF.pdb]. Repeatable.")
    p.add_argument("--out", type=Path, default=None, help="JSON output path.")
    args = p.parse_args()

    results = []
    for spec in args.case:
        if len(spec) not in (3, 4):
            p.error(f"--case needs 3 or 4 args, got {len(spec)}: {spec!r}")
        name, rfa, van, *ref = spec
        results.append(compute_case(name, Path(rfa), Path(van), Path(ref[0]) if ref else None))

    table = format_table(results)
    print(table)

    if args.out:
        args.out.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nWrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
