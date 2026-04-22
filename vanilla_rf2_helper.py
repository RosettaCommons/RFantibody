#!/usr/bin/env python3
"""Helper for the RFantibody/vanilla-RF2 Makefile.

Subcommands
-----------
  msa NAME SEQ OUT_DIR
      Fetch real MSA from ColabFold's public MMseqs2 API. Writes
      <NAME>_uniref.a3m and <NAME>_envbfd.a3m into OUT_DIR.

  verify --rf2-dir DIR
      Build a vanilla RoseTTAFoldModule and load RF2_jan24.pt with
      strict=True to confirm the install is coherent.

  rmsd PRED_PDB REF_PDB [--chain C]
      Superpose PRED onto REF by Cα and print Å RMSD.
"""
import argparse
import io
import pathlib
import sys
import tarfile
import time


def cmd_msa(args):
    import requests

    api = "https://api.colabfold.com"
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[submit] {len(args.sequence)} aa -> {api}/ticket/msa (mode=env)",
          flush=True)
    r = requests.post(
        f"{api}/ticket/msa",
        data={"q": f">{args.name}\n{args.sequence}", "mode": "env"},
    )
    r.raise_for_status()
    tid = r.json()["id"]

    status = None
    for _ in range(240):
        time.sleep(5)
        status = requests.get(f"{api}/ticket/{tid}").json()
        if status.get("status") in ("COMPLETE", "ERROR"):
            break
    else:
        sys.exit("ColabFold MSA timed out")
    if status["status"] == "ERROR":
        sys.exit(f"ColabFold MSA errored: {status}")

    print("[download] fetching tarball", flush=True)
    tb = requests.get(f"{api}/result/download/{tid}").content
    with tarfile.open(fileobj=io.BytesIO(tb)) as t:
        for member in t.getmembers():
            if not member.name.endswith(".a3m"):
                continue
            stem = pathlib.Path(member.name).stem
            out_name = {
                "uniref": f"{args.name}_uniref.a3m",
                "bfd.mgnify30.metaeuk30.smag30": f"{args.name}_envbfd.a3m",
            }.get(stem, f"{args.name}_{stem}.a3m")
            path = out_dir / out_name
            with t.extractfile(member) as src:
                path.write_bytes(src.read())
            n = sum(1 for line in path.open() if line.startswith(">"))
            print(f"[a3m] {path.name}: {n} sequences")


def cmd_verify(args):
    import torch

    network_dir = pathlib.Path(args.rf2_dir) / "network"
    weights = network_dir / "weights" / "RF2_jan24.pt"
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}")

    sys.path.insert(0, str(network_dir))
    from RoseTTAFoldModel import RoseTTAFoldModule  # noqa: E402

    model_param = dict(
        n_extra_block=4, n_main_block=36, n_ref_block=4,
        d_msa=256, d_pair=128, d_templ=64,
        n_head_msa=8, n_head_pair=4, n_head_templ=4,
        d_hidden=32, d_hidden_templ=32, p_drop=0.0,
        SE3_param_full=dict(num_layers=1, num_channels=48, num_degrees=2,
                            l0_in_features=32, l0_out_features=32,
                            l1_in_features=2, l1_out_features=2,
                            num_edge_features=32, div=4, n_heads=4),
        SE3_param_topk=dict(num_layers=1, num_channels=128, num_degrees=2,
                            l0_in_features=64, l0_out_features=64,
                            l1_in_features=2, l1_out_features=2,
                            num_edge_features=64, div=4, n_heads=4),
    )
    model = RoseTTAFoldModule(**model_param)
    ck = torch.load(str(weights), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    print("OK: vanilla RF2 RoseTTAFoldModule + RF2_jan24.pt load cleanly "
          "(strict=True)")


def cmd_rmsd(args):
    import biotite.structure.io.pdb as pdbio
    from biotite.structure import filter_amino_acids, rmsd, superimpose

    def ca(path, chain=None):
        atoms = pdbio.PDBFile.read(path).get_structure(model=1)
        atoms = atoms[filter_amino_acids(atoms)]
        if chain is not None:
            atoms = atoms[atoms.chain_id == chain]
        return atoms[atoms.atom_name == "CA"]

    pred = ca(args.pred_pdb)
    ref = ca(args.ref_pdb, chain=args.chain)
    n = min(len(pred), len(ref))
    if n == 0:
        sys.exit("No Ca atoms found in one of the inputs")
    pred, ref = pred[:n], ref[:n]
    fitted, _ = superimpose(ref, pred)
    print(f"Ca RMSD ({n} residues): {rmsd(ref, fitted):.3f} A")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("msa", help="fetch MSA from ColabFold API")
    pm.add_argument("name")
    pm.add_argument("sequence")
    pm.add_argument("out_dir")
    pm.set_defaults(func=cmd_msa)

    pv = sub.add_parser("verify", help="strict=True load test")
    pv.add_argument("--rf2-dir", required=True)
    pv.set_defaults(func=cmd_verify)

    pr = sub.add_parser("rmsd", help="Ca RMSD between two PDBs")
    pr.add_argument("pred_pdb")
    pr.add_argument("ref_pdb")
    pr.add_argument("--chain", default=None)
    pr.set_defaults(func=cmd_rmsd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
