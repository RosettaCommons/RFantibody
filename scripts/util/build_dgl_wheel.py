"""Rebuild the DGL 2.4.0 (torch 2.3 / CUDA 11.8 / CPython 3.10) wheel used by RFantibody.

RFantibody originally installed DGL from
https://data.dgl.ai/wheels/torch-2.3/cu118/dgl-2.4.0%2Bcu118-cp310-cp310-manylinux1_x86_64.whl
but the dgl.ai domain has expired (see GitHub issue #108). The same build is
still published by the DGL team as a conda package on anaconda.org
(channel ``dglteam``). This script downloads that package, verifies its
published sha256, and repackages its ``site-packages`` contents unchanged as a
standard wheel. The resulting file is what RFantibody hosts as a GitHub release
asset and pins in ``pyproject.toml``/``uv.lock``.

The output is byte-for-byte reproducible, so anyone can rerun this script and
check that the hosted wheel's sha256 matches.

Usage (no Python environment needed beyond uv):

    uv run --no-project --python 3.10 python scripts/util/build_dgl_wheel.py [OUT_DIR]
"""

import argparse
import base64
import email.parser
import hashlib
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

CONDA_URL = (
    "https://anaconda.org/dglteam/dgl/2.4.0.th23.cu118/download/"
    "linux-64/dgl-2.4.0.th23.cu118-py310_0.tar.bz2"
)
# sha256 published by anaconda.org for the file above
CONDA_SHA256 = "ae5c7fd93395bac12b3c321fc72ea0bf2ad2382ee00512e6c587f0c1b1d5a058"

VERSION = "2.4.0+cu118"
WHEEL_NAME = f"dgl-{VERSION}-cp310-cp310-manylinux1_x86_64.whl"
SITE_PACKAGES = "lib/python3.10/site-packages/"
EGG_INFO = f"{SITE_PACKAGES}dgl-{VERSION}-py3.10.egg-info/"
LICENSE_MEMBER = "info/licenses/LICENSE0.txt"
DIST_INFO = f"dgl-{VERSION}.dist-info/"
# Fixed timestamp (DGL 2.4.0 release month) so the zip output is reproducible.
ZIP_DATE = (2024, 8, 30, 0, 0, 0)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_conda_package(dest):
    print(f"Downloading {CONDA_URL}", file=sys.stderr)
    # anaconda.org rejects urllib's default User-Agent with 403
    req = urllib.request.Request(CONDA_URL, headers={"User-Agent": "rfantibody-build-dgl-wheel"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            out.write(chunk)


def build_metadata(pkg_info, requires_txt):
    msg = email.parser.Parser().parsestr(pkg_info)
    lines = ["Metadata-Version: 2.1", "Name: dgl", f"Version: {VERSION}"]
    for key in ("Summary", "Home-page", "License"):
        if msg.get(key) and msg[key] != "UNKNOWN":
            lines.append(f"{key}: {msg[key]}")
    lines.append("License-File: LICENSE")
    for req in requires_txt.splitlines():
        req = req.strip()
        if req.startswith("["):  # extras sections are not used by RFantibody
            break
        if req:
            lines.append(f"Requires-Dist: {req}")
    return "\n".join(lines) + "\n"


def convert(conda_pkg, out_dir):
    files = {}
    pkg_info = requires_txt = license_text = None
    with tarfile.open(conda_pkg) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if member.name == LICENSE_MEMBER:
                license_text = tar.extractfile(member).read()
                continue
            if not member.name.startswith(SITE_PACKAGES):
                continue
            data = tar.extractfile(member).read()
            if member.name.startswith(EGG_INFO):
                if member.name.endswith("/PKG-INFO"):
                    pkg_info = data.decode()
                elif member.name.endswith("/requires.txt"):
                    requires_txt = data.decode()
                continue
            rel = member.name[len(SITE_PACKAGES):]
            if "__pycache__" in rel or rel.endswith(".pyc"):
                continue
            files[rel] = (data, member.mode)

    if pkg_info is None or requires_txt is None or license_text is None:
        sys.exit("Unexpected conda package layout: metadata or license not found")

    files[DIST_INFO + "METADATA"] = (build_metadata(pkg_info, requires_txt).encode(), 0o644)
    files[DIST_INFO + "WHEEL"] = (
        b"Wheel-Version: 1.0\n"
        b"Generator: rfantibody-build_dgl_wheel\n"
        b"Root-Is-Purelib: false\n"
        b"Tag: cp310-cp310-manylinux1_x86_64\n",
        0o644,
    )
    files[DIST_INFO + "top_level.txt"] = (b"dgl\n", 0o644)
    files[DIST_INFO + "LICENSE"] = (license_text, 0o644)

    wheel_path = Path(out_dir) / WHEEL_NAME
    record = []
    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(files):
            data, mode = files[rel]
            info = zipfile.ZipInfo(rel, date_time=ZIP_DATE)
            info.external_attr = (0o100000 | (mode & 0o777)) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
            record.append(f"{rel},sha256={digest.decode()},{len(data)}")
        record.append(f"{DIST_INFO}RECORD,,")
        info = zipfile.ZipInfo(DIST_INFO + "RECORD", date_time=ZIP_DATE)
        info.external_attr = 0o100644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, "\n".join(record) + "\n")
    return wheel_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", nargs="?", default=".", help="where to write the wheel")
    parser.add_argument("--conda-package", help="use an already-downloaded conda .tar.bz2")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        conda_pkg = args.conda_package or str(Path(tmp) / "dgl.tar.bz2")
        if not args.conda_package:
            fetch_conda_package(conda_pkg)
        actual = sha256_file(conda_pkg)
        if actual != CONDA_SHA256:
            sys.exit(f"sha256 mismatch for conda package: {actual} != {CONDA_SHA256}")
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        wheel = convert(conda_pkg, args.out_dir)

    print(f"{wheel}\nsha256: {sha256_file(wheel)}")


if __name__ == "__main__":
    main()
