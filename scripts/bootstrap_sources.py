"""Fetch pinned upstream sources without overwriting an existing checkout."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run(args,cwd=ROOT,check=True)


if __name__ == "__main__":
    pins=json.loads((ROOT/"third_party/versions.json").read_text())
    for name,source in pins.items():
        folder=ROOT/"third_party"/name
        if folder.exists():
            actual=subprocess.check_output(["git","-c",f"safe.directory={folder.as_posix()}","-C",str(folder),"rev-parse","HEAD"],text=True).strip()
            if actual!=source["commit"]:
                raise SystemExit(f"{folder} is {actual}; expected {source['commit']}. Existing checkout left intact.")
        else:
            run("git","clone","--filter=blob:none","--no-checkout",source["url"],str(folder))
            run("git","-C",str(folder),"checkout",source["commit"])
        if name=="splatad":
            run("git","-C",str(folder),"submodule","update","--init","--recursive")
        print(f"{name}: {source['commit']}")
    patcher = ROOT / "scripts/apply_upstream_patches.py"
    if patcher.is_file():
        run(sys.executable, str(patcher))
