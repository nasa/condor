"""Execute all example scripts in docs/*_src"""

import subprocess
import sys
from pathlib import Path


def printfl(*args, **kwargs):
    print(*args, **(kwargs | {"flush": True}))


docdir = Path(__file__).parent
fails = []
for srcdir in sorted(docdir.glob("*_src")):
    printfl(80 * "=")
    printfl(f"Running scripts in {srcdir.stem}")
    printfl(80 * "=")

    for srcfile in sorted(srcdir.glob("*.py")):
        if srcfile.name.startswith("_"):
            continue

        name = f"{srcdir.name}/{srcfile.name}"
        printfl("\n" + 80 * "-")
        printfl(f"Running {name}")
        printfl(80 * "-" + "\n")

        proc = subprocess.run([sys.executable, srcfile], check=False)  # noqa: S603
        if proc.returncode:
            fails.append(name)

if fails:
    printfl("\n\n")
    printfl(80 * "=")
    printfl("Failed cases:")
    printfl(*fails, sep="\n")
    printfl(80 * "=")

    sys.exit(1)
