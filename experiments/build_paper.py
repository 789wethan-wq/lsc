"""Build the SSRN working-paper PDF from the patched draft (M5).

Converts PAPER_DRAFT.md into paper_assets/lsc_wp.pdf via pandoc + tectonic:
the leading H1 becomes title-page metadata (title / author / date), the
in-body `## Abstract` and JEL/keywords render beneath it, sections are
numbered, and derivation pointers live in the appendices. Deterministic:
the title-page date is fixed (passed in or defaulted) so `make paper`
reproduces byte-stably given the same draft.

Usage: python experiments/build_paper.py [--date "July 2026"]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFT = ROOT / "PAPER_DRAFT.md"
OUT = ROOT / "paper_assets" / "lsc_wp.pdf"
AUTHOR = "Ethan Wuang, E.W. Research  ·  789wethan@gmail.com"


# The main text font (STIX Two Text) covers the Greek letters,
# super/subscripts, and fractions the draft uses, but delegates a handful
# of *math* operators/relations to a math font that text mode does not
# reach. Rewrite just those to inline math (pandoc runs with
# tex_math_dollars); the draft contains no $-math, so this cannot collide.
# "√F" is handled before the bare radical so the argument is set under it.
MATH_SUBS = [
    ("√F", r"$\sqrt{F}$"),
    ("√(m²−4)", r"$\sqrt{m^2-4}$"),
    ("√", r"$\surd$"),
    ("∞", r"$\infty$"),
    ("∈", r"$\in$"),
    ("→", r"$\rightarrow$"),
    ("≤", r"$\le$"),
    ("≥", r"$\ge$"),
    ("≈", r"$\approx$"),
    ("≡", r"$\equiv$"),
    ("∎", r"$\blacksquare$"),
]


def apply_math_subs(text: str) -> str:
    for a, b in MATH_SUBS:
        # pandoc's tex_math_dollars refuses a closing $ immediately
        # followed by a digit ("$\ge$200" renders literally), so pull a
        # directly-adjacent number run inside the math span first.
        if b.startswith("$") and b.endswith("$"):
            text = re.sub(re.escape(a) + r"(\d[\d.]*)",
                          lambda m: f"${b[1:-1]} {m.group(1)}$", text)
        text = text.replace(a, b)
    return text


def yaml_escape(s: str) -> str:
    return s.replace('"', '\\"')


def build(date: str) -> None:
    lines = DRAFT.read_text().splitlines()
    if not lines or not lines[0].startswith("# "):
        raise SystemExit("draft must begin with an H1 title line")
    title = lines[0][2:].strip()
    body = apply_math_subs("\n".join(lines[1:]).lstrip("\n"))

    meta = (
        "---\n"
        f'title: "{yaml_escape(title)}"\n'
        f'author: "{yaml_escape(AUTHOR)}"\n'
        f'date: "{yaml_escape(date)}"\n'
        "geometry: margin=1in\n"
        "fontsize: 11pt\n"
        # STIX Two Text covers the Greek, sub/superscript, fraction, and
        # math symbols the draft uses as literal Unicode (the draft has no
        # $-math); without it the default Latin Modern text font silently
        # drops them. Ships with macOS 12+.
        'mainfont: "STIX Two Text"\n'
        "linkcolor: blue\n"
        "urlcolor: blue\n"
        "---\n\n"
    )
    # LaTeX preamble via --include-in-header: always inserted raw, so it
    # works with the gfm reader (which does not support the raw_tex
    # extension). Shrinks wide reference tables and eases line breaking.
    preamble = ("\\usepackage{etoolbox}\n"
                "\\AtBeginEnvironment{longtable}{\\tiny}\n"
                "\\emergencystretch=3em\n"
                # figure captions carry their own hand-set "Figure N."
                # labels (auto-numbering is off to match the draft's
                # hand-numbered sections), so drop LaTeX's "Figure N:"
                "\\usepackage{caption}\n"
                "\\captionsetup[figure]{labelformat=empty}\n")

    tmp = Path(tempfile.mkstemp(suffix=".md", dir=str(ROOT))[1])
    hdr = Path(tempfile.mkstemp(suffix=".tex", dir=str(ROOT))[1])
    tmp.write_text(meta + body)
    hdr.write_text(preamble)

    cmd = [
        "pandoc", str(tmp),
        "--pdf-engine=tectonic",
        # the draft carries its own section numbers ("1.", "2.", ...) and
        # appendix letters, so auto-numbering is off; shift the body's H2
        # sections up to proper \section level (the H1 title lives in
        # metadata, not the body).
        "--shift-heading-level-by=-1",
        "--include-in-header", str(hdr),
        # implicit_figures renders the draft's standalone images as
        # captioned figures (alt text -> caption)
        "--from", "gfm+tex_math_dollars+yaml_metadata_block+footnotes"
                  "+implicit_figures",
        "-V", "colorlinks=true",
        "-o", str(OUT),
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        tmp.unlink(missing_ok=True)
        hdr.unlink(missing_ok=True)
    print(f"wrote {OUT.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="July 2026")
    a = ap.parse_args()
    if not DRAFT.exists():
        sys.exit(f"missing {DRAFT}")
    build(a.date)


if __name__ == "__main__":
    main()
