"""Handcrafted-TF figure: FFGS + TTT (14v) against ground truth on hand-picked scenes and target views.

Rows are FFGS + TTT and ground truth, columns are samples. Writes, into --out:
  - one PNG per (sample, row), named handcrafted_s<idx>_v<view>_{ffgs_ttt,gt}.png, for upload to Overleaf;
  - handcrafted_comparison.tex, a figure* that includes them (needs only graphicx);
  - handcrafted_comparison.{pdf,png}, a matplotlib preview of the same layout.

A sample is given as its scene index (position in the eval JSON = `images/*_0-0-<idx>.png`) and a target view index
(target_<view> = eval view 50 + view). Run from the repo root:

    python methods/TokenGS/run_scripts/10.28.handcrafted.tfn.premul/10.28.handcrafted.figure.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

T = "methods/TokenGS"
RUN = f"{T}/results/10.28.handcrafted.tfn.premul/handcrafted_premul_14v"
EVAL_JSON = f"{T}/assets/10.28.handcrafted.tfn.premul/handcrafted_premul_eval_14v.json"
# (row label, file tag, file kind)
ROWS = [("FFGS + TTT", "ffgs_ttt", "pred"), ("Ground truth", "gt", "gt")]
# (scene index, target view index)
SAMPLES = [(1, 8), (2, 17), (23, 7), (26, 11), (22, 0), (27, 16)]


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{T}/results/10.28.handcrafted.tfn.premul/figures/handcrafted_14v")
    ap.add_argument("--tex_dir", default="figures/handcrafted", help="image folder as seen from the Overleaf project")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scenes = list(json.load(open(EVAL_JSON)))

    names = [[None] * len(SAMPLES) for _ in ROWS]
    imgs = [[None] * len(SAMPLES) for _ in ROWS]
    legend = []
    for j, (idx, v) in enumerate(SAMPLES):
        scene = scenes[idx]
        for i, (_, tag, kind) in enumerate(ROWS):
            img = load(f"{RUN}/views/{scene}/target_{v:02d}_{kind}.png")
            names[i][j] = f"handcrafted_s{idx:02d}_v{v:02d}_{tag}.png"
            Image.fromarray((img * 255).round().astype(np.uint8)).save(f"{args.out}/{names[i][j]}")
            imgs[i][j] = img
        psnr = -10 * np.log10(max(float(((imgs[0][j] - imgs[1][j]) ** 2).mean()), 1e-10))
        legend.append(f"s{idx:02d} v{v:02d} = {scene}, target view {v} (PSNR {psnr:.2f})")
        print(legend[-1])

    n = len(SAMPLES)
    lines = [
        f"% Requires \\usepackage{{graphicx}}. Upload the PNGs to {args.tex_dir}/ in the Overleaf project.",
        "% Columns, left to right:",
        *[f"%   {s}" for s in legend],
        "\\begin{figure*}[t]",
        "    \\centering",
        "    % \\himg{file}: one image, 0.155\\textwidth wide, vertically centered on the row label",
        "    \\newcommand{\\himg}[1]{\\raisebox{-0.5\\height}{\\includegraphics[width=0.155\\textwidth]{%s/#1}}}"
        % args.tex_dir,
        "    \\setlength{\\tabcolsep}{0.6pt}",
        "    \\begin{tabular}{c%s}" % ("c" * n),
    ]
    for i, (label, *_) in enumerate(ROWS):
        row = " & ".join(f"\\himg{{{f}}}" for f in names[i])
        lines.append(f"        \\rotatebox[origin=c]{{90}}{{{label}}}~ & {row} \\\\")
        if i < len(ROWS) - 1:
            lines.append("        \\noalign{\\vskip 0.6pt}")
    lines += [
        "    \\end{tabular}",
        "    \\caption{FFGS + TTT (14 context views) on handcrafted transfer functions for Nyx and CQ500, compared",
        "    with ground truth at novel target views.}",
        "    \\label{fig:handcrafted_tf}",
        "\\end{figure*}",
    ]
    open(f"{args.out}/handcrafted_comparison.tex", "w").write("\n".join(lines) + "\n")

    # matplotlib preview with the same layout
    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    width, label_w = 7.0, 0.3  # inches
    cell = (width - label_w) / n
    fig = plt.figure(figsize=(width, cell * len(ROWS)))
    W, H = fig.get_size_inches()
    for i, (label, *_) in enumerate(ROWS):
        for j in range(n):
            ax = fig.add_axes([(label_w + j * cell) / W, 1 - (i + 1) * cell / H, cell / W, cell / H])
            ax.axis("off")
            ax.imshow(imgs[i][j], interpolation="lanczos")
            if j == 0:
                ax.text(-0.04, 0.5, label, ha="right", va="center", rotation=90, transform=ax.transAxes)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}/handcrafted_comparison.{ext}", dpi=300)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
