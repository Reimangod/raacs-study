"""Dependency-free scientific plots of saved tables; not a circuit diagram."""
from collections import defaultdict
from html import escape
from pathlib import Path
from .evidence import number

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#333333", "#777777")


def write_figures(summary, output: Path):
    groups = defaultdict(list)
    for row in summary:
        # Avoid a mean over a successful subset or a mixture of readout contexts.
        if str(row["all_recorded_replicates_D_matched"]) not in ("True", "true") or row["n_D_matched"] < 3:
            continue
        groups[(row["cohort"], row["case"], row["context"], row["final_shots"])].append(row)
    written = []
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    for key, points in sorted(groups.items()):
        methods = sorted({r["display_name"] for r in points})
        xs = [number(r["D_target"]) for r in points]
        ys = [number(r["energy_Ha_mean_D_matched"]) for r in points]
        if len(set(xs)) < 2 or any(x is None for x in xs + ys):
            continue
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        pad = max((ymax-ymin)*.06, 1e-8)
        ymin, ymax = ymin-pad, ymax+pad
        def xy(x, y):
            return 90 + 560*(x-xmin)/(xmax-xmin), 355 - 275*(y-ymin)/(ymax-ymin)
        lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="450" viewBox="0 0 1000 450">',
                 '<rect width="1000" height="450" fill="white"/>',
                 '<g font-family="Arial,sans-serif" font-size="13" fill="black">',
                 f'<text x="90" y="24">{escape(" / ".join(key))}</text>',
                 '<text x="90" y="46">Physical CI energy; mean of 3 recorded D-matched readout replicates</text>',
                 '<path d="M90 80 V355 H650" fill="none" stroke="black"/>',
                 '<text x="280" y="413">Actual determinant dimension D</text>',
                 '<text x="16" y="245" transform="rotate(-90 16 245)">Energy [Ha]</text>']
        for i in range(5):
            x=xmin+(xmax-xmin)*i/4; px,_=xy(x,ymin)
            y=ymin+(ymax-ymin)*i/4; _,py=xy(xmin,y)
            lines += [f'<text x="{px:.2f}" y="377" text-anchor="middle">{x:g}</text>',
                      f'<text x="82" y="{py+4:.2f}" text-anchor="end">{y:.6f}</text>']
        for j, method in enumerate(methods):
            color=COLORS[j%len(COLORS)]
            p=sorted([r for r in points if r["display_name"]==method],key=lambda r:number(r["D_target"]))
            coords=[xy(number(r["D_target"]),number(r["energy_Ha_mean_D_matched"]))for r in p]
            lines.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}"for x,y in coords)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
            for x,y in coords:lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" fill="{color}"/>')
            lines.append(f'<text x="680" y="{88+j*25}" fill="{color}">{escape(method)}</text>')
        lines += ['<text x="90" y="440">Saved-data derivative, not new sampling. Missing sources are listed separately.</text>', '</g></svg>']
        name="__".join(key).replace("/","_").replace(" ","_")+".svg"
        (folder/name).write_text("\n".join(lines)+"\n",encoding="utf-8")
        written.append("figures/"+name)
    return written
