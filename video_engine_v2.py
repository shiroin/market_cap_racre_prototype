import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ease_in_out(t):
    t = np.clip(float(t), 0.0, 1.0)
    return 3 * t * t - 2 * t * t * t


def _style_axis(ax, bg, text, grid):
    ax.set_facecolor(bg)
    ax.tick_params(colors=text, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="y", color=grid, linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)


def render_story_frame(df, scene, ratio, bg, text, grid, cmap, progress=1.0):
    """Render one smooth frame. Bars grow vertically and lines are progressively drawn."""
    metric = scene["metric"]
    work = df[["date", "company", metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna()
    dates = list(dict.fromkeys(work["date"].astype(str)))
    companies = list(dict.fromkeys(work["company"].astype(str)))
    pivot = work.pivot_table(index="date", columns="company", values=metric, aggfunc="sum").reindex(dates).fillna(0)

    sizes = {"9:16": (5.4, 9.6), "1:1": (7, 7), "16:9": (9.6, 5.4)}
    fig = plt.figure(figsize=sizes[ratio], dpi=120)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0.10, 0.17, 0.80, 0.64])
    _style_axis(ax, bg, text, grid)
    fig.text(0.08, 0.93, scene["title"], color=text, fontsize=scene.get("title_size", 22), fontweight="bold", ha="left")
    if scene.get("subtitle"):
        fig.text(0.08, 0.885, scene["subtitle"], color=text, fontsize=max(8, scene.get("title_size", 22)-7), ha="left", alpha=.82)
    if scene.get("source"):
        fig.text(0.08, 0.055, f"出典: {scene['source']}", color=text, fontsize=7, ha="left", alpha=.62)

    x = np.arange(len(dates), dtype=float)
    p = ease_in_out(progress)
    chart = scene["chart"]

    # Fixed axes prevent visual jumping during animation.
    if chart == "100%積み上げ":
        ymax = 100.0
    else:
        ymax = float(max(1.0, pivot.sum(axis=1).max() if chart == "積み上げ棒" else pivot.to_numpy().max())) * 1.18
    ax.set_ylim(0, ymax)

    if chart in ("積み上げ棒", "100%積み上げ"):
        shown = pivot.copy()
        if chart == "100%積み上げ":
            shown = shown.div(shown.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
        bottom = np.zeros(len(dates))
        for company in companies:
            vals = shown[company].to_numpy(float) * p if company in shown else np.zeros(len(dates))
            ax.bar(x, vals, bottom=bottom, color=cmap[company], width=.72, label=company)
            bottom += vals
        if scene.get("latest_values", True) and progress >= .98 and chart == "積み上げ棒":
            ax.text(x[-1], bottom[-1] + ymax*.018, f"{bottom[-1]:,.0f}{scene.get('unit','')}", color=text, ha="center", va="bottom", fontsize=9, fontweight="bold")

    elif chart == "棒グラフ":
        width = .78 / max(1, len(companies))
        for i, company in enumerate(companies):
            vals = pivot[company].to_numpy(float) * p if company in pivot else np.zeros(len(dates))
            ax.bar(x + (i-(len(companies)-1)/2)*width, vals, width=width, color=cmap[company], label=company)

    else:  # 折れ線
        # Draw continuously inside the current segment instead of revealing whole dates abruptly.
        pos = p * max(0, len(dates)-1)
        whole = int(np.floor(pos))
        frac = pos - whole
        for company in companies:
            vals = pivot[company].to_numpy(float) if company in pivot else np.zeros(len(dates))
            if len(dates) == 1:
                ax.scatter([0], [vals[0]], color=cmap[company], s=22)
                end_x, end_y = 0, vals[0]
            else:
                xs = list(x[:whole+1]); ys = list(vals[:whole+1])
                if whole < len(dates)-1:
                    end_x = x[whole] + frac
                    end_y = vals[whole] + (vals[whole+1]-vals[whole])*frac
                    xs.append(end_x); ys.append(end_y)
                else:
                    end_x, end_y = x[-1], vals[-1]
                ax.plot(xs, ys, color=cmap[company], linewidth=2.8, solid_capstyle="round")
                ax.scatter([end_x], [end_y], color=cmap[company], s=18, zorder=4)
            if scene.get("end_labels", True) and progress >= .98:
                label = company
                if scene.get("latest_values", True):
                    label += f"  {vals[-1]:,.1f}{scene.get('unit','')}"
                ax.text(x[-1]+.15, vals[-1], label, color=cmap[company], fontsize=8, va="center", fontweight="bold")
        ax.set_xlim(-.2, max(1, len(dates)-1)+2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(dates, color=text)
    ax.set_ylabel(scene.get("unit", ""), color=text, fontsize=9)
    if scene.get("legend", False):
        leg = ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
        for t in leg.get_texts(): t.set_color(text)
    return fig


def save_scene_v2(df, scene, path, ratio, fps, bg, text, grid, cmap):
    frames = max(2, int(scene.get("duration", 2.5)*fps))
    hold = max(0, int(scene.get("hold", 1.0)*fps))
    folder = Path(tempfile.mkdtemp(prefix="story_frames_"))
    try:
        for i in range(frames+hold):
            p = 1.0 if i >= frames else (i+1)/frames
            fig = render_story_frame(df, scene, ratio, bg, text, grid, cmap, p)
            fig.savefig(folder/f"frame_{i:05d}.png", facecolor=bg)
            plt.close(fig)
        subprocess.run(["ffmpeg","-y","-framerate",str(fps),"-i",str(folder/"frame_%05d.png"),"-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def concat_with_crossfade(paths, durations, output, transition=.45):
    """Join scenes with xfade. Falls back to concat when only one scene is supplied."""
    if len(paths) == 1:
        shutil.copyfile(paths[0], output)
        return
    cmd = ["ffmpeg", "-y"]
    for p in paths: cmd += ["-i", str(p)]
    filters = []
    previous = "[0:v]"
    elapsed = durations[0]
    for i in range(1, len(paths)):
        out = f"[v{i}]"
        offset = max(.01, elapsed-transition)
        filters.append(f"{previous}[{i}:v]xfade=transition=fade:duration={transition}:offset={offset:.3f}{out}")
        previous = out
        elapsed += durations[i]-transition
    cmd += ["-filter_complex", ";".join(filters), "-map", previous, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
