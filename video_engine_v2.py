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


def fade_window(p, start=0.0, end=0.18):
    if end <= start:
        return 1.0
    return ease_in_out(np.clip((p-start)/(end-start), 0, 1))


def _style_axis(ax, bg, text, grid):
    ax.set_facecolor(bg)
    ax.tick_params(colors=text, labelsize=9, length=0, pad=7)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="y", color=grid, linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)


def _spread_label_positions(values, ymin, ymax, min_gap_ratio=.055):
    """Greedy label spreading in data coordinates, preserving vertical order."""
    if not values:
        return []
    gap = max((ymax-ymin)*min_gap_ratio, 1e-9)
    order = np.argsort(values)
    placed = np.array(values, dtype=float)
    low = ymin + (ymax-ymin)*.025
    high = ymax - (ymax-ymin)*.025
    last = low-gap
    for idx in order:
        placed[idx] = max(values[idx], last+gap, low)
        last = placed[idx]
    overflow = placed[order[-1]]-high
    if overflow > 0:
        placed -= overflow
    for j in range(len(order)-2, -1, -1):
        a, b = order[j], order[j+1]
        placed[a] = min(placed[a], placed[b]-gap)
    under = low-placed[order[0]]
    if under > 0:
        placed += under
    return placed.tolist()


def render_story_frame(df, scene, ratio, bg, text, grid, cmap, progress=1.0):
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
    # More right room for direct labels; portrait gets a slightly taller chart.
    rect = [0.10, 0.15, 0.72, 0.66] if ratio == "9:16" else [0.09, 0.16, 0.75, 0.65]
    ax = fig.add_axes(rect)
    _style_axis(ax, bg, text, grid)

    raw_p = np.clip(float(progress), 0, 1)
    p = ease_in_out(raw_p)
    title_alpha = fade_window(raw_p, 0.0, .16)
    subtitle_alpha = fade_window(raw_p, .06, .24) * .84
    chart_alpha = fade_window(raw_p, .10, .28)
    label_alpha = fade_window(raw_p, .80, .98)

    fig.text(0.075, 0.93, scene["title"], color=text, fontsize=scene.get("title_size", 22), fontweight="bold", ha="left", alpha=title_alpha)
    if scene.get("subtitle"):
        fig.text(0.075, 0.885, scene["subtitle"], color=text, fontsize=max(8, scene.get("title_size", 22)-7), ha="left", alpha=subtitle_alpha)
    if scene.get("source"):
        fig.text(0.075, 0.052, f"出典: {scene['source']}", color=text, fontsize=7, ha="left", alpha=.58*label_alpha)

    x = np.arange(len(dates), dtype=float)
    chart = scene["chart"]
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
        grow = ease_in_out(np.clip((raw_p-.10)/.72, 0, 1))
        for company in companies:
            vals = shown[company].to_numpy(float) * grow if company in shown else np.zeros(len(dates))
            ax.bar(x, vals, bottom=bottom, color=cmap[company], width=.68, label=company, alpha=chart_alpha)
            bottom += vals
        if scene.get("latest_values", True) and raw_p >= .82 and chart == "積み上げ棒":
            ax.text(x[-1], bottom[-1] + ymax*.018, f"{bottom[-1]:,.0f}{scene.get('unit','')}", color=text, ha="center", va="bottom", fontsize=9, fontweight="bold", alpha=label_alpha)

    elif chart == "棒グラフ":
        width = .76 / max(1, len(companies))
        grow = ease_in_out(np.clip((raw_p-.10)/.72, 0, 1))
        for i, company in enumerate(companies):
            vals = pivot[company].to_numpy(float) * grow if company in pivot else np.zeros(len(dates))
            ax.bar(x + (i-(len(companies)-1)/2)*width, vals, width=width, color=cmap[company], label=company, alpha=chart_alpha)

    else:
        draw_p = ease_in_out(np.clip((raw_p-.10)/.72, 0, 1))
        pos = draw_p * max(0, len(dates)-1)
        whole = int(np.floor(pos)); frac = pos-whole
        endpoints = []
        for company in companies:
            vals = pivot[company].to_numpy(float) if company in pivot else np.zeros(len(dates))
            if len(dates) == 1:
                end_x, end_y = 0, vals[0]
                ax.scatter([0], [end_y], color=cmap[company], s=22, alpha=chart_alpha)
            else:
                xs = list(x[:whole+1]); ys = list(vals[:whole+1])
                if whole < len(dates)-1:
                    end_x = x[whole]+frac
                    end_y = vals[whole]+(vals[whole+1]-vals[whole])*frac
                    xs.append(end_x); ys.append(end_y)
                else:
                    end_x, end_y = x[-1], vals[-1]
                ax.plot(xs, ys, color=cmap[company], linewidth=2.8, solid_capstyle="round", alpha=chart_alpha)
                ax.scatter([end_x], [end_y], color=cmap[company], s=18, zorder=4, alpha=chart_alpha)
            endpoints.append((company, vals[-1]))

        if scene.get("end_labels", True) and raw_p >= .80:
            actual = [v for _, v in endpoints]
            adjusted = _spread_label_positions(actual, 0, ymax, scene.get("label_gap", .055))
            for (company, actual_y), label_y in zip(endpoints, adjusted):
                # Leader line makes collision avoidance visually explicit.
                ax.plot([x[-1]+.04, x[-1]+.22], [actual_y, label_y], color=cmap[company], linewidth=.9, alpha=.65*label_alpha)
                label = company
                if scene.get("latest_values", True):
                    decimals = int(scene.get("value_decimals", 0))
                    label += f"  {actual_y:,.{decimals}f}{scene.get('unit','')}"
                ax.text(x[-1]+.27, label_y, label, color=cmap[company], fontsize=scene.get("end_label_size", 8), va="center", fontweight="bold", alpha=label_alpha)
        ax.set_xlim(-.2, max(1, len(dates)-1)+2.45)

    ax.set_xticks(x)
    ax.set_xticklabels(dates, color=text, alpha=chart_alpha)
    ax.set_ylabel(scene.get("unit", ""), color=text, fontsize=9, alpha=chart_alpha)
    if scene.get("legend", False):
        leg = ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
        for t in leg.get_texts():
            t.set_color(text); t.set_alpha(chart_alpha)
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
    if len(paths) == 1:
        shutil.copyfile(paths[0], output)
        return
    transition = max(.05, float(transition))
    cmd = ["ffmpeg", "-y"]
    for path in paths:
        cmd += ["-i", str(path)]
    filters = []
    previous = "[0:v]"
    elapsed = float(durations[0])
    for i in range(1, len(paths)):
        out = f"[v{i}]"
        safe_t = min(transition, max(.05, durations[i-1]/2), max(.05, durations[i]/2))
        offset = max(.01, elapsed-safe_t)
        filters.append(f"{previous}[{i}:v]xfade=transition=fade:duration={safe_t:.3f}:offset={offset:.3f}{out}")
        previous = out
        elapsed += float(durations[i])-safe_t
    cmd += ["-filter_complex", ";".join(filters), "-map", previous, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
