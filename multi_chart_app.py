import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import font_manager

st.set_page_config(page_title="複数グラフ動画ジェネレーター", layout="wide")


def setup_jp_font():
    preferred = ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return "sans-serif"


FONT = setup_jp_font()

DEFAULT_DATA = pd.DataFrame({
    "date": ["2022Q1", "2022Q2", "2022Q3", "2022Q4", "2023Q1", "2023Q2"] * 4,
    "company": np.repeat(["カチタス", "スター・マイカHD", "ムゲンエステート", "インテリックスHD"], 6),
    "inventory": [520, 545, 570, 600, 635, 680, 650, 690, 720, 760, 805, 860, 330, 350, 380, 410, 450, 490, 250, 265, 280, 300, 320, 345],
    "sales": [310, 320, 330, 345, 360, 375, 250, 260, 275, 290, 300, 315, 180, 190, 205, 220, 235, 250, 160, 170, 175, 185, 195, 205],
})
DEFAULT_DATA["inventory_months"] = DEFAULT_DATA["inventory"] / DEFAULT_DATA["sales"] * 3

PRESETS = {
    "Strainer風": {"bg": "#F4F1EA", "text": "#25313C", "grid": "#D8D4CC"},
    "白": {"bg": "#FFFFFF", "text": "#172033", "grid": "#E6E8EC"},
    "ダーク": {"bg": "#101B2C", "text": "#FFFFFF", "grid": "#314057"},
}
DEFAULT_COLORS = ["#264D6D", "#C94F27", "#758F3D", "#D59B28", "#6A5ACD", "#3A8D8D", "#A64B7A", "#777777"]


def fig_size(ratio):
    return {"9:16": (5.4, 9.6), "1:1": (7, 7), "16:9": (9.6, 5.4)}[ratio]


def clean_data(df):
    required = {"date", "company"}
    if not required.issubset(df.columns):
        raise ValueError("date / company 列が必要です。")
    out = df.copy()
    out["date"] = out["date"].astype(str)
    out["company"] = out["company"].astype(str)
    return out


def color_map(companies, palette_text):
    palette = [x.strip() for x in palette_text.split(",") if x.strip()]
    if not palette:
        palette = DEFAULT_COLORS
    return {c: palette[i % len(palette)] for i, c in enumerate(companies)}


def style_axis(ax, bg, text, grid):
    ax.set_facecolor(bg)
    ax.tick_params(colors=text, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color=grid, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def scene_figure(df, scene, ratio, bg, text, grid, cmap, progress=1.0):
    metric = scene["metric"]
    if metric not in df.columns:
        raise ValueError(f"{metric} 列がありません。")
    work = df[["date", "company", metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna()
    dates = list(dict.fromkeys(work["date"].tolist()))
    companies = list(dict.fromkeys(work["company"].tolist()))
    if not dates or not companies:
        raise ValueError("表示できるデータがありません。")

    fig = plt.figure(figsize=fig_size(ratio), dpi=120)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0.11, 0.17, 0.78, 0.65])
    style_axis(ax, bg, text, grid)
    fig.text(0.08, 0.93, scene["title"], color=text, fontsize=scene["title_size"], fontweight="bold", ha="left")
    if scene["subtitle"]:
        fig.text(0.08, 0.885, scene["subtitle"], color=text, fontsize=max(8, scene["title_size"] - 7), ha="left", alpha=0.85)
    if scene["source"]:
        fig.text(0.08, 0.055, f"出典: {scene['source']}", color=text, fontsize=7, ha="left", alpha=0.65)

    pivot = work.pivot_table(index="date", columns="company", values=metric, aggfunc="sum").reindex(dates).fillna(0)
    n = max(1, int(np.ceil(len(dates) * np.clip(progress, 0.01, 1.0))))
    shown_dates = dates[:n]
    shown = pivot.loc[shown_dates]
    x = np.arange(len(shown_dates))

    if scene["chart"] == "積み上げ棒":
        bottom = np.zeros(len(shown_dates))
        for company in companies:
            vals = shown[company].to_numpy() if company in shown.columns else np.zeros(len(shown_dates))
            ax.bar(x, vals, bottom=bottom, color=cmap[company], width=0.72, label=company)
            bottom += vals
    elif scene["chart"] == "棒グラフ":
        width = 0.78 / max(1, len(companies))
        for i, company in enumerate(companies):
            vals = shown[company].to_numpy() if company in shown.columns else np.zeros(len(shown_dates))
            ax.bar(x + (i - (len(companies)-1)/2) * width, vals, width=width, color=cmap[company], label=company)
    elif scene["chart"] == "100%積み上げ":
        denom = shown.sum(axis=1).replace(0, np.nan)
        pct = shown.div(denom, axis=0).fillna(0) * 100
        bottom = np.zeros(len(shown_dates))
        for company in companies:
            vals = pct[company].to_numpy() if company in pct.columns else np.zeros(len(shown_dates))
            ax.bar(x, vals, bottom=bottom, color=cmap[company], width=0.72, label=company)
            bottom += vals
        ax.set_ylim(0, 100)
    else:
        for company in companies:
            vals = shown[company].to_numpy() if company in shown.columns else np.zeros(len(shown_dates))
            ax.plot(x, vals, color=cmap[company], linewidth=2.7, marker="o", markersize=4, label=company)
            if scene["end_labels"] and len(vals):
                ax.text(x[-1] + 0.12, vals[-1], company, color=cmap[company], fontsize=8, va="center", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(shown_dates, rotation=0, color=text)
    ax.set_ylabel(scene["unit"], color=text, fontsize=9)
    if scene["legend"]:
        leg = ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
        for t in leg.get_texts():
            t.set_color(text)
    if scene["chart"] == "折れ線" and scene["end_labels"]:
        ax.set_xlim(-0.2, max(1, len(shown_dates)-1) + 1.4)
    return fig


def save_scene_video(df, scene, path, ratio, fps, bg, text, grid, cmap):
    setup_jp_font()
    frames = max(2, int(scene["duration"] * fps))
    hold = max(0, int(scene["hold"] * fps))
    fig = plt.figure(figsize=fig_size(ratio), dpi=120)
    plt.close(fig)

    # Rebuild the figure per frame. It is slower than artist updates but robust across chart types.
    tmp_dir = Path(tempfile.mkdtemp(prefix="scene_frames_"))
    try:
        for i in range(frames + hold):
            progress = min(1.0, (i + 1) / frames)
            frame_fig = scene_figure(df, scene, ratio, bg, text, grid, cmap, progress)
            frame_fig.savefig(tmp_dir / f"frame_{i:05d}.png", facecolor=bg, bbox_inches=None)
            plt.close(frame_fig)
        cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp_dir / "frame_%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def concat_videos(paths, output):
    list_file = Path(tempfile.gettempdir()) / "chart_video_concat.txt"
    list_file.write_text("\n".join([f"file '{p.as_posix()}'" for p in paths]), encoding="utf-8")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(output)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


st.title("複数グラフ動画ジェネレーター v1.0 prototype")
st.caption(f"Google Sheets / CSV → 複数Scene → MP4 ｜ 使用フォント: {FONT}")

with st.sidebar:
    st.header("動画全体")
    ratio = st.selectbox("縦横比", ["9:16", "1:1", "16:9"], index=1)
    fps = st.select_slider("FPS", [24, 30, 60], value=30)
    preset_name = st.selectbox("デザイン", list(PRESETS.keys()))
    preset = PRESETS[preset_name]
    bg = st.color_picker("背景", preset["bg"])
    text = st.color_picker("文字", preset["text"])
    grid = st.color_picker("グリッド", preset["grid"])
    palette_text = st.text_input("企業カラー（カンマ区切り）", ",".join(DEFAULT_COLORS[:4]))

source = st.radio("データソース", ["直接編集", "CSVアップロード", "Google Sheets"], horizontal=True)
df = DEFAULT_DATA.copy()
if source == "CSVアップロード":
    uploaded = st.file_uploader("CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
elif source == "Google Sheets":
    url = st.text_input("CSV公開URL（Google SheetsのCSV出力URL）")
    if url:
        try:
            df = pd.read_csv(url)
        except Exception as e:
            st.error(f"読み込み失敗: {e}")

st.subheader("データ")
st.caption("必須: date / company。数値列はSceneの指標として選べます。")
edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)

try:
    cleaned = clean_data(edited)
    metric_columns = [c for c in cleaned.columns if c not in ["date", "company"]]
    companies = list(dict.fromkeys(cleaned["company"].tolist()))
    cmap = color_map(companies, palette_text)
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Scene構成")
scene_count = st.number_input("Scene数", min_value=1, max_value=8, value=3, step=1)
scenes = []
default_titles = ["上場4社の在庫", "各社の在庫推移", "在庫は売上の何か月分か"]
default_charts = ["積み上げ棒", "折れ線", "折れ線"]
default_metrics = ["inventory", "inventory", "inventory_months"]

for i in range(int(scene_count)):
    with st.expander(f"Scene {i+1}", expanded=i == 0):
        c1, c2 = st.columns(2)
        chart = c1.selectbox("グラフ種類", ["積み上げ棒", "折れ線", "棒グラフ", "100%積み上げ"], index=["積み上げ棒", "折れ線", "棒グラフ", "100%積み上げ"].index(default_charts[i] if i < 3 else "折れ線"), key=f"chart_{i}")
        preferred_metric = default_metrics[i] if i < 3 and default_metrics[i] in metric_columns else (metric_columns[0] if metric_columns else "")
        metric = c2.selectbox("指標列", metric_columns, index=metric_columns.index(preferred_metric) if preferred_metric in metric_columns else 0, key=f"metric_{i}")
        title = st.text_input("タイトル", default_titles[i] if i < 3 else f"Scene {i+1}", key=f"title_{i}")
        subtitle = st.text_input("サブタイトル", "", key=f"subtitle_{i}")
        unit = st.text_input("単位", "億円" if metric != "inventory_months" else "か月", key=f"unit_{i}")
        source_text = st.text_input("出典", "", key=f"source_{i}")
        c3, c4, c5 = st.columns(3)
        duration = c3.slider("描画時間（秒）", 0.5, 8.0, 2.5, 0.5, key=f"duration_{i}")
        hold = c4.slider("静止時間（秒）", 0.0, 5.0, 1.0, 0.5, key=f"hold_{i}")
        title_size = c5.slider("タイトルサイズ", 12, 34, 22, key=f"title_size_{i}")
        c6, c7 = st.columns(2)
        legend = c6.checkbox("凡例を表示", True, key=f"legend_{i}")
        end_labels = c7.checkbox("折れ線の右端に企業名", chart == "折れ線", key=f"end_labels_{i}")
        scenes.append({"chart": chart, "metric": metric, "title": title, "subtitle": subtitle, "unit": unit, "source": source_text, "duration": duration, "hold": hold, "title_size": title_size, "legend": legend, "end_labels": end_labels})

preview_scene = st.selectbox("プレビューするScene", range(1, len(scenes)+1), format_func=lambda x: f"Scene {x}")
try:
    preview = scene_figure(cleaned, scenes[preview_scene-1], ratio, bg, text, grid, cmap, 1.0)
    st.pyplot(preview, use_container_width=False)
    plt.close(preview)
except Exception as e:
    st.warning(f"プレビューできません: {e}")

if st.button("MP4を生成", type="primary", use_container_width=True):
    if shutil.which("ffmpeg") is None:
        st.error("FFmpegが見つかりません。ローカルでは `brew install ffmpeg`、デプロイ環境では packages.txt に ffmpeg を追加してください。")
    else:
        try:
            with st.spinner("Sceneを描画してMP4を生成しています…"):
                workdir = Path(tempfile.mkdtemp(prefix="multi_chart_video_"))
                scene_paths = []
                for i, scene in enumerate(scenes):
                    p = workdir / f"scene_{i:02d}.mp4"
                    save_scene_video(cleaned, scene, p, ratio, fps, bg, text, grid, cmap)
                    scene_paths.append(p)
                output = workdir / "multi_chart_video.mp4"
                concat_videos(scene_paths, output)
            st.video(str(output))
            st.download_button("MP4を保存", output.read_bytes(), "multi_chart_video.mp4", "video/mp4", use_container_width=True)
        except subprocess.CalledProcessError as e:
            st.error("FFmpeg処理に失敗しました。")
        except Exception as e:
            st.error(str(e))

with st.expander("CSV例"):
    st.code("date,company,inventory,sales,inventory_months\n2022Q1,カチタス,520,310,5.03\n2022Q2,カチタス,545,320,5.11\n2022Q1,スター・マイカHD,650,250,7.80", language="text")
