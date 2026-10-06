import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import font_manager

from video_engine_v2 import render_story_frame, save_scene_v2, concat_with_crossfade

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
    "date": ["2022Q1", "2022Q2", "2022Q3", "2022Q4", "2023Q1", "2023Q2"]*4,
    "company": np.repeat(["カチタス", "スター・マイカHD", "ムゲンエステート", "インテリックスHD"], 6),
    "inventory": [520,545,570,600,635,680,650,690,720,760,805,860,330,350,380,410,450,490,250,265,280,300,320,345],
    "sales": [310,320,330,345,360,375,250,260,275,290,300,315,180,190,205,220,235,250,160,170,175,185,195,205]
})
DEFAULT_DATA["inventory_months"] = DEFAULT_DATA["inventory"]/DEFAULT_DATA["sales"]*3
PRESETS = {
    "Strainer風": {"bg":"#F4F1EA", "text":"#25313C", "grid":"#D8D4CC"},
    "白": {"bg":"#FFFFFF", "text":"#172033", "grid":"#E6E8EC"},
    "ダーク": {"bg":"#101B2C", "text":"#FFFFFF", "grid":"#314057"}
}
PALETTES = {
    "定番": ["#264D6D", "#C94F27", "#758F3D", "#D59B28", "#6A5ACD", "#3A8D8D", "#A64B7A", "#777777"],
    "ブルー": ["#163A5F", "#2E6594", "#5A8DB8", "#89B4D3", "#B7D3E8", "#375A7F", "#6F8FAF", "#9DB8CE"],
    "ビビッド": ["#2563EB", "#E4572E", "#2E8B57", "#F2B134", "#7C3AED", "#0891B2", "#DB2777", "#64748B"],
    "くすみ": ["#526D82", "#A26769", "#6B7D5C", "#B08B57", "#766C91", "#5F8585", "#956C7B", "#7B7B73"],
    "モノクロ": ["#1F2933", "#4B5563", "#6B7280", "#9CA3AF", "#374151", "#7C8795", "#B0B7C0", "#D1D5DB"]
}
DEFAULT_COLORS = PALETTES["定番"]


def clean_data(df):
    if not {"date", "company"}.issubset(df.columns):
        raise ValueError("date / company 列が必要です。")
    out = df.copy()
    out["date"] = out["date"].astype(str)
    out["company"] = out["company"].astype(str)
    return out


def parse_palette(text):
    return [x.strip() for x in text.split(",") if x.strip()] or DEFAULT_COLORS


st.title("複数グラフ動画ジェネレーター v1.3")
st.caption(f"Google Sheets / CSV → 複数Scene → 高速MP4 ｜ 使用フォント: {FONT}")

if "palette_text" not in st.session_state:
    st.session_state.palette_text = ",".join(DEFAULT_COLORS[:4])

with st.sidebar:
    st.header("動画全体")
    ratio = st.selectbox("縦横比", ["9:16", "1:1", "16:9"], index=0)
    render_mode = st.selectbox("生成品質", ["高速プレビュー", "標準", "高画質"], index=0, help="まず高速プレビューで確認し、最後だけ標準/高画質がおすすめです。")
    mode_settings = {"高速プレビュー":(12,"preview"), "標準":(24,"standard"), "高画質":(30,"high")}
    fps, quality = mode_settings[render_mode]
    st.caption({"高速プレビュー":"360×640相当 / 12fps", "標準":"720×1280相当 / 24fps", "高画質":"1080×1920相当 / 30fps"}[render_mode])
    transition = st.slider("Scene間クロスフェード（秒）", .10, 1.20, .45, .05)
    preset_name = st.selectbox("デザイン", list(PRESETS.keys()))
    preset = PRESETS[preset_name]
    bg = st.color_picker("背景", preset["bg"])
    text = st.color_picker("文字", preset["text"])
    grid = st.color_picker("グリッド", preset["grid"])
    st.markdown("**グラフ色プリセット**")
    pcols = st.columns(2)
    for idx, (name, colors) in enumerate(PALETTES.items()):
        if pcols[idx % 2].button(name, key=f"palette_btn_{name}", use_container_width=True):
            st.session_state.palette_text = ",".join(colors)
    palette_text = st.text_input("カラーコード（直接編集も可）", key="palette_text")

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
    if not metric_columns:
        raise ValueError("数値指標列を1つ以上追加してください。")
    companies = list(dict.fromkeys(cleaned["company"].tolist()))
    palette = parse_palette(palette_text)
    cmap = {c: palette[i % len(palette)] for i, c in enumerate(companies)}
except Exception as e:
    st.error(str(e)); st.stop()

with st.sidebar:
    with st.expander("企業別カラー", expanded=False):
        st.caption("プリセット選択後に、各社だけ微調整できます。")
        for i, company in enumerate(companies):
            cmap[company] = st.color_picker(company, cmap[company], key=f"company_color_{company}")

st.subheader("Scene構成")
scene_count = st.number_input("Scene数", 1, 8, 3, 1)
scenes = []
default_titles = ["上場買取再販4社の在庫", "各社の在庫キャッシュフロー", "在庫は売上の何か月分か"]
default_subtitles = ["販売用不動産（仕掛販売用不動産を含む）の期末残高", "販売用不動産の期末残高", "販売用不動産の期末残高÷売上高×12"]
default_charts = ["積み上げ棒", "折れ線", "折れ線"]
default_metrics = ["inventory", "inventory", "inventory_months"]
chart_options = ["積み上げ棒", "折れ線", "棒グラフ", "100%積み上げ"]

for i in range(int(scene_count)):
    with st.expander(f"Scene {i+1}", expanded=i == 0):
        c1, c2 = st.columns(2)
        default_chart = default_charts[i] if i < 3 else "折れ線"
        chart = c1.selectbox("グラフ種類", chart_options, index=chart_options.index(default_chart), key=f"chart_{i}")
        preferred = default_metrics[i] if i < 3 and default_metrics[i] in metric_columns else metric_columns[0]
        metric = c2.selectbox("指標列", metric_columns, index=metric_columns.index(preferred), key=f"metric_{i}")
        title = st.text_input("タイトル", default_titles[i] if i < 3 else f"Scene {i+1}", key=f"title_{i}")
        subtitle = st.text_input("サブタイトル", default_subtitles[i] if i < 3 else "", key=f"subtitle_{i}")
        c3, c4 = st.columns(2)
        unit = c3.text_input("単位", "か月" if metric == "inventory_months" else "億円", key=f"unit_{i}")
        source_text = c4.text_input("出典", "", key=f"source_{i}")
        c5, c6, c7 = st.columns(3)
        duration = c5.slider("描画時間（秒）", .5, 8.0, 2.8, .1, key=f"duration_{i}")
        hold = c6.slider("静止時間（秒）", 0., 5., 1.2, .1, key=f"hold_{i}")
        title_size = c7.slider("タイトルサイズ", 12, 34, 22, key=f"title_size_{i}")
        c8, c9, c10 = st.columns(3)
        legend = c8.checkbox("凡例", chart != "折れ線", key=f"legend_{i}")
        end_labels = c9.checkbox("右端ラベル", chart == "折れ線", key=f"end_labels_{i}")
        latest_values = c10.checkbox("最新値を表示", True, key=f"latest_{i}")
        c11, c12, c13 = st.columns(3)
        end_label_size = c11.slider("右端ラベルサイズ", 6, 16, 8, key=f"label_size_{i}")
        label_gap = c12.slider("ラベル間隔", .025, .12, .055, .005, key=f"label_gap_{i}")
        value_decimals = c13.selectbox("小数桁", [0,1,2], index=0, key=f"decimals_{i}")

        bar_animation = "左→右"
        data_labels = "自動"
        data_label_size = 7
        if chart in ("積み上げ棒", "100%積み上げ", "棒グラフ"):
            st.markdown("**棒グラフ表示**")
            b1, b2, b3 = st.columns(3)
            bar_animation = b1.selectbox("表示パターン", ["左→右", "右→左", "一気に表示"], index=0, key=f"bar_animation_{i}")
            data_labels = b2.selectbox("データラベル", ["自動", "すべて", "合計のみ", "なし"], index=0, key=f"data_labels_{i}", help="自動は狭い積み上げ部分の数値を省略し、重なりを防ぎます。")
            data_label_size = b3.slider("データラベルサイズ", 6, 14, 8, key=f"data_label_size_{i}")

        scenes.append({
            "chart":chart, "metric":metric, "title":title, "subtitle":subtitle, "unit":unit, "source":source_text,
            "duration":duration, "hold":hold, "title_size":title_size, "legend":legend, "end_labels":end_labels,
            "latest_values":latest_values, "end_label_size":end_label_size, "label_gap":label_gap,
            "value_decimals":value_decimals, "bar_animation":bar_animation, "data_labels":data_labels,
            "data_label_size":data_label_size
        })

preview_scene = st.selectbox("プレビューするScene", range(1, len(scenes)+1), format_func=lambda x:f"Scene {x}")
preview_progress = st.slider("アニメーション位置", .05, 1.0, 1.0, .05)
try:
    preview = render_story_frame(cleaned, scenes[preview_scene-1], ratio, bg, text, grid, cmap, preview_progress, "preview")
    st.pyplot(preview, use_container_width=False)
    plt.close(preview)
except Exception as e:
    st.warning(f"プレビューできません: {e}")

if st.button(f"MP4を生成（{render_mode}）", type="primary", use_container_width=True):
    if shutil.which("ffmpeg") is None:
        st.error("FFmpegが見つかりません。macOSでは `brew install ffmpeg` を実行してください。")
    else:
        try:
            progress = st.progress(0, text="動画生成を開始します…")
            workdir = Path(tempfile.mkdtemp(prefix="multi_chart_video_")); paths = []; durations = []
            for i, scene in enumerate(scenes):
                progress.progress(int(i/max(1, len(scenes))*85), text=f"Scene {i+1}/{len(scenes)} を生成中…")
                p = workdir/f"scene_{i:02d}.mp4"
                save_scene_v2(cleaned, scene, p, ratio, fps, bg, text, grid, cmap, quality)
                paths.append(p); durations.append(scene["duration"]+scene["hold"])
            progress.progress(90, text="Sceneを結合中…")
            output = workdir/"multi_chart_video_v13.mp4"
            concat_with_crossfade(paths, durations, output, transition)
            progress.progress(100, text="完成しました")
            st.video(str(output))
            st.download_button("MP4を保存", output.read_bytes(), "multi_chart_video_v13.mp4", "video/mp4", use_container_width=True)
        except Exception as e:
            st.error(f"生成に失敗しました: {e}")

with st.expander("入力CSV例"):
    st.code("date,company,inventory,sales,inventory_months\n2022Q1,カチタス,520,310,5.03\n2022Q2,カチタス,545,320,5.11\n2022Q1,スター・マイカHD,650,250,7.80", language="text")
