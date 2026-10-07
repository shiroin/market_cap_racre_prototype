import io
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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


def google_sheets_csv_url(url):
    """Accept a normal Google Sheets share/edit URL or an existing CSV export URL."""
    url = url.strip()
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        return url
    sheet_id = match.group(1)
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    gid = query.get("gid", [None])[0]
    if gid is None and parsed.fragment:
        frag = parse_qs(parsed.fragment)
        gid = frag.get("gid", [None])[0]
    export = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    if gid:
        export += f"&gid={gid}"
    return export


st.title("複数グラフ動画ジェネレーター v1.3")
st.caption(f"Google Sheets / CSV / Excel → 複数Scene → 高速MP4 ｜ 使用フォント: {FONT}")

if "palette_text" not in st.session_state:
    st.session_state.palette_text = ",".join(DEFAULT_COLORS[:4])

with st.sidebar:
    st.header("動画全体")
    ratio = st.selectbox("縦横比", ["元動画 (64:139)", "9:16", "4:5", "1:1", "5:4", "16:9"], index=2, help="デフォルトは4:5です。元動画は512×1112px＝64:139です。")
    render_mode = st.selectbox("生成品質", ["高速プレビュー", "標準", "高画質"], index=0, help="まず高速プレビューで確認し、最後だけ標準/高画質がおすすめです。")
    mode_settings = {"高速プレビュー":(12,"preview"), "標準":(24,"standard"), "高画質":(30,"high")}
    fps, quality = mode_settings[render_mode]
    quality_caption = {
        "元動画 (64:139)": {"高速プレビュー":"384×834相当 / 12fps", "標準":"768×1668相当 / 24fps", "高画質":"1024×2224 / 30fps"},
        "9:16": {"高速プレビュー":"360×640相当 / 12fps", "標準":"720×1280相当 / 24fps", "高画質":"1080×1920相当 / 30fps"},
        "4:5": {"高速プレビュー":"400×500相当 / 12fps", "標準":"800×1000相当 / 24fps", "高画質":"1080×1350相当 / 30fps"},
        "1:1": {"高速プレビュー":"400×400相当 / 12fps", "標準":"800×800相当 / 24fps", "高画質":"1080×1080相当 / 30fps"},
        "5:4": {"高速プレビュー":"500×400相当 / 12fps", "標準":"1000×800相当 / 24fps", "高画質":"1350×1080相当 / 30fps"},
        "16:9": {"高速プレビュー":"640×360相当 / 12fps", "標準":"1280×720相当 / 24fps", "高画質":"1920×1080相当 / 30fps"},
    }
    st.caption(quality_caption[ratio][render_mode])
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

source = st.radio("データソース", ["直接編集", "ファイルアップロード", "Google Sheets"], horizontal=True)
df = DEFAULT_DATA.copy()
if source == "ファイルアップロード":
    uploaded = st.file_uploader("CSV / Excel", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            suffix = Path(uploaded.name).suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(uploaded)
            else:
                excel = pd.ExcelFile(uploaded)
                sheet = st.selectbox("読み込むシート", excel.sheet_names, key="excel_sheet")
                df = pd.read_excel(excel, sheet_name=sheet)
            st.success(f"読み込み成功: {uploaded.name}（{len(df):,}行）")
        except Exception as e:
            st.error(f"ファイル読み込み失敗: {type(e).__name__}: {e}")
            st.stop()
elif source == "Google Sheets":
    url = st.text_input("Google Sheets URL", help="通常の共有URL（/edit?usp=sharing）をそのまま貼り付けられます。CSV出力URLにも対応します。")
    if url:
        try:
            csv_url = google_sheets_csv_url(url)
            df = pd.read_csv(csv_url)
            st.success(f"Google Sheets読み込み成功（{len(df):,}行）")
        except Exception as e:
            st.error(f"読み込み失敗: {type(e).__name__}: {e}")
            st.caption("共有設定が「リンクを知っている全員が閲覧可」になっているか確認してください。通常の /edit URL はアプリ側でCSV URLへ自動変換します。")

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
chart_options = ["積み上げ棒", "折れ線", "棒グラフ", "100%積み上げ", "横比較ランキング", "年表"]

for i in range(int(scene_count)):
    with st.expander(f"Scene {i+1}", expanded=i == 0):
        c1, c2 = st.columns(2)
        default_chart = default_charts[i] if i < 3 else "折れ線"
        chart = c1.selectbox("グラフ種類", chart_options, index=chart_options.index(default_chart), key=f"chart_{i}")
        preferred = default_metrics[i] if i < 3 and default_metrics[i] in metric_columns else metric_columns[0]
        metric = c2.selectbox("指標列", metric_columns, index=metric_columns.index(preferred), key=f"metric_{i}", disabled=chart == "年表")
        title = st.text_input("タイトル", default_titles[i] if i < 3 else f"Scene {i+1}", key=f"title_{i}")
        subtitle = st.text_area("サブタイトル", default_subtitles[i] if i < 3 else "", key=f"subtitle_{i}", height=80, help="長い場合は任意の位置で改行できます。")
        c3, c4 = st.columns(2)
        unit = c3.text_input("単位", "か月" if metric == "inventory_months" else "億円", key=f"unit_{i}")
        source_text = c4.text_input("出典", "", key=f"source_{i}")
        scene_note = st.text_area("補足", "", key=f"scene_note_{i}", help="Scene下部に小さく表示します。複数行入力できます。")
        st.markdown("**下部コメント**")
        cc1,cc2 = st.columns(2)
        scene_comment_1 = cc1.text_area("下部コメント①", "", key=f"scene_comment_1_{i}", height=70, help="グラフ下部の専用スペースに表示します。")
        scene_comment_2 = cc2.text_area("下部コメント②", "", key=f"scene_comment_2_{i}", height=70, help="下部コメント①の後に表示します。")
        scene_comment_size = st.slider("下部コメントサイズ", 8, 24, 12, key=f"scene_comment_size_{i}")
        scene_comment_delay = st.slider("グラフ完了後→下部コメント①まで（秒）", 0.0, 3.0, 0.7, 0.1, key=f"scene_comment_delay_{i}", help="グラフの描画が完了してから下部コメント①のフェード開始までの待ち時間です。")
        c5, c6, c7, c7b = st.columns(4)
        duration = c5.slider("描画時間（秒）", .5, 30.0, 2.8, .1, key=f"duration_{i}")
        hold = c6.slider("静止時間（秒）", 0., 10., 1.2, .1, key=f"hold_{i}")
        title_size = c7.slider("タイトルサイズ", 12, 34, 22, key=f"title_size_{i}")
        subtitle_size = c7b.slider("サブタイトルサイズ", 6, 24, 12, key=f"subtitle_size_{i}")
        c8, c9, c10 = st.columns(3)
        legend = c8.checkbox("凡例", chart != "折れ線", key=f"legend_{i}")
        end_labels = c9.checkbox("右端ラベル", chart == "折れ線", key=f"end_labels_{i}")
        latest_values = c10.checkbox("最新値を表示", True, key=f"latest_{i}")
        c11, c12, c13 = st.columns(3)
        end_label_size = c11.slider("右端ラベルサイズ", 6, 16, 8, key=f"label_size_{i}")
        label_gap = c12.slider("ラベル間隔", .025, .12, .055, .005, key=f"label_gap_{i}")
        value_decimals = c13.selectbox("小数桁", [0,1,2], index=0, key=f"decimals_{i}")

        timeline_events = []
        timeline_start = 2018
        timeline_end = 2025
        timeline_note = ""
        timeline_summary = ""
        timeline_summary_2 = ""
        timeline_summary_size = 12
        if chart == "年表":
            st.markdown("**年表データ**")
            st.caption("元動画風：左に年、中央に縦軸、右に「日付 → 見出し → 補足」。イベントは上から順に1件ずつ表示されます。")
            default_timeline = pd.DataFrame([
                {"year":2018.45,"date":"2018年6月","title":"運営会社を設立","description":"プロジェクトの運営会社などが設立","badge":""},
                {"year":2020.05,"date":"2020年1月","title":"140億円","description":"出資を受け、プロジェクトが本格始動","badge":"出資"},
                {"year":2022.72,"date":"2022年9月","title":"80億円","description":"出資を決定","badge":"資金調達"},
                {"year":2023.86,"date":"2023年11月","title":"366億円の協調融資","description":"複数の金融機関による協調融資","badge":"融資"},
                {"year":2025.04,"date":"2025年1月","title":"開業日を発表","description":"同日に経済効果の試算も公表","badge":""},
                {"year":2025.56,"date":"2025年7月25日","title":"開業","description":"開業を迎える","badge":""},
            ])
            timeline_df = st.data_editor(default_timeline, num_rows="dynamic", use_container_width=True, key=f"timeline_data_{i}")
            timeline_events = timeline_df.to_dict("records")
            t1,t2 = st.columns(2)
            timeline_start = int(t1.number_input("開始年", 1900, 2200, 2018, 1, key=f"timeline_start_{i}"))
            timeline_end = int(t2.number_input("終了年", 1900, 2200, 2025, 1, key=f"timeline_end_{i}"))
            timeline_summary = ""
            timeline_summary_2 = ""
            timeline_summary_size = 12
            timeline_note = st.text_area("出典・補足注記", "出典・補足事項をここに入力できます。", key=f"timeline_note_{i}")

        ranking_sort = "大きい順"
        ranking_reference = 0.0
        ranking_highlight = 3
        if chart == "横比較ランキング":
            st.markdown("**横比較ランキング表示**")
            r1,r2,r3 = st.columns(3)
            ranking_sort = r1.selectbox("並び順", ["大きい順","小さい順"], key=f"ranking_sort_{i}")
            ranking_reference = float(r2.number_input("基準値（0で非表示）", value=0.0, key=f"ranking_reference_{i}"))
            ranking_highlight = int(r3.number_input("強調する上位件数", 0, 50, 3, 1, key=f"ranking_highlight_{i}"))
            st.caption("対象名は company 列、比較値は選択した指標列を使用します。同一対象に複数行ある場合は最新行の値を使います。")

        bar_animation = "左→右"
        data_labels = "自動"
        data_label_size = 7
        bar_gap = 0.32
        if chart in ("積み上げ棒", "100%積み上げ", "棒グラフ"):
            st.markdown("**棒グラフ表示**")
            b1, b2, b3 = st.columns(3)
            bar_animation = b1.selectbox("表示パターン", ["左→右", "右→左", "一気に表示"], index=0, key=f"bar_animation_{i}")
            data_labels = b2.selectbox("データラベル", ["自動", "すべて", "合計のみ", "なし"], index=0, key=f"data_labels_{i}", help="自動は狭い積み上げ部分の数値を省略し、重なりを防ぎます。")
            data_label_size = b3.slider("データラベルサイズ", 6, 14, 8, key=f"data_label_size_{i}")
            bar_gap = st.slider("棒と棒の隙間", 0.0, 0.80, 0.32, 0.02, key=f"bar_gap_{i}", help="0にすると隣り合う期間の棒がぴったり接します。値を大きくすると棒の間隔が広がります。")

        scenes.append({
            "chart":chart, "metric":metric, "title":title, "subtitle":subtitle, "unit":unit, "source":source_text, "scene_note":scene_note, "scene_comment_1":scene_comment_1, "scene_comment_2":scene_comment_2, "scene_comment_size":scene_comment_size, "scene_comment_delay":scene_comment_delay,
            "duration":duration, "hold":hold, "title_size":title_size, "subtitle_size":subtitle_size, "legend":legend, "end_labels":end_labels,
            "latest_values":latest_values, "end_label_size":end_label_size, "label_gap":label_gap,
            "value_decimals":value_decimals, "bar_animation":bar_animation, "data_labels":data_labels,
            "data_label_size":data_label_size, "bar_gap":bar_gap, "timeline_events":timeline_events,
            "timeline_start":timeline_start, "timeline_end":timeline_end, "timeline_note":timeline_note,
            "timeline_summary":timeline_summary, "timeline_summary_2":timeline_summary_2, "timeline_summary_size":timeline_summary_size,
            "ranking_sort":ranking_sort, "ranking_reference":ranking_reference, "ranking_highlight":ranking_highlight
        })

preview_scene = st.selectbox("プレビューするScene", range(1, len(scenes)+1), format_func=lambda x:f"Scene {x}")
preview_progress = st.slider("アニメーション位置", .05, 1.0, 1.0, .05)
try:
    preview = render_story_frame(cleaned, scenes[preview_scene-1], ratio, bg, text, grid, cmap, preview_progress, "preview")
    st.pyplot(preview, use_container_width=False)
    plt.close(preview)
except Exception as e:
    st.warning(f"プレビューできません: {e}")

st.markdown("**サムネイル画像**")
thumbnail_progress = st.slider("サムネイルのアニメーション位置", 0.0, 1.0, 0.30, 0.05, help="タイトル・軸・補足は常時表示し、データ本体をどこまで描画した状態でPNGにするか指定します。")
try:
    thumbnail = render_story_frame(cleaned, scenes[preview_scene-1], ratio, bg, text, grid, cmap, thumbnail_progress, "high")
    thumbnail_buffer = io.BytesIO()
    thumbnail.savefig(thumbnail_buffer, format="png", facecolor=thumbnail.get_facecolor(), bbox_inches=None, pad_inches=0)
    thumbnail_buffer.seek(0)
    plt.close(thumbnail)
    st.download_button(
        "サムネイルPNGを保存",
        thumbnail_buffer.getvalue(),
        f"scene_{preview_scene:02d}_thumbnail.png",
        "image/png",
        use_container_width=True,
    )
except Exception as e:
    st.warning(f"サムネイルPNGを生成できません: {e}")

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
