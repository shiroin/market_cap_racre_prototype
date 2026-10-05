import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import font_manager
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import tempfile, re, json

st.set_page_config(page_title="企業比較動画ジェネレーター", layout="wide")

def setup_jp_font():
    preferred = ["Hiragino Sans","Hiragino Kaku Gothic ProN","Yu Gothic",
                 "Noto Sans CJK JP","Noto Sans JP","IPAexGothic","IPAGothic"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for f in preferred:
        if f in installed:
            plt.rcParams["font.family"] = f
            plt.rcParams["axes.unicode_minus"] = False
            return f
    return "sans-serif"


FONT = setup_jp_font()

PRESET_FILE = Path(__file__).with_name("my_presets.json")

def load_my_presets():
    if not PRESET_FILE.exists():
        return {}
    try:
        return json.loads(PRESET_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_my_presets(data):
    PRESET_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def apply_saved_preset(preset):
    # Values are placed into session_state before widgets are rendered on rerun.
    for k, v in preset.items():
        st.session_state[k] = v

def sheets_csv_url(url, sheet_name=None):
    # Supports normal Google Sheets share/edit URLs.
    m = re.search(r"/spreadsheets/d/([^/]+)", url)
    if not m:
        raise ValueError("Google Sheets URLを確認してください。")
    sid = m.group(1)
    gid = None
    if "gid=" in url:
        gid = parse_qs(urlparse(url).fragment).get("gid", [None])[0]
        if gid is None:
            gid = parse_qs(urlparse(url).query).get("gid", [None])[0]
    if sheet_name:
        return f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    if gid:
        return f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}"
    return f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv"

st.title("企業比較動画ジェネレーター v0.3")
st.caption(f"Google Sheets / CSV → 縦型比較動画 → MP4　｜　使用フォント: {FONT}")

source = st.radio("データソース", ["Google Sheets", "CSVアップロード", "直接編集"], horizontal=True)

default = pd.DataFrame({
    "company":["トヨタ自動車","三菱UFJ FG","ソニーグループ","日立製作所","キーエンス","ファーストリテイリング"],
    "value":[48.2,28.7,22.4,19.8,18.6,16.2]
})
df = default.copy()

if source == "Google Sheets":
    sheet_url = st.text_input("Google Sheets URL（閲覧可能な共有設定にしてください）")
    sheet_name = st.text_input("シート名（空欄ならURLのgidを使用）")
    if sheet_url:
        try:
            df = pd.read_csv(sheets_csv_url(sheet_url, sheet_name or None))
            st.success("Google Sheetsを読み込みました。")
        except Exception as e:
            st.warning(f"読み込み待ち/失敗: {e}")
elif source == "CSVアップロード":
    up = st.file_uploader("CSV", type="csv")
    if up:
        df = pd.read_csv(up)

PRESETS = {
    "元動画風": {
        "bg":"#101B2C","target":"#D9DFE8","stack":"#EF6262","alt":"#D94F5C",
        "edge":"#101B2C","text":"#FFFFFF","added":"#FFFFFF"
    },
    "Strainer風": {
        "bg":"#F4F1EA","target":"#2E4057","stack":"#E07A5F","alt":"#81B29A",
        "edge":"#F4F1EA","text":"#25313C","added":"#25313C"
    },
    "Bloomberg風": {
        "bg":"#080808","target":"#FF8C00","stack":"#00A6D6","alt":"#7AC943",
        "edge":"#080808","text":"#FFFFFF","added":"#FF8C00"
    },
    "モノクロ": {
        "bg":"#F5F5F5","target":"#222222","stack":"#666666","alt":"#A0A0A0",
        "edge":"#F5F5F5","text":"#111111","added":"#111111"
    },
    "ダーク＋赤": {
        "bg":"#111111","target":"#E6E6E6","stack":"#E53935","alt":"#8E2424",
        "edge":"#111111","text":"#FFFFFF","added":"#FF6B6B"
    },
    "ネイビー＋ブルー": {
        "bg":"#0B132B","target":"#E0E7FF","stack":"#3A86FF","alt":"#5BC0EB",
        "edge":"#0B132B","text":"#FFFFFF","added":"#8ECAE6"
    },
}

my_presets = load_my_presets()

with st.sidebar:
    st.header("マイプリセット")
    saved_names = ["（選択してください）"] + sorted(my_presets.keys())
    selected_saved = st.selectbox("保存済みデザイン", saved_names, key="saved_preset_selector")
    c_load, c_delete = st.columns(2)
    if c_load.button("呼び出す", use_container_width=True, disabled=selected_saved=="（選択してください）"):
        apply_saved_preset(my_presets[selected_saved])
        st.rerun()
    if c_delete.button("削除", use_container_width=True, disabled=selected_saved=="（選択してください）"):
        del my_presets[selected_saved]
        save_my_presets(my_presets)
        st.success("削除しました。")
        st.rerun()

    st.header("比較対象")
    target_name = st.text_input("名称", "NVIDIA", key="target_name")
    target_value = st.number_input("値", min_value=0.01, value=890.0, step=10.0, key="target_value")
    unit = st.selectbox("単位", ["兆円","億円","十億ドル","百万ドル"], key="unit")
    title = st.text_input("タイトル", "NVIDIAに日本企業を何社足せば届く？", key="title")

    st.header("動画設定")
    ratio = st.selectbox("縦横比", ["9:16","1:1","16:9"], key="ratio")
    fps = st.select_slider("FPS", [24,30,60], value=30, key="fps")
    speed = st.slider("通常速度（秒/社）", .15, 1.2, .50, .05, key="speed")
    accelerate_after = st.number_input("高速化を始める社数", 1, 100, 12, key="accelerate_after")
    fast_speed = st.slider("高速時（秒/社）", .08, .50, .18, .02, key="fast_speed")
    final_hold = st.slider("最後の静止", .5, 4.0, 1.5, .5, key="final_hold")

    st.header("デザイン")
    preset_name = st.selectbox(
        "配色プリセット",
        ["元動画風","Strainer風","Bloomberg風","モノクロ","ダーク＋赤","ネイビー＋ブルー"]
    )
    p = PRESETS[preset_name]
    st.caption("プリセット選択後も下の各色を個別に変更できます。")
    bg = st.color_picker("背景", p["bg"], key=f"bg_{preset_name}")
    target_color = st.color_picker("比較対象の棒", p["target"], key=f"target_{preset_name}")
    stack_color = st.color_picker("積み上げ棒（基本色）", p["stack"], key=f"stack_{preset_name}")
    alt_stack_color = st.color_picker("交互色 / 第2色", p["alt"], key=f"alt_{preset_name}")
    edge_color = st.color_picker("棒の境界線", p["edge"], key=f"edge_{preset_name}")
    text_color = st.color_picker("文字", p["text"], key=f"text_{preset_name}")
    added_text_color = st.color_picker("追加企業テキスト", p["added"], key=f"added_{preset_name}")
    color_mode = st.selectbox("積み上げ棒の配色", ["単色", "交互2色", "企業ごと"])
    company_palette = st.text_input(
        "企業ごとの色（HEX、カンマ区切り）",
        "#EF6262,#F08A5D,#B83B5E,#6A2C70,#355C7D"
    )
    show_labels = st.checkbox("棒の企業名", True)
    show_remaining = st.checkbox("あと○○を表示", True)
    count_up = st.checkbox("数字を滑らかにカウントアップ", True)

    st.header("レイアウト・文字")
    title_fontsize = st.slider("タイトル文字サイズ", 10, 32, 18)
    value_fontsize = st.slider("金額・社数の文字サイズ", 8, 28, 15)
    company_fontsize = st.slider("棒内の企業名サイズ", 5, 18, 9)
    remaining_fontsize = st.slider("「あと○○」文字サイズ", 8, 28, 16)
    added_fontsize = st.slider("追加企業の文字サイズ", 8, 24, 13)

    bar_width = st.slider("棒の太さ", 0.25, 0.90, 0.62, 0.01)
    bar_gap = st.slider("左右の棒の間隔", 0.50, 2.00, 1.00, 0.05)
    title_y = st.slider("タイトル位置（上下）", 0.85, 0.995, 0.965, 0.005)
    chart_top_margin = st.slider("グラフ上余白", 0.02, 0.25, 0.10, 0.01)
    bottom_text_y = st.slider("「あと○○」位置", 0.01, 0.15, 0.035, 0.005)
    added_text_y = st.slider("追加企業テキスト位置", 0.75, 0.98, 0.94, 0.01)

    st.divider()
    st.subheader("現在のデザインを保存")
    new_preset_name = st.text_input("プリセット名", placeholder="例：一平 X投稿 9:16")
    if st.button("マイプリセットに保存", use_container_width=True):
        if not new_preset_name.strip():
            st.warning("プリセット名を入力してください。")
        else:
            current = {
                "target_name": target_name,
                "target_value": target_value,
                "unit": unit,
                "title": title,
                "ratio": ratio,
                "fps": fps,
                "speed": speed,
                "accelerate_after": accelerate_after,
                "fast_speed": fast_speed,
                "final_hold": final_hold,
            }
            # Store design values by their variable names; these are applied where possible.
            current.update({
                "saved_design_preset": preset_name,
                "saved_bg": bg,
                "saved_target_color": target_color,
                "saved_stack_color": stack_color,
                "saved_alt_stack_color": alt_stack_color,
                "saved_edge_color": edge_color,
                "saved_text_color": text_color,
                "saved_added_text_color": added_text_color,
                "saved_color_mode": color_mode,
                "saved_company_palette": company_palette,
                "saved_title_fontsize": title_fontsize,
                "saved_value_fontsize": value_fontsize,
                "saved_company_fontsize": company_fontsize,
                "saved_remaining_fontsize": remaining_fontsize,
                "saved_added_fontsize": added_fontsize,
                "saved_bar_width": bar_width,
                "saved_bar_gap": bar_gap,
                "saved_title_y": title_y,
                "saved_chart_top_margin": chart_top_margin,
                "saved_bottom_text_y": bottom_text_y,
                "saved_added_text_y": added_text_y,
            })
            my_presets[new_preset_name.strip()] = current
            save_my_presets(my_presets)
            st.success(f"「{new_preset_name.strip()}」を保存しました。")

    if my_presets:
        preset_json = json.dumps(my_presets, ensure_ascii=False, indent=2)
        st.download_button(
            "マイプリセットをJSONで書き出す",
            preset_json.encode("utf-8"),
            "company_video_presets.json",
            "application/json",
            use_container_width=True
        )

with st.expander("マイプリセットをJSONから読み込む"):
    imported = st.file_uploader("company_video_presets.json", type="json", key="preset_json_upload")
    if imported is not None and st.button("プリセットをインポート"):
        try:
            incoming = json.load(imported)
            if not isinstance(incoming, dict):
                raise ValueError("JSON形式が不正です。")
            merged = load_my_presets()
            merged.update(incoming)
            save_my_presets(merged)
            st.success("インポートしました。")
            st.rerun()
        except Exception as e:
            st.error(f"インポートできません: {e}")

st.subheader("データ")
st.caption("列名は company / value。ここで直接修正できます。")
if "market_cap" in df.columns and "value" not in df.columns:
    df = df.rename(columns={"market_cap":"value"})
edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)

st.subheader("プレビュー")
preview_count = st.slider(
    "何社積み上げた状態をプレビューするか",
    1,
    max(1, len(edited)),
    min(5, max(1, len(edited)))
)

def figsize(r):
    return {"9:16":(5.4,9.6),"1:1":(7,7),"16:9":(9.6,5.4)}[r]

def ease(t):
    return 1 - (1-t)**3

def pick_stack_color(i):
    if color_mode == "単色":
        return stack_color
    if color_mode == "交互2色":
        return stack_color if i % 2 == 0 else alt_stack_color
    palette = [c.strip() for c in company_palette.split(",") if c.strip()]
    return palette[i % len(palette)] if palette else stack_color

def generate(data, output):
    setup_jp_font()
    d = data[["company","value"]].copy()
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna().sort_values("value", ascending=False).reset_index(drop=True)
    if d.empty: raise ValueError("company / value の有効なデータがありません。")
    names = d.company.astype(str).to_numpy()
    vals = d.value.to_numpy(float)
    cum = np.cumsum(vals)

    frames_each = [max(2, int(fps*(speed if i < accelerate_after else fast_speed))) for i in range(len(d))]
    starts = np.cumsum([0]+frames_each[:-1]).tolist()
    anim_frames = sum(frames_each)
    total_frames = anim_frames + int(fps*final_hold)
    ymax = max(target_value, cum[-1])*1.18

    fig = plt.figure(figsize=figsize(ratio), dpi=120)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([.10,.10,.80,max(.55,.88-chart_top_margin)])

    def state(frame):
        if frame >= anim_frames: return len(d), 0, 1.0
        idx = 0
        for i,s in enumerate(starts):
            if frame >= s: idx=i
        local = frame-starts[idx]
        frac = min(1, (local+1)/frames_each[idx])
        return idx, idx, ease(frac)

    def draw(frame):
        ax.clear(); ax.set_facecolor(bg); ax.axis("off")
        ax.set_xlim(-.75, bar_gap + .75); ax.set_ylim(0,ymax)

        if frame >= anim_frames:
            completed=len(d); active=None; frac=1
        else:
            active=0
            for i,s in enumerate(starts):
                if frame >= s: active=i
            completed=active
            frac=ease(min(1,(frame-starts[active]+1)/frames_each[active]))

        ax.bar(0,target_value,width=bar_width,color=target_color)
        ax.text(0,target_value+ymax*.018,f"{target_name}\n約{target_value:,.0f}{unit}",
                ha="center",va="bottom",color=text_color,fontsize=value_fontsize,fontweight="bold")

        bottom=0.0
        upto=len(d) if active is None else active+1
        for i in range(upto):
            val=vals[i] if active is None or i<active else vals[i]*frac
            ax.bar(bar_gap,val,bottom=bottom,width=bar_width,color=pick_stack_color(i),edgecolor=edge_color,linewidth=.9)
            if show_labels and val>ymax*.035:
                ax.text(bar_gap,bottom+val/2,names[i],ha="center",va="center",
                        color=text_color,fontsize=company_fontsize,fontweight="bold")
            bottom += val

        shown_count = len(d) if active is None else active+1
        display_total = bottom if count_up else (cum[shown_count-1] if shown_count else 0)
        ax.text(bar_gap,bottom+ymax*.018,f"{shown_count}社\n約{display_total:,.0f}{unit}",
                ha="center",va="bottom",color=text_color,fontsize=value_fontsize,fontweight="bold")

        if show_remaining:
            rem=target_value-display_total
            txt=f"あと約{rem:,.0f}{unit}" if rem>=0 else f"約{-rem:,.0f}{unit}上回る"
            ax.text(bar_gap/2,ymax*bottom_text_y,txt,ha="center",color=text_color,fontsize=remaining_fontsize,fontweight="bold")

        if active is not None:
            ax.text(bar_gap/2,ymax*added_text_y,f"+ {names[active]}  {vals[active]:,.1f}{unit}",
                    ha="center",color=added_text_color,fontsize=added_fontsize,fontweight="bold")

        fig.suptitle(title,y=title_y,color=text_color,fontsize=title_fontsize,fontweight="bold")

    ani=FuncAnimation(fig,draw,frames=total_frames,interval=1000/fps)
    ani.save(output,writer=FFMpegWriter(fps=fps,bitrate=5000))
    plt.close(fig)


def make_preview(data, count):
    setup_jp_font()
    d = data[["company","value"]].copy()
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna().sort_values("value", ascending=False).reset_index(drop=True)
    if d.empty:
        raise ValueError("company / value の有効なデータがありません。")

    count = min(count, len(d))
    names = d.company.astype(str).to_numpy()
    vals = d.value.to_numpy(float)
    total = vals[:count].sum()
    ymax = max(target_value, vals.sum()) * 1.18

    fig = plt.figure(figsize=figsize(ratio), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([.10,.10,.80,max(.55,.88-chart_top_margin)])
    ax.set_facecolor(bg)
    ax.axis("off")
    ax.set_xlim(-.75, bar_gap + .75)
    ax.set_ylim(0,ymax)

    ax.bar(0,target_value,width=bar_width,color=target_color)
    ax.text(0,target_value+ymax*.018,f"{target_name}\n約{target_value:,.0f}{unit}",
            ha="center",va="bottom",color=text_color,fontsize=value_fontsize,fontweight="bold")

    bottom=0.0
    for i in range(count):
        val=vals[i]
        ax.bar(bar_gap,val,bottom=bottom,width=bar_width,color=pick_stack_color(i),
               edgecolor=edge_color,linewidth=.9)
        if show_labels and val>ymax*.035:
            ax.text(bar_gap,bottom+val/2,names[i],ha="center",va="center",
                    color=text_color,fontsize=company_fontsize,fontweight="bold")
        bottom += val

    ax.text(bar_gap,bottom+ymax*.018,f"{count}社\n約{total:,.0f}{unit}",
            ha="center",va="bottom",color=text_color,fontsize=value_fontsize,fontweight="bold")

    if show_remaining:
        rem=target_value-total
        txt=f"あと約{rem:,.0f}{unit}" if rem>=0 else f"約{-rem:,.0f}{unit}上回る"
        ax.text(bar_gap/2,ymax*bottom_text_y,txt,ha="center",color=text_color,fontsize=remaining_fontsize,fontweight="bold")

    if count < len(d):
        ax.text(bar_gap/2,ymax*added_text_y,f"次: {names[count]}  {vals[count]:,.1f}{unit}",
                ha="center",color=added_text_color,fontsize=added_fontsize,fontweight="bold")

    fig.suptitle(title,y=title_y,color=text_color,fontsize=title_fontsize,fontweight="bold")
    return fig

try:
    preview_fig = make_preview(edited, preview_count)
    st.pyplot(preview_fig, use_container_width=False)
    plt.close(preview_fig)
    st.caption("設定変更はこのプレビューに即時反映されます。MP4生成前のデザイン確認に使えます。")
except Exception as e:
    st.warning(f"プレビューできません: {e}")

if st.button("動画を生成", type="primary", use_container_width=True):
    try:
        with st.spinner("MP4を生成中…"):
            out=Path(tempfile.gettempdir())/"company_comparison_v03.mp4"
            generate(edited,out)
        st.video(str(out))
        st.download_button("MP4を保存",out.read_bytes(),"company_comparison_v03.mp4","video/mp4",
                           use_container_width=True)
    except Exception as e:
        st.error(str(e))

with st.expander("Google Sheetsの作り方"):
    st.markdown("""
1行目を次の2列にします。

| company | value |
|---|---:|
| トヨタ自動車 | 48.2 |
| 三菱UFJ FG | 28.7 |

Google Sheetsの共有設定を「リンクを知っている全員が閲覧可」にしてURLを貼り付けます。
""")
