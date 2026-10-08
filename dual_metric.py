"""Two-metric horizontal comparison video mode."""
import io
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import streamlit as st

EXAMPLE = pd.DataFrame([
    {"year": y, "company": name, "metric_a": a, "metric_b": b}
    for y, rows in {
        2021: [("企業A",100,100),("企業B",100,100),("企業C",100,100)],
        2022: [("企業A",105,118),("企業B",90,111),("企業C",110,106)],
        2023: [("企業A",98,145),("企業B",94,123),("企業C",125,109)],
        2024: [("企業A",112,171),("企業B",104,135),("企業C",119,125)],
        2025: [("企業A",120,210),("企業B",108,148),("企業C",135,131)],
    }.items() for name, a, b in rows
])

def validate(df):
    required = {"year","company","metric_a","metric_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("必要な列がありません: " + ", ".join(sorted(missing)))
    d = df[list(required)].copy()
    d["year"] = pd.to_numeric(d["year"], errors="raise").astype(int)
    for col in ("metric_a","metric_b"):
        d[col] = pd.to_numeric(d[col], errors="raise")
    if d[["metric_a","metric_b"]].isna().any().any() or not np.isfinite(d[["metric_a","metric_b"]].to_numpy()).all():
        raise ValueError("指標に欠損または無限大があります")
    if d.duplicated(["year","company"]).any():
        raise ValueError("year × company に重複があります")
    if (d[["metric_a","metric_b"]] < 0).any().any():
        raise ValueError("横棒の値は0以上にしてください")
    years = sorted(d.year.unique())
    companies = list(dict.fromkeys(d.company.astype(str)))
    if len(years)<1 or len(companies)<1:
        raise ValueError("データがありません")
    if len(d)!=len(years)*len(companies):
        raise ValueError("全ての年に全企業の2指標を入力してください")
    return d, years, companies

def render_mode(sheets_csv_url, setup_jp_font):
    st.header("2指標・横棒比較")
    st.caption("year / company / metric_a / metric_b の縦持ちデータ。実数値・基準年倍率・基準年比成長率に対応。")
    source = st.radio("データ入力", ["直接編集","CSVアップロード","Google Sheets"], horizontal=True, key="dual_source")
    df = EXAMPLE.copy()
    if source == "CSVアップロード":
        uploaded = st.file_uploader("CSVファイル", type=["csv"], key="dual_csv")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
    elif source == "Google Sheets":
        url = st.text_input("公開・閲覧可能なGoogle Sheets URL", key="dual_url")
        sheet = st.text_input("シート名（任意）", key="dual_sheet")
        if url:
            try:
                df = pd.read_csv(sheets_csv_url(url, sheet or None))
            except Exception as e:
                st.error(f"読み込み失敗: {e}")
                return
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="dual_editor")
    st.download_button("CSVテンプレート", EXAMPLE.to_csv(index=False).encode("utf-8-sig"), "dual_metric_template.csv", "text/csv")
    try:
        data, years, companies = validate(edited)
    except Exception as e:
        st.error(str(e))
        return
    with st.sidebar:
        st.subheader("2指標モード設定")
        title = st.text_input("タイトル", "企業別・2指標の比較", key="dual_title")
        name_a = st.text_input("指標Aの名称", "戸数", key="dual_name_a")
        name_b = st.text_input("指標Bの名称", "1戸あたり売値", key="dual_name_b")
        mode = st.selectbox("表示形式", ["基準年倍率","実数値","基準年比成長率"], key="dual_mode")
        axis = st.radio("横軸", ["同一軸","別軸"], key="dual_axis")
        sort = st.selectbox("企業の並び", ["入力順","指標Aの最新値順","指標Bの最新値順"], key="dual_sort")
        ratio = st.selectbox("画面比率", ["16:9","1:1","9:16"], key="dual_ratio")
        color_a = st.color_picker("指標Aの色", "#8799B1", key="dual_color_a")
        color_b = st.color_picker("指標Bの色", "#D95E37", key="dual_color_b")
        bg = st.color_picker("背景色", "#F7F5F0", key="dual_bg")
        fg = st.color_picker("文字色", "#202B38", key="dual_fg")
        fps = st.selectbox("FPS", [24,30,60], key="dual_fps")
        duration = st.slider("年ごとの移行時間（秒）", .5, 3.0, 1.2, .1, key="dual_duration")
        hold = st.slider("最後の静止（秒）", .5, 4.0, 1.5, .5, key="dual_hold")
    if sort != "入力順":
        col = "metric_a" if "A" in sort else "metric_b"
        latest = data[data.year==years[-1]].set_index("company")[col]
        companies = sorted(companies, key=lambda c: -latest.loc[c])
    lookup = {(int(r.year),str(r.company)):(float(r.metric_a),float(r.metric_b)) for r in data.itertuples()}
    base = {c:lookup[(years[0],c)] for c in companies}
    def transformed(y,c):
        values = lookup[(y,c)]
        if mode == "実数値":
            return values
        if any(x==0 for x in base[c]):
            raise ValueError("基準年倍率・成長率は基準年の値が0だと計算できません")
        vals = [values[i]/base[c][i] for i in range(2)]
        return tuple(v if mode=="基準年倍率" else (v-1)*100 for v in vals)
    try:
        values = np.array([[transformed(y,c) for c in companies] for y in years])
    except ValueError as e:
        st.error(str(e)); return
    # Growth rates may be negative: use symmetric limits with a zero baseline.
    lows = values.min(axis=(0,1)); highs = values.max(axis=(0,1))
    def limits(k):
        lo,hi = (min(0,lows[k]),max(0,highs[k])) if axis=="別軸" else (min(0,lows.min()),max(0,highs.max()))
        pad = max((hi-lo)*.14,.1)
        return lo-pad,hi+pad
    lim_a,lim_b = limits(0),limits(1)
    size = {"16:9":(12.8,7.2),"1:1":(9,9),"9:16":(7.2,12.8)}[ratio]
    setup_jp_font()
    def draw(fig, frame_values, label):
        fig.clear()
        fig.patch.set_facecolor(bg)
        ax1 = fig.add_axes([.23,.14,.69,.70])
        ax1.set_facecolor(bg)
        ax1.set_xlim(*lim_a)
        y = np.arange(len(companies))
        h = .34
        ax1.barh(y-h/2,frame_values[:,0],height=h,color=color_a,label=name_a)
        ax1.axvline(0,color=fg,linewidth=.7,alpha=.4)
        ax1.set_yticks(y,companies,color=fg)
        ax1.invert_yaxis()
        ax1.set_ylim(len(companies)-.4,-.7)
        ax1.tick_params(axis="x",colors=fg,labelsize=9)
        ax1.tick_params(axis="y",length=0,labelsize=10)
        for spine in ax1.spines.values(): spine.set_visible(False)
        ax2 = ax1.twiny() if axis=="別軸" else ax1
        if axis=="別軸":
            ax2.set_xlim(*lim_b)
            ax2.tick_params(axis="x",colors=color_b,labelsize=9)
            for spine in ax2.spines.values(): spine.set_visible(False)
        ax2.barh(y+h/2,frame_values[:,1],height=h,color=color_b,label=name_b)
        def fmt(v):
            return f"{v:+.0f}%" if mode=="基準年比成長率" else (f"{v:.2f}倍" if mode=="基準年倍率" else f"{v:,.1f}")
        for k,ax in enumerate((ax1,ax2)):
            low,high = (lim_a,lim_b)[k]
            offset=(high-low)*.012
            for i,v in enumerate(frame_values[:,k]):
                ax.text(v+(offset if v>=0 else -offset),i+(-h/2 if k==0 else h/2),fmt(v),
                        va="center",ha="left" if v>=0 else "right",fontsize=8,color=fg)
        fig.text(.5,.96,title,ha="center",va="top",color=fg,fontsize=17,weight="bold")
        fig.text(.5,.89,str(label),ha="center",color=fg,fontsize=18,weight="bold")
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=color_a,label=name_a),Patch(color=color_b,label=name_b)],
                   loc="lower center",ncol=2,frameon=False,labelcolor=fg,bbox_to_anchor=(.5,.02))
        if axis=="別軸":
            fig.text(.5,.095,"下軸: "+name_a+"   /   上軸: "+name_b,ha="center",fontsize=9,color=fg)
    st.subheader("プレビュー")
    selected = st.select_slider("表示年",options=years,value=years[-1],key="dual_preview_year")
    fig=plt.figure(figsize=size,dpi=100)
    draw(fig,values[years.index(selected)],selected)
    st.pyplot(fig)
    plt.close(fig)
    if st.button("2指標比較MP4を生成",type="primary",key="dual_generate"):
        try:
            with st.spinner("動画をレンダリング中..."):
                frames_per=max(2,round(fps*duration))
                count=max(1,(len(years)-1)*frames_per)+round(fps*hold)
                fig=plt.figure(figsize=size,dpi=110)
                def update(frame):
                    if len(years)==1:
                        current=values[0]; label=years[0]
                    else:
                        segment=min(frame//frames_per,len(years)-2)
                        t=min(1,(frame-segment*frames_per)/frames_per)
                        t=t*t*(3-2*t)
                        current=values[segment]*(1-t)+values[segment+1]*t
                        label=years[segment] if t<.5 else years[segment+1]
                    draw(fig,current,label)
                animation=FuncAnimation(fig,update,frames=count,interval=1000/fps)
                with tempfile.TemporaryDirectory() as temp:
                    path=Path(temp)/"dual_metric_comparison.mp4"
                    animation.save(str(path),writer=FFMpegWriter(fps=fps,bitrate=3500))
                    video=path.read_bytes()
                plt.close(fig)
            st.video(video)
            st.download_button("MP4をダウンロード",video,"dual_metric_comparison.mp4","video/mp4",key="dual_download")
        except Exception as e:
            st.error(f"動画生成に失敗しました: {e}")
