"""Direct entry point for the two-metric company comparison mode."""
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager
import re
from urllib.parse import urlparse, parse_qs

st.set_page_config(page_title="2指標・企業横比較", layout="wide")

def setup_jp_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Hiragino Sans", "Yu Gothic", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic"):
        if name in installed:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return "sans-serif"

def sheets_csv_url(url, sheet_name=None):
    match = re.search(r"/spreadsheets/d/([^/]+)", url)
    if not match:
        raise ValueError("Google Sheets URLを確認してください。")
    sid = match.group(1)
    if sheet_name:
        return f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    parsed = urlparse(url)
    gid = parse_qs(parsed.fragment).get("gid", [None])[0] or parse_qs(parsed.query).get("gid", [None])[0]
    return f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv" + (f"&gid={gid}" if gid else "")

from dual_metric import render_mode
setup_jp_font()
render_mode(sheets_csv_url, setup_jp_font)
