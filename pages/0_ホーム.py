import streamlit as st

st.set_page_config(page_title="動画グラフ生成ツール", layout="wide")
st.title("動画グラフ生成ツール")
st.write("左のページメニューから利用するツールを選択してください。")
st.page_link("app.py", label="企業比較動画ジェネレーター")
st.page_link("pages/2_複数グラフ動画.py", label="複数グラフ動画ジェネレーター")

st.page_link("pages/3_2指標・企業横比較.py", label="2指標・企業横比較")
