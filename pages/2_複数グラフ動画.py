# 複数グラフ動画ジェネレーター
#
# 実装本体は multi_chart_app.py に残し、既存URL/ローカル起動との互換性を保ちます。
# Streamlit の pages/ から選択されたときだけ本体を実行します。
exec(open("multi_chart_app.py", encoding="utf-8").read(), globals())
