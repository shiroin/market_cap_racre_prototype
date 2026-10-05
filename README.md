# 企業比較動画ジェネレーター v0.8

## 新機能：マイプリセット
調整した設定を名前付きで保存し、次回起動時にも呼び出せます。

- マイプリセット保存
- 保存済みプリセット一覧
- 呼び出し
- 削除
- JSONエクスポート
- JSONインポート

`my_presets.json` にローカル保存します。

## 起動
```bash
cd ~/Downloads/company_comparison_video_v08
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

FFmpeg未導入:
```bash
brew install ffmpeg
```

※ フォルダを削除するとローカルの my_presets.json も消えるため、
重要なプリセットはJSONエクスポートでバックアップしてください。
