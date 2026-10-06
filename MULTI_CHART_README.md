# 複数グラフ動画ジェネレーター v1.0 prototype

既存の時価総額レースとは別に、複数のチャートSceneを順番に描画して1本のMP4にする試作版です。

## 起動

```bash
pip install -r requirements.txt
streamlit run multi_chart_app.py
```

FFmpegが必要です。macOSでは `brew install ffmpeg`。デプロイ環境向けの `packages.txt` には既に ffmpeg が入っています。

## 入力

long形式のCSVを使います。

- `date`: 期間
- `company`: 企業名
- その他の数値列: Sceneで選択する指標

`sample_timeseries.csv` を参照してください。

## v1.0 prototypeでできること

- Sceneを1〜8個作成
- 積み上げ棒
- 複数折れ線
- 横並び棒
- 100%積み上げ
- Sceneごとに指標・タイトル・サブタイトル・単位・出典を指定
- 描画時間・静止時間を指定
- 企業カラーを共通化
- 9:16 / 1:1 / 16:9
- 24 / 30 / 60fps
- 複数SceneをFFmpegで結合してMP4出力

## 次の実装候補

- Sceneのドラッグ並べ替え
- Scene間のクロスフェード / Morph風トランジション
- 最新値ラベルの詳細設定
- PPTX出力
- Google Sheets通常共有URLの直接読み込み
- Sceneテンプレート保存
