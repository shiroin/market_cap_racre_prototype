# v2 animation upgrade

`video_engine_v2.py` を追加しました。

## 改善点

- 棒グラフ: 期間が1本ずつ突然出る方式ではなく、0→実数値へ滑らかに伸長
- 折れ線: 日付単位のカクカクした表示ではなく、線分内を連続補間して描画
- easing: smoothstep (`3t² - 2t³`)
- 折れ線右端: 企業名 + 最新値 + 単位を直接表示可能
- 積み上げ棒: 最新合計値を表示可能
- Scene切替: FFmpeg `xfade` によるクロスフェードを実装
- 軸レンジを動画中固定し、アニメーション中のガタつきを防止

## multi_chart_app.py への組み込み方

先頭で次をimportします。

```python
from video_engine_v2 import render_story_frame, save_scene_v2, concat_with_crossfade
```

プレビューの `scene_figure(...)` を `render_story_frame(...)` に置換します。

MP4生成では `save_scene_video(...)` を `save_scene_v2(...)` に置換し、各Sceneの実時間を

```python
durations = [s["duration"] + s["hold"] for s in scenes]
```

として、最後の `concat_videos(...)` を

```python
concat_with_crossfade(scene_paths, durations, output, transition=0.45)
```

に置換します。

Scene設定に以下を追加するとラベル表示を切り替えられます。

```python
latest_values = st.checkbox("最新値を表示", True, key=f"latest_values_{i}")
```

scene dictへ `"latest_values": latest_values` を追加してください。

## 次段階

元動画にさらに寄せる場合は、右端ラベルの衝突回避、タイトルのフェードイン、Scene間でタイトル位置を固定、注目系列だけ太くする機能を追加します。
