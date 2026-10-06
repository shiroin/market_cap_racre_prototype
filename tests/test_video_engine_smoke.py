import tempfile
from pathlib import Path

import pandas as pd

from video_engine_v2 import render_story_frame


def test_render_story_frame_smoke():
    df = pd.DataFrame({
        "date": ["2025", "2026", "2025", "2026"],
        "company": ["A", "A", "B", "B"],
        "value": [10, 12, 7, 9],
    })
    scene = {
        "metric": "value", "chart": "積み上げ棒", "title": "test", "subtitle": "",
        "unit": "", "bar_animation": "左→右", "data_labels": "自動",
    }
    fig = render_story_frame(df, scene, "9:16", "#FFFFFF", "#172033", "#E6E8EC", {"A":"#264D6D", "B":"#C94F27"}, 1.0, "preview")
    fig.canvas.draw()
    assert fig.canvas.get_width_height() == (360, 640)
