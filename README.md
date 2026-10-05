# IR Webcast Transcriber v15

v14 + Q4 Events の字幕HLS (`captions/.../subtitles.m3u8`) 対応。

## v15 修正点
- `subtitles.m3u8` / `/captions/` を音声M3U8として ffmpeg に渡さない。
- HLS manifest を読み、WebVTT字幕セグメントを直接取得してTXT化。
- 既存字幕を使う場合はOpenAI Transcription APIを呼ばない（API費用なし）。
- 字幕URLではMP3ダウンロードを表示しない。
- 通常の音声M3U8は従来どおり ffmpeg → MP3 → OpenAI API。
- YouTube / Vimeo / SmartVision / IR Webcasting / TS等の既存対応は維持。

## 原因
Q4の `.../captions/.../subtitles.m3u8` は音声プレイリストではなく字幕(WebVTT)プレイリストです。v14は拡張子だけで通常M3U8と判定してffmpegでMP3化しようとしていたため、文字起こし経路が不適切でした。
