# Reference video motion model

Frame-by-frame review of `ScreenRecording_10-06-2026 11-33-36_1(2).mp4` shows that the stacked-bar scene is primarily a **horizontal reveal**, not a sequence of bars growing from zero height.

Implementation target:

- Historical bars keep their full data height.
- The animation front moves horizontally through time.
- Completed periods are shown at full height.
- The current period is revealed horizontally across the bar width; its height is already the period's full value.
- Series labels sit immediately to the right of the reveal front and use the current period's full values.
- When the front travels through the gap between periods, label Y positions interpolate from the previous period's full segment centers to the next period's full segment centers.
- No bottom-to-top label sweep.
- No collision lane or leader lines for stacked bars.
- The total label follows the top of the current full stack.
- `一気に表示` remains a separate mode; the reference-style rule applies to chronological left/right reveal.

This is intentionally different from the previous `value * reveal_factor` implementation, which made bars and labels rise vertically from zero.