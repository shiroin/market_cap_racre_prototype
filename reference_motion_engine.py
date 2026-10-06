import numpy as np


def reference_front(raw_p, count, mode="左→右"):
    """Horizontal time wipe. Fraction controls bar width, never bar height."""
    if count <= 0:
        return 0, 0, 0.0, 0.0
    p = np.clip((float(raw_p) - 0.10) / 0.72, 0.0, 1.0)
    p = 3 * p * p - 2 * p * p * p
    u = p * count
    completed = min(int(np.floor(u)), count)
    frac = 1.0 if completed >= count else float(u - completed)
    if mode == "右→左":
        active = max(0, count - 1 - completed)
        front_x = active + 0.34 - 0.68 * frac
    else:
        active = min(count - 1, completed)
        front_x = active - 0.34 + 0.68 * frac
    return completed, active, frac, front_x


def stacked_full_geometry(shown, companies, period):
    running = 0.0
    result = []
    for company in companies:
        value = float(shown[company].iloc[period]) if company in shown else 0.0
        result.append((company, value, running + value / 2.0))
        running += value
    return result, running
