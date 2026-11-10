"""holo_graph — investor-grade SVG visuals, generated in pure Python.

No external deps, no JS: every graph is a self-contained SVG that renders
inline in Hugging Face model/dataset/Space cards and on GitHub. The style
targets top-tier technical presentation: dark canvas, neon gradient traces,
glow effects, labeled axes, and a receipts line so every visual is honest.
"""

from __future__ import annotations
import math


_DARK = "#0a0e1a"
_NEON = ["#22d3ee", "#a78bfa", "#34d399", "#f472b6", "#fbbf24"]


def _poly(points, color, width=2.5, glow=True, fill=False, opacity=1.0):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    g = f'<filter id="g{color[1:]}"><feGaussianBlur stdDeviation="3" result="b"/>\
<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>' if glow else ""
    f = f' filter="url(#g{color[1:]})"' if glow else ""
    area = f'<polygon points="{pts} {_points_area(points)}" fill="{color}" opacity="0.12"/>' if fill else ""
    return f'{g}<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" \
stroke-linecap="round" stroke-linejoin="round"{f} opacity="{opacity}"/>{area}'


def _points_area(points):
    if not points:
        return ""
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    return f"{min(xs):.1f},{ys[0]:.1f} {max(xs):.1f},{ys[0]:.1f}"


def line_chart(title: str, series: dict, *, width=720, height=360,
               x_labels=None, y_label="", source_line="") -> str:
    """series: {name: [values]}. Returns a complete SVG string."""
    pad_l, pad_b, pad_t = 56, 40, 52
    w, h = width, height
    all_vals = [v for vs in series.values() for v in vs]
    vmax = max(all_vals) if all_vals else 1
    vmin = min(0, min(all_vals)) if all_vals else 0
    n = max((len(vs) for vs in series.values()), default=1) or 1

    def X(i):
        return pad_l + (w - pad_l - 16) * (i / max(n - 1, 1))

    def Y(v):
        return pad_t + (h - pad_t - pad_b) * (1 - (v - vmin) / max(vmax - vmin, 1e-9))

    grid = ""
    for k in range(5):
        y = pad_t + (h - pad_t - pad_b) * k / 4
        val = vmax - (vmax - vmin) * k / 4
        grid += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-16}" y2="{y:.1f}" stroke="#1e293b" stroke-width="1"/>\
<text x="{pad_l-8}" y="{y+4:.1f}" fill="#64748b" font-size="11" text-anchor="end" font-family="monospace">{val:.3g}</text>'
    if x_labels:
        for i, lab in enumerate(x_labels):
            if i % max(1, len(x_labels) // 8) == 0:
                grid += f'<text x="{X(i):.1f}" y="{h-14}" fill="#64748b" font-size="10" text-anchor="middle" font-family="monospace">{lab}</text>'

    body = ""
    for idx, (name, vals) in enumerate(series.items()):
        color = _NEON[idx % len(_NEON)]
        pts = [(X(i), Y(v)) for i, v in enumerate(vals)]
        body += _poly(pts, color, fill=True)
        # end-point marker
        if pts:
            body += f'<circle cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="4" fill="{color}"/>'
        body += f'<text x="{pts[-1][0]+8:.1f}" y="{pts[-1][1]:.1f}" fill="{color}" font-size="11" font-family="monospace">{name}</text>'

    legend = "".join(
        f'<circle cx="{pad_l+10+i*150}" cy="20" r="4" fill="{_NEON[i%len(_NEON)]}"/>\
<text x="{pad_l+20+i*150}" y="24" fill="#94a3b8" font-size="11" font-family="monospace">{n}</text>'
        for i, n in enumerate(list(series)[: max(1, (w - pad_l) // 150)])
    )
    src = f'<text x="{w-16}" y="{h-6}" fill="#475569" font-size="9" text-anchor="end" font-family="monospace">{source_line}</text>' if source_line else ""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<rect width="{w}" height="{h}" rx="12" fill="{_DARK}"/>
<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="#0a0e1a"/><stop offset="100%" stop-color="#101728"/></linearGradient></defs>
<rect width="{w}" height="{h}" rx="12" fill="url(#bg)"/>
<text x="{pad_l}" y="26" fill="#e2e8f0" font-size="15" font-weight="bold" font-family="monospace">{title}</text>
{legend}{grid}{body}
<text x="{pad_l}" y="{h-6}" fill="#475569" font-size="9" font-family="monospace">{y_label}</text>{src}
</svg>'''


def radar(title: str, axes: list[str], values: list[float], *, width=360, height=360) -> str:
    """Radar/spider chart, values in [0,1]. Holographic style."""
    cx, cy, R = width / 2, height / 2 + 8, min(width, height) / 2 - 56
    n = len(axes)
    pts = []
    for i, v in enumerate(values):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        pts.append((cx + R * v * math.cos(ang), cy + R * v * math.sin(ang)))
    grid = ""
    for ring in (0.25, 0.5, 0.75, 1.0):
        rp = " ".join(
            f"{cx + R*ring*math.cos(-math.pi/2+2*math.pi*i/n):.1f},{cy + R*ring*math.sin(-math.pi/2+2*math.pi*i/n):.1f}"
            for i in range(n))
        grid += f'<polygon points="{rp}" fill="none" stroke="#1e293b" stroke-width="1"/>'
    labels = ""
    for i, a in enumerate(axes):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        lx, ly = cx + (R + 18) * math.cos(ang), cy + (R + 18) * math.sin(ang)
        labels += f'<text x="{lx:.1f}" y="{ly:.1f}" fill="#94a3b8" font-size="10" text-anchor="middle" font-family="monospace">{a}</text>'
    shape = f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}" fill="#22d3ee" fill-opacity="0.18" stroke="#22d3ee" stroke-width="2.5" filter="url(#glow)"/>'
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#22d3ee"/>' for x, y in pts)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="#0a0e1a"/><stop offset="100%" stop-color="#101728"/></linearGradient>
<filter id="glow"><feGaussianBlur stdDeviation="2.5" result="b"/>
<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
<rect width="{width}" height="{height}" rx="12" fill="url(#bg)"/>
<text x="{width/2}" y="24" fill="#e2e8f0" font-size="13" font-weight="bold" text-anchor="middle" font-family="monospace">{title}</text>
{grid}{labels}{shape}{dots}
</svg>'''
