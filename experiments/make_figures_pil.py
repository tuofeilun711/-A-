# -*- coding: utf-8 -*-
"""Regenerate figures/ from results/*.csv without matplotlib (PIL only).

PIL-only figure script; use it when matplotlib is unavailable. Works on
matplotlib. Same file names so the report can reference either version.
"""
import csv, os, sys
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")
F = lambda n: ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", n, index=0)
NAVY, BLUE, ORANGE, GREY = (15, 36, 56), (31, 111, 235), (232, 131, 58), (110, 120, 135)
TEAL = (42, 157, 143)


def read_csv(name):
    with open(os.path.join(RES, name), newline="") as f:
        return list(csv.DictReader(f))


def frame():
    W, H = 1500, 820
    img = Image.new("RGB", (W, H), "white")
    return img, ImageDraw.Draw(img), 150, W - 70, 110, H - 140


def axes(d, L, R, T, B, ymax, ylabel, xticks, xlabels=None, yfmt="{:.3f}"):
    pw, ph = R - L, B - T
    for i in range(6):
        y = T + ph - ph * i / 5
        d.line([(L, y), (R, y)], fill=(240, 243, 247), width=2)
        d.text((L - 16, y), yfmt.format(ymax * i / 5), font=F(26), fill=GREY, anchor="rm")
    for i, t in enumerate(xticks):
        x = L + pw * (i + 0.5) / len(xticks)
        d.line([(x, B), (x, B + 10)], fill=(170, 180, 192), width=3)
        lab = xlabels[i] if xlabels else str(t)
        d.text((x, B + 22), lab, font=F(24), fill=GREY, anchor="ma")
    d.line([(L, B), (R, B)], fill=(150, 160, 172), width=3)
    d.line([(L, T), (L, B)], fill=(150, 160, 172), width=3)
    d.text((L + 10, T - 56), ylabel, font=F(28), fill=GREY, anchor="lm")
    return pw, ph


def bar_chart(name, labels, values, ylabel, colors, ymax):
    img, d, L, R, T, B = frame()
    pw, ph = axes(d, L, R, T, B, ymax, ylabel, labels, labels)
    bw = pw / len(values) * 0.55
    for i, v in enumerate(values):
        cx = L + pw * (i + 0.5) / len(values)
        h = ph * v / ymax
        d.rectangle([cx - bw / 2, B - h, cx + bw / 2, B], fill=colors[i])
        d.text((cx, B - h - 10), "%.4f" % v, font=F(24), fill=NAVY, anchor="mb")
    img.save(os.path.join(OUT, name))


def line_chart(name, xs, series, ylabel, xlabel, ymax):
    img, d, L, R, T, B = frame()
    pw, ph = axes(d, L, R, T, B, ymax, ylabel, [str(x) for x in xs], [str(x) for x in xs])
    for label, ys, color in series:
        pts = [(L + pw * (i + 0.5) / len(xs), B - ph * min(y / ymax, 1.0)) for i, y in enumerate(ys)]
        d.line(pts, fill=color, width=6, joint="curve")
        for p in pts:
            d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=color)
    d.text((L + pw / 2, B + 70), xlabel, font=F(28), fill=NAVY, anchor="ma")
    lx = L + 420
    for label, _, color in series:
        d.line([(lx, T - 54), (lx + 40, T - 54)], fill=color, width=8)
        d.text((lx + 50, T - 54), label, font=F(26), fill=NAVY, anchor="lm")
        lx += 90 + len(label) * 28
    img.save(os.path.join(OUT, name))


def main():
    os.makedirs(OUT, exist_ok=True)
    test = read_csv("common_reference.csv")
    bar_chart("window_rmse.png", ["%s@%s" % (r["trajectory"].replace("test_", ""), r["start"]) for r in test],
              [float(r["rmse_mean"]) for r in test], "RMSE",
              [BLUE if r["trajectory"] == "test_a2" else ORANGE for r in test], 0.14)
    rel = read_csv("reliability_error_growth.csv")
    hs = [int(r["horizon"]) for r in rel]
    growth_series = [
        ("POD-AR", [float(r["rmse_mean"]) for r in rel], BLUE),
        ("重复最后一帧", [float(r["persistence_rmse"]) for r in rel], ORANGE),
    ]
    if rel and "ensemble_spread" in rel[0]:
        growth_series.append(
            ("20 成员离散度", [float(r["ensemble_spread"]) for r in rel], TEAL)
        )
    line_chart(
        "error_growth.png", hs, growth_series,
        "RMSE / 集成离散度", "预测步长（Δt*=0.1）", 0.24
    )
    trans = read_csv("transition_eval.csv")
    keys = ["reverse_a0", "reverse_a3"]
    vals = [sum(float(r["rmse_mean"]) for r in trans if r["trajectory"] == k) /
            len([r for r in trans if r["trajectory"] == k]) for k in keys]
    pers = [sum(float(r["persistence_rmse_mean"]) for r in trans if r["trajectory"] == k) /
            len([r for r in trans if r["trajectory"] == k]) for k in keys]
    img, d, L, R, T, B = frame()
    ymax = 0.24
    pw, ph = axes(d, L, R, T, B, ymax, "RMSE", keys, keys)
    for i, k in enumerate(keys):
        cx = L + pw * (i + 0.5) / len(keys)
        for j, (v, c) in enumerate([(vals[i], BLUE), (pers[i], ORANGE)]):
            x0 = cx + (j - 0.5) * 0.5 * pw / len(keys) * 0.5
            h = ph * v / ymax
            d.rectangle([x0 - 40, B - h, x0 + 40, B], fill=c)
            d.text((x0, B - h - 8), "%.4f" % v, font=F(24), fill=NAVY, anchor="mb")
    img.save(os.path.join(OUT, "transition_rmse.png"))
    sw = os.path.join(RES, "validation_sweep.csv")
    if os.path.exists(sw):
        rows = read_csv("validation_sweep.csv")
        alphas = [float(r["ridge_alpha"]) for r in rows]
        line_chart("validation_sweep.png", alphas,
                   [("validation RMSE", [float(r["validation_rmse"]) for r in rows], BLUE)],
                   "RMSE", "ridge alpha", 0.12)
    stable_path = os.path.join(RES, "stability_sweep.csv")
    if os.path.exists(stable_path):
        rows = [
            r for r in read_csv("stability_sweep.csv")
            if int(r["ar_order"]) == 3 and float(r["ridge_alpha"]) == 3.0
        ]
        modes = [int(r["n_modes"]) for r in rows]
        line_chart(
            "mode_sweep.png",
            modes,
            [("验证集 RMSE", [float(r["validation_rmse"]) for r in rows], BLUE)],
            "验证集 RMSE",
            "POD 模态数 k（AR(3), α=3.0）",
            0.14,
        )
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
