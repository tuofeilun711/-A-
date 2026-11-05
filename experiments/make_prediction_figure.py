"""Generate the field-comparison figure used in the defence deck."""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PYLIBS = os.path.join(ROOT, ".pylibs")
if os.path.isdir(PYLIBS):
    sys.path.insert(0, PYLIBS)

from data.h5_loader import BridgeFlowData  # noqa: E402
from models.pod_ar import PODFlowPredictor  # noqa: E402


FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
NAVY = (20, 38, 56)
GREY = (105, 118, 132)


def font(size):
    return ImageFont.truetype(FONT_PATH, size, index=0)


def vorticity(field, x, z):
    w, u = field[0], field[1]
    dwdx = np.gradient(w, x, axis=1)
    dudz = np.gradient(u, z, axis=0)
    return dwdx - dudz


def vorticity_rgb(values, mask, limit=15.0):
    values = np.clip(values, -limit, limit) / limit
    rgb = np.ones(values.shape + (3,), dtype=np.uint8) * 255
    positive = values > 0
    negative = values < 0
    magnitude = np.abs(values)
    rgb[..., 0] = np.where(positive, 255 - 10 * magnitude, 255 - 210 * magnitude)
    rgb[..., 1] = np.where(positive, 255 - 210 * magnitude, 255 - 145 * magnitude)
    rgb[..., 2] = np.where(negative, 255 - 10 * magnitude, 255 - 205 * magnitude)
    rgb[~mask] = np.array([55, 55, 55], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB").resize((760, 380), Image.Resampling.BILINEAR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5", default=os.path.join(ROOT, "bridge_a_flow_light.h5")
    )
    parser.add_argument("--trajectory", default="test_a2")
    parser.add_argument("--start", type=int, default=199)
    args = parser.parse_args()

    bd = BridgeFlowData(args.h5).load()
    mask = bd.get_fluid_mask()
    train = [
        bd.normalize(data["field"])
        for data in bd.trajectories["train"].values()
    ]
    model = PODFlowPredictor(50, 3, 3.0).fit(train, mask)

    trajectory = bd.trajectories["test"][args.trajectory]
    normalized = bd.normalize(trajectory["field"])
    history = normalized[args.start - 9 : args.start + 1]
    prediction = bd.denormalize(model.predict(history, 20))
    truth = trajectory["field"][args.start + 1 : args.start + 21]
    x = np.asarray(trajectory["x"])
    z = np.asarray(trajectory["z"])

    panels = [
        ("真值  m=1", truth[0]),
        ("预测  m=1", prediction[0]),
        ("真值  m=20", truth[-1]),
        ("预测  m=20", prediction[-1]),
    ]
    canvas = Image.new("RGB", (1700, 1050), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (80, 32),
        f"{args.trajectory}，起点帧 {args.start}：流场预测对比",
        font=font(38),
        fill=NAVY,
    )
    draw.text(
        (80, 84),
        "涡量 ω = ∂w/∂x − ∂u/∂z；红色为顺时针，蓝色为逆时针",
        font=font(23),
        fill=GREY,
    )

    positions = [(80, 155), (870, 155), (80, 590), (870, 590)]
    for (label, field), (left, top) in zip(panels, positions):
        panel = vorticity_rgb(vorticity(field, x, z), mask)
        canvas.paste(panel, (left, top))
        draw.text((left, top - 44), label, font=font(29), fill=NAVY)

    draw.text(
        (80, 1002),
        "色标截断范围：ω ∈ [−15, 15]；黑色区域为固体边界",
        font=font(20),
        fill=GREY,
    )
    output = os.path.join(ROOT, "figures", "prediction_example.png")
    canvas.save(output)
    print(output)


if __name__ == "__main__":
    main()
