"""Create lightweight PNG charts from the checked-in benchmark CSV files."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks"
OUT = ROOT / "images"


def font(size: int, bold: bool = False):
    candidates = (
        [
            Path("C:/Windows/Fonts/segoeuib.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
        ]
        if bold
        else [
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


TITLE = font(30, True)
SUBTITLE = font(17)
LABEL = font(18, True)
SMALL = font(15)
TINY = font(13)


def median(values):
    return statistics.median([float(value) for value in values])


def read_rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def grouped(rows, key_fields, value_field):
    values = defaultdict(list)
    for row in rows:
        values[tuple(row[field] for field in key_fields)].append(row[value_field])
    return {key: median(items) for key, items in values.items()}


def draw_base(title: str, subtitle: str, ylabel: str):
    image = Image.new("RGB", (1400, 820), "#10141c")
    draw = ImageDraw.Draw(image)
    draw.text((70, 34), title, fill="#f3f6fb", font=TITLE)
    draw.text((72, 82), subtitle, fill="#aeb9c9", font=SUBTITLE)
    left, top, right, bottom = 120, 150, 1330, 700
    draw.line((left, top, left, bottom), fill="#dce5f2", width=2)
    draw.line((left, bottom, right, bottom), fill="#dce5f2", width=2)
    ylabel_image = Image.new("RGBA", (500, 50), (0, 0, 0, 0))
    ylabel_draw = ImageDraw.Draw(ylabel_image)
    ylabel_draw.text((0, 0), ylabel, fill="#dce5f2", font=LABEL)
    ylabel_image = ylabel_image.rotate(90, expand=True)
    image.paste(ylabel_image, (27, 225), ylabel_image)
    return image, draw, (left, top, right, bottom)


def log_mapper(values, start, end, logarithmic=True):
    if logarithmic:
        low = math.log10(min(values))
        high = math.log10(max(values))
        return lambda value: start + (math.log10(value) - low) / (high - low) * (end - start)
    low, high = min(values), max(values)
    return lambda value: start + (value - low) / (high - low) * (end - start)


def add_legend(draw, entries, x=980, y=175):
    for index, (label, color) in enumerate(entries):
        yy = y + index * 32
        draw.line((x, yy + 10, x + 32, yy + 10), fill=color, width=5)
        draw.text((x + 44, yy), label, fill="#e9eef7", font=SMALL)


def scaling_chart():
    rows = read_rows(DATA / "results_agent_cuda.csv")
    medians = grouped(rows, ["mode", "objects"], "steps_per_second")
    objects = sorted({int(key[1]) for key in medians})
    image, draw, (left, top, right, bottom) = draw_base(
        "CUDA flocking scaling",
        "Median of 5 CUDA-event samples per point; Release, RTX 3060 Laptop GPU; OpenGL excluded",
        "steps / second",
    )
    x = log_mapper(objects, left + 35, right - 40)
    y_values = [value for value in medians.values() if value > 0]
    y = log_mapper(y_values, bottom - 25, top + 30)
    colors = {"naive": "#ff8b8b", "scattered": "#71c7ff", "coherent": "#71e6a5"}
    labels = {"naive": "Naive", "scattered": "Scattered grid", "coherent": "Coherent grid"}

    for object_count in objects:
        xx = x(object_count)
        draw.line((xx, top, xx, bottom), fill="#273142", width=1)
        draw.text((xx - 24, bottom + 16), f"{object_count // 1000}k", fill="#c5cfdd", font=TINY)
    for tick in (10, 100, 1000, 10000):
        if min(y_values) <= tick <= max(y_values):
            yy = y(tick)
            draw.line((left, yy, right, yy), fill="#273142", width=1)
            draw.text((left - 72, yy - 9), f"{tick:,}", fill="#c5cfdd", font=TINY)
    for mode, color in colors.items():
        points = []
        for object_count in objects:
            value = medians.get((mode, str(object_count)))
            if value is not None:
                points.append((x(object_count), y(value)))
        if len(points) > 1:
            draw.line(points, fill=color, width=5)
        for xx, yy in points:
            draw.ellipse((xx - 6, yy - 6, xx + 6, yy + 6), fill=color)
    add_legend(draw, [(labels[key], value) for key, value in colors.items()])
    draw.text((610, 748), "boids (log scale)", fill="#dce5f2", font=LABEL)
    image.save(OUT / "performance_scaling.png")


def block_chart():
    medians = {}
    rows = read_rows(DATA / "blocksize_results_clean.csv")
    by_block = defaultdict(list)
    for row in rows:
        by_block[int(row["block_size"])].append(row["step_ms"])
    for block_size, values in by_block.items():
        medians[block_size] = median(values)
    blocks = sorted(medians)
    image, draw, (left, top, right, bottom) = draw_base(
        "Coherent grid block-size sweep",
        "50,000 boids; median CUDA-event step time across 7 samples; lower is better",
        "step time (ms)",
    )
    x = log_mapper(blocks, left + 45, right - 45, logarithmic=False)
    y_values = list(medians.values())
    y = log_mapper(y_values, bottom - 25, top + 30, logarithmic=False)
    points = [(x(block_size), y(medians[block_size])) for block_size in blocks]
    low = min(y_values)
    high = max(y_values)
    for index in range(5):
        tick = low + (high - low) * index / 4
        yy = y(tick)
        draw.line((left, yy, right, yy), fill="#273142", width=1)
        draw.text((left - 70, yy - 9), f"{tick:.3f}", fill="#c5cfdd", font=TINY)
    for block_size in blocks:
        xx = x(block_size)
        draw.line((xx, top, xx, bottom), fill="#273142", width=1)
        draw.text((xx - 16, bottom + 16), str(block_size), fill="#c5cfdd", font=TINY)
    draw.line(points, fill="#f7c66a", width=5)
    for xx, yy in points:
        draw.ellipse((xx - 7, yy - 7, xx + 7, yy + 7), fill="#f7c66a")
    for block_size, value in medians.items():
        draw.text((x(block_size) - 24, y(value) - 32), f"{value:.3f}", fill="#f7c66a", font=TINY)
    draw.text((650, 748), "threads per block", fill="#dce5f2", font=LABEL)
    image.save(OUT / "performance_block_size.png")


def runtime_chart():
    datasets = {
        "VISUALIZE=1": (DATA / "runtime_fps.csv", "#71c7ff"),
        "VISUALIZE=0": (DATA / "runtime_fps_novis.csv", "#71e6a5"),
    }
    medians = {}
    for label, (path, _color) in datasets.items():
        by_run = defaultdict(list)
        for row in read_rows(path):
            by_run[int(row["run"])].append(float(row["fps"]))
        medians[label] = {run: median(values) for run, values in by_run.items()}
    runs = sorted({run for values in medians.values() for run in values})
    image, draw, (left, top, right, bottom) = draw_base(
        "Interactive runtime with and without drawing",
        "Coherent grid, 10,000 boids; median window-title FPS over 3 launches",
        "FPS",
    )
    maximum = max(value for values in medians.values() for value in values.values()) * 1.2
    bar_width = 92
    group_gap = 58
    bar_gap = 12
    group_width = 2 * bar_width + bar_gap
    start = left + 140
    for run_index, run in enumerate(runs):
        group_x = start + run_index * (group_width + group_gap)
        draw.text((group_x + 48, bottom + 16), f"Run {run}", fill="#c5cfdd", font=SMALL)
        for series_index, (label, (_path, color)) in enumerate(datasets.items()):
            value = medians[label][run]
            xx = group_x + series_index * (bar_width + bar_gap)
            height = (value / maximum) * (bottom - top)
            draw.rectangle((xx, bottom - height, xx + bar_width, bottom), fill=color)
            draw.text((xx + 17, bottom - height - 35), f"{value:.0f}", fill=color, font=TINY)
    add_legend(draw, [(label, color) for label, (_path, color) in datasets.items()], x=1010, y=175)
    draw.text((570, 748), "repeat", fill="#dce5f2", font=LABEL)
    image.save(OUT / "performance_runtime_fps.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    scaling_chart()
    block_chart()
    runtime_chart()
    print("Wrote performance_scaling.png, performance_block_size.png, and performance_runtime_fps.png")
