from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable
from xml.sax.saxutils import escape

import openpyxl


MODELS = [
    "ResNet-56",
    "SE-PreResNet-110",
    "SE-ResNet-20",
    "DenseNet-BC-40",
    "RoR-3-110",
    "DIA-PreResNet-56",
]

TARGET_HEADERS = [
    "ResNet-56",
    "SE-PreResNet-110",
    "SE-ResNet-20",
    "DenseNet-BC-40",
    "RoR-3-110",
    "DIA-PreResNet-56",
]

BETA_LEVELS = ["MIFGSM", 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
N_LEVELS = [0, 5, 10, 15, 20]
METHOD_LEVELS = ["FGSM", "IFGSM", "MIFGSM"]

BETA_START_ROW = 4
BETA_START_COL = 4
BETA_LOSS_START_ROW = 67
BETA_ROW_STEP = len(BETA_LEVELS) + 1

N_START_ROW = 4
N_START_COL = 14
N_LOSS_START_ROW = 67
N_ROW_STEP = len(N_LEVELS) + 1

METHOD_START_ROW = 4
METHOD_START_COL = 24
METHOD_LOSS_START_ROW = 67
METHOD_ROW_STEP = len(METHOD_LEVELS) + 1

OUT_DIR = "paper_figures_svg"

ACCENT_RED = "#8f2d2d"
ACCENT_BLUE = "#1f4e79"
ACCENT_GOLD = "#8a6a2f"
ACCENT_GREEN = "#355e3b"
ACCENT_TEAL = "#2d6a73"
ACCENT_SLATE = "#4c566a"
GRID_COLOR = "#e6e9ef"
AXIS_COLOR = "#6b7280"
FRAME_COLOR = "#cfd6df"
TEXT_COLOR = "#1f2937"
PANEL_BG = "#ffffff"


@dataclass
class Series:
    name: str
    values: list[float]
    color: str
    stroke_width: float = 3.0
    dash: str | None = None


class SvgCanvas:
    def __init__(self, width: int, height: int, background: str = "#ffffff") -> None:
        self.width = width
        self.height = height
        self._parts = [
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}">'
            ),
            (
                '<style>'
                'text{font-family:"Times New Roman","SimSun","STSong","Microsoft YaHei",serif;fill:#1f2937}'
                '.title{font-size:22px;font-weight:700}'
                '.subtitle{font-size:12px;fill:#6b7280}'
                '.axis{font-size:11px;fill:#6b7280}'
                '.legend{font-size:11px}'
                '.cell-label{font-size:10px;font-weight:600}'
                "</style>"
            ),
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="{background}"/>',
        ]

    def add(self, element: str) -> None:
        self._parts.append(element)

    def text(
        self,
        x: float,
        y: float,
        text: str,
        css_class: str = "",
        anchor: str = "start",
        fill: str | None = None,
    ) -> None:
        attrs = [f'x="{x:.2f}"', f'y="{y:.2f}"', f'text-anchor="{anchor}"']
        if css_class:
            attrs.append(f'class="{css_class}"')
        if fill:
            attrs.append(f'fill="{fill}"')
        self.add(f"<text {' '.join(attrs)}>{escape(text)}</text>")

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = "#cbd2d9",
        stroke_width: float = 1.0,
        dash: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"{dash_attr}/>'
        )

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 1.0,
        rx: float = 0.0,
    ) -> None:
        stroke_attr = ""
        if stroke:
            stroke_attr = f' stroke="{stroke}" stroke-width="{stroke_width:.2f}"'
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            f'fill="{fill}" rx="{rx:.2f}"{stroke_attr}/>'
        )

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        fill: str,
        stroke: str = "#ffffff",
        stroke_width: float = 1.5,
    ) -> None:
        self.add(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"/>'
        )

    def polyline(
        self,
        points: Iterable[tuple[float, float]],
        stroke: str,
        stroke_width: float = 3.0,
        fill: str = "none",
        dash: str | None = None,
    ) -> None:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.2f}" stroke-linecap="round" '
            f'stroke-linejoin="round"{dash_attr}/>'
        )

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self._parts + ["</svg>"]), encoding="utf-8")


def find_workbook() -> Path:
    matches = [p for p in Path(".").glob("*.xlsx") if not p.name.startswith("~$")]
    if not matches:
        raise FileNotFoundError("No .xlsx workbook found in the current directory.")
    return matches[0]


def read_table(
    ws,
    start_row: int,
    start_col: int,
    row_keys: list,
    row_step: int,
) -> dict[str, dict[object, list[float]]]:
    result: dict[str, dict[object, list[float]]] = {}
    for model_idx, model in enumerate(MODELS):
        block_start = start_row + model_idx * row_step
        result[model] = {}
        for offset, key in enumerate(row_keys):
            row = block_start + offset
            values = [float(ws.cell(row=row, column=start_col + 2 + i).value) for i in range(6)]
            result[model][key] = values
    return result


def compute_means(table: dict[str, dict[object, list[float]]], row_keys: list) -> tuple[list[float], list[float]]:
    whitebox = []
    blackbox = []
    for key in row_keys:
        white_vals = []
        black_vals = []
        for model_idx, model in enumerate(MODELS):
            vals = table[model][key]
            white_vals.append(vals[model_idx])
            black_vals.extend(v for idx, v in enumerate(vals) if idx != model_idx)
        whitebox.append(mean(white_vals))
        blackbox.append(mean(black_vals))
    return whitebox, blackbox


def compute_source_blackbox(table: dict[str, dict[object, list[float]]], row_keys: list) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    for model_idx, model in enumerate(MODELS):
        series[model] = []
        for key in row_keys:
            vals = table[model][key]
            series[model].append(mean(v for idx, v in enumerate(vals) if idx != model_idx))
    return series


def nice_ticks(values: list[float], tick_count: int = 5) -> list[float]:
    v_min = min(values)
    v_max = max(values)
    if abs(v_max - v_min) < 1e-9:
        return [v_min]
    span = v_max - v_min
    pad = span * 0.08
    lo = v_min - pad
    hi = v_max + pad
    step = (hi - lo) / max(tick_count - 1, 1)
    return [lo + step * i for i in range(tick_count)]


def map_x(index: int, count: int, left: float, width: float) -> float:
    if count == 1:
        return left + width / 2
    return left + index * (width / (count - 1))


def map_y(value: float, ticks: list[float], top: float, height: float) -> float:
    lo = min(ticks)
    hi = max(ticks)
    if abs(hi - lo) < 1e-9:
        return top + height / 2
    ratio = (value - lo) / (hi - lo)
    return top + height - ratio * height


def draw_panel_frame(canvas: SvgCanvas, x: float, y: float, width: float, height: float) -> None:
    canvas.rect(x, y, width, height, fill=PANEL_BG, stroke=FRAME_COLOR, stroke_width=1.1, rx=10)


def draw_legend_box(canvas: SvgCanvas, x: float, y: float, width: float, height: float) -> None:
    canvas.rect(x, y, width, height, fill="#ffffff", stroke=FRAME_COLOR, stroke_width=0.9, rx=8)


def draw_line_panel(
    canvas: SvgCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    x_labels: list[str],
    series_list: list[Series],
    y_formatter: str,
) -> None:
    draw_panel_frame(canvas, x, y, width, height)
    canvas.text(x + 20, y + 28, title, css_class="title")

    plot_x = x + 64
    plot_y = y + 62
    plot_w = width - 92
    plot_h = height - 120

    values = [value for series in series_list for value in series.values]
    ticks = nice_ticks(values, 5)

    for tick in ticks:
        ty = map_y(tick, ticks, plot_y, plot_h)
        canvas.line(plot_x, ty, plot_x + plot_w, ty, stroke=GRID_COLOR)
        canvas.text(plot_x - 10, ty + 4, y_formatter.format(tick), css_class="axis", anchor="end")

    canvas.line(plot_x, plot_y, plot_x, plot_y + plot_h, stroke=AXIS_COLOR, stroke_width=1.2)
    canvas.line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, stroke=AXIS_COLOR, stroke_width=1.2)

    for idx, label in enumerate(x_labels):
        tx = map_x(idx, len(x_labels), plot_x, plot_w)
        canvas.line(tx, plot_y + plot_h, tx, plot_y + plot_h + 5, stroke=AXIS_COLOR, stroke_width=1.0)
        canvas.text(tx, plot_y + plot_h + 21, label, css_class="axis", anchor="middle")

    for series in series_list:
        points = [
            (map_x(idx, len(x_labels), plot_x, plot_w), map_y(value, ticks, plot_y, plot_h))
            for idx, value in enumerate(series.values)
        ]
        canvas.polyline(points, stroke=series.color, stroke_width=series.stroke_width, dash=series.dash)
        for px, py in points:
            canvas.circle(px, py, 3.6, fill=series.color)

    legend_x = x + width - 178
    legend_y = y + 16
    legend_h = 18 + len(series_list) * 20
    draw_legend_box(canvas, legend_x - 14, legend_y - 12, 154, legend_h)
    for idx, series in enumerate(series_list):
        ly = legend_y + idx * 22
        canvas.line(legend_x, ly, legend_x + 24, ly, stroke=series.color, stroke_width=series.stroke_width, dash=series.dash)
        canvas.circle(legend_x + 12, ly, 3.6, fill=series.color)
        canvas.text(legend_x + 34, ly + 4, series.name, css_class="legend")


def draw_bar_panel(
    canvas: SvgCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    labels: list[str],
    values: list[float],
    fill: str,
    y_formatter: str,
) -> None:
    draw_panel_frame(canvas, x, y, width, height)
    canvas.text(x + 20, y + 28, title, css_class="title")

    plot_x = x + 64
    plot_y = y + 62
    plot_w = width - 92
    plot_h = height - 120
    ticks = nice_ticks(values, 5)

    for tick in ticks:
        ty = map_y(tick, ticks, plot_y, plot_h)
        canvas.line(plot_x, ty, plot_x + plot_w, ty, stroke=GRID_COLOR)
        canvas.text(plot_x - 10, ty + 4, y_formatter.format(tick), css_class="axis", anchor="end")

    canvas.line(plot_x, plot_y, plot_x, plot_y + plot_h, stroke=AXIS_COLOR, stroke_width=1.2)
    canvas.line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, stroke=AXIS_COLOR, stroke_width=1.2)

    slot_w = plot_w / len(labels)
    bar_w = slot_w * 0.48
    for idx, (label, value) in enumerate(zip(labels, values)):
        cx = plot_x + slot_w * idx + slot_w / 2
        bx = cx - bar_w / 2
        by = map_y(value, ticks, plot_y, plot_h)
        bar_fill = blend_color("#d7e3f2", fill, idx / max(len(labels) - 1, 1))
        canvas.rect(bx, by, bar_w, plot_y + plot_h - by, fill=bar_fill, stroke=fill, stroke_width=0.8, rx=6)
        canvas.text(cx, plot_y + plot_h + 21, label, css_class="axis", anchor="middle")
        canvas.text(cx, by - 8, y_formatter.format(value), css_class="axis", anchor="middle")


def blend_color(hex_lo: str, hex_hi: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    lo = [int(hex_lo[i : i + 2], 16) for i in (1, 3, 5)]
    hi = [int(hex_hi[i : i + 2], 16) for i in (1, 3, 5)]
    rgb = [round(lo[i] + (hi[i] - lo[i]) * ratio) for i in range(3)]
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def draw_heatmap_panel(
    canvas: SvgCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    matrix: list[list[float]],
    low_color: str,
    high_color: str,
    formatter: str,
) -> None:
    draw_panel_frame(canvas, x, y, width, height)
    canvas.text(x + 20, y + 28, title, css_class="title")

    grid_x = x + 128
    grid_y = y + 72
    cell_w = (width - 156) / len(col_labels)
    cell_h = (height - 110) / len(row_labels)

    flat = [item for row in matrix for item in row]
    lo = min(flat)
    hi = max(flat)
    span = hi - lo if hi != lo else 1.0

    for idx, label in enumerate(col_labels):
        canvas.text(grid_x + idx * cell_w + cell_w / 2, grid_y - 12, label, css_class="axis", anchor="middle")

    for row_idx, row_label in enumerate(row_labels):
        cy = grid_y + row_idx * cell_h + cell_h / 2 + 4
        canvas.text(x + 112, cy, row_label, css_class="axis", anchor="end")
        for col_idx, value in enumerate(matrix[row_idx]):
            ratio = (value - lo) / span
            fill = blend_color(low_color, high_color, ratio)
            cx = grid_x + col_idx * cell_w
            cy_top = grid_y + row_idx * cell_h
            canvas.rect(cx, cy_top, cell_w - 3, cell_h - 3, fill=fill, stroke="#ffffff", stroke_width=1.0, rx=4)
            text_fill = "#0f1720" if ratio < 0.55 else "#fffdf8"
            canvas.text(
                cx + cell_w / 2 - 1.5,
                cy_top + cell_h / 2 + 4,
                formatter.format(value),
                css_class="cell-label",
                anchor="middle",
                fill=text_fill,
            )


def render_beta_figure(beta_acc, beta_loss, out_dir: Path) -> Path:
    acc_white, acc_black = compute_means(beta_acc, BETA_LEVELS)
    loss_white, loss_black = compute_means(beta_loss, BETA_LEVELS)
    labels = ["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4"]

    canvas = SvgCanvas(1400, 560)
    canvas.text(60, 48, "Beta Ablation", css_class="title")
    canvas.text(60, 72, "Baseline 0 denotes MI-FGSM (no noisy neighborhood sampling).", css_class="subtitle")

    draw_line_panel(
        canvas,
        48,
        100,
        630,
        410,
        "Mean Accuracy",
        labels,
        [
            Series("Black-box mean", acc_black, ACCENT_RED),
            Series("White-box mean", acc_white, ACCENT_BLUE, dash="7 5"),
        ],
        "{:.3f}",
    )
    draw_line_panel(
        canvas,
        722,
        100,
        630,
        410,
        "Mean Loss",
        labels,
        [
            Series("Black-box mean", loss_black, ACCENT_BLUE),
            Series("White-box mean", loss_white, ACCENT_GOLD, dash="7 5"),
        ],
        "{:.2f}",
    )

    path = out_dir / "beta_ablation.svg"
    canvas.save(path)
    return path


def render_n_figure(n_acc, n_loss, out_dir: Path) -> Path:
    acc_white, acc_black = compute_means(n_acc, N_LEVELS)
    loss_white, loss_black = compute_means(n_loss, N_LEVELS)
    labels = [str(v) for v in N_LEVELS]

    canvas = SvgCanvas(1400, 560)
    canvas.text(60, 48, "N Ablation", css_class="title")
    canvas.text(60, 72, "Beta is fixed at 2.5. N = 0 falls back to plain MI-FGSM.", css_class="subtitle")

    draw_line_panel(
        canvas,
        48,
        100,
        630,
        410,
        "Mean Accuracy",
        labels,
        [
            Series("Black-box mean", acc_black, ACCENT_RED),
            Series("White-box mean", acc_white, ACCENT_BLUE, dash="7 5"),
        ],
        "{:.3f}",
    )
    draw_line_panel(
        canvas,
        722,
        100,
        630,
        410,
        "Mean Loss",
        labels,
        [
            Series("Black-box mean", loss_black, ACCENT_BLUE),
            Series("White-box mean", loss_white, ACCENT_GOLD, dash="7 5"),
        ],
        "{:.2f}",
    )

    path = out_dir / "n_ablation.svg"
    canvas.save(path)
    return path


def render_method_figure(method_acc, method_loss, out_dir: Path) -> Path:
    _, acc_black = compute_means(method_acc, METHOD_LEVELS)
    _, loss_black = compute_means(method_loss, METHOD_LEVELS)

    canvas = SvgCanvas(1400, 560)
    canvas.text(60, 48, "Attack Method Comparison", css_class="title")
    canvas.text(60, 72, "Bars show mean black-box transfer performance across six source models.", css_class="subtitle")

    draw_bar_panel(
        canvas,
        48,
        100,
        630,
        410,
        "Black-box Mean Accuracy",
        METHOD_LEVELS,
        acc_black,
        fill=ACCENT_RED,
        y_formatter="{:.3f}",
    )
    draw_bar_panel(
        canvas,
        722,
        100,
        630,
        410,
        "Black-box Mean Loss",
        METHOD_LEVELS,
        loss_black,
        fill=ACCENT_BLUE,
        y_formatter="{:.2f}",
    )

    path = out_dir / "method_comparison.svg"
    canvas.save(path)
    return path


def render_sourcewise_method_figure(method_acc, method_loss, out_dir: Path) -> Path:
    acc_by_source = compute_source_blackbox(method_acc, METHOD_LEVELS)
    loss_by_source = compute_source_blackbox(method_loss, METHOD_LEVELS)

    acc_matrix = [acc_by_source[model] for model in MODELS]
    loss_matrix = [loss_by_source[model] for model in MODELS]

    short_rows = ["R56", "SE-PR110", "SE-R20", "Dense40", "RoR110", "DIA-PR56"]

    canvas = SvgCanvas(1400, 610)
    canvas.text(60, 48, "Source-wise Method Comparison", css_class="title")
    canvas.text(60, 72, "Each cell is the mean black-box transfer result for one source model.", css_class="subtitle")

    draw_heatmap_panel(
        canvas,
        48,
        110,
        630,
        450,
        "Accuracy Heatmap",
        short_rows,
        METHOD_LEVELS,
        acc_matrix,
        low_color="#fbe3e3",
        high_color=ACCENT_RED,
        formatter="{:.3f}",
    )
    draw_heatmap_panel(
        canvas,
        722,
        110,
        630,
        450,
        "Loss Heatmap",
        short_rows,
        METHOD_LEVELS,
        loss_matrix,
        low_color="#e3edf8",
        high_color=ACCENT_BLUE,
        formatter="{:.2f}",
    )

    path = out_dir / "method_sourcewise.svg"
    canvas.save(path)
    return path


def render_beta_sourcewise_figure(beta_acc, out_dir: Path) -> Path:
    source_series = compute_source_blackbox(beta_acc, BETA_LEVELS)
    labels = ["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4"]
    colors = [ACCENT_RED, ACCENT_BLUE, ACCENT_GOLD, ACCENT_GREEN, ACCENT_TEAL, ACCENT_SLATE]

    canvas = SvgCanvas(1400, 620)
    canvas.text(60, 48, "Beta Ablation by Source Model", css_class="title")
    canvas.text(60, 72, "Lines are mean black-box accuracies for each source model.", css_class="subtitle")

    draw_line_panel(
        canvas,
        48,
        100,
        1304,
        470,
        "Source-wise Black-box Mean Accuracy",
        labels,
        [
            Series(model, source_series[model], colors[idx])
            for idx, model in enumerate(MODELS)
        ],
        "{:.3f}",
    )

    path = out_dir / "beta_sourcewise.svg"
    canvas.save(path)
    return path


def main() -> None:
    workbook_path = find_workbook()
    wb = openpyxl.load_workbook(workbook_path, data_only=True)
    ws = wb["Sheet1"]

    beta_acc = read_table(ws, BETA_START_ROW, BETA_START_COL, BETA_LEVELS, BETA_ROW_STEP)
    beta_loss = read_table(ws, BETA_LOSS_START_ROW, BETA_START_COL, BETA_LEVELS, BETA_ROW_STEP)
    n_acc = read_table(ws, N_START_ROW, N_START_COL, N_LEVELS, N_ROW_STEP)
    n_loss = read_table(ws, N_LOSS_START_ROW, N_START_COL, N_LEVELS, N_ROW_STEP)
    method_acc = read_table(ws, METHOD_START_ROW, METHOD_START_COL, METHOD_LEVELS, METHOD_ROW_STEP)
    method_loss = read_table(ws, METHOD_LOSS_START_ROW, METHOD_START_COL, METHOD_LEVELS, METHOD_ROW_STEP)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(exist_ok=True)

    outputs = [
        render_beta_figure(beta_acc, beta_loss, out_dir),
        render_beta_sourcewise_figure(beta_acc, out_dir),
        render_n_figure(n_acc, n_loss, out_dir),
        render_method_figure(method_acc, method_loss, out_dir),
        render_sourcewise_method_figure(method_acc, method_loss, out_dir),
    ]

    print("Workbook:", workbook_path.name)
    for path in outputs:
        print("Generated:", path.as_posix())


if __name__ == "__main__":
    main()
