"""產生 GitHub README 與 Social Preview 使用的可重現圖片。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "social-preview.png"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _candles(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    """繪製固定 seed 的示意 K 線，不使用或暗示真實績效。"""
    left, top, right, bottom = box
    rng = np.random.default_rng(20260917)
    returns = rng.normal(0.0004, 0.012, 42) + 0.003 * np.sin(np.arange(42) / 4)
    close = 100 * np.exp(np.cumsum(returns))
    open_price = np.r_[close[0], close[:-1]]
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.002, 0.012, 42))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.002, 0.012, 42))
    minimum, maximum = float(low.min()), float(high.max())

    for line in range(5):
        y = top + (bottom - top) * line // 4
        draw.line((left, y, right, y), fill="#22303d", width=1)
    step = (right - left) / len(close)
    candle_width = max(int(step * 0.55), 3)
    scale = (bottom - top) / max(maximum - minimum, 1e-9)
    for index, (opening, highest, lowest, closing) in enumerate(
        zip(open_price, high, low, close, strict=True)
    ):
        x = int(left + step * (index + 0.5))
        y_high = int(bottom - (highest - minimum) * scale)
        y_low = int(bottom - (lowest - minimum) * scale)
        y_open = int(bottom - (opening - minimum) * scale)
        y_close = int(bottom - (closing - minimum) * scale)
        color = "#36d399" if closing >= opening else "#ff6b72"
        draw.line((x, y_high, x, y_low), fill=color, width=2)
        draw.rectangle(
            (x - candle_width // 2, min(y_open, y_close), x + candle_width // 2, max(y_open, y_close) + 1),
            fill=color,
        )


def main() -> int:
    canvas = Image.new("RGB", (1280, 640), "#0b1118")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 12, 640), fill="#36c5d8")
    draw.text((54, 42), "BTC AI QUANT RESEARCH", font=_font(42, bold=True), fill="#f2f7fb")
    draw.text(
        (56, 96),
        "MULTI-TIMEFRAME TRANSFORMER V3  /  SAC  /  INDEPENDENT RISK ENGINE",
        font=_font(18, bold=True),
        fill="#8fa5b8",
    )

    chart_box = (56, 158, 780, 474)
    draw.rounded_rectangle((40, 136, 798, 500), radius=8, fill="#101923", outline="#273645")
    draw.text((58, 150), "CAUSAL 15m MARKET VIEW", font=_font(16, bold=True), fill="#9fb4c6")
    _candles(draw, chart_box)

    panels = [
        ("ANALYST", "Transformer V3", "return / volatility / regime", "#36c5d8"),
        ("TRADER", "SAC exposure", "long / short / close / hold", "#8b9cff"),
        ("RISK", "Hard safety gates", "cost / drawdown / kill switch", "#e9b949"),
    ]
    y = 138
    for label, title, detail, color in panels:
        draw.rounded_rectangle((830, y, 1238, y + 104), radius=8, fill="#121c27", outline="#2a3948")
        draw.text((852, y + 15), label, font=_font(14, bold=True), fill=color)
        draw.text((852, y + 38), title, font=_font(23, bold=True), fill="#f2f7fb")
        draw.text((852, y + 72), detail, font=_font(15), fill="#98aabc")
        y += 122

    stages = ["MARKET DATA", "CAUSAL FEATURES", "TRANSFORMER", "SAC", "RISK + EXECUTION"]
    x = 48
    for index, stage in enumerate(stages):
        width = 210 if index < 4 else 236
        draw.rounded_rectangle((x, 544, x + width, 598), radius=7, fill="#17222d", outline="#314353")
        draw.text((x + 16, 560), stage, font=_font(14, bold=True), fill="#dce8f1")
        x += width + 12
    draw.text((1015, 45), "RESEARCH ONLY", font=_font(15, bold=True), fill="#e9b949")
    draw.text((1090, 72), "LIVE LOCKED", font=_font(13, bold=True), fill="#ff6b72")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, optimize=True)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
