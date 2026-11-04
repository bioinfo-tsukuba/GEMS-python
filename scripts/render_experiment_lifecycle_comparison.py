"""Render the three lifecycle concepts for Issue #24; no runtime implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version


def render(font_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    image = Image.new("RGB", (2400, 1340), "#f5f7fb")
    draw = ImageDraw.Draw(image)
    ink = "#172b4d"
    connector = "#64748b"

    def text(cx: int, cy: int, value: str, size: int = 34, color: str = ink) -> None:
        draw.multiline_text(
            (cx, cy), value, font=ImageFont.truetype(str(font_path), size),
            fill=color, anchor="mm", align="center", spacing=12,
        )

    def box(
        cx: int, cy: int, width: int, height: int, value: str,
        fill: str = "#eaf1ff", size: int = 35,
    ) -> None:
        draw.rounded_rectangle(
            (cx - width // 2, cy - height // 2, cx + width // 2, cy + height // 2),
            radius=24, fill=fill, outline="#b9c7da", width=3,
        )
        text(cx, cy, value, size)

    def down(cx: int, y1: int, y2: int, *, dashed: bool = False) -> None:
        if dashed:
            for y in range(y1, y2 - 18, 24):
                draw.line((cx, y, cx, min(y + 12, y2 - 18)), fill=connector, width=5)
        else:
            draw.line((cx, y1, cx, y2 - 18), fill=connector, width=5)
        draw.polygon(((cx, y2), (cx - 12, y2 - 20), (cx + 12, y2 - 20)), fill=connector)

    text(1200, 75, "Experimentの終了と分割", 50)
    text(1200, 142, "従来の仕様と追加仕様の比較", 31, "#52637a")
    for left, right in ((40, 780), (830, 1570), (1620, 2360)):
        draw.rounded_rectangle((left, 210, right, 1180), radius=30, fill="white")

    text(410, 265, "1. 従来", 40)
    text(1200, 265, "2. 自己消去", 40)
    text(1990, 265, "3. 試料分割", 40)

    # Existing identity and history remain registered until explicit deletion.
    box(410, 410, 580, 150, "Experiment A\n履歴 H")
    box(410, 620, 580, 120, "状態遷移")
    down(410, 485, 560)
    draw.line(((700, 620), (742, 620), (742, 410), (720, 410)), fill=connector, width=5)
    draw.polygon(((700, 410), (720, 398), (720, 422)), fill=connector)
    box(410, 825, 580, 130, "終了後もAが残る", "#fff2df")
    down(410, 680, 760, dashed=True)
    box(410, 1030, 580, 120, "手動で削除", "#eef0f4")
    down(410, 890, 970, dashed=True)
    text(410, 1130, "同じUUIDで進行", 29, "#52637a")

    # Explicit completion removes the runtime object, preserving a separate record.
    box(1200, 410, 580, 150, "Experiment A\n履歴 H")
    box(1200, 620, 580, 120, "終了")
    down(1200, 485, 560)
    box(1200, 825, 580, 130, "Aを自動削除", "#e7f6eb")
    down(1200, 680, 760)
    box(1200, 1030, 580, 120, "終了記録・履歴 H", "#eef0f4")
    down(1200, 890, 970)
    text(1200, 1130, "子は作らない", 29, "#52637a")

    # H is immutable provenance; H1/H2 belong to the separate child Experiments.
    box(1990, 410, 580, 150, "親 Experiment P\n履歴 H")
    box(1990, 620, 580, 120, "細胞の取り分け等")
    down(1990, 485, 560)
    draw.line(((1990, 680), (1990, 715)), fill=connector, width=5)
    draw.line(((1815, 715), (2165, 715)), fill=connector, width=5)
    down(1815, 715, 770)
    down(2165, 715, 770)
    box(1815, 855, 300, 170, "子 C1\n出自 H\n観測 H1", "#e7f6eb", 33)
    box(2165, 855, 300, 170, "子 C2\n出自 H\n観測 H2", "#e7f6eb", 33)
    text(1990, 975, "独立したExperiment", 31, "#3a6c45")
    box(1990, 1060, 580, 110, "Pを削除・履歴 Hを保存", "#eef0f4", 31)
    # This final box describes the atomic split, not a later operation after child execution.
    text(1990, 1145, "親削除と全子登録を一括確定", 28, "#52637a")

    text(1200, 1230, "自己消去＝管理対象からの削除。履歴・出自の記録は残す。", 32, "#52637a")
    text(1200, 1290, "H：分割前の親履歴 ／ H1・H2：分割後の各子の観測履歴", 30, "#52637a")
    png = output / "comparison.png"
    image.save(png)
    metadata = {
        "issue": 24,
        "kind": "conceptual design; not implemented behavior",
        "font_sha256": hashlib.sha256(font_path.read_bytes()).hexdigest(),
        "image_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
        "width": image.width,
        "height": image.height,
        "python": platform.python_version(),
        "pillow": pillow_version,
        "labels": {
            "H": "Parent history retained as provenance, or the existing experiment history.",
            "H1": "Child C1 observations after splitting.",
            "H2": "Child C2 observations after splitting.",
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, required=True, help="Japanese-capable font file")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if not args.font.is_file():
        parser.error(f"font is not a file: {args.font}")
    render(args.font, args.output)


if __name__ == "__main__":
    main()
