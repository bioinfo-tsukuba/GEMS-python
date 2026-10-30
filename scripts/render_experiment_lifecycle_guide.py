"""Render the reader guide from recorded Issue #24 inputs, without a solver run."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gems-lifecycle-guide-mpl"))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

from gems_python.common.scheduling import validate_groups
from gems_python.multi_machine_problem_interval_task.lifecycle import fingerprint
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments


COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
TEXT_COLORS = ["#0072B2", "#956000", "#007D5A", "#9C4A78"]
INK, MUTED, BORDER, PALE = "#243444", "#536678", "#CBD5DF", "#F2F5F8"
FONT = None
OUTPUT = None
RENDERED = []


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Canvas:
    def __init__(self, name, title, subtitle="", height=480):
        self.name, self.title = name, title
        self.width, self.height = 1200, height
        self.fig = plt.figure(figsize=(12, height / 100), dpi=100, facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 1), ylim=(0, 1))
        self.ax.axis("off")
        self.texts = []
        self.text(.035, 1 - 32 / height, title, size=24, ha="left")
        if subtitle:
            self.text(.035, 1 - 80 / height, subtitle, size=16, color=MUTED, ha="left")

    def text(self, x, y, text, size=20, color=INK, ha="center", va="center"):
        artist = self.ax.text(x, y, text, fontsize=size, fontproperties=FONT,
                              color=color, ha=ha, va=va, linespacing=1.45, zorder=4)
        self.texts.append(artist)
        return artist

    def box(self, x, y, width, height, text="", color=INK, fill=PALE, size=20,
            layer=2, text_color=None):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.5, edgecolor=color, facecolor=fill, zorder=layer))
        if text:
            self.text(x + width / 2, y + height / 2, text, size=size,
                      color=color if text_color is None else text_color)

    def arrow(self, x1, y1, x2, y2, color=MUTED, dashed=False):
        self.ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=20, linewidth=1.8,
            linestyle="--" if dashed else "-", color=color, zorder=1))

    def line(self, x1, y1, x2, y2, color=BORDER):
        self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.8, zorder=1)

    def dot(self, x, y, diameter=.022, color=COLORS[0]):
        self.ax.add_patch(Ellipse(
            (x, y), diameter, diameter * self.width / self.height,
            facecolor=color, edgecolor="white", linewidth=.5, zorder=3))

    def device(self, x, y, width=.16, height=.17, label="装置A"):
        self.box(x, y, width, height, fill="white", color=MUTED)
        self.ax.add_patch(Rectangle((x + .016, y + height - .026), width - .032, .012,
                                    facecolor=BORDER, linewidth=0, zorder=3))
        self.text(x + width / 2, y + height / 2 - .012, label, size=18)

    def child(self, index, x, y, width=.19, height=.17, steps=True):
        color = COLORS[index % 4]
        self.box(x, y, width, height, fill="white", color=color)
        self.text(x + width / 2, y + height * (.65 if steps else .5),
                  f"子{index + 1}", size=23, color=TEXT_COLORS[index % 4])
        if steps:
            self.text(x + width / 2, y + height * .25, "工程1 → 2 → 3", size=16)

    def save(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for artist in self.texts:
            box = artist.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= self.width and box.y1 <= self.height, (
                self.name, artist.get_text(), box.bounds)
        paths = []
        for extension in ("png", "svg"):
            path = OUTPUT / f"{self.name}.{extension}"
            metadata = {"Software": "GEMS lifecycle reader guide"} if extension == "png" else {"Date": None}
            self.fig.savefig(path, dpi=100, metadata=metadata, facecolor="white")
            paths.append(dict(file=path.name, sha256=sha256(path)))
        RENDERED.append(dict(name=self.name, title=self.title, files=paths))
        plt.close(self.fig)


def overview():
    c = Canvas("01_experiments", "分割すると、子がそれぞれ独立した実験になります",
               "2親から4子をつくる例。子の色と番号は、後の計画例でも同じです。", height=600)
    for parent in range(2):
        y = .59 if parent == 0 else .24
        c.box(.06, y, .23, .20, f"親{parent + 1}\nこれまでの観測", fill=PALE, size=21)
        for branch in range(2):
            index = parent * 2 + branch
            child_y = y + (.115 if branch == 0 else -.065)
            c.child(index, .47, child_y, .22, .135, steps=False)
            c.arrow(.30, y + .10, .455, child_y + .067, color=COLORS[index])
            c.box(.76, child_y, .18, .135, "自分の観測\n自分の3工程", fill="white", size=18)
            c.arrow(.70, child_y + .067, .748, child_y + .067, color=COLORS[index])
    c.text(.5, .075, "親の履歴を引き継ぎ、分割後の観測と進行は子ごとに管理します。", size=19)
    c.save()


def sequence(config):
    c = Canvas("02_task_sequence", "1つの子が行う、1セット分の作業",
               "各子がこの3工程を持ちます。横幅は時間に比例します。", height=420)
    durations, intervals = config["processing_minutes"], config["interval_minutes"]
    total = sum(durations) + sum(intervals)
    cursor = 0
    for i, (duration, wait) in enumerate(zip(durations, intervals)):
        x = .09 + .82 * cursor / total
        width = .82 * wait / total
        c.ax.add_patch(Rectangle((x, .41), width, .21, facecolor=PALE, edgecolor=BORDER))
        c.text(x + width / 2, .31, f"待ち\n{wait}分", size=18)
        cursor += wait
        x = .09 + .82 * cursor / total
        width = .82 * duration / total
        c.box(x + .004, .41, width - .008, .21, f"工程{i + 1}\n{duration}分",
              fill="white", size=21)
        cursor += duration
    for time in (0, 2, 7, 9, 12, 13, 17):
        x = .09 + .82 * time / total
        c.text(x, .17, str(time), size=17)
    c.text(.5, .075, "群の基準時刻からの経過時間（分）", size=18, color=MUTED)
    c.save()


def shared_devices(count):
    many = count == 8
    name = "07_many_candidates" if many else "03_contention"
    title = "同じ作業で、8台から装置を選べます" if many else "4つの子が、同じ2台を共有します"
    c = Canvas(name, title, "全工程に同じ種類の装置を使います。各工程をどの1台に割り当てるかを決めます。",
               height=590 if many else 490)
    positions = [.045, .285, .525, .765]
    for i, x in enumerate(positions):
        c.child(i, x, .63, height=.15)
        c.line(x + .095, .62, x + .095, .53)
    c.line(.14, .53, .86, .53)
    c.text(.5, .47, "全子・全工程が、この装置群を共有", size=20)
    if many:
        for index in range(count):
            x = .095 + (index % 4) * .205
            y = .225 if index < 4 else .025
            c.device(x, y, .17, .15, f"装置{chr(65 + index)}")
        c.arrow(.5, .425, .5, .39)
    else:
        c.device(.28, .13, .19, .20, "装置A")
        c.device(.53, .13, .19, .20, "装置B")
        c.arrow(.43, .425, .375, .345)
        c.arrow(.57, .425, .625, .345)
        c.text(.5, .055, "同じ装置で、同時に複数の工程を実行することはできません。", size=18)
    c.save()


def heterogeneous():
    c = Canvas("05_heterogeneous", "工程ごとに、使う装置の種類が変わります",
               "どの子も同じ経路を通り、種類ごとの装置を共有します。", height=600)
    for stage, x in enumerate((.065, .385, .705)):
        c.box(x, .62, .23, .15, f"工程{stage + 1}", fill="white", size=23)
        c.arrow(x + .115, .60, x + .115, .49)
        if stage < 2:
            c.arrow(x + .25, .695, x + .30, .695)
        count = 2 if stage < 2 else 1
        c.text(x + .115, .435, f"この工程用：{count}台", size=20)
        for device in range(count):
            c.device(x + .035, .255 - device * .18, .16, .13,
                     f"装置{chr(65 + device)}")
    c.text(.5, .035, "工程3用は1台なので、子の工程3はこの1台に順番に入ります。", size=19)
    c.save()


def child_legend(c, y):
    for i, x in enumerate((.18, .39, .60, .81)):
        c.dot(x - .045, y, .022, COLORS[i])
        c.text(x, y, f"子{i + 1}", size=19, color=TEXT_COLORS[i], ha="left")


def schedule(pattern, source, target, all_examples):
    names = {
        "contention": ("04_contention_schedule", "保存された計画例：2台に順番に入る4つの子"),
        "heterogeneous": ("06_heterogeneous_schedule", "保存された計画例：工程3を1台で順に実行"),
        "replanning": ("10_replanning_schedule", "保存された計画例：固定予約を残して計画"),
    }
    name, title = names[pattern]
    fixed = pattern == "replanning"
    machine_count = len(target.machine_list.machines)
    c = Canvas(name, title, "2親・各2子の保存出力を使用。棒の番号は工程、色は子を表します。",
               height=610 if machine_count > 2 else 530)
    child_legend(c, .78)
    ax = c.fig.add_axes([.155, .25, .81, .48])
    ids = [m.machine_id for m in target.machine_list.machines]
    labels = []
    for m in target.machine_list.machines:
        number = int(m.machine_id.rsplit("m", 1)[1])
        labels.append(f"工程{m.machine_type + 1}用 {chr(65 + number)}"
                      if pattern == "heterogeneous" else f"装置{chr(65 + number)}")
    for i, group in enumerate(target.task_groups):
        old = source.task_groups[i]
        for j, task in enumerate(group.tasks):
            if task.completed:
                continue
            y = machine_count - 1 - ids.index(task.allocated_machine_id)
            start = task.scheduled_time - target.reference_time
            immutable = not old.is_not_started()
            ax.add_patch(Rectangle(
                (start, y - .3), task.processing_time, .6, facecolor=COLORS[i],
                edgecolor=INK if immutable else "white", linewidth=2 if immutable else 1,
                hatch="//" if immutable else None, zorder=3))
            ax.text(start + task.processing_time / 2, y, str(j + 1),
                    fontproperties=FONT, fontsize=17, ha="center", va="center",
                    color="white" if i in (0, 2) else INK, zorder=4)
    right = max(t.scheduled_time + t.processing_time - lab.reference_time
                for lab in all_examples.values() for g in lab.task_groups
                for t in g.tasks if not t.completed)
    right = math.ceil(right / 5) * 5
    ax.set(xlim=(-5 if fixed else 0, right), ylim=(-.7, machine_count - .3))
    ax.set_yticks(range(machine_count), labels[::-1], fontproperties=FONT, fontsize=17)
    ax.set_xticks(np.arange(-5 if fixed else 0, right + 1, 5))
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(FONT)
        tick.set_fontsize(16)
    ax.set_xlabel("計画入力の基準時刻からの経過時間（分）", fontproperties=FONT, fontsize=18, labelpad=12)
    ax.grid(axis="x", color=BORDER, linewidth=.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if fixed:
        ax.axvspan(-5, 0, color=PALE, zorder=0)
        ax.axvline(0, color=INK, linestyle="--", linewidth=1.5)
        c.text(.5, .065, "0分が「いま」。斜線は動かせない予約で、工程1は完了済みです。", size=19)
    else:
        c.text(.5, .065, "同じ装置の棒は重なりません。空白の時間は、その装置が空いています。", size=19)
    c.save()


def preferences(config, lab):
    c = Canvas("08_start_preferences", "子ごとに、開始時刻の好みを変える条件",
               "装置は同型2台。工程は同じで、群の開始時刻に対する費用を変えます。", height=800)
    shifts = np.arange(-6, 6.1, .1)
    for i, y in enumerate((.60, .33)):
        c.text(.035, y + .065, f"子{i + 1}", size=23, color=TEXT_COLORS[i], ha="left")
        c.text(.035, y + .015, "時刻のずれに比例" if i == 0 else "±2分までは許容",
               size=18, ha="left")
        ax = c.fig.add_axes([.30, y - .015, .64, .17])
        penalty = lab.task_groups[i].penalty_type
        values = [penalty.calculate_penalty(float(shift), 0) for shift in shifts]
        ax.plot(shifts, values, color=COLORS[i], linewidth=2.8)
        if i == 1:
            ax.axvspan(-2, 2, color="#FFF3D6", zorder=0)
        ax.axvline(0, color=BORDER, linestyle="--")
        ax.set(xlim=(-6, 6), ylim=(-1, 21), xticks=(-6, -2, 0, 2, 6), yticks=(0, 10, 20))
        ax.set_ylabel("費用", fontproperties=FONT, fontsize=15)
        ax.set_xlabel("希望する開始時刻からのずれ（分）", fontproperties=FONT, fontsize=15)
        for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            tick.set_fontproperties(FONT)
            tick.set_fontsize(14)
        ax.spines[["top", "right"]].set_visible(False)
    c.text(.035, .155, "子3", size=23, color=TEXT_COLORS[2], ha="left")
    c.text(.035, .105, "周期的な休止帯", size=18, ha="left")
    ax = c.fig.add_axes([.30, .12, .64, .075])
    ax.set(xlim=(0, config["rest_cycle_minutes"]), ylim=(0, 1))
    ax.axhspan(.15, .85, color=PALE)
    for lower, upper in config["rest_ranges"]:
        ax.add_patch(Rectangle((lower, .15), upper - lower, .7, color="#D35F68", hatch="//"))
    ax.set_xticks([0, 20, 30, 90, 105, 120])
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(FONT)
        tick.set_fontsize(13)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    c.text(.62, .035, "120分周期の位置（分）。赤い休止帯では大きな費用を課します。", size=16)
    c.save()


def replanning():
    c = Canvas("09_replanning", "すでに決まった予約を残して、ほかの子を計画します",
               "4子の例。子1は工程1を終え、工程2・3は装置Aに予約済みです。", height=580)
    c.box(.045, .54, .21, .22, "子1\n工程1は完了", color=COLORS[0], fill="white", size=22)
    c.arrow(.265, .65, .365, .65, color=COLORS[0])
    c.box(.38, .54, .26, .22, "長い待ちの後\n工程2・3の予約", color=COLORS[0], fill="white", size=21)
    c.arrow(.655, .65, .725, .65, color=COLORS[0])
    c.device(.74, .54, .20, .22, "装置A")
    c.text(.50, .46, "この予約の時刻・装置は変えない", size=19, color=COLORS[0])
    for index, x in enumerate((.045, .29, .535), start=1):
        c.child(index, x, .20, .19, .16)
    c.device(.80, .18, .14, .18, "装置B")
    c.text(.39, .12, "子2・3・4の開始時刻と割当を決める", size=19)
    c.text(.5, .045, "子2・3・4は、装置Aの固定予約と装置Bの空きを見ながら計画します。", size=17)
    c.save()


def dot_grid(c, x, y, count, columns, step=.030, color=COLORS[0], diameter=.016):
    for i in range(count):
        c.dot(x + i % columns * step, y - i // columns * step * c.width / c.height,
              diameter=diameter, color=color)


def child_counts():
    c = Canvas("11_child_counts", "親8個から、子を何個つくるかを変えます",
               "3つは別々の測定条件です。丸1つが1つのExperimentを表します。", height=740)
    for index, count in enumerate((2, 4, 8)):
        x = .035 + index * .32
        c.box(x, .065, .29, .70, fill="white", color=BORDER, layer=0)
        c.text(x + .145, .70, f"各親から{count}子", size=23)
        dot_grid(c, x + .083, .60, 8, 4, color="#7F8C99", diameter=.018)
        c.text(x + .145, .47, "親8個", size=19, color=MUTED)
        c.arrow(x + .145, .435, x + .145, .405)
        columns = 4 if count == 2 else 8
        left = x + .145 - (columns - 1) * .018 / 2
        dot_grid(c, left, .38, 8 * count, columns, step=.018, diameter=.014)
        c.text(x + .145, .10, f"子{8 * count}個", size=27, color=COLORS[0])
    c.save()


def generations():
    c = Canvas("12_two_generations", "2世代では、子がもう一度分割して孫になります",
               "8親 → 16子 → 32孫。最後に計画する活動中Experimentは32個です。", height=570)
    for index, (count, name, color) in enumerate(
            ((8, "親8個", "#7F8C99"), (16, "子16個", COLORS[0]), (32, "孫32個", COLORS[2]))):
        x = .055 + index * .32
        c.box(x, .25, .24, .48, fill="white", color=BORDER, layer=0)
        c.text(x + .12, .66, name, size=26, color=color)
        columns = 4 if index == 0 else 8
        left = x + .12 - (columns - 1) * .023 / 2
        dot_grid(c, left, .53, count, columns, step=.023, color=color, diameter=.018)
        c.text(x + .12, .31, "分割後は終了記録へ" if index < 2 else "それぞれ独立に計画",
               size=16, color=MUTED)
        if index < 2:
            c.arrow(x + .25, .49, x + .305, .49)
    c.text(.5, .10, "親の履歴と子の履歴を引き継ぎ、孫の新しい観測は孫ごとに管理します。", size=18)
    c.save()


def history_amounts():
    c = Canvas("13_history_amounts", "分割前に持たせる履歴量を変えます",
               "8親・各2子・1世代は固定。親に与える完了観測は合成入力です。", height=680)
    for index, interval in enumerate((1, 2, 4)):
        x = .035 + index * .32
        c.box(x, .095, .29, .67, fill="white", color=BORDER, layer=0)
        c.text(x + .145, .705, f"親の観測 {3 * interval}行", size=23)
        for row in range(3 * interval):
            c.ax.add_patch(Rectangle((x + .075, .62 - row * .017), .14, .010,
                                    facecolor="#A8B5C1", linewidth=0))
        c.arrow(x + .145, .385, x + .145, .32)
        c.box(x + .035, .15, .22, .15, "各子へ履歴を継承\n子の初期観測は1行",
              fill=PALE, color=MUTED, size=17)
    c.text(.5, .045, "履歴量に加えて基準時刻も変わるため、周期的な費用の位相も変わり得ます。", size=17)
    c.save()


def flat_comparison():
    c = Canvas("14_flat_comparison", "比較では、計画する作業と装置を揃えています",
               "どちらも活動中の4子・同じ3工程・同じ共有装置。履歴の持ち方が違います。", height=650)
    for side in range(2):
        x = .045 + side * .49
        c.box(x, .08, .42, .70, fill="white", color=BORDER, layer=0)
        c.text(x + .21, .72, "分割・履歴継承あり" if side == 0 else "flat：履歴を結合", size=24)
        if side == 0:
            c.box(x + .09, .525, .24, .10, "親の履歴・終了記録", color=MUTED, fill=PALE, size=18)
        else:
            c.text(x + .21, .575, "親子の記録は持たない", size=18, color=MUTED)
        if side == 0:
            c.line(x + .09, .575, x + .06, .575, color=MUTED)
            c.line(x + .06, .575, x + .06, .205, color=MUTED)
        for i in range(4):
            child_x, child_y = x + .12, .43 - i * .085
            c.box(child_x, child_y, .26, .06, f"子{i + 1}",
                  color=COLORS[i], text_color=TEXT_COLORS[i], fill="white", size=20)
            if side == 0:
                c.arrow(x + .06, child_y + .030, child_x - .010, child_y + .030,
                        color=MUTED, dashed=True)
        c.text(x + .21, .12, "継承履歴＋自分の観測" if side == 0 else "同じ履歴値を1つに結合", size=17)
    c.save()


def load_examples(source, config):
    rows = [json.loads(line) for line in (source / "results.jsonl").read_text().splitlines()]
    examples, provenance = {}, []
    for pattern in config["patterns"]:
        matched = [r for r in rows if r["pattern"] == pattern and r["series"] == "scan"
                   and r["parents"] == 2 and r["children"] == 2 and r["group_interval"] == 2
                   and r["generations"] == 1 and r["budget"] == 1 and r["mode"] == "lifecycle"
                   and r["repetition"] == 0]
        assert len(matched) == 1 and matched[0]["outcome"] == "VALID_ON_TIME"
        row = matched[0]
        directory = source / row["artifact_directory"]
        before = Experiments.from_json((directory / "input.json").read_text())
        after = Experiments.from_json((directory / "output.json").read_text())
        assert fingerprint(before) == row["input_hash"]
        assert len(after.experiments) == 4
        assert len(after.machine_list.machines) == sum(config["machine_counts"][pattern])
        fixed = {id(new): (old.scheduled_time, old.allocated_machine_id)
                 for old_group, new_group in zip(before.task_groups, after.task_groups)
                 if not old_group.is_not_started()
                 for old, new in zip(old_group.tasks, new_group.tasks)}
        validate_groups(after.task_groups, after.reference_time, after.machine_list, fixed)
        for i, group in enumerate(before.task_groups):
            assert [t.processing_time for t in group.tasks] == config["processing_minutes"]
            expected = config["interval_minutes"].copy()
            if pattern == "replanning" and i % 4 == 0:
                expected[1] += len(before.experiments) * 20
            assert [t.interval for t in group.tasks] == expected
        examples[pattern] = before, after
        provenance.append(dict(
            pattern=pattern, artifact_directory=row["artifact_directory"], job={
                key: row[key] for key in ("series", "parents", "children", "group_interval",
                                         "generations", "budget", "mode", "repetition")},
            input_sha256=sha256(directory / "input.json"), input_fingerprint=row["input_hash"],
            output_sha256=sha256(directory / "output.json"), reference_time=after.reference_time))
    return examples, provenance


def main():
    global FONT, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Unpacked recorded capacity study")
    parser.add_argument("--font", type=Path, required=True, help="Japanese font installed in the environment")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    FONT, OUTPUT = FontProperties(fname=str(args.font)), args.output
    OUTPUT.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "gems-lifecycle-reader-guide",
                         "axes.unicode_minus": False})
    config_path = Path(__file__).with_name("scheduler_capacity_config.json")
    variation_path = Path(__file__).with_name("experiment_lifecycle_capacity_config.json")
    config, variations = json.loads(config_path.read_text()), json.loads(variation_path.read_text())
    assert config["processing_minutes"] == [5, 3, 4] and config["interval_minutes"] == [2, 2, 1]
    assert variations["variation_parents"] == 8 and variations["variation_child_counts"] == [4, 8]
    assert variations["variation_group_intervals"] == [1, 4]
    assert variations["generations"] == [1, 2]
    assert config["machine_counts"] == {
        "contention": [2], "heterogeneous": [2, 2, 1], "many_candidates": [8],
        "mixed_penalties": [2], "replanning": [2]}
    assert config["rest_cycle_minutes"] == 120 and config["rest_ranges"] == [[20, 30], [90, 105]]
    examples, provenance = load_examples(args.data, config)
    afters = {name: labs[1] for name, labs in examples.items()}
    overview()
    sequence(config)
    shared_devices(config["machine_counts"]["contention"][0])
    schedule("contention", *examples["contention"], afters)
    heterogeneous()
    schedule("heterogeneous", *examples["heterogeneous"], afters)
    shared_devices(config["machine_counts"]["many_candidates"][0])
    preferences(config, examples["mixed_penalties"][0])
    replanning()
    schedule("replanning", *examples["replanning"], afters)
    child_counts()
    generations()
    history_amounts()
    flat_comparison()
    metadata = dict(
        source_commit=json.loads((args.data / "metadata.json").read_text())["commit"],
        source_examples=provenance, solver_called=False,
        renderer_sha256=sha256(Path(__file__)),
        font=dict(name=FONT.get_name(), file=args.font.name, sha256=sha256(args.font),
                  source="Environment-installed font; font file is not redistributed"),
        matplotlib_version=matplotlib.__version__,
        configuration_sha256={path.name: sha256(path) for path in (config_path, variation_path)},
        design=dict(example_parents=2, example_children_per_parent=2, canvas_width=1200,
                    identity="Child number and color, consistent across the recorded examples",
                    gantt_time="Minutes relative to the recorded manager reference time",
                    schematic="Overview, resource pools, scale, history and flat panels are explanatory diagrams"),
        figures=RENDERED)
    (OUTPUT / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(figures=len(RENDERED), formats=["png", "svg"], solver_called=False)))


if __name__ == "__main__":
    main()
