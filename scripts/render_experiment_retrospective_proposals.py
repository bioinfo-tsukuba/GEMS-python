"""Draw four post-run review proposals from an explicitly synthetic fixture."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts import render_experiment_lifecycle_guide as drawing

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


COLORS = {"A": "#0072B2", "B": "#A66A00"}
TAG_NUMBER = 0


def tag(artist, experiment, kind):
    global TAG_NUMBER
    TAG_NUMBER += 1
    artist.set_gid(f"pick-{experiment}-{kind}-{TAG_NUMBER}")
    return artist


def validate(data):
    assert data["kind"] == "design_fixture" and not data["observed_run"]
    records = {e["id"]: e for e in data["experiments"]}
    assert len(records) == len(data["experiments"]) == 8
    machines = {m["id"] for m in data["machines"]}
    assert machines == {"imager", "processor"}
    assert len({t["id"] for t in data["tasks"]}) == len(data["tasks"]) == 24
    for record in records.values():
        assert 0 <= record["born"] < record["ended"] <= 120
        assert record["reason"] in ("finished", "split")
        parent = record["parent"]
        if parent:
            assert records[parent]["reason"] == "split"
            assert record["born"] == records[parent]["ended"]
            assert record["family"] == records[parent]["family"]
            visited = {record["id"]}
            while parent:
                assert parent not in visited
                visited.add(parent)
                parent = records[parent]["parent"]
    assert len(data["splits"]) == 3
    for split in data["splits"]:
        parent = records[split["parent"]]
        assert parent["ended"] == split["time"] and parent["reason"] == "split"
        assert set(split["children"]) == {e["id"] for e in records.values()
                                         if e["parent"] == parent["id"]}
    assert {e["id"] for e in records.values() if e["reason"] == "split"} == {
        s["parent"] for s in data["splits"]}
    for task in data["tasks"]:
        record = records[task["experiment"]]
        assert task["machine"] in machines
        assert record["born"] <= task["start"] < task["end"] <= record["ended"]
    for key, values in (("machine", machines), ("experiment", records)):
        for value in values:
            tasks = sorted((t for t in data["tasks"] if t[key] == value),
                           key=lambda t: t["start"])
            assert all(left["end"] <= right["start"] for left, right in zip(tasks, tasks[1:]))
    for record in records.values():
        assert max(t["end"] for t in data["tasks"] if t["experiment"] == record["id"]) == record["ended"]
    return records


def axes(c, area, count):
    ax = c.fig.add_axes(area)
    ax.set(xlim=(0, 120), ylim=(-.65, count - .35))
    ax.set_xticks(range(0, 121, 20))
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(drawing.FONT)
        tick.set_fontsize(16)
    ax.set_xlabel("最初の登録からの経過時間（分）", fontproperties=drawing.FONT, fontsize=17, labelpad=10)
    ax.grid(axis="x", color=drawing.BORDER, linewidth=.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    return ax


def color(record, selected=None):
    return COLORS[record["family"]] if selected is None or record["id"] in selected else "#CDD3D9"


def lifetime(c, data, area, rows=None, connect=True, show_labels=True):
    rows = data["experiments"] if rows is None else rows
    records, count = {e["id"]: e for e in rows}, len(rows)
    positions = {e["id"]: count - 1 - i for i, e in enumerate(rows)}
    ax = axes(c, area, count)
    for record in rows:
        y, shade = positions[record["id"]], color(record)
        ax.add_patch(Rectangle((record["born"], y - .24),
                               record["ended"] - record["born"], .48,
                               facecolor=shade, alpha=.13, linewidth=0, zorder=1))
        ax.scatter(record["ended"], y, marker="D" if record["reason"] == "split" else "s",
                   s=65, facecolor=shade, edgecolor="white", linewidth=.7, zorder=5)
    for task in data["tasks"]:
        if task["experiment"] not in records:
            continue
        y = positions[task["experiment"]]
        record = records[task["experiment"]]
        ax.add_patch(Rectangle((task["start"], y - .24), task["end"] - task["start"], .48,
                               facecolor=color(record), edgecolor="white", linewidth=.8, zorder=3))
        ax.text((task["start"] + task["end"]) / 2, y,
                "撮" if task["machine"] == "imager" else "処",
                fontproperties=drawing.FONT, fontsize=13, color="white",
                ha="center", va="center", zorder=4)
    if connect:
        for split in data["splits"]:
            if split["parent"] not in records:
                continue
            parent_y = positions[split["parent"]]
            for child in split["children"]:
                if child in positions:
                    child_y = positions[child]
                    ax.plot([split["time"], split["time"]], [parent_y, child_y],
                            color=color(records[child]), linewidth=1.8, zorder=2)
                    ax.scatter(split["time"], child_y, marker="o", s=20,
                               color=color(records[child]), zorder=4)
    labels = [e["name"] for e in rows][::-1] if show_labels else [""] * count
    ax.set_yticks(range(count), labels, fontproperties=drawing.FONT, fontsize=20)
    return ax


def lineage(c, data, compact=False, selected=None):
    positions = {
        "A": (.255, .68), "A1": (.12, .43), "A2": (.39, .43),
        "A2a": (.285, .18), "A2b": (.495, .18),
        "B": (.79, .68), "B1": (.685, .43), "B2": (.895, .43),
    }
    if compact:
        positions = {uid: (x, .82 if y == .68 else .49 if y == .43 else y)
                     for uid, (x, y) in positions.items()}
    height = .15 if compact else .19
    for record in data["experiments"]:
        uid, parent = record["id"], record["parent"]
        if parent:
            x1, y1 = positions[parent]
            x2, y2 = positions[uid]
            tag(c.ax.add_patch(FancyArrowPatch(
                (x1, y1 - height / 2 - .006), (x2, y2 + height / 2 + .010),
                arrowstyle="-|>", mutation_scale=17, linewidth=1.8,
                color=color(record, selected), zorder=1)), uid, "edge")
    for record in data["experiments"]:
        uid = record["id"]
        x, y = positions[uid]
        shade = color(record, selected)
        tag(c.ax.add_patch(FancyBboxPatch(
            (x - .085, y - height / 2), .17, height,
            boxstyle="round,pad=0.006,rounding_size=0.015",
            facecolor="white", edgecolor=shade, linewidth=2, zorder=2)), uid, "node")
        tag(c.text(x, y + (.025 if compact else .050), record["name"],
                   size=20 if compact else 22, color=shade), uid, "label")
        tag(c.text(x, y - (.030 if compact else .009),
                   f'{record["born"]}–{record["ended"]}分', size=14 if compact else 16,
                   color=drawing.MUTED if selected is None or uid in selected else shade), uid, "label")
        if not compact:
            tag(c.text(x, y - .061, "分割して終了" if record["reason"] == "split" else "終了",
                       size=15, color=shade), uid, "label")


def devices(c, data, area, selected=None):
    ax = axes(c, area, len(data["machines"]))
    records = {e["id"]: e for e in data["experiments"]}
    positions = {m["id"]: len(data["machines"]) - 1 - i for i, m in enumerate(data["machines"])}
    labels = []
    for task in data["tasks"]:
        record, y = records[task["experiment"]], positions[task["machine"]]
        shade = color(record, selected)
        tag(ax.add_patch(Rectangle(
            (task["start"], y - .20), task["end"] - task["start"], .40,
            facecolor=shade, edgecolor="white", linewidth=.9, zorder=3)),
            record["id"], "task")
        label = tag(ax.text((task["start"] + task["end"]) / 2, y + .40, record["id"],
                    fontproperties=drawing.FONT, fontsize=12.5, ha="center", va="center",
                    color=shade,
                    zorder=4), record["id"], "label")
        labels.append((task, label))
    ax.set_yticks(range(len(positions)), [m["name"] for m in data["machines"]][::-1],
                  fontproperties=drawing.FONT, fontsize=20)
    ax.set_ylim(-.65, len(positions) - .05)
    c.fig.canvas.draw()
    renderer = c.fig.canvas.get_renderer()
    for machine in data["machines"]:
        right = float("-inf")
        for task, label in sorted(labels, key=lambda item: item[0]["start"]):
            if task["machine"] != machine["id"]:
                continue
            bbox = label.get_window_extent(renderer)
            y = positions[task["machine"]]
            offset = .62 if bbox.x0 < right + 8 else .40
            if offset == .40:
                right = bbox.x1
            label.set_y(y + offset)
            center = (task["start"] + task["end"]) / 2
            tag(ax.plot([center, center], [y + .20, y + offset - .15],
                        color=color(records[task["experiment"]], selected),
                        linewidth=.9, zorder=2)[0], task["experiment"], "leader")
    return ax


def family_tree(c, rows, area):
    ax = c.fig.add_axes(area)
    ax.set(xlim=(-.15, 2.9), ylim=(-.65, len(rows) - .35))
    ax.axis("off")
    positions, depths = {}, {}
    for i, record in enumerate(rows):
        depths[record["id"]] = 0 if not record["parent"] else depths[record["parent"]] + 1
        positions[record["id"]] = depths[record["id"]] * .90, len(rows) - 1 - i
    for record in rows:
        x, y = positions[record["id"]]
        if record["parent"]:
            px, py = positions[record["parent"]]
            ax.plot([px + .62, px + .75, px + .75, x], [py, py, y, y],
                    color=color(record), linewidth=1.8, zorder=1)
        ax.add_patch(FancyBboxPatch((x, y - .27), .63, .54,
                                    boxstyle="round,pad=0.03", facecolor="white",
                                    edgecolor=color(record), linewidth=1.7))
        ax.text(x + .315, y, record["id"], fontproperties=drawing.FONT,
                fontsize=19, color=color(record), ha="center", va="center")


def render(data):
    c = drawing.Canvas("01_time_lineage", "案①：親子のつながりを、時間軸の上で読む",
                       "説明用データ：8 Experiment・3分割・24作業。全件終了、全体116分。", height=850)
    c.text(.05, .80, "青：親Aの家族　橙：親Bの家族　◆ 分割して終了　■ 終了", size=18, ha="left")
    lifetime(c, data, [.19, .20, .77, .56])
    c.text(.5, .09, "淡い帯＝登録から終了まで。濃い棒＝予約された作業（撮：撮像、処：処理）。", size=17)
    c.text(.5, .035, "作業のない区間の理由は、この記録だけでは区別しません。実機の実績ではありません。",
           size=16, color=drawing.MUTED)
    c.save()

    c = drawing.Canvas("02_lineage", "親子の対応：誰から誰が生まれたか",
                       "説明用データ。上下は世代、横の位置は時刻を表しません。", height=660)
    lineage(c, data)
    c.text(.5, .035, "親AからA1・A2、A2からA2a・A2b。親BからB1・B2。番号は次の時間図と共通です。", size=16)
    c.save()

    c = drawing.Canvas("03_device_schedule", "共有装置：どのExperimentが、いつ使ったか",
                       "説明用データ。棒の上の番号がExperiment、色が家族。同じ装置の予約は重なりません。",
                       height=530)
    devices(c, data, [.13, .27, .83, .46])
    c.text(.5, .075, "作業の並行、予約の詰まり、装置ごとの空白を読みます。棒は予約区間です。", size=17)
    c.save()

    for family in ("A", "B"):
        rows = [e for e in data["experiments"] if e["family"] == family]
        c = drawing.Canvas(f"04_family_{family.lower()}", f"案③：親{family}の家族だけを、1ページで振り返る",
                           f"説明用データ。左に親子、右に登録〜終了と作業。同じ番号を同じ行で読む。",
                           height=700 if family == "A" else 540)
        c.text(.08, .765, "親子の対応", size=18, ha="left")
        c.text(.39, .765, "時間の流れ", size=18, ha="left")
        area = [.04, .27, .27, .43]
        family_tree(c, rows, area)
        lifetime(c, data, [.36, .27, .60, .43], rows, connect=False, show_labels=False)
        c.text(.5, .09, "家族ごとにページを分けます。時間軸は全ページで0〜120分に揃えます。", size=17)
        c.text(.5, .035, "他家族の作業を省略しているため、装置全体の空きは別の装置図で確認します。",
               size=16, color=drawing.MUTED)
        c.save()

    c = drawing.Canvas("05_linked_selection", "案④：1つを選び、系譜と装置図を一緒にたどる",
                       "説明用画面。A2を選択し、祖先Aと子A2a・A2bを強調した例。", height=980)
    selected = {"A", "A2", "A2a", "A2b"}
    original = c.ax
    c.ax = c.fig.add_axes([.015, .47, .97, .40])
    c.ax.set(xlim=(0, 1), ylim=(0, 1)); c.ax.axis("off")
    lineage(c, data, compact=True, selected=selected)
    c.ax = original
    c.box(.05, .40, .90, .055, "選択：A2　親：A　子：A2a・A2b　24〜50分、分割して終了", size=18)
    devices(c, data, [.13, .18, .83, .18], selected=selected)
    c.text(.5, .065, "薄い作業も残して、ほかのExperimentとの時間関係を保ちます。", size=18)
    c.text(.5, .025, "HTML試作では番号を選択できます。予定と実績の重ね表示は、実績記録を追加した後の設計です。",
           size=15, color=drawing.MUTED)
    c.save()


JAVASCRIPT = r"""
function matching(data, id, scope) {
  if (id === null) return data.experiments.map(e => e.id);
  const index = new Map(data.experiments.map(e => [e.id, e]));
  if (!index.has(id)) throw new Error("Unknown Experiment");
  if (scope === "self") return [id];
  if (scope === "family")
    return data.experiments.filter(e => e.family === index.get(id).family).map(e => e.id);
  if (scope !== "relatives") throw new Error("Unknown scope");
  const result = new Set([id]);
  let parent = index.get(id).parent;
  while (parent) { result.add(parent); parent = index.get(parent).parent; }
  function children(parentId) {
    for (const e of data.experiments.filter(e => e.parent === parentId)) {
      result.add(e.id); children(e.id);
    }
  }
  children(id);
  return data.experiments.filter(e => result.has(e.id)).map(e => e.id);
}
if (typeof module !== "undefined") module.exports = {matching, data: DATA};
if (typeof document !== "undefined") {
  let selected = "A2";
  const scope = document.querySelector("#scope");
  function render() {
    const visible = new Set(matching(DATA, selected, scope.value));
    document.querySelectorAll("[data-exp]").forEach(node => {
      node.classList.toggle("dimmed", !visible.has(node.dataset.exp));
      node.classList.toggle("selected", selected === node.dataset.exp);
    });
    document.querySelectorAll("button[data-select]").forEach(button => {
      const value = button.dataset.select || null;
      button.setAttribute("aria-pressed", String(value === selected));
    });
    const record = DATA.experiments.find(e => e.id === selected);
    const summary = record
      ? `${record.name}：親 ${record.parent || "なし"}／子 ${DATA.experiments.filter(e => e.parent === selected).map(e => e.id).join("・") || "なし"}／${record.born}〜${record.ended}分／${record.reason === "split" ? "分割して終了" : "終了"}`
      : "全8 Experiment・3分割・24作業。全件終了、全体116分。";
    document.querySelector("#summary").textContent = summary;
    document.querySelector("#focus-list").textContent = "強調：" + [...visible].join("・");
  }
  function choose(id) { selected = id || null; render(); }
  document.addEventListener("click", event => {
    const button = event.target.closest("button[data-select]");
    const graphic = event.target.closest("[data-exp]");
    if (button) choose(button.dataset.select);
    else if (graphic) choose(graphic.dataset.exp);
  });
  document.addEventListener("keydown", event => {
    const graphic = event.target.closest("[data-exp][tabindex]");
    if (graphic && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault(); choose(graphic.dataset.exp);
    }
  });
  scope.addEventListener("change", render);
  render();
}
"""


def inline_svg(path, prefix):
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(path.read_text())
    identifiers = {node.attrib["id"]: prefix + node.attrib["id"]
                   for node in root.iter() if "id" in node.attrib}
    for node in root.iter():
        gid = node.attrib.get("id", "")
        if gid.startswith("pick-"):
            _, uid, kind, _ = gid.split("-")
            node.set("data-exp", uid)
            node.set("data-kind", kind)
            if kind in ("node", "task"):
                node.set("tabindex", "0")
                node.set("role", "button")
                node.set("aria-label", uid + "を選択する")
                title = ET.SubElement(node, "{http://www.w3.org/2000/svg}title")
                title.text = uid + "を選択する"
        for key, value in list(node.attrib.items()):
            if key == "id": node.set(key, identifiers[value])
            elif value.startswith("#") and value[1:] in identifiers:
                node.set(key, "#" + identifiers[value[1:]])
            elif "url(#" in value:
                for old, new in identifiers.items():
                    value = value.replace(f"url(#{old})", f"url(#{new})")
                node.set(key, value)
    root.set("role", "group")
    return ET.tostring(root, encoding="unicode")


def prototype(data, output):
    controls = ['<button data-select="">全体</button>'] + [
        f'<button data-select="{e["id"]}">{html.escape(e["name"])}</button>' for e in data["experiments"]]
    tree = inline_svg(output / "02_lineage.svg", "tree-")
    schedule = inline_svg(output / "03_device_schedule.svg", "schedule-")
    document = f"""<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>全終了後の振り返り：選択連動の試作</title>
<style>
body{{margin:0;background:#f3f5f7;color:#243444;font-family:system-ui,sans-serif}}
main{{max-width:1200px;margin:auto;padding:24px}}
h1{{font-size:26px;margin:0 0 12px}} p{{line-height:1.7}}
.controls{{display:flex;flex-wrap:wrap;gap:8px;align-items:center}}
button,select{{font:inherit;padding:9px 14px;border:1px solid #81909e;border-radius:6px;background:white;color:inherit}}
button[aria-pressed=true]{{background:#243444;color:white}}
button:focus-visible,select:focus-visible{{outline:3px solid #0072b2;outline-offset:3px}}
.summary,.panel{{background:white;border:1px solid #cbd5df;border-radius:10px;margin:18px 0;padding:14px}}
.summary p{{margin:4px 0}} svg{{width:100%;height:auto;display:block}}
[data-exp]{{cursor:pointer}} .dimmed{{opacity:.16}} [data-exp][tabindex]:focus{{outline:2px solid #243444}}
[data-kind=node].selected path,[data-kind=task].selected path{{stroke:#243444!important;stroke-width:2.5!important}}
#focus-list{{color:#536678}} .note{{color:#536678;font-size:14px}}
</style>
<main><h1>全終了後の振り返り：選んだ実験のつながりを追う</h1>
<p>説明用の仮想記録です。8件すべて終了しています。棒は予約区間で、実機の実績ではありません。</p>
<div class="controls" aria-label="Experimentの選択">{"".join(controls)}</div>
<p><label for="scope">一緒に強調する範囲：</label>
<select id="scope"><option value="relatives">祖先と子孫</option>
<option value="family">同じ家族すべて</option><option value="self">本人だけ</option></select></p>
<div class="summary" aria-live="polite"><p id="summary"></p><p id="focus-list"></p></div>
<p class="note">図中の番号や作業を選ぶと、親子の図と装置の時間図を同時に強調します。
薄い作業も残し、ほかの実験との時間関係を保ちます。ボタンとキーボードでも選べます。</p>
<div class="panel">{tree}</div><div class="panel">{schedule}</div>
<p class="note">外部サービスへの通信はありません。実績との比較には、実績の開始・終了を別途記録する必要があります。</p>
<noscript><p>選択機能にはJavaScriptが必要です。上の図は全体表示として読めます。</p></noscript>
</main><script src="prototype.js"></script></html>"""
    (output / "prototype.html").write_text(
        "\n".join(line.rstrip() for line in document.splitlines()) + "\n")
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    (output / "prototype.js").write_text("const DATA = " + encoded + ";\n" + JAVASCRIPT.strip() + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    validate(data)
    args.output.mkdir(parents=True, exist_ok=False)
    drawing.FONT, drawing.OUTPUT, drawing.RENDERED = FontProperties(fname=str(args.font)), args.output, []
    plt.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "gems-retrospective-proposals",
                         "axes.unicode_minus": False})
    render(data)
    prototype(data, args.output)
    metadata = {
        "design_fixture": True, "observed_run": False, "solver_called": False,
        "sample_sha256": drawing.sha256(args.data),
        "renderer_sha256": drawing.sha256(Path(__file__)),
        "drawing_helpers_sha256": drawing.sha256(Path(drawing.__file__)),
        "font": {"name": drawing.FONT.get_name(), "file": args.font.name, "sha256": drawing.sha256(args.font)},
        "matplotlib_version": drawing.matplotlib.__version__,
        "figures": drawing.RENDERED,
        "prototype": [{"file": p.name, "sha256": drawing.sha256(p)}
                      for p in sorted(args.output.glob("prototype.*"))],
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"figures": len(drawing.RENDERED), "interactive": "prototype.html",
                      "observed_run": False, "solver_called": False}))


if __name__ == "__main__":
    main()
