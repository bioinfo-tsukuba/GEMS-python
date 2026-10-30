"""Post-run lineage and final reservations, including standalone linked selection.

Times are manager logical times in the caller's unit. A completed task's last
reservation is retained; these are deliberately not presented as actual device
start/end measurements. Inherited scientific history is never plotted as tasks.
"""
from collections import Counter
import hashlib
import html
import io
import json
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import matplotlib
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle


PALETTE = ("#0072B2", "#A66A00", "#009E73", "#CC79A7", "#D55E00", "#5952A3")
SVG = "http://www.w3.org/2000/svg"
ROWS_PER_PAGE = 24


def review_data(collection):
    """Extract only recorded facts; older snapshots explicitly remain partial."""
    if collection.experiments:
        raise ValueError("Finish all Experiments before rendering the retrospective")
    trace = collection.retrospective_trace
    if trace["version"] != 1:
        raise ValueError("Unsupported retrospective trace version")
    ended = collection.ended_experiments
    registrations = trace["experiments"]
    incoming = {}
    for split in collection.split_requests.values():
        if split.status == "COMMITTED":
            for uid, spec in zip(split.child_uuids, split.children):
                incoming[uid] = (spec.experiment_name, split.logical_time)
    records = {}
    for uid, ending in ended.items():
        saved = registrations.get(uid, {})
        fallback = incoming.get(uid, (uid, None))
        records[uid] = dict(id=uid, name=saved.get("name", fallback[0]),
                            born=saved.get("born", fallback[1]), ended=ending.logical_time,
                            parent=ending.parent_uuid, reason=ending.reason)
    children = {uid: [] for uid in records}
    roots = []
    for uid, record in records.items():
        parent = record["parent"]
        if parent is None or parent not in records:
            roots.append(uid)
        else:
            children[parent].append(uid)
    sort_key = lambda uid: (records[uid]["born"] is None,
                            records[uid]["born"] or 0, records[uid]["name"], uid)
    ordered, visited = [], set()

    def visit(uid, root, depth, ancestors):
        if uid in ancestors:
            raise ValueError("Cyclic experiment lineage")
        visited.add(uid)
        records[uid].update(family=root, depth=depth)
        ordered.append(records[uid])
        for child in sorted(children[uid], key=sort_key):
            visit(child, root, depth + 1, ancestors | {uid})

    for root in sorted(roots, key=sort_key):
        visit(root, root, 0, set())
    if visited != set(records):
        raise ValueError("Cyclic experiment lineage")
    counts = Counter(record["name"] for record in ordered)
    occurrences = Counter()
    for index, record in enumerate(ordered):
        occurrences[record["name"]] += 1
        record["label"] = record["name"] + (
            f" #{occurrences[record['name']]}" if counts[record["name"]] > 1 else "")
        record["pick"] = f"e{index}"  # DOM ids never depend on user-supplied names/UUIDs.
    tasks, omitted = [], 0
    for key, saved in sorted(trace["tasks"].items()):
        if saved["experiment"] not in records:
            continue
        if saved["start"] is None or saved["end"] is None or saved["machine"] is None:
            omitted += 1
            continue
        if saved["end"] < saved["start"]:
            raise ValueError(f"Negative recorded duration: {key}")
        tasks.append(dict(saved, id=key))
    machine_names = {m.machine_id: (m.description if m.description not in (None, "", "None")
                                   else str(m.machine_id))
                     for m in collection.machine_list.machines}
    machines = [dict(id=mid, name=machine_names.get(mid, str(mid)))
                for mid in sorted({t["machine"] for t in tasks}, key=str)]
    complete = trace["complete"] and set(ended) == set(registrations) and not omitted
    if not complete:
        warnings.warn("Incomplete retrospective trace: missing registrations or reservations "
                      "are not reconstructed from inherited history.", RuntimeWarning)
    return dict(version=1, complete=complete, omitted_tasks=omitted,
                time_basis="final_reservation", experiments=ordered, tasks=tasks,
                machines=machines, logical_end=collection.reference_time)


class Renderer:
    def __init__(self, data, font, time_unit):
        self.data = data
        self.font = FontProperties(fname=str(font)) if font else FontProperties()
        self.unit = time_unit
        self.records = {e["id"]: e for e in data["experiments"]}
        roots = list(dict.fromkeys(e["family"] for e in data["experiments"]))
        self.colors = {root: PALETTE[i % len(PALETTE)] for i, root in enumerate(roots)}
        starts = [e["born"] for e in data["experiments"] if e["born"] is not None]
        self.left = min(starts + [t["start"] for t in data["tasks"]] + [0])
        self.right = max([e["ended"] for e in data["experiments"]] +
                         [t["end"] for t in data["tasks"]] + [data["logical_end"], 1])
        self.span = max(1, self.right - self.left)
        self.number = 0
        self.tags = {}

    def color(self, record):
        return self.colors[record["family"]]

    def with_ancestors(self, rows):
        """Keep cross-page parent relationships visible without altering the ledger."""
        ids = {e["id"] for e in rows}
        for record in rows:
            parent = self.records.get(record["parent"])
            while parent:
                ids.add(parent["id"])
                parent = self.records.get(parent["parent"])
        return [e for e in self.data["experiments"] if e["id"] in ids]

    def tag(self, artist, record, kind, title):
        key = f"pick{self.number}"
        self.number += 1
        artist.set_gid(key)
        self.tags[key] = (record["pick"], kind, title)
        return artist

    def text(self, ax, x, y, value, **kwargs):
        return ax.text(x, y, value, fontproperties=self.font, **kwargs)

    def figure(self, count, width=16):
        fig = Figure(figsize=(width, max(3.5, 1.9 + count * .48)), facecolor="white")
        return fig

    def time_axes(self, fig, area, count):
        ax = fig.add_axes(area)
        ax.set(xlim=(self.left, self.right + self.span * .025),
               ylim=(-.7, max(count - .3, .7)))
        ax.set_xlabel(f"Reservation ({self.unit})", fontproperties=self.font, fontsize=11)
        ax.tick_params(axis="y", length=0, pad=12)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", color="#dce2e8", linewidth=.7, zorder=0)
        for label in ax.get_xticklabels():
            label.set_fontproperties(self.font)
        return ax

    def legend(self, fig):
        handles = [Patch(color="#6b7785", label="Task"),
                   Line2D([], [], color="#6b7785", marker="D", linestyle="", label="Split"),
                   Line2D([], [], color="#6b7785", marker="s", linestyle="", label="End")]
        fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False,
                   prop=self.font, bbox_to_anchor=(.98, .99))

    def lifetime(self, fig, area, rows, labels=True):
        ax = self.time_axes(fig, area, len(rows))
        positions = {e["id"]: len(rows) - 1 - i for i, e in enumerate(rows)}
        for record in rows:
            y, color = positions[record["id"]], self.color(record)
            if record["born"] is not None:
                self.tag(ax.add_patch(Rectangle((record["born"], y - .23),
                         record["ended"] - record["born"], .46, facecolor=color,
                         alpha=.12, edgecolor="none", zorder=1)), record, "life", record["label"])
                ax.scatter(record["born"], y, s=18, color=color, zorder=4)
            split = any(e["parent"] == record["id"] for e in self.data["experiments"])
            self.tag(ax.scatter(record["ended"], y, marker="D" if split else "s",
                                s=45, color=color, edgecolors="white", linewidths=.6, zorder=5),
                     record, "end", record["reason"])
            if record["parent"] in positions and record["born"] is not None:
                parent_y = positions[record["parent"]]
                ax.plot([record["born"], record["born"]], [parent_y, y],
                        color=color, linewidth=1.4, alpha=.65, zorder=2)
        for task in self.data["tasks"]:
            if task["experiment"] not in positions:
                continue
            record = self.records[task["experiment"]]
            y = positions[record["id"]]
            width = task["end"] - task["start"]
            artist = ax.add_patch(Rectangle((task["start"], y - .23), width, .46,
                                           color=self.color(record), linewidth=0, zorder=3))
            self.tag(artist, record, "task", self.task_title(task, record))
            if width == 0:
                self.tag(ax.scatter(task["start"], y, marker="|", s=90, color=self.color(record)),
                         record, "task", self.task_title(task, record))
        ax.set_yticks(list(positions.values()),
                      [e["label"] for e in rows] if labels else [""] * len(rows))
        for label in ax.get_yticklabels():
            label.set_fontproperties(self.font)
            label.set_fontsize(10)
        if not rows:
            self.text(ax, (self.left + self.right) / 2, 0, "∅", ha="center", fontsize=24)
        return ax

    def tree(self, fig, area, rows):
        ax = fig.add_axes(area)
        positions = {e["id"]: len(rows) - 1 - i for i, e in enumerate(rows)}
        depth = max([e["depth"] for e in rows] + [0])
        # Horizontal placement uses depth; vertical placement exactly matches the time rows.
        ax.set(xlim=(-.55, depth + .55), ylim=(-.7, max(len(rows) - .3, .7)))
        ax.axis("off")
        for record in rows:
            y, x = positions[record["id"]], record["depth"]
            color = self.color(record)
            parent = self.records.get(record["parent"])
            if parent and parent["id"] in positions:
                px, py = parent["depth"], positions[parent["id"]]
                edge = ax.plot([px + .38, px + .49, px + .49, x - .38],
                               [py, py, y, y], color=color, linewidth=1.5, zorder=1)[0]
                self.tag(edge, record, "edge", record["label"])
            box = ax.add_patch(FancyBboxPatch((x - .38, y - .24), .76, .48,
                               boxstyle="round,pad=.015", facecolor="white",
                               edgecolor=color, linewidth=1.5, zorder=2))
            self.tag(box, record, "node", record["label"])
            label = record["label"]
            text = self.text(ax, x, y, label if len(label) <= 18 else label[:17] + "…",
                             ha="center", va="center", fontsize=9, color="#243444", zorder=3)
            self.tag(text, record, "label", record["label"])
        return ax

    @staticmethod
    def task_title(task, record):
        return (f'{record["label"]} · {task["operation"]} · '
                f'{task["start"]}–{task["end"]} · {task["machine"]}')

    def devices(self, fig):
        machines = self.data["machines"]
        ax = self.time_axes(fig, [.13, .21, .84, .66], len(machines))
        positions = {m["id"]: len(machines) - 1 - i for i, m in enumerate(machines)}
        for task in self.data["tasks"]:
            record = self.records[task["experiment"]]
            y = positions[task["machine"]]
            width = task["end"] - task["start"]
            box = ax.add_patch(Rectangle((task["start"], y - .3), width, .6,
                                        facecolor=self.color(record), edgecolor="white",
                                        linewidth=.8, zorder=3))
            self.tag(box, record, "task", self.task_title(task, record))
            if width == 0:
                self.tag(ax.scatter(task["start"], y, marker="|", s=90, color=self.color(record)),
                         record, "task", self.task_title(task, record))
            # Narrow bars expose their name on hover and focus, avoiding colliding labels.
            if width / self.span * 950 > max(35, len(record["label"]) * 7):
                label = self.text(ax, task["start"] + width / 2, y, record["label"],
                                  ha="center", va="center", fontsize=9, color="white", zorder=4)
                self.tag(label, record, "label", self.task_title(task, record))
        ax.set_yticks(list(positions.values()), [m["name"] for m in machines])
        for label in ax.get_yticklabels():
            label.set_fontproperties(self.font)
        return ax

    def save(self, fig, output, name):
        paths = []
        for suffix in ("png", "svg"):
            path = output / f"{name}.{suffix}"
            metadata = {"Software": "GEMS"} if suffix == "png" else {"Date": None, "Creator": "GEMS"}
            fig.savefig(path, dpi=140, metadata=metadata)
            if suffix == "svg":
                path.write_text("\n".join(line.rstrip() for line in
                                          path.read_text(encoding="utf-8").splitlines()) + "\n",
                                encoding="utf-8")
            paths.append(path)
        return paths

    def inline(self, fig, prefix):
        stream = io.StringIO()
        fig.savefig(stream, format="svg", metadata={"Date": None, "Creator": "GEMS"})
        ET.register_namespace("", SVG)
        ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
        root = ET.fromstring(stream.getvalue())
        identifiers = {node.attrib["id"]: prefix + node.attrib["id"]
                       for node in root.iter() if "id" in node.attrib}
        for node in root.iter():
            gid = node.attrib.get("id")
            if gid in self.tags:
                pick, kind, title = self.tags[gid]
                node.set("data-exp", pick)
                node.set("data-kind", kind)
                node.set("aria-label", title)
                ET.SubElement(node, f"{{{SVG}}}title").text = title
                if kind in ("node", "task"):
                    node.set("tabindex", "0")
                    node.set("role", "button")
            for key, value in list(node.attrib.items()):
                if key == "id":
                    node.set(key, identifiers[value])
                elif value.startswith("#") and value[1:] in identifiers:
                    node.set(key, "#" + identifiers[value[1:]])
                elif "url(#" in value:
                    for old, new in identifiers.items():
                        value = value.replace(f"url(#{old})", f"url(#{new})")
                    node.set(key, value)
        root.set("role", "group")
        return ET.tostring(root, encoding="unicode")


JAVASCRIPT = """
const records = new Map(DATA.experiments.map(e => [e.pick, e]));
const byId = new Map(DATA.experiments.map(e => [e.id, e]));
let selected = "";
const scope = document.getElementById("scope");
const choice = document.getElementById("choice");
function matching(pick, mode) {
  if (!pick) return new Set(records.keys());
  const e = records.get(pick);
  if (mode === "self") return new Set([pick]);
  if (mode === "family") return new Set(DATA.experiments.filter(r => r.family === e.family).map(r => r.pick));
  const result = new Set([pick]);
  let parent = byId.get(e.parent);
  while (parent) { result.add(parent.pick); parent = byId.get(parent.parent); }
  const stack = [e.id];
  while (stack.length) {
    const id = stack.pop();
    for (const child of DATA.experiments.filter(r => r.parent === id)) {
      result.add(child.pick); stack.push(child.id);
    }
  }
  return result;
}
function render() {
  const visible = matching(selected, scope.value);
  document.querySelectorAll("[data-exp]").forEach(node => {
    node.classList.toggle("dimmed", !visible.has(node.dataset.exp));
    node.classList.toggle("selected", selected === node.dataset.exp);
    if (node.getAttribute("role") === "button")
      node.setAttribute("aria-pressed", String(selected === node.dataset.exp));
  });
  choice.value = selected;
  document.getElementById("selected").textContent = selected ? records.get(selected).label : "全体";
}
function choose(pick) { selected = records.has(pick) ? pick : ""; render(); }
choice.addEventListener("change", () => choose(choice.value));
scope.addEventListener("change", render);
document.getElementById("reset").addEventListener("click", () => choose(""));
document.addEventListener("click", event => {
  const graphic = event.target.closest("[data-exp]");
  if (graphic) choose(graphic.dataset.exp);
});
document.addEventListener("keydown", event => {
  const graphic = event.target.closest("[data-exp][tabindex]");
  if (graphic && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault(); choose(graphic.dataset.exp);
  }
});
render();
"""


def interactive(renderer, output):
    rows = renderer.data["experiments"]
    tree = renderer.figure(len(rows), width=max(12, 3 * (max([e["depth"] for e in rows] + [0]) + 1)))
    renderer.tree(tree, [.04, .08, .92, .86], rows)
    device = renderer.figure(len(renderer.data["machines"]), width=16)
    renderer.devices(device)
    options = '<option value="">全体</option>' + "".join(
        f'<option value="{e["pick"]}">{html.escape(e["label"])}</option>' for e in rows)
    encoded = json.dumps(renderer.data, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    document = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Experiment</title><style>
body{{margin:0;background:#f4f6f8;color:#243444;font-family:system-ui,sans-serif}}
main{{max-width:1400px;margin:auto;padding:16px}}
.controls{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;position:sticky;top:0;z-index:3;background:#f4f6f8;padding:8px 0}}
select,button{{font:inherit;background:white;color:inherit;border:1px solid #8996a3;border-radius:5px;padding:7px 10px;max-width:100%}}
.panel{{background:white;margin:12px 0;border:1px solid #dce2e8;border-radius:8px;overflow:auto}}
.sr-only{{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)}}
svg{{display:block;width:100%;height:auto;min-width:750px}}
[data-exp]{{cursor:pointer}}[data-kind=label]{{pointer-events:none}}.dimmed{{opacity:.13}}[data-kind=node].selected path,[data-kind=task].selected path{{stroke:#172b4d!important;stroke-width:2.5!important}}
[data-exp]:focus-visible path{{stroke:#172b4d!important;stroke-width:3!important}}select:focus-visible,button:focus-visible{{outline:3px solid #0072b2}}
</style></head><body><main>
<div class="controls"><select id="choice" aria-label="Experiment">{options}</select>
<select id="scope" aria-label="範囲"><option value="relatives">祖先・子孫</option><option value="family">家系</option><option value="self">選択のみ</option></select>
<button id="reset">全体</button><span id="selected" class="sr-only" aria-live="polite"></span></div>
<div class="panel" id="lineage" aria-label="親子">{renderer.inline(tree, "tree-")}</div>
<div class="panel" id="schedule" aria-label="装置・予約時間">{renderer.inline(device, "schedule-")}</div>
<noscript>JavaScript を有効にしてください。</noscript>
</main><script>const DATA = {encoded};
{JAVASCRIPT}</script></body></html>"""
    path = output / "review.html"
    path.write_text("\n".join(line.rstrip() for line in document.splitlines()) + "\n", encoding="utf-8")
    return path


def render_retrospective(collection, output, *, font=None, time_unit="min"):
    """Export selected views; output must be a new directory to preserve earlier runs.

    ``time_unit`` labels manager logical time without converting it. Set it to
    your protocol's unit. Supply a font containing the experiment-name glyphs.
    """
    data = review_data(collection)
    output = Path(output)
    if font and not Path(font).is_file():
        raise FileNotFoundError(font)
    output.mkdir(parents=True, exist_ok=False)
    files = []
    with matplotlib.rc_context({"svg.fonttype": "path", "svg.hashsalt": "gems-retrospective-v1"}):
        renderer = Renderer(data, font, time_unit)
        rows = data["experiments"]
        pages = [rows[i:i + ROWS_PER_PAGE] for i in range(0, len(rows), ROWS_PER_PAGE)] or [[]]
        for i, page in enumerate(pages, 1):
            page = renderer.with_ancestors(page)
            fig = renderer.figure(len(page))
            renderer.lifetime(fig, [.17, .16, .8, .74], page)
            renderer.legend(fig)
            files.extend(renderer.save(fig, output, f"overview-{i:02d}"))
        families = list(dict.fromkeys(e["family"] for e in rows))
        for i, family in enumerate(families, 1):
            members = [e for e in rows if e["family"] == family]
            for j, start in enumerate(range(0, len(members), ROWS_PER_PAGE), 1):
                page = members[start:start + ROWS_PER_PAGE]
                page = renderer.with_ancestors(page)
                depth = max(e["depth"] for e in page) + 1
                tree_fraction = min(.55, max(.24, .12 * depth))
                fig = renderer.figure(len(page), width=18)
                renderer.tree(fig, [.025, .16, tree_fraction - .035, .74], page)
                renderer.lifetime(fig, [tree_fraction + .03, .16, .945 - tree_fraction, .74],
                                  page, labels=False)
                renderer.legend(fig)
                files.extend(renderer.save(fig, output, f"family-{i:03d}-{j:02d}"))
        files.append(interactive(renderer, output))
    data_path = output / "review-data.json"
    data_path.write_text(json.dumps(dict(data, time_unit=time_unit), ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    files.append(data_path)
    manifest = dict(version=1, time_basis=data["time_basis"], complete=data["complete"],
                    matplotlib=matplotlib.__version__,
                    renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(collection.to_json().encode()).hexdigest(),
                    font=(None if not font else hashlib.sha256(Path(font).read_bytes()).hexdigest()),
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return files + [manifest_path]
