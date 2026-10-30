const DATA = {"kind": "design_fixture", "observed_run": false, "description": "4案で共通に使う説明用の仮想記録。測定値・実機の実行記録ではない。", "time_unit": "minute", "origin": "first registration", "machines": [{"id": "imager", "name": "撮像機"}, {"id": "processor", "name": "処理機"}], "experiments": [{"id": "A", "name": "親A", "parent": null, "family": "A", "born": 0, "ended": 24, "reason": "split"}, {"id": "A1", "name": "A1", "parent": "A", "family": "A", "born": 24, "ended": 66, "reason": "finished"}, {"id": "A2", "name": "A2", "parent": "A", "family": "A", "born": 24, "ended": 50, "reason": "split"}, {"id": "A2a", "name": "A2a", "parent": "A2", "family": "A", "born": 50, "ended": 110, "reason": "finished"}, {"id": "A2b", "name": "A2b", "parent": "A2", "family": "A", "born": 50, "ended": 100, "reason": "finished"}, {"id": "B", "name": "親B", "parent": null, "family": "B", "born": 0, "ended": 36, "reason": "split"}, {"id": "B1", "name": "B1", "parent": "B", "family": "B", "born": 36, "ended": 85, "reason": "finished"}, {"id": "B2", "name": "B2", "parent": "B", "family": "B", "born": 36, "ended": 116, "reason": "finished"}], "splits": [{"parent": "A", "children": ["A1", "A2"], "time": 24}, {"parent": "B", "children": ["B1", "B2"], "time": 36}, {"parent": "A2", "children": ["A2a", "A2b"], "time": 50}], "tasks": [{"id": "A-1", "experiment": "A", "machine": "imager", "start": 2, "end": 7}, {"id": "A-2", "experiment": "A", "machine": "processor", "start": 10, "end": 13}, {"id": "A-3", "experiment": "A", "machine": "imager", "start": 20, "end": 24}, {"id": "B-1", "experiment": "B", "machine": "imager", "start": 7, "end": 12}, {"id": "B-2", "experiment": "B", "machine": "processor", "start": 20, "end": 23}, {"id": "B-3", "experiment": "B", "machine": "imager", "start": 32, "end": 36}, {"id": "A1-1", "experiment": "A1", "machine": "imager", "start": 26, "end": 31}, {"id": "A1-2", "experiment": "A1", "machine": "processor", "start": 47, "end": 50}, {"id": "A1-3", "experiment": "A1", "machine": "imager", "start": 62, "end": 66}, {"id": "A2-1", "experiment": "A2", "machine": "imager", "start": 36, "end": 41}, {"id": "A2-2", "experiment": "A2", "machine": "processor", "start": 43, "end": 46}, {"id": "A2-3", "experiment": "A2", "machine": "imager", "start": 46, "end": 50}, {"id": "A2a-1", "experiment": "A2a", "machine": "imager", "start": 52, "end": 57}, {"id": "A2a-2", "experiment": "A2a", "machine": "processor", "start": 70, "end": 73}, {"id": "A2a-3", "experiment": "A2a", "machine": "imager", "start": 106, "end": 110}, {"id": "A2b-1", "experiment": "A2b", "machine": "imager", "start": 57, "end": 62}, {"id": "A2b-2", "experiment": "A2b", "machine": "processor", "start": 74, "end": 77}, {"id": "A2b-3", "experiment": "A2b", "machine": "imager", "start": 96, "end": 100}, {"id": "B1-1", "experiment": "B1", "machine": "imager", "start": 66, "end": 71}, {"id": "B1-2", "experiment": "B1", "machine": "processor", "start": 78, "end": 81}, {"id": "B1-3", "experiment": "B1", "machine": "imager", "start": 81, "end": 85}, {"id": "B2-1", "experiment": "B2", "machine": "imager", "start": 71, "end": 76}, {"id": "B2-2", "experiment": "B2", "machine": "processor", "start": 85, "end": 88}, {"id": "B2-3", "experiment": "B2", "machine": "imager", "start": 112, "end": 116}]};
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
