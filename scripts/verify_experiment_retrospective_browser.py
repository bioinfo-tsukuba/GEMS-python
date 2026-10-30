"""Verify linked SVG selection in Chromium, including HTML copied on its own.

Requires the optional verification tool playwright and its Chromium browser;
neither is a runtime dependency. The caller supplies browser environment paths.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from playwright.sync_api import sync_playwright


def expected(records, pick, scope):
    index = {e["id"]: e for e in records}
    selected = next(e for e in records if e["pick"] == pick)
    if scope == "self":
        return {pick}
    if scope == "family":
        return {e["pick"] for e in records if e["family"] == selected["family"]}
    ids = {selected["id"]}
    parent = selected["parent"]
    while parent in index:
        ids.add(parent)
        parent = index[parent]["parent"]
    # Descendants only: siblings of selected are not added via ancestors.
    for e in records:
        parent = e["parent"]
        while parent in index:
            if parent == selected["id"]:
                ids.add(e["id"])
                break
            parent = index[parent]["parent"]
    return {index[uid]["pick"] for uid in ids}


def verify(page, records, pick, scope):
    wanted = expected(records, pick, scope)
    for panel in ("lineage", "schedule"):
        observed = page.locator(f"#{panel} [data-exp]").evaluate_all(
            """nodes => [...new Set(nodes.filter(n => !n.classList.contains("dimmed"))
                    .map(n => n.dataset.exp))]""")
        represented = set(page.locator(f"#{panel} [data-exp]").evaluate_all(
            "nodes => nodes.map(n => n.dataset.exp)"))
        assert set(observed) == wanted & represented, (panel, pick, scope, observed, wanted)
        selected = page.locator(f"#{panel} .selected[data-exp]").evaluate_all(
            "nodes => nodes.map(n => n.dataset.exp)")
        assert set(selected) == ({pick} & represented), (panel, selected)
        assert page.locator(f"#{panel} .dimmed").evaluate_all(
            "nodes => nodes.every(n => Number(getComputedStyle(n).opacity) < .2)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.data.read_text())["experiments"]
    if not records:
        raise ValueError("Browser verification needs at least one Experiment")
    report = dict(scenarios=0, errors=[], requests=[])
    with tempfile.TemporaryDirectory() as directory, sync_playwright() as p:
        single = Path(directory) / "review.html"
        shutil.copyfile(args.html, single)
        browser = p.chromium.launch()
        report["browser"] = browser.version
        page = browser.new_page(viewport={"width":1440, "height":1100})
        page.on("pageerror", lambda error: report["errors"].append(str(error)))
        page.on("request", lambda request: report["requests"].append(request.url))
        page.goto(single.as_uri())
        assert page.locator("script[src],link[href],img[src]").count() == 0
        ids = page.locator("[id]").evaluate_all("nodes => nodes.map(n => n.id)")
        assert len(ids) == len(set(ids)), "Duplicate SVG/DOM identifiers"
        for record in records:
            for scope in ("relatives", "family", "self"):
                page.locator("#scope").select_option(scope)
                page.locator("#choice").select_option(record["pick"])
                verify(page, records, record["pick"], scope)
                report["scenarios"] += 1
        for panel in ("lineage", "schedule"):
            targets = page.locator(f"#{panel} [data-exp][tabindex]")
            for i in range(targets.count()):
                target = targets.nth(i)
                pick = target.get_attribute("data-exp")
                page.locator("#reset").click()
                target.click()
                verify(page, records, pick, "self")
                target.focus()
                page.keyboard.press("Enter")
                verify(page, records, pick, "self")
                page.keyboard.press("Space")
                verify(page, records, pick, "self")
                report["scenarios"] += 3
        page.locator("#reset").click()
        assert page.locator(".dimmed,.selected").count() == 0
        page.set_viewport_size({"width":390, "height":844})
        page.locator("#choice").select_option(records[-1]["pick"])
        verify(page, records, records[-1]["pick"], "self")
        report["scenarios"] += 2
        page.set_viewport_size({"width":1440, "height":1100})
        selected = next((e for e in records if e["parent"] and
                         any(c["parent"] == e["id"] for c in records)), records[0])
        page.locator("#scope").select_option("relatives")
        page.locator("#choice").select_option(selected["pick"])
        verify(page, records, selected["pick"], "relatives")
        page.screenshot(path=str(args.screenshot), full_page=True)
        report["screenshot_selection"] = selected["label"]
        report["scenarios"] += 1
        assert not report["errors"], report["errors"]
        assert report["requests"] == [single.as_uri()], report["requests"]
        browser.close()
    report["standalone"] = True
    report["requests"] = ["file:///review.html"]
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
