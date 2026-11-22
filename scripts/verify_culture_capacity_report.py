"""Exercise the portable capacity report in an optional Chromium environment."""

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

from playwright.sync_api import sync_playwright


def verify_report(report, output):
    output.mkdir(parents=True, exist_ok=False)
    data = json.loads(re.search(r"const DATA=(.*?);\nlet selected", report.read_text(),
                               re.DOTALL)[1])
    verification = {"scenarios": 0, "errors": [], "external_requests": []}
    with tempfile.TemporaryDirectory() as directory, sync_playwright() as playwright:
        portable = Path(directory) / "report.html"
        shutil.copyfile(report, portable)
        browser = playwright.chromium.launch()
        verification["browser"] = browser.version
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("pageerror", lambda error: verification["errors"].append(str(error)))
        page.on("request", lambda request:
                verification["external_requests"].append(request.url)
                if not request.url.startswith(("file:", "data:")) else None)
        page.goto(portable.as_uri())
        assert page.locator("script[src],link[href]").count() == 0
        assert page.locator("img").evaluate_all(
            "nodes => nodes.every(n => n.complete && n.naturalWidth > 0)")
        buttons = page.locator("#grid button")
        assert buttons.count() == len(data["hek_counts"])*len(data["ips_counts"])
        for condition in data["conditions"]:
            h, i = condition["hek"], condition["ips"]
            selector = f'#grid button[data-h="{h}"][data-i="{i}"]'
            page.locator(selector).click()
            rows = [r for r in data["runs"] if r["hek"] == h and r["ips"] == i]
            assert page.locator("#replicates tr").count() == len(rows)+1
            for criterion in ["completed", "primary_accepted", "supplementary_accepted"]:
                page.locator("#criterion").select_option(criterion)
                assert page.locator(selector).inner_text() == (
                    f"{condition[criterion]}/{condition['repetitions_recorded']}")
                verification["scenarios"] += 1
            representative = data["representatives"][f"{h}_{i}"]
            assert page.locator("#timeline svg").count() == 1
            assert str(representative["repetition"]) in page.locator(
                "#timelinecaption").inner_text()
            page.locator("#window").select_option("day")
            for day in [0, max(0, int(representative["end"]//1440)-1)]:
                page.locator("#day").evaluate(
                    "(node, value) => {node.value=value;node.dispatchEvent(new Event('input'));}",
                    day)
                expected = [
                    e for e in representative["events"] if e[1] > day*1440
                    and e[0] < min(representative["end"], (day+1)*1440) and e[0] != e[1]]
                assert page.locator("#timeline rect").count() == len(expected)
                verification["scenarios"] += 1
            page.locator("#window").select_option("all")
        page.locator("#criterion").select_option("primary_accepted")
        page.screenshot(path=str(output / "desktop.png"), full_page=False)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.locator("#timeline svg").is_visible()
        page.screenshot(path=str(output / "mobile.png"), full_page=False)
        verification["scenarios"] += 1
        assert not verification["errors"], verification["errors"]
        assert not verification["external_requests"], verification["external_requests"]
        browser.close()
    verification["standalone_copy_verified"] = True
    (output / "verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    print(json.dumps(verification, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_report(args.html, args.output)


if __name__ == "__main__":
    main()
