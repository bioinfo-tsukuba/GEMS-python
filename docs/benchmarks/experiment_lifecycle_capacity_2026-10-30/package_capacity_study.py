"""Package completed local evidence; no solver or Git mutation."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

root = Path(__file__).resolve().parent
source = root / "capacity-a719506"
target = root / "archive/docs/benchmarks/experiment_lifecycle_capacity_2026-10-30"
assert json.loads((root / "pipeline_status.json").read_text())["stage"] == "complete"
assert json.loads((root / "final_checks_status.json").read_text())["stage"] == "complete"
files = sorted(p for p in source.rglob("*") if p.is_file())
assert not any(p.is_symlink() for p in source.rglob("*"))
manifest = {}
for path in files:
    manifest[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
(target / "raw-manifest.json").write_text(json.dumps(
    dict(file_count=len(files), source_commit=json.loads((source / "metadata.json").read_text())["commit"],
         sha256=manifest), indent=2, sort_keys=True))
with (target / "raw.tar.gz").open("wb") as stream:
    with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for path in sorted(source.rglob("*")):
                info = archive.gettarinfo(str(path), path.relative_to(source).as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o755 if info.isdir() else 0o644
                if info.isfile():
                    with path.open("rb") as content:
                        archive.addfile(info, content)
                else:
                    archive.addfile(info)
shutil.copytree(root / "statistics", target / "statistics")
for name in ("verification.json", "boundaries.json"):
    shutil.copy2(source / name, target / name)
logs = target / "test-logs"
logs.mkdir()
for path in sorted(root.glob("*.txt")):
    if path.name in ("install.txt", "bootstrap.txt", "latest-partial-directory.txt"):
        continue
    with (logs / f"{path.name}.gz").open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            compressed.write(path.read_bytes())
shutil.copy2(Path(__file__), target / "package_capacity_study.py")
shutil.copy2(root / "finish_capacity_study.py", target / "session_orchestration.py")
shutil.copy2(root / "final_checks.py", target / "session_final_checks.py")
hashes = []
for path in sorted(p for p in target.rglob("*") if p.is_file()):
    hashes.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(target).as_posix()}")
(target / "SHA256SUMS.txt").write_text("\n".join(hashes)+"\n")
print(json.dumps(dict(raw_files=len(files), raw_archive_bytes=(target / "raw.tar.gz").stat().st_size,
                      public_files=len(hashes))))
