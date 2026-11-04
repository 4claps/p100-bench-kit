"""Model manifest, local model discovery, checksum verification and downloads.

A manifest model counts as present when every file exists under the models
dir, either at its manifest name or as any .gguf with the same size whose
SHA256 matches (so existing downloads with other names are picked up).
Hashes are cached by (path, size, mtime) because a 25 GB file takes minutes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HF_URL = "https://huggingface.co/{repo}/resolve/main/{name}"
CHUNK = 8 * 1024 * 1024
SHARD_RE = re.compile(r"-(\d{5})-of-\d{5}\.gguf$")


def load_manifest(path: Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text())
    if data.get("schema") != 1:
        raise ValueError("unsupported manifest schema")
    models = data["models"]
    for m in models:
        m["size"] = sum(f["size"] for f in m["files"])
    return models


def sha256_file(path: Path, progress=None) -> str:
    h = hashlib.sha256()
    done = 0
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(done)
    return h.hexdigest()


@dataclass
class Job:
    model_id: str
    kind: str                 # download | verify
    total: int
    done: int = 0
    status: str = "running"   # running | done | failed
    error: str | None = None
    started: float = field(default_factory=time.time)

    def view(self) -> dict:
        return {"model_id": self.model_id, "kind": self.kind, "total": self.total, "done": self.done,
                "status": self.status, "error": self.error, "started": self.started}


class ModelStore:
    def __init__(self, models_dir: Path, data_dir: Path, manifest_path: Path):
        self.dir = Path(models_dir)
        self.cache_path = Path(data_dir) / "hashes.json"
        self.manifest = load_manifest(manifest_path)
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        try:
            self._cache = json.loads(self.cache_path.read_text())
        except (OSError, ValueError):
            self._cache = {}

    # ---------------- hashing cache ----------------
    def _key(self, p: Path) -> str:
        st = p.stat()
        return f"{p}|{st.st_size}|{int(st.st_mtime)}"

    def cached_sha(self, p: Path) -> str | None:
        try:
            return self._cache.get(self._key(p))
        except OSError:
            return None

    def _store_sha(self, p: Path, sha: str) -> None:
        with self._lock:
            self._cache[self._key(p)] = sha
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=1))

    # ---------------- discovery ----------------
    def _ggufs(self) -> list[Path]:
        if not self.dir.exists():
            return []
        return sorted(p for p in self.dir.rglob("*.gguf") if p.is_file())

    def resolve(self, m: dict) -> list[Path | None]:
        """Local path for each manifest file (None if missing)."""
        files = self._ggufs()
        out = []
        for f in m["files"]:
            exact = self.dir / f["name"]
            if exact.exists() and exact.stat().st_size == f["size"]:
                out.append(exact)
                continue
            cand = [p for p in files if p.stat().st_size == f["size"]]
            match = next((p for p in cand if self.cached_sha(p) == f["sha256"]), None)
            out.append(match or (cand[0] if len(cand) == 1 else None))
        return out

    def status(self, m: dict) -> dict:
        paths = self.resolve(m)
        present = all(paths)
        verified = present and all(self.cached_sha(p) == f["sha256"] for p, f in zip(paths, m["files"]))
        job = self.jobs.get(m["id"])
        return {"id": m["id"], "label": m["label"], "kind": m["kind"], "mtp": m["mtp"],
                "vram_gb": m["vram_gb"], "size": m["size"], "repo": m["repo"],
                "present": present, "verified": bool(verified),
                "path": str(paths[0]) if paths and paths[0] else None,
                "job": job.view() if job else None}

    def list(self) -> dict:
        standard = [self.status(m) for m in self.manifest]
        known = {s["path"] for s in standard if s["path"]}
        for m in self.manifest:
            known.update(str(p) for p in self.resolve(m) if p)
        custom = []
        for p in self._ggufs():
            shard = SHARD_RE.search(p.name)
            if str(p) in known or (shard and shard.group(1) != "00001"):
                continue  # only the first shard of a split model is loadable
            custom.append({"id": "custom:" + str(p.relative_to(self.dir)), "label": p.name,
                           "path": str(p), "size": p.stat().st_size, "custom": True})
        free = shutil.disk_usage(self.dir).free if self.dir.exists() else None
        return {"standard": standard, "custom": custom, "models_dir": str(self.dir), "free_bytes": free}

    def get(self, model_id: str) -> dict | None:
        return next((m for m in self.manifest if m["id"] == model_id), None)

    def model_path(self, model_id: str) -> tuple[Path | None, bool]:
        """(path to load, comparable). Custom ids map to a file under the models dir."""
        if model_id.startswith("custom:"):
            p = (self.dir / model_id[7:]).resolve()
            if self.dir.resolve() not in p.parents or not p.exists():
                return None, False
            return p, False
        m = self.get(model_id)
        if not m:
            return None, False
        paths = self.resolve(m)
        if not all(paths):
            return None, True
        return paths[0], True

    # ---------------- jobs ----------------
    def _start(self, model_id: str, kind: str, total: int, target) -> Job:
        cur = self.jobs.get(model_id)
        if cur and cur.status == "running":
            raise RuntimeError(f"a {cur.kind} job is already running for {model_id}")
        job = Job(model_id, kind, total)
        self.jobs[model_id] = job

        def wrap():
            try:
                target(job)
                job.status = "done"
            except Exception as e:  # noqa: BLE001
                job.status, job.error = "failed", str(e)
        threading.Thread(target=wrap, daemon=True, name=f"{kind}-{model_id}").start()
        return job

    def verify(self, model_id: str) -> Job:
        m = self.get(model_id)
        if not m:
            raise KeyError(model_id)
        paths = self.resolve(m)
        if not all(paths):
            raise RuntimeError("model files are not all present")

        def work(job: Job):
            base = 0
            bad = []
            for p, f in zip(paths, m["files"]):
                sha = sha256_file(p, lambda d, b=base: setattr(job, "done", b + d))
                base += f["size"]
                self._store_sha(p, sha)
                if sha != f["sha256"]:
                    bad.append(p.name)
            if bad:
                raise RuntimeError("checksum mismatch: " + ", ".join(bad))
        return self._start(model_id, "verify", m["size"], work)

    def download(self, model_id: str) -> Job:
        import httpx
        m = self.get(model_id)
        if not m:
            raise KeyError(model_id)
        paths = self.resolve(m)
        need = sum(f["size"] for p, f in zip(paths, m["files"]) if not p)
        self.dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.dir).free
        if need > free - 2 * 1024**3:
            raise RuntimeError(f"not enough disk: need {need / 1e9:.1f} GB, free {free / 1e9:.1f} GB")

        def work(job: Job):
            base = 0
            for p, f in zip(paths, m["files"]):
                if p:
                    base += f["size"]
                    job.done = base
                    continue
                dest = self.dir / f["name"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                part = dest.with_name(dest.name + ".part")
                h = hashlib.sha256()
                have = part.stat().st_size if part.exists() else 0
                if have:
                    with open(part, "rb") as fh:
                        while chunk := fh.read(CHUNK):
                            h.update(chunk)
                headers = {"Range": f"bytes={have}-"} if have else {}
                url = HF_URL.format(repo=m["repo"], name=f["name"])
                with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                                  timeout=httpx.Timeout(60, read=300)) as r:
                    if have and r.status_code != 206:
                        have, h = 0, hashlib.sha256()
                    r.raise_for_status()
                    with open(part, "ab" if have else "wb") as out:
                        written = have
                        for chunk in r.iter_bytes(CHUNK):
                            out.write(chunk)
                            h.update(chunk)
                            written += len(chunk)
                            job.done = base + written
                if part.stat().st_size != f["size"]:
                    raise RuntimeError(f"{f['name']}: size {part.stat().st_size} != {f['size']}")
                if h.hexdigest() != f["sha256"]:
                    part.unlink()
                    raise RuntimeError(f"{f['name']}: checksum mismatch, partial file removed")
                os.replace(part, dest)
                self._store_sha(dest, f["sha256"])
                base += f["size"]
        return self._start(model_id, "download", m["size"], work)
