#!/usr/bin/env python3
"""
publish.py — publish cache/results/evaluation to GitHub so the deployed
Reflex Cloud app reads fresh data via raw URLs (see data_urls.py).

Flow
----
1. Copy ``predictions_cache.json``, ``results.csv``, ``evaluation_report.json``
   into the data repo (``AUGO_DATA_REPO`` or ``./data_repo``).
2. ``git add`` + commit + push (with ``--push``).
3. Write ``remote_data_urls.json`` in this project with the raw GitHub URLs so
   the cloud app picks them up (env vars still override in data_urls.py).

Usage
-----
    python publish.py            # copy + write remote_data_urls.json (no push)
    python publish.py --push     # also git commit & push

Config (env / .env)
    AUGO_DATA_REPO  path to the git repo that hosts the data files
    AUGO_RAW_BASE   raw base URL, e.g. https://raw.githubusercontent.com/<owner>/<repo>/main
                    (if omitted, derived from the repo's origin remote)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

from dotenv import load_dotenv

load_dotenv()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_REPO = os.getenv("AUGO_DATA_REPO") or os.path.join(APP_DIR, "data_repo")
RAW_BASE = os.getenv("AUGO_RAW_BASE", "").strip()
URLS_FILE = os.path.join(APP_DIR, "remote_data_urls.json")

DATA_FILES = [
    "predictions_cache.json",
    "results.csv",
    "evaluation_report.json",
]


def _run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _derive_raw_base(repo_dir: str) -> str | None:
    r = _run(["git", "config", "--get", "remote.origin.url"], cwd=repo_dir)
    url = r.stdout.strip()
    if not url:
        return None
    url = url.removesuffix(".git")
    if "github.com/" in url:
        owner_repo = url.split("github.com/")[-1].strip("/")
        return f"https://raw.githubusercontent.com/{owner_repo}/main"
    return None


def _git_ok(repo_dir: str) -> bool:
    return _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_dir).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="git commit + push after copying")
    args = parser.parse_args()

    os.makedirs(DATA_REPO, exist_ok=True)
    copied = []
    for name in DATA_FILES:
        src = os.path.join(APP_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(DATA_REPO, name))
            copied.append(name)
    print(f"Copied {len(copied)} file(s) -> {DATA_REPO}: {copied or 'none'}")

    raw_base = RAW_BASE or _derive_raw_base(DATA_REPO)
    if raw_base:
        urls = {
            "predictions_cache_url": f"{raw_base}/predictions_cache.json",
            "results_url": f"{raw_base}/results.csv",
        }
        with open(URLS_FILE, "w", encoding="utf-8") as f:
            json.dump(urls, f, indent=2)
        print(f"Wrote {URLS_FILE} -> {urls}")
    else:
        print("⚠️  No AUGO_RAW_BASE set and no git origin found — remote_data_urls.json not updated.")

    if args.push:
        if not _git_ok(DATA_REPO):
            print("❌  Not a git repo — run:  git init && git remote add origin <url>")
            return
        _run(["git", "add", "-A"], cwd=DATA_REPO)
        _run(["git", "commit", "-m", "update Augo data", "--allow-empty"], cwd=DATA_REPO)
        r = _run(["git", "push"], cwd=DATA_REPO)
        if r.returncode == 0:
            print("✓ Pushed to origin.")
        else:
            print(f"❌  Push failed: {r.stderr.strip() or r.stdout.strip()}")


if __name__ == "__main__":
    main()
