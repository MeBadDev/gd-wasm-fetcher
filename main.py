#!/usr/bin/env python3
"""
Godot WASM/JS fetcher for web exports.

Features
- Discovers ALL stable Godot releases (>= 3.0.0) via GitHub API at runtime.
- Downloads each release's export templates (.tpz) asset.
- Extracts only the files needed for Web exports (WASM/JS, including worker JS) from the archive.
- Stores artifacts in a versioned folder: <dest>/<version>/
- Idempotent: skips already-processed versions unless the upstream asset changed.
- Cron-friendly logging and zero non-stdlib dependencies.

Environment
- Optional: GITHUB_TOKEN to increase GitHub API rate limits.

Usage
  python main.py --dest ./store

"""
from __future__ import annotations

import argparse
import hashlib
import json
import io
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


GITHUB_API = "https://api.github.com"
GODOT_REPO = "godotengine/godot"


@dataclass(frozen=True, order=True)
class Version:
	major: int
	minor: int
	patch: int = 0

	@staticmethod
	def parse(tag: str) -> Optional["Version"]:
		"""Parse tag names like '4.2.2-stable' or '3.0-stable' into Version.
		Returns None if the tag isn't a stable release.
		"""
		# Ensure it's stable, not alpha/beta/rc
		if not tag.endswith("-stable"):
			return None
		core = tag[:-7]  # strip '-stable'
		parts = core.split(".")
		try:
			if len(parts) == 2:
				major, minor = int(parts[0]), int(parts[1])
				patch = 0
			elif len(parts) == 3:
				major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
			else:
				return None
		except ValueError:
			return None
		return Version(major, minor, patch)

	def __str__(self) -> str:
		if self.patch:
			return f"{self.major}.{self.minor}.{self.patch}"
		return f"{self.major}.{self.minor}"


def http_get_json(url: str, token: Optional[str] = None, retries: int = 3, timeout: int = 30):
	headers = {
		"Accept": "application/vnd.github+json",
		"User-Agent": "gd-wasm-fetcher/1.0",
	}
	if token:
		headers["Authorization"] = f"Bearer {token}"
	backoff = 2
	for attempt in range(1, retries + 1):
		try:
			req = Request(url, headers=headers)
			with urlopen(req, timeout=timeout) as resp:
				data = resp.read()
				return json.loads(data.decode("utf-8"))
		except HTTPError as e:
			if e.code in (429, 500, 502, 503, 504) and attempt < retries:
				logging.warning("HTTP %s on %s (attempt %d/%d), retrying in %ds...", e.code, url, attempt, retries, backoff)
				time.sleep(backoff)
				backoff *= 2
				continue
			raise
		except URLError as e:
			if attempt < retries:
				logging.warning("Network error on %s (attempt %d/%d): %s; retrying in %ds...", url, attempt, retries, e, backoff)
				time.sleep(backoff)
				backoff *= 2
				continue
			raise


def http_download(url: str, dest_path: Path, token: Optional[str] = None, retries: int = 3, timeout: int = 60):
	headers = {
		"User-Agent": "gd-wasm-fetcher/1.0",
	}
	# GitHub's browser_download_url does not require auth, but allow token for robustness.
	if token:
		headers["Authorization"] = f"Bearer {token}"
	tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
	backoff = 2
	for attempt in range(1, retries + 1):
		try:
			req = Request(url, headers=headers)
			with urlopen(req, timeout=timeout) as resp, open(tmp_path, "wb") as f:
				shutil.copyfileobj(resp, f)
			tmp_path.replace(dest_path)
			return
		except HTTPError as e:
			if e.code in (429, 500, 502, 503, 504) and attempt < retries:
				logging.warning("HTTP %s downloading %s (attempt %d/%d), retrying in %ds...", e.code, url, attempt, retries, backoff)
				time.sleep(backoff)
				backoff *= 2
				continue
			raise
		except URLError as e:
			if attempt < retries:
				logging.warning("Network error downloading %s (attempt %d/%d): %s; retrying in %ds...", url, attempt, retries, e, backoff)
				time.sleep(backoff)
				backoff *= 2
				continue
			raise
		finally:
			try:
				if tmp_path.exists():
					tmp_path.unlink()
			except Exception:
				pass


def sha256_file(path: Path) -> str:
	h = hashlib.sha256()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(1024 * 1024), b""):
			h.update(chunk)
	return h.hexdigest()


def list_godot_stable_releases(token: Optional[str]) -> List[Dict]:
	"""Return a list of release dicts from GitHub API for stable releases only.
	Includes only releases that have an export templates asset.
	"""
	releases: List[Dict] = []
	page = 1
	while True:
		url = f"{GITHUB_API}/repos/{GODOT_REPO}/releases?per_page=100&page={page}"
		data = http_get_json(url, token=token)
		if not data:
			break
		releases.extend(data)
		page += 1

	stable: List[Dict] = []
	for r in releases:
		if r.get("draft") or r.get("prerelease"):
			continue
		tag = (r.get("tag_name") or "").strip()
		ver = Version.parse(tag)
		if not ver:
			continue
		# Find export templates asset (non-mono)
		assets = r.get("assets") or []
		asset = None
		for a in assets:
			name = a.get("name") or ""
			if name.endswith("_export_templates.tpz") and "mono" not in name.lower():
				asset = a
				break
		if not asset:
			continue
		stable.append({
			"tag": tag,
			"version": ver,
			"release": r,
			"asset": asset,
		})

	# Sort by version ascending
	stable.sort(key=lambda x: x["version"])
	return stable


def ensure_min_version(entries: List[Dict], min_version: Version) -> List[Dict]:
	return [e for e in entries if e["version"] >= min_version]


WEB_PATH_HINTS = (
	"web",           # Godot 4.x 'web' folder
	"webassembly",   # Godot 3.x 'webassembly'
	"html",          # Some archives include 'html' in path
	"javascript",    # Historical
)


def is_web_file(path: str) -> bool:
	p = path.lower()
	if not p.startswith("templates/"):
		return False
	if "/mono/" in p:
		return False
	if not (p.endswith(".wasm") or p.endswith(".js")):
		return False
	# Must also live under a web-related path or be clearly wasm/js in templates
	return any(h in p for h in WEB_PATH_HINTS) or p.endswith(".wasm") or p.endswith(".js")


def extract_web_artifacts(tpz_path: Path, dest_dir: Path) -> List[Dict[str, str]]:
	"""Extract WASM/JS files from the export templates archive into dest_dir.
	Returns a list of {"name": <filename>, "sha256": <hash>} entries.
	"""
	extracted: List[Dict[str, str]] = []
	def add_file_from_bytes(base: str, data: bytes, existing_hashes: Dict[str, List[Tuple[str, str]]]):
		# existing_hashes maps basename -> list[(filename, sha256)] already in dest_dir
		desired_hash = hashlib.sha256(data).hexdigest()
		# Check if an existing file with same base has identical content
		for fname, fhash in existing_hashes.get(base, []):
			if fhash == desired_hash:
				# Identical file already present; skip writing duplicate
				return
		# Determine a unique output name
		out_name = base
		if base in existing_hashes:
			# Only suffix if a different content under same basename exists
			index = len(existing_hashes[base])
			if index > 0:
				out_name = f"{index}_{base}"
		out_path = dest_dir / out_name
		out_path.parent.mkdir(parents=True, exist_ok=True)
		with open(out_path, "wb") as dst:
			dst.write(data)
		out_hash = sha256_file(out_path)
		existing_hashes.setdefault(base, []).append((out_name, out_hash))
		extracted.append({"name": out_name, "sha256": out_hash})

	with zipfile.ZipFile(tpz_path) as zf:
		members = zf.namelist()
		# Avoid filename collisions across direct and nested sources
		existing_hashes: Dict[str, List[Tuple[str, str]]] = {}
		for p in dest_dir.glob("*.wasm"):
			base = p.name
			existing_hashes.setdefault(base, []).append((p.name, sha256_file(p)))
		for p in dest_dir.glob("*.js"):
			base = p.name
			existing_hashes.setdefault(base, []).append((p.name, sha256_file(p)))

		# 1) Direct files in templates/ that are wasm/js
		for m in members:
			if not is_web_file(m):
				continue
			base = os.path.basename(m)
			if not base:
				continue
			with zf.open(m) as src:
				data = src.read()
				add_file_from_bytes(base, data, existing_hashes)

		# 2) Nested HTML/web zips in templates/ (common in 3.x: html.zip, webassembly.zip)
		for m in members:
			ml = m.lower()
			if not ml.endswith(".zip"):
				continue
			if not ml.startswith("templates/"):
				continue
			if not any(h in ml for h in ("html", "html5", "web", "webassembly", "javascript")):
				continue
			try:
				with zf.open(m) as src:
					data = src.read()
				with zipfile.ZipFile(io.BytesIO(data)) as inner:
					for name in inner.namelist():
						nl = name.lower()
						if nl.endswith(".wasm") or nl.endswith(".js"):
							base = os.path.basename(name)
							if not base:
								continue
							with inner.open(name) as src2:
								data2 = src2.read()
								add_file_from_bytes(base, data2, existing_hashes)
			except Exception as e:
				logging.debug("Skipping nested zip %s due to error: %s", m, e)
	return extracted


def save_json(path: Path, data: Dict):
	tmp = path.with_suffix(path.suffix + ".tmp")
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, sort_keys=True)
	tmp.replace(path)


def load_json(path: Path) -> Optional[Dict]:
	try:
		with open(path, "r", encoding="utf-8") as f:
			return json.load(f)
	except FileNotFoundError:
		return None


def collect_present_web_files(version_dir: Path) -> List[Dict[str, str]]:
	"""Collect existing .wasm and .js files from a version directory with hashes."""
	files: List[Dict[str, str]] = []
	if not version_dir.exists():
		return files
	for p in sorted(list(version_dir.glob("*.wasm")) + list(version_dir.glob("*.js"))):
		if p.is_file():
			files.append({"name": p.name, "sha256": sha256_file(p)})
	return files


def process_release(entry: Dict, dest_root: Path, token: Optional[str]) -> bool:
	"""Process a single release entry.
	Returns True if new work was done, False if skipped.
	"""
	ver: Version = entry["version"]
	tag: str = entry["tag"]
	asset = entry["asset"]
	version_dir = dest_root / str(ver)
	version_dir.mkdir(parents=True, exist_ok=True)
	meta_path = version_dir / "metadata.json"
	meta = load_json(meta_path)
	asset_id = asset.get("id")
	asset_name = asset.get("name")
	asset_url = asset.get("browser_download_url")
	if meta and meta.get("asset_id") == asset_id:
		logging.info("%s already up-to-date, skipping", tag)
		return False

	# If files already present for this version, skip downloading and ensure metadata exists.
	present_files = collect_present_web_files(version_dir)
	if present_files and not meta:
		out = {
			"tag": tag,
			"version": str(ver),
			"asset_id": None,
			"asset_name": None,
			"asset_url": None,
			"published_at": entry["release"].get("published_at"),
			"fetched_at": int(time.time()),
			"files": present_files,
		}
		save_json(meta_path, out)
		logging.info("%s files already present; wrote metadata and skipped download", tag)
		return False
	elif present_files and meta:
		logging.info("%s files already present; skipping download", tag)
		return False

	logging.info("Downloading %s -> %s", asset_name, version_dir)
	with tempfile.TemporaryDirectory() as td:
		tpz_path = Path(td) / asset_name
		http_download(asset_url, tpz_path, token=token)
		files = extract_web_artifacts(tpz_path, version_dir)
	out = {
		"tag": tag,
		"version": str(ver),
		"asset_id": asset_id,
		"asset_name": asset_name,
		"asset_url": asset_url,
		"published_at": entry["release"].get("published_at"),
		"fetched_at": int(time.time()),
		"files": files,
	}
	save_json(meta_path, out)
	logging.info("%s processed: %d files", tag, len(out["files"]))
	return True


def build_index(dest_root: Path):
	index = {}
	for child in sorted(dest_root.iterdir() if dest_root.exists() else []):
		if not child.is_dir():
			continue
		meta_path = child / "metadata.json"
		meta = load_json(meta_path)
		if not meta:
			continue
		index[str(child.name)] = meta
	save_json(dest_root / "index.json", index)


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Fetch Godot web export WASM/JS assets for all stable releases >= 3.0")
	parser.add_argument("--dest", default=str(Path.cwd() / "store"), help="Destination directory to store extracted files (default: ./store)")
	parser.add_argument("--min-version", default="3.0", help="Minimum version to include, e.g., 3.0 or 3.2.2 (default: 3.0)")
	parser.add_argument("--log", default=os.environ.get("LOG_LEVEL", "INFO"), help="Log level (DEBUG, INFO, WARNING, ERROR)")
	parser.add_argument("--max-downloads", type=int, default=None, help="Process at most N releases (for testing); default: no limit")
	parser.add_argument("--list-only", action="store_true", help="List matching releases and exit without downloading")
	args = parser.parse_args(argv)

	logging.basicConfig(
		level=getattr(logging, args.log.upper(), logging.INFO),
		format="%(asctime)s %(levelname)s %(message)s",
	)

	# Parse minimum version
	m = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?$", args.min_version.strip())
	if not m:
		logging.error("Invalid --min-version: %s", args.min_version)
		return 2
	min_version = Version(int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))
	if (min_version.major, min_version.minor) < (3, 0):
		logging.warning("Raising minimum to 3.0 as requested by requirements")
		min_version = Version(3, 0, 0)

	dest_root = Path(args.dest).resolve()
	dest_root.mkdir(parents=True, exist_ok=True)
	token = os.environ.get("GITHUB_TOKEN") or None
	if not token:
		logging.info("No GITHUB_TOKEN provided; GitHub API rate limits may apply.")

	logging.info("Discovering Godot stable releases via GitHub API...")
	entries = list_godot_stable_releases(token)
	entries = ensure_min_version(entries, min_version)
	if not entries:
		logging.warning("No stable releases found >= %s", min_version)
		return 0

	logging.info("Found %d stable releases from %s to %s", len(entries), entries[0]["tag"], entries[-1]["tag"])
	if args.list_only:
		for e in entries:
			v: Version = e["version"]
			a = e["asset"]
			print(f"{e['tag']:>16}  v={str(v):<8}  asset={a.get('name')}")
		return 0
	changed = 0
	processed = 0
	for entry in entries:
		try:
			if process_release(entry, dest_root, token):
				changed += 1
			processed += 1
			if args.max_downloads is not None and processed >= args.max_downloads:
				logging.info("Reached --max-downloads=%d limit", args.max_downloads)
				break
		except Exception as e:
			logging.exception("Failed processing %s: %s", entry["tag"], e)

	build_index(dest_root)
	logging.info("Done. Updated %d release(s). Output: %s", changed, dest_root)
	return 0


if __name__ == "__main__":
	sys.exit(main())

