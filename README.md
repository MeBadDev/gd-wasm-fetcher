## Godot WASM Fetcher
This project discovers all stable Godot releases (>= 3.0) on GitHub and downloads the HTML export's WebAssembly and JS files for each version.

# Requirements
* Python >= 3.10.17

## Usage

- Optional: export a GitHub token to raise API rate limits.
	export GITHUB_TOKEN=ghp_yourtoken

- Run once:
	python main.py --dest ./store

- Options:
	--dest PATH          Destination directory (default: ./store)
	--min-version X.Y    Minimum version to include (default: 3.0)
	--log LEVEL          Log level (DEBUG, INFO, WARNING, ERROR)

Artifacts are stored under ./store/<version> with a metadata.json and the extracted .wasm/.js files. An index.json at the root summarizes all versions.

## Cron example (Debian)

Edit crontab:
	crontab -e

Nightly at 02:30, with a Python venv and optional token:
	30 2 * * * cd /home/youruser/ProgrammingProjects/gd-wasm-fetcher && \
		/usr/bin/env -S bash -lc 'source env/bin/activate && export GITHUB_TOKEN=... && python main.py --dest ./store --log INFO >> fetch.log 2>&1'

Notes
- No external dependencies. Uses GitHub API for discovery; falls back to rate-limited access if no token.
- Idempotent: skips versions already processed unless the upstream asset changed (tracked by asset_id in metadata.json).