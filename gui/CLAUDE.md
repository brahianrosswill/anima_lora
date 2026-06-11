# gui/CLAUDE.md

Guidance for the PySide6 (Qt6) desktop GUI. Scoped to `gui/` — read the root `CLAUDE.md` for the training/config/daemon contracts this GUI drives. ~12k lines across 29 Python files; the two big tabs (`config_tab.py` 1654, `preprocess_tab.py` 1366) hold most of the surface.

## What it is

A thin **front-end over the existing pipeline** — it edits TOML configs and submits jobs to the daemon; it does **not** contain training/torch logic. `config_io.py` and `_paths.py` are deliberately **Qt-free** (no PySide6 import) so they stay headless-unit-testable; keep them that way. The only `library/` import is the dataset-key linter (`library.config.dataset_keys.lint_dataset_sections`) — don't pull torch/training code into the GUI or you slow startup.

## Launch

- `make gui` → `tasks.py gui` → `scripts/tasks/gui.py::cmd_gui` → `python -m gui`.
- `python -m gui` → `gui/__main__.py` → `gui/__init__.py::main` → `gui/app.py::main`.
- `app.py::main` (gui/app.py:457): `load_language()` → `ensure_daemon_quietly()` → build `MainWindow` → Qt loop.
- The legacy CLI `make lora-gui GUI_PRESETS=<variant>` is a *training* entry (runs `gui-methods/` configs directly); it is not this GUI. The GUI submits to the daemon instead.

## Architecture

- **`app.py::MainWindow`** (gui/app.py:181) — top bar (Guidebook / Models / Update / Experimental + TensorBoard toggles / language) + a `QStackedWidget` swapping a **standard** tab set (ConfigTab, Preprocess, Dataset, Merge, Queue) and an **experimental** set (MethodsTab picker, EasyControl, SPD/Turbo distill). Dark `QPalette` via `_dark()`.
- **Tabs** (`gui/tabs/`) inherit `LazyTabMixin` (widgets built on first view) — **except** `ConfigTab` + the TensorBoard panel, which are pre-built before the window shows so the daemon + Train button are live immediately. `EasyControlTab` and `DistillConfigTab` extend `ConfigTab`.
- **`config_io.py`** — config discovery + merge + lint, all pure TOML/pathlib. `merged_gui_variant_preset(variant, preset)` returns `(dict, origin_map)` (origin_map = which key came from base/preset/variant). Variants are **auto-discovered** from `configs/gui-methods/*.toml` `[variant]` blocks (`family`/`order`) — adding a variant is one new file, no map to update; custom ones live in `gui-methods/custom/`.
- **`daemon.py`** — GUI-side client wrapper over `scripts.daemon.client`. `submit_training()` / `submit_command()` POST to the localhost daemon; the GUI only **observes** jobs by polling files on disk (job.json / progress.jsonl / stdout.log) via a `QTimer`. No background thread, no SSE. `active_job_id()` re-attaches to a job from a previous session / the ComfyUI node / CLI on restart.
- **`widgets.py`** — the field factory: `_widget(value, key)` maps a TOML value → Qt widget by type, `_read(widget)` maps back. Custom widgets for target_res / sample-prompts. `LazyTabMixin` lives here.
- **`i18n/`** — one module per language (`en/ko/ja/cn.py`), each exporting `STRINGS: dict[str,str]` (~370 keys). `t(key, **kwargs)` (gui/i18n/__init__.py:72) looks up current lang then **falls back to English, then to the key itself** — so a key missing from `ko.py` silently shows English, not an error. Register new languages in `TRANSLATIONS`.
- **`explanations/`** — lazy-loaded help: `guides/<lang>/_fields.json` (per-field tooltips) + `_preprocess_fields.json` + `<method>.html` (per-method overviews). Same English fallback.
- Support modules: `progress.py` (JSONL/tqdm parse), `process.py` (`kill_process_tree` via psutil), `tensorboard.py`, `validation.py`, `dialogs.py`, `discovery.py`, `system_dialog.py` (update + model manager).

## Gotchas

- **Save is comment-destructive.** `config_io._save` round-trips via `toml.dumps()`, stripping comments. Don't route hand-commented files (e.g. `base.toml`) through a GUI save if the comments matter — edit presets/variants instead.
- **Tab ownership is partitioned — edit the right one.** `_SKIP` keys (`target_res`, `drop_lowres_images`, `min_pixels`) are hidden from ConfigTab because **PreprocessingTab** owns them (they persist to `preprocess.toml`, not the training config). `_VIRTUAL_KEYS` (`use_valid`, `validation_split_num`) are not flat TOML keys — ConfigTab writes them into per-dataset `[[datasets]]` overrides. `_BASIC` (config_io.py:269) controls the collapsible "Advanced" fold. Putting a knob in the wrong tab causes silent drift.
- **i18n key parity is manual.** The 4 language files are independent; nothing enforces that they share keys. A missing key just falls back to English. When you add a string, add it to all four (and the matching `_fields.json` / `.html` if it's help text) — see the `translator` agent for propagating English → ko/ja/cn.
- **Daemon outlives the GUI.** Closing the window does not stop training; jobs live in the daemon. The GUI is a pure observer of on-disk job files.
- **Process kill must walk the tree.** Training is a grandchild of any directly-spawned `QProcess`, so `QProcess.kill()` alone leaks it — use `process.py::kill_process_tree` (psutil). Daemon jobs are stopped via `daemon.stop_job()`.
- **`gui_settings.json`** holds UI state (language, 6h update-check cache, preprocess knobs) — separate from `configs/` so it survives a config reset.

## Common changes

- **New training field**: add the TOML key → it surfaces in ConfigTab via `_widget()` auto-mapping (check `_SKIP`/`_BASIC` placement) → add a tooltip to `guides/en/_fields.json` and replicate to ko/ja/cn.
- **New variant**: drop `configs/gui-methods/<name>.toml` with a `[variant] family="…"` block — auto-discovered.
- **New language**: `gui/i18n/<code>.py` with `STRINGS`, register in `TRANSLATIONS`, add `guides/<code>/` files.
- **Change job submission**: `ConfigTab._on_train` and `PreprocessingTab`'s run handler — both go through `daemon.submit_training` / `submit_command`.
