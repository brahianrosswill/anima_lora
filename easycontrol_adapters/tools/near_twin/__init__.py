"""near_twin — in-artist near-twin variant-pair miner (EasyControl curation tool).

Mines near-duplicate *variant* pairs within a single artist where the two members
differ by a **specified attribute** (e.g. one has a speech bubble, the other
doesn't). See ``near_twin.__main__`` for the full pipeline + CLI. Run from the
repo root with ``python -m easycontrol_adapters.tools.near_twin``.

Layout:
  - ``engine``  — discovery → embed/cache → Stage-B grid match → discriminators → run_artist
  - ``outputs`` — HTML / TSV / pair-tree export / dataset blueprint
  - ``__main__``— config ([miner] toml) + argparse CLI + orchestration
"""
