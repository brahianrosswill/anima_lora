"""FLAIR — training-free inverse-problem solver on the Anima flow prior.

Phase 0 (port validation): see ``sanity_sr.py``. Proposal:
``docs/proposal/flair_inverse.md``. Nothing here is wired into the inference
engine yet — the solver lives in the bench so the port can be validated before
promotion to ``library/inference/corrections/flair.py``.
"""
