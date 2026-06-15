# Bundled GUI fonts

**Pretendard** — the primary UI sans for the desktop GUI (loaded at startup by
`gui/theme.py::_load_bundled_fonts`). A multilingual (Latin + Hangul) face that
unifies the EN/KO look of this cross-platform Qt tool into one consistent font,
rather than depending on whatever the host OS ships. The system CJK + emoji
families stay in the fallback chain (`apply_theme`) for JA/ZH glyphs and emoji
markers Pretendard doesn't cover.

Three static weights were instanced from the upstream `PretendardVariable`
variable font (axis `wght` 45–930) at the weights the design tokens use:

| File | Weight | Used by |
|------|--------|---------|
| `Pretendard-Regular.ttf` | 400 | body / control text |
| `Pretendard-Medium.ttf`  | 500 | tab labels (`font-weight: 500`) |
| `Pretendard-Bold.ttf`    | 700 | groupbox titles, action buttons |

All three register under the single Qt family `"Pretendard"`, so a normal
`QFont("Pretendard")` + `setWeight()` picks the right instance.

## License

Pretendard is licensed under the **SIL Open Font License 1.1**.

- Upstream: https://github.com/orioncactus/pretendard
- Copyright © 2021 Kil Hyung-jin (길형진), with reserved font name "Pretendard".

The OFL permits bundling and redistribution with software. Keep this notice
alongside the font files.
