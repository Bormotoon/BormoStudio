# Changelog

All notable changes to Studio will be documented in this file.

## Unreleased

### Fixed

- Worker cancellation remains responsive without stdout output, both pipes are
  drained concurrently, and buffered events receive the same protocol checks.
- Worker diagnostics use stderr; failures emit a protocol-valid failed event.
- Workers receive a private snapshot of effective settings, including CLI overrides;
  manifests record the settings actually used, without modifying project YAML.
- Invalid CLI settings return exit code 2 through the installed console entry
  point and write diagnostics to stderr, keeping JSON stdout clean.

### Added

- LLM rough-cut detectors (start, slips, semantic fillers, take delivery) with
  verbatim-quote evidence; suggestions by default, report of rejected findings.
- Whole-timeline cut overview, with-cut/original audition without a master, and
  duration-preserving audio crossfades on program joins (`program.crossfade_ms`).
- Recorder track lanes and project speaker cameras with `uvf tracks`/`uvf cameras`
  and a grouped Material screen with drag-and-drop regrouping.
- Pinned, resumable, verified Whisper and Gemma 4 GGUF downloads from the app.
- `tools/migrate_settings.py` for WhisperSync/Forge settings and
  `tools/legacy_suites.py` for per-test comparison runs of the frozen suites.
- Batch/channel webhook and desktop notifications; yt-dlp is cancelled
  mid-download and interrupted downloads resume.
- Opt-in text stage: conservative proofreading with word realignment, article with
  faithfulness report, and term check with offline fixes and separate network opt-in.
- Reels selection with verified quotes, episode overview, cleanup/judge review,
  bounded retries/backoff and audio features; `reel_render` with captions, portrait
  framing and optional YuNet/Light-ASD tracking in the managed vision environment.
- Native PyQt subtitle editor with video preview, cue/style editing and
  `program_subtitles` captioned review master.
- Desktop Settings (basic + Advanced YAML), setup wizard, Modules and models page
  with progress/cancellation, compute inspection and GGUF picker with dirty guards.
- Managed llama-server session shared across adjacent LLM stages and batch
  projects; per-task LLM role routing; persistent project LLM response cache.
- `uvf channel` queue with resumable acquisition report, `tools/night_run.sh`,
  frozen `--gui/--cli/--worker` entry and `packaging/install_linux.sh`.

- Frozen-source characterization tests and golden transcript/export contracts,
  including lock interruption, fingerprint adoption, and subprocess errors.
- Frozen source snapshots for WhisperSync and Podcast Reels Forge.
- Reproducible Python 3.12 lock files for both legacy applications.
- Initial installable Studio package and packaged UI resources.
- Deterministic two-camera, recorder, drift, pause, retake, and GoPro chapter fixture generator.
- CI for Python 3.10-3.13, both frozen legacy test baselines, clean wheel installs, PyInstaller, and cross-platform artifact imports.
- Recoverable stage publication with private worker outputs, rollback on
  cancellation or errors, and journal recovery before CLI planning.
- Discovery plan gating validates stage identity, fingerprint, successful status,
  and artifact checksums instead of accepting manifest file existence.
- Python 3.10-compatible string enums and UTC timestamps; corrected CI system
  dependencies and installed-wheel resource smoke command.
- Lossless WhisperSync rendering utilities migrated into Studio, with a typed
  render-plan facade and sample-for-sample frozen-engine parity checks.
