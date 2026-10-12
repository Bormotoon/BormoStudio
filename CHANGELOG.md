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
- A recorder loss seen by two clips is one loss even when one of them only
  bracketed it between two windows; the fine location searches half a tracking
  window earlier. On sony_run the largest loss (1.1 s) was never repaired:
  the two observations, 6 s apart, contradicted each other.
- Tracking windows that touch the recorder's digital silence are not measured:
  the windows next to a dropout produced a false 2.2 s loss.
- The acoustic model no longer cuts a slope into a staircase of 0 ms steps
  (about 130 "sync jumps by +0 ms" warnings on sony_run).
- The file-order check ends a placed clip where its map ends, so the next
  chapter may start inside what the recorder lost (a correct text match of
  dji-0927 was dropped and transcribed again in quality).
- The sync check measures clips under 120 s on a denser grid, and stretches
  where the recorder wrote zeros do not count against its coverage.
- In clips under 120 s the sync check counts overlapping windows once per
  stretch of one window length, rejects a window whose peak does not stand
  2.5 times above every other lag, and correlates the whole clip against the
  voice: a decisive whole-clip peak within the threshold stands for the
  windows the clip lacks, one beyond it fails the clip. On sony_run
  mvi-0848, 0849 and 0856 pass (they were failed or inconclusive); a voice
  shifted by 30-800 ms fails or stays inconclusive.

### Added

- `scan.min_duration_s` (3 s): shorter files are false starts (a camera
  switched on and off by mistake) and get the role `ignore` with a scan
  warning; a manual role still wins, 0 keeps everything.
- Clips neither words nor the neighbour search placed are searched by their
  camera's clock (`stages/sync_camera_clock.py`): the placed clips of a camera
  tell whether its `recorded_at` stamps the start (DJI) or the end (Canon) of
  a recording, the clip's start is interpolated between its neighbours and its
  whole sound is correlated a few seconds around it. A clip too short for a
  fit keeps the found start (`map: clock` in `second_pass.acoustic_search`).

- Text-first synchronization (`sync.placement: text`, the default): every camera
  is transcribed fast (`transcribe.camera_mode`) and placed by its words against
  the whole recorder, as WhisperSync did; the acoustics then refine around that
  place, an acoustic match that contradicts the words is dropped, clips the words
  did not place are searched acoustically between their neighbouring files and
  then transcribed again in quality (`sync.rescue`), as is and with cleaned-up sound.
- Camera words are matched against a recorder transcript in the cameras' Whisper
  mode (`match_transcripts`): on sony_run the fast recorder transcript gave
  anchors with a third of the p90 timing error of the quality one.
- Short clips get a denser acoustic grid, and a few windows that agree on one
  offset confirm a text placement (`used: acoustic offset`).
- Recorder clock repair (`sync.recorder_repair`, `sync.recorder_fill`): samples a
  recorder lost (a phone dropping buffers) are found by tracking every camera
  against it, put back from the camera's sound or as silence, and every later
  stage reads the repaired recorder and its transcript.
- One map of recorder jumps from all cameras (`recorder_jumps` in
  `stages/sync/output.json`); clips without acoustics of their own follow it.
- The drift check and refinement see lags up to `sync.refine_max_lag_s` (1 s):
  recorder jumps of 0.2–0.8 s were invisible at 0.25 s.
- Failed or inconclusive sync checks are stage warnings and set the run report
  to `needs_review`; `report.json` lists every stage warning.

### Changed

- `recorded_at` prefers `com.apple.quicktime.creationdate` (the recording time)
  over `creation_time`, which is often the time the files were copied.
- `locate` keeps a group's clock delta as a hint only when most of its clips agree,
  narrow searches need a stricter vote, and a position that breaks the order of
  the camera's files is searched again between its neighbours.

- Reel layouts (`reels.layout`): a hook-title band over the framed picture and an
  audiogram that needs no camera; per-reel framing, crop, tracking and title
  (`uvf reel`, Subtitles page) with an instant crop preview over an unframed still.
- Forge caption looks for reels (`reels.caption_preset` and `reels.caption_*`):
  word-timed cues, word/fill/reveal/pop/karaoke highlights, case, punctuation,
  censoring and fades, burned with the packaged fonts.
- Speaker identities matched across recorder files and diarized sessions by voice.
- Owned llama-server tuning (`llm.*`), per-role sampling, a streaming watchdog
  with retries, and their migration from Forge `llama_cpp.*`.
- Measured Whisper/llama tuning (`uvf tune`, Settings) and a whole-edit preview
  proxy on the Edit page (`uvf preview`).
- Mac: MLX Whisper (`transcribe.device=mlx`) and VideoToolbox encoding; not yet
  run on Apple hardware.

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
