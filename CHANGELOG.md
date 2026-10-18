# Changelog

All notable changes to Studio will be documented in this file.

## Unreleased

### Added

- WhisperSync's lip-sync calibration: `sync.camera_av_offset_ms` and
  `sync.camera_av_offset_ms_by_camera` move a camera's picture against all
  sound in the exports, program, whole-edit preview and reels; changing them
  never synchronizes again.
- WhisperSync's pause ducking (`sync.pause_duck_*`, off by default; **Lower
  voices in pauses** in the advanced form) lowers rendered voices where neither
  the camera nor the recorder has words. Settings migration now carries both.
- Program, preview and reels show recorder-only stretches (the recorder
  started before the cameras) as black picture with the recorder's sound
  instead of failing with "no camera covers retained range".

### Fixed

- Reels selection with gemma4 failed on every chunk ("returned no text
  content"): its chat template thinks first and spent the whole token budget
  on reasoning. Requests now ask templates not to think, as Forge's prefill did.
- Program, preview and reels decoded every source from its start to cut a
  piece (13+ minutes for 18 s of a two-hour recording); inputs are now sought
  before decoding and trimmed frame-exactly after.
- The diarization module could not import: unpinned torch 2.14, torchaudio
  2.11 and huggingface_hub 1.x break pyannote.audio 3.4. The recipe pins
  them, the worker allows pyannote's Lightning checkpoints under torch >= 2.6,
  and a module installed from an older recipe is installed again.
- A caption could start with punctuation Whisper split off ("?» меня…"); it
  now stays with the word before it.
- A stage that failed in a run was reported as "disabled by user".

- BormoStudio picks the most capable ffmpeg on PATH at startup (it runs,
  then libass, then NVENC; `BSTUDIO_FFMPEG_DIR` overrides): on the owner's
  machine Homebrew's ffmpeg 9 comes first but lacks libass and NVENC, which
  /usr/bin's has. `bstudio doctor` reports libass.
- The NVENC probe encoded a 128x128 frame, below NVENC's minimum, so NVENC
  always read as unavailable and the program and reels encoded on the CPU.
  It probes 256x256 now; reels smaller than 256 px on a side use libx264.

- Reels prompts were sent with `{requirements}`, `{target_candidates}` and
  `{chunk_json}` left unfilled, and cleanup/judge were asked only to "keep
  grounded, complete moments": every placeholder is now filled as in Forge.

- Final Cut import of sony_run: the timeline format carries Final Cut's own
  name (`FFVideoFormat1080p2997`), which removes the "unexpected value
  format=r1" warning; media links start at the document's published folder
  (they pointed into `_studio/export`, so every clip was offline); asset
  lengths are the files' real lengths on their own frame grid (a 25 fps clip
  rounded to 29.97 frames ran one frame past its file, so relinking failed
  with "not long enough"); clips are shortened where rounding passes the end.
- One camera with a recorder exports as a Final Cut synchronized clip, not
  a one-angle multicam. The event also lists video files sync did not place
  and ignored false starts, under the keyword "Not synchronized".
- The Final Cut event lists the multicam clip and every file, not only the
  project; event, project and multicam take the material folder's name
  (`export.name`, also in the GUI settings) instead of always "Studio".
- The repaired recorder is written as `repaired-<recorder>.wav` and shown as
  "Recorder (repaired)": relinking by name no longer offers the shorter
  original recorder for it.
- The NLE export carries the whole (repaired) recorder as its own track;
  per-camera voices rendered from it are no longer exported beside it.

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
- The repair and content checks measure clips under 120 s on the same dense
  grid as the sync check.

### Added

- WhisperSync's tempo-conform method, seam snapping and self-check Whisper
  mode are settings (`sync.stretch_method`, `seam_snap_max_s`,
  `self_check_transcribe_mode`), scan takes extra media suffixes
  (`scan.extra_extensions`), and the migration maps `audio_source_camera`,
  `render_workers`, `video_exts`/`audio_exts` and gives every other WhisperSync
  key a stated reason. Settings added later stay out of fingerprints at their
  defaults (`settings.LATER_DEFAULTS`), so existing results are reused.

- The remaining Forge and WhisperSync settings get Studio homes: article
  faithfulness thresholds and window (`text.article_*`), term check
  occurrences and pause (`text.term_min_occurrences`, `term_pause_s`), Whisper
  `best_of`, `patience` and `condition_on_previous_text` (left out of
  fingerprints at their defaults, so cached transcripts stay valid), pyannote
  `speakers.num_speakers`, and JSON-schema constrained reels answers
  (`reels.strict_json_schema`, Forge's schemas, stepping down to simpler ones
  when the server cannot compile a schema). The migration report of Forge's
  own `config.yaml` now names a Studio key or a reason for every key.

- Forge's `host_memory`: `compute.host_memory` shrinks running libvirt VMs by
  balloon and shuts down `compute.vm_stop_domains` (GPU-passthrough VMs) for a
  run or a whole queue, gives everything back at the end, on signals and, after a
  crash, on the next run or `bstudio host-memory restore` (`status` lists the VMs).
  Forge's own tests of the module run against the port.

- Reel face tracking takes Forge's options (`reels.face_min_size`,
  `face_follow`, `two_speaker_layout`, `speaker_switch`) and rejects reels with
  too few faces (`face_min_ratio`); the framing report carries the face rate and
  the tracking plan.

- Styled reel captions split at speaker changes and colour each speaker
  (`reels.caption_split_on_speaker`, `caption_speaker_colors`), move with
  `caption_vertical_align`/`caption_vertical_offset`, and can be checked with
  a second Whisper pass over each reel (`caption_whisper_sync`, Forge's
  `whisper_sync`, with a parity test of the matching and retiming).

- Reel rendering does what Forge's cut stage did: edges fitted to the speech
  (`reels.edge_*`), the quality filters again at render time with
  `rejected.json` and optional `render_rejected`, QA of every reel
  (`reels.qa`, `qa_blackdetect`), WebM/GIF/M4A exports, a `.md` description
  and `.txt` caption per reel, `reels-preview.mp4`, and Forge's encoder
  settings (`reels.encoder`, `video_bitrate`, `audio_bitrate`, `preset`,
  `nvenc_cq`, `nvenc_preset`) for every reel encode.

- Reels selection follows Forge's staged analysis: duration-scaled clip quotas
  per type (`reels.clips_per_hour`, `story_count` ... `highlight_count`), the
  clip ask in every prompt, two quote gates (`scout_quote_ratio`,
  `min_quote_ratio`), pre-cleanup cap and pool top-up, audio measured before
  the judge, judge batches with each clip's first and last words
  (`judge_context`), quality filters (`min_score`, `min_duration_s`,
  `max_duration_s`), MMR diversity and overlap policy (`diversity`,
  `max_topic_similarity`, `mmr_lambda`, `max_overlap_ratio`), ranking weights
  (`scoring_weights`), prompt variants, `rejected_candidates.json`, Forge's
  metrics, derived titles/captions/hashtags and its `reels_summary.md`. A test
  runs the frozen Forge and Studio on the same transcript and scripted LLM and
  requires the same moments and rejections. `tools/migrate_settings.py`
  migrates the matching Forge keys.

- Desktop basic mode (the default): choose the material folder, check the
  speech language, pause removal, cut markers and the review video, press
  **Process**; Studio runs the whole pipeline and names the Final Cut and
  Premiere files when it is done. The Work page shows the current step and its
  progress. **Advanced mode** (remembered per user) shows every page and a
  settings form with every setting, in sections with names, explanations,
  a filter and bold names for values that differ from the defaults; it saves
  only the fields that were edited.
- `export.marker_min_s`: cuts shorter than it get no marker in Final Cut or
  Premiere (default 0, every accepted cut is marked).

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

- Renamed to BormoStudio: command `bstudio` (`bstudio-gui`), distribution and
  binary `bormostudio`, `BSTUDIO_*` variables, `~/.cache/bormostudio`. `uvf`,
  `uvf-gui` and `UVF_*` still work.

- Whisper engines with the same model, device and compute type share one
  loaded model per process: the sync stage's rescue pass, self-check and
  repair no longer each load their own copy.

- `export.targets` defaults to `[multicam, xmeml]`: `Studio.fcpxml` is a
  multicam clip (one angle per camera group, one for the recorder; the edit
  lives in the project over uncut angles). `export.markers` (default `cuts`)
  leaves the filler and untranscribed-sound hints out of the NLE.

- `recorded_at` prefers `com.apple.quicktime.creationdate` (the recording time)
  over `creation_time`, which is often the time the files were copied.
- `locate` keeps a group's clock delta as a hint only when most of its clips agree,
  narrow searches need a stricter vote, and a position that breaks the order of
  the camera's files is searched again between its neighbours.

- Reel layouts (`reels.layout`): a hook-title band over the framed picture and an
  audiogram that needs no camera; per-reel framing, crop, tracking and title
  (`bstudio reel`, Subtitles page) with an instant crop preview over an unframed still.
- Forge caption looks for reels (`reels.caption_preset` and `reels.caption_*`):
  word-timed cues, word/fill/reveal/pop/karaoke highlights, case, punctuation,
  censoring and fades, burned with the packaged fonts.
- Speaker identities matched across recorder files and diarized sessions by voice.
- Owned llama-server tuning (`llm.*`), per-role sampling, a streaming watchdog
  with retries, and their migration from Forge `llama_cpp.*`.
- Measured Whisper/llama tuning (`bstudio tune`, Settings) and a whole-edit preview
  proxy on the Edit page (`bstudio preview`).
- Mac: MLX Whisper (`transcribe.device=mlx`) and VideoToolbox encoding; not yet
  run on Apple hardware.

- LLM rough-cut detectors (start, slips, semantic fillers, take delivery) with
  verbatim-quote evidence; suggestions by default, report of rejected findings.
- Whole-timeline cut overview, with-cut/original audition without a master, and
  duration-preserving audio crossfades on program joins (`program.crossfade_ms`).
- Recorder track lanes and project speaker cameras with `bstudio tracks`/`bstudio cameras`
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
- `bstudio channel` queue with resumable acquisition report, `tools/night_run.sh`,
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
