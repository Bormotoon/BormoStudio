# Implementation status

This file distinguishes implemented contracts from plan items that still need
legacy migration or external acceptance. It intentionally does not mark a
phase complete merely because a similarly named module exists.

## Current snapshot — 2026-10-10

This section is the authoritative current inventory. The dated continuation entries
below are historical evidence: their test counts and remaining-gap lists describe
that point in development, not today's backlog. Plan phase acceptance remains open.

| Plan phase | Implemented locally | Remaining implementation / evidence |
|---|---|---|
| 0 — Preparation | Imported histories and preserved source snapshots/patch, fixture generator, locks/CI definitions, local wheel/frozen smoke | Executed CI matrix, clean installed-artifact tests, private acceptance recording, release interpreter qualification |
| 1 — Contracts and pipeline | Versioned project/manifests, four time domains, planner/worker/events, recoverable publication, scan/prepare/transcribe/complex sync, CLI and Qt shell | Complete legacy golden coverage and real end-to-end GUI/CLI parity |
| 2 — Sync and exports | Camera/simple/auto/complex paths, acoustic fallback, overlapping and explicitly positioned disjoint sessions, reference-camera choice, selected-stream matching/rendering and detached camera PCM for NLE, voice/warp outputs, neutral FCPXML/XMEML, persistent material overrides, RecorderTrack lanes with GUI/CLI, drag-and-drop groups | Full legacy source-selection workflow parity, NLE import and relinking |
| 3 — Rough cut | Energy-gated pauses, retakes, manual accept/reject, timeline cut overview, with-cut/original audition with or without master, markers, placement-aware CFR program/transcript with duration-preserving join crossfades | Real word-integrity/listening acceptance, end-to-end re-export check |
| 4 — Speakers/multicam | Microphone attribution/smoothing from named lanes, isolated pyannote facade, project speaker-camera choices (GUI/CLI), wide-shot policy, retimed multicam FCPXML | Nonlinear warped mic attribution, cross-file identities, real pyannote and NLE acceptance |
| 5 — Forge workflow | Text/term-check, task model routing, verified reels/captions, cached audio probes with voice/placement/edited mapping, native editor, vision, fetch/channel/batch with webhook/desktop notifications and mid-download cancellation, reports/night helper, owned llama lifecycle, settings migration, legacy comparison runner | Raw recorder warp replay, Studio homes for the remaining reels/llm legacy keys, phase-wide reuse, full Forge golden migration |
| 6 — Modules/distribution | Managed modules, pinned verified YuNet/Light-ASD/Whisper/GGUF downloads with resume, LLM rough-cut detectors, native setup/settings/GGUF picker and dirty guards, compute probe, Linux frozen/local installer | Real-model LLM edit acceptance, download step in first-run wizard, measured tuning, bundled dependencies, clean-machine/Python 3.12/platform qualification |
| 7 — Follow-up | Source-video reel rendering and native editor foundations already available | Speaker-angle reels, vision auto-assignment, OTIO, MLX/VideoToolbox, LUFS/covers/audiograms/YouTube replay data |

### Current verification

- Latest combined source suite: **447 passed, 2 skipped** on Python **3.10.20**
  with offscreen Qt (2026-10-10); Ruff passes. After the type-check repair,
  `mypy studio tools` passes on Python 3.10 and 3.12 (124 source files).
  The remote CI result still needs a new run with these working-tree changes.
- Frozen WhisperSync suite run through `tools/legacy_suites.py` in a Python 3.12
  environment from its lock file: **403/403**, matching the snapshot. The Forge
  suite was not run locally (its lock pins torch); CI runs it through the same tool.
- `whisper-tiny` was downloaded for real through `uvf models install` and matched
  every pinned checksum. Larger catalog entries were not downloaded here.
- The Linux one-file binary at `/tmp/opencode/uvf-dist/` predates all 2026-10-10
  work and must be rebuilt. Python **3.12** remains the intended release baseline.
- No plan phase is accepted solely on these results. Legacy snapshots remain intact.

### Prioritized local backlog

1. Confirm the repaired type check in a new CI run.
2. Real acceptance of the new local pieces: LLM edit suggestions on a real model,
   listening to with-cut previews and program join crossfades, Material screen
   interaction (lanes, drag and drop, cameras), an overnight channel run with
   notifications and a cancelled-then-resumed download.
3. Raw recorder warp replay without voice artifacts; Studio sections (batch, fetch,
   hooks, remaining reels/llm options) for keys the migration still reports as
   unmapped; phase-wide model ownership/reuse.
4. Wide-shot policy for speaker cameras, download step in the first-run wizard,
   measured Whisper/llama tuning.
5. Studio-vs-legacy golden parity on the comparison suites, installed/frozen
   real-processing checks, production packaging, private sample acceptance.

### Acceptance still required

- Real Whisper/llama/pyannote/vision runs, including managed environments and frozen
  inference; compare WAV/XML/moments against fixed legacy inputs/config/models.
- Private 20–40 minute multi-camera/recorder sample; beginning/middle/end lip sync,
  cut word integrity, listening quality, and actual GUI interaction.
- Final Cut/Premiere import, roles, channel links, multicam, speed and relinking on
  explicitly selected versions; overnight unattended channel run.
- CI matrix, Python 3.12 qualification and clean Linux installation first;
  Windows/macOS and process-tree cancellation afterward per owner platform priority.
- Third-party model/font/ffmpeg license inventory before distribution.

### Type-check repair — 2026-10-10

- Completed the existing working-tree typing repair: stage protocol signatures,
  optional Qt/process values, frozen runtime paths, completion-provider protocol,
  typed Markdown sections, NumPy return boundaries and distinct CLI local names.
- Missing-import overrides cover only optional untyped ML libraries, including
  installed `faster_whisper`; project checks retain their existing strict flags.
- Verification: mypy passes for all 124 source files on Python 3.10 and 3.12;
  Ruff passes; offscreen source tests on Python 3.10: 447 passed, 2 skipped.
  Python 3.12 verification used the existing legacy environment with the declared
  `types-PyYAML` development dependency added; this is not release qualification.

### Phase 1–2 fixture continuation — 2026-10-10

- Probe now records the absolute ffprobe stream index (not its ordinal among
  audio streams). `transcribe.audio_streams` overrides that choice per asset;
  synchronization privately decodes the same stream losslessly with all channels.
  Simple mode prepares camera transcripts before alignment instead of failing later.
- Session planning accepts camera/recorder pairs independently, rejects nonoverlap,
  connects clock maps through shared recordings and reports rejected pairs. Truly
  disconnected sessions require explicit `sync.session_offsets` (asset/group,
  seconds in the reference camera clock); their relative placement is not guessed.
  `sync.reference_camera` selects that reference by asset/group. Voice selection
  prioritizes covered duration, then residual/inliers; old sync-derived outputs
  are removed from the project output registry when recomputing synchronization.
- Timeline transcription includes disjoint recorder coverage without duplicating
  overlapping microphone speech. A single-camera timeline creates its identity
  placement, allowing export when synchronization is intentionally skipped.
- Whisper forwards glossary hotwords, repetition penalty and n-gram controls;
  segment text survives missing word alignment. Transcript cache/producer versions
  invalidate results generated before these decoding changes.
- `--only` can consume a verified cached dependency; explicit skips and damaged
  manifests still block required dependents. A real QProcess bridge regression
  checks identical GUI/CLI exports, unchanged mtimes on reuse and regeneration
  after export corruption. PCM legacy parity covers all three strategies with
  positive/negative/zero offsets and both directions of drift; overlap FCPXML
  intervals are compared with the frozen golden fixture.
- Frozen WhisperSync baseline rerun: 403 passed, matches snapshot. This is oracle
  validation, not proof that all baseline tests exercise Studio implementations.
- Owner confirms fixture-only acceptance material and no NLE access currently.
  Real recording, real model GUI/CLI parity and NLE acceptance remain open, as do
  the full golden workflow matrix, full timeline preview and complete source/NLE
  workflow parity. Phases 1 and 2 are not marked accepted.
- Camera-only NLE export now detaches the selected stream to published lossless
  PCM and mutes embedded camera audio in both formats. XML references final paths,
  never private worker paths; artifacts participate in manifest validation.
- Camera mode prepares and uses camera transcripts even when recorder files exist;
  it no longer accidentally selects recorder speech. Export includes additional
  synchronized recorder lanes when `recorder_mode=all` rather than dropping them.
- Edit page includes a keyboard-navigable source overview with group, interval,
  in-point, clock rate and coverage; stale cut displays clear on project changes.
  This is a read-only overview, not full audiovisual timeline/simulator parity.
- Final checks for this continuation: 465 passed, 2 skipped on Python 3.10,
  Ruff passes, mypy passes on Python 3.12 (126 source files), diff whitespace passes.

### OTIO export — 2026-10-10

- `export_otio.py` writes OTIO JSON (Timeline/Stack/Track/Clip.1/Gap/Marker.1)
  from the neutral sequence on whole sequence frames: video lanes, embedded camera
  audio lanes (disabled where synced voice replaces it), voice lanes, markers,
  `LinearTimeWarp` for drift-corrected clips; overlapping clips on one lane fail.
  Opt-in through `export.targets: otio`. OpenTimelineIO 0.18 reads the file (dev
  dependency for that test). Resolve import not verified.

### Reel loudness, covers and most-replayed — 2026-10-10

- `reel_finish.py`: two-pass EBU R128 `loudnorm` to `reels.loudness_lufs`
  (−14 LUFS, −1 dBTP, video stream copied; no-audio/silent reels untouched) and a
  cover JPEG per reel. Real-ffmpeg tests confirm a quiet clip lands within 1.5 LU.
- `reels/analysis/replay.py`: yt-dlp `heatmap` → duration-weighted span intensity
  scaled by the episode peak, mapped through the media placement (and program
  mapping for edited transcripts). New `replay` scoring factor, neutral (0.5)
  when absent so rankings without a heatmap keep their order.

### Fetch and batch settings — 2026-10-10

- New `fetch` section: `download` video/audio, `max_height`, `cookies_file`,
  `retries`, `rate_limit` become yt-dlp options for `uvf fetch`, channel downloads
  and the fetch stage (which previously had no valid settings section). Channel
  discovery filters `exclude`, `skip_live`, `min/max_duration_s` before the limit,
  looking up to five times deeper; unknown durations are kept, as in Forge.
- New `batch` section: notification defaults (CLI flags win) and
  `min_free_disk_gb`, which fails the next project and leaves the rest not started.
- Migration maps `youtube.*`, `autonomy.notify.*`, `autonomy.min_free_disk_gb` and
  WhisperSync `repetition_penalty`/`no_repeat_ngram_size`; replaced Forge keys
  (archive, download_dir, lock/log options, cli.*) carry a concrete replacement.
  Remaining unmapped keys are reels/llm tuning. yt-dlp behaviour not run live.

### Stage requirements and install navigation — 2026-10-10

- `stages/requirements.py` checks declared `Requirement`s: binaries on PATH,
  Python packages, managed modules, catalog/recipe models and, for managed
  llama, the executable and GGUF path (external endpoints are not probed).
  pyannote speakers now declare the diarization module; reel tracking declares
  vision/YuNet/Light-ASD as optional (it falls back to a static crop).
- The planner attaches missing requirements to stages that will run; `uvf plan`
  and run reports list them with a hint and install target. `uvf run` records a
  stage with a missing mandatory requirement as failed without starting it (its
  dependents skip) and emits a warning event. The Plan page lists them and
  **Install missing requirement** opens Modules and models with it preselected.

### First-run model step — 2026-10-10

- The setup wizard has a Models page: it lists catalog models already present,
  embeds the same verified/resumable downloader as Modules and models (preselected
  Whisper large-v3-turbo), and Finish saves the chosen Whisper model and a
  downloaded GGUF path to the project. Finish waits for a running download; Cancel
  stops it and keeps the partial file for resume. Not run on a clean machine.

### Wide-shot policy — 2026-10-10

- `@wide` in speaker cameras (Material page "Wide shot", `uvf cameras`, settings)
  names the wide camera. `CameraPolicy` (from `program.*`) selects it for
  overlaps and busy exchanges (`max_switches` within `switch_window_s`), inserts
  it into shots longer than `max_shot_s`, and switches `cut_lead_s` ahead of
  turns by shifting whole turns (no false overlaps). Program and multicam export
  share it; manual camera choices still win. Real switching quality not judged.

### Rough-cut detectors and presets — 2026-10-10

- `unknown-sound` markers cover audio that is neither detected silence nor a
  transcript word, limited to the placed source coverage; an empty transcript keeps
  its whole duration and marks it instead of cutting it. Overlapping word
  timestamps no longer produce false pauses, and one boundary guard drops any cut
  (pause, retake, LLM) whose edge falls inside a word. "как бы" is a two-word
  filler marker; an LLM filler at the same word replaces the heuristic marker.
- Packaged presets `roughcut-gentle/balanced/tight` (`studio/resources/presets`,
  in wheel and PyInstaller data) merge into settings.yaml via `uvf preset` or the
  Settings page; unrelated keys survive.
- `tools/check_program.py` re-transcribes the master and diffs words against the
  edited transcript plus `self_check` timing diagnosis. Not yet run on a real model.
- Checks: 474 passed, 2 skipped on Python 3.10 (offscreen); Ruff and mypy pass.

### Continuation — 2026-10-10: rough-cut LLM, preview, tracks, models, migration, batch

- **LLM rough-cut detectors** (`stages/roughcut_llm.py`): start of conversation,
  slips, semantic fillers and take delivery through the routed `roughcut` role.
  The model names phrase numbers and verbatim quotes only; quotes must match
  transcript words, cut times come from those words, and rejected or failed
  findings go to `llm-report.json`. Default `llm_apply: suggest` adds unaccepted
  cuts and delivery markers; fillers stay markers. Runner ownership of the managed
  llama server follows `uses_llm`, so a non-LLM roughcut does not start it.
- **Review**: `gui/cut_timeline.py` overview synced with the decision list;
  `core/preview.py` renders with-cut/original previews from the master, a synced
  camera voice or the transcript source via placement. **Program** joins use
  handled overlap-add crossfades (`program_audio.py`), memory-mapped decoding,
  exact sample counts, constant-power on edit joins and linear on camera changes.
- **Material**: `RecorderTrack` and `speaker_cameras` in `project.json`
  (additive fields, schema 1 loads older projects), `core/recorder_tracks.py`,
  `uvf tracks` / `uvf cameras`, multi-asset `uvf material`, and `gui/material.py`
  with grouped tree and drag-and-drop regrouping.
- **Models**: `modules/catalog.py` (pinned HF commits, per-file SHA256 or git blob
  SHA1) and `modules/downloads.py` (resume via ranges, cancellation, verified
  publication); `transcribe` prefers a downloaded catalog model; Modules page lists
  the catalog and fills settings.
- **Tools**: `tools/migrate_settings.py` (table 6.3, report of every key, Studio
  validation) and `tools/legacy_suites.py` (per-test outcomes, snapshot totals,
  run-to-run comparison; used by CI).
- **Batch**: `cli/notify.py` webhook/desktop notifications recorded in reports;
  `run_cancellable` stops yt-dlp mid-download, partial downloads resume by URL,
  cancelled channel runs write a `cancelled` report and exit 130.
- Verification: 447 passed, 2 skipped; Ruff passes; mypy 186 (pre-existing).

### Repository state — 2026-10-10

- Phase 5–6 work (text/term-check, reels/vision/render, native subtitle editor,
  settings/setup/modules pages, LLM session/roles/retries, channel queue, frozen
  entry, Linux installer) existed only as uncommitted working-tree changes. It is
  now committed by area and pushed to `origin/main`.
- GitHub CI (run 37990897469, commit `571cc8f`): frozen WhisperSync baseline
  (403) and Forge baseline (620) pass; Studio smoke passes on Python 3.10, 3.11
  and 3.13; wheel/PyInstaller and installed-artifact smoke pass on macOS and
  Windows. **Failing:** `mypy studio tools` in the Python 3.12 job — 195 errors in
  36 files, introduced after `8ccdef7` (last green run, 2026-10-08) because local
  checks ran pytest and Ruff only. Type-check repair is the next CI task; until
  then the local "Ruff passes" statements do not imply a green CI.
- Intermediate area commits are not individually guaranteed to pass the suite;
  shared modules (settings, CLI, main window) changed across areas. The final
  commit state is the verified one.
- Removed a stale agent worktree (`Ultimate-Video-Forge.worktrees/d73852f3…`)
  whose branch was fully contained in `main` and had no local changes.
- The verification venv (`/tmp/opencode/studio-py310`) and frozen binary
  (`/tmp/opencode/uvf-dist/`) live in volatile `/tmp`. The binary predates the
  final source guard changes and must be rebuilt; a persistent Python 3.12
  environment is still needed for release qualification.
- README now keeps the Russian overview only; the operational feature reference
  moved to [USAGE.md](USAGE.md).

### Reconciliation checklist — 2026-10-09

- [x] Re-ran full combined source checks: **410 passed, 2 skipped**, Ruff passes;
  `bash -n` passes for installer/night helper; `git diff --check` passes.
- [x] Reconciled phase 0–7 task checkboxes against source and continuation evidence.
  Plan explicitly labels partial tasks and names their remaining implementation;
  acceptance-only tasks are separate from completed local implementation.
- [x] Updated current tables/backlog to include audio cache/mapping, role routing,
  term-check, channel/night helper, dirty guards, audio audition and local installer.
- [ ] Acceptance: complete legacy suites, real ML/night/NLE runs and clean machine
  remain unverified. Current frozen smoke predates the final source guard changes.
- [x] Source snapshot README labels historical pre-owner decisions explicitly and
  links current owner policy while preserving provenance.

### Persistent audio measurements — 2026-10-09

- Successful probes persist atomically under project `cache/audio_features`.
  Identity includes full source SHA256, exact interval, selected stream, detector
  thresholds and implementation version. Candidate IDs and timeout do not invalidate
  successful measurements. Invalid cached schema/types/ranges are remeasured;
  failed probes are not cached. Cache write/hash errors are reported without
  discarding successful measurements.
- Verification: 402 passed, 2 skipped. Tests verify reuse, same-size source changes,
  threshold changes and invalid cache replacement. Source-only placement/warp probing
  remains open. Frozen binary predates audio changes and needs a later rebuild.

### Placement-aware source audio — 2026-10-09

- Timeline audio features now fall back to the transcript's explicit source asset
  and SourcePlacement when no master exists. Inverse offset/in-point mapping and
  chained atempo conversion measure in timeline time; uncovered candidates are
  reported rather than probed at wrong offsets. Explicit probed stream is selected.
- Source content, placement and stream participate in fingerprint and probe cache.
  This closes affine source fallback only: piecewise recorder AudioWarpMaps,
  cross-source spans and edited probing without program remain open.
- Verification: 403 passed, 2 skipped; regression checks nonzero offset/in-point,
  k=2 tempo and explicit stream selection. Real-model acceptance remains open.

### Extended analysis inputs and task routing — 2026-10-09

- Analysis assembles temporary timeline audio from synchronized voice outputs,
  with coverage selection from the existing master renderer. Piecewise warp results
  are consumed as rendered voices, not reimplemented against raw recorder files.
  Edited audio without program media uses validated matching render-report mapping.
  Cross-source joins/gaps are handled by the temporary assembly; cleanup is scoped.
- Added validated per-task model/endpoint routing for proofread/article and
  context/scout/cleanup/judge, including requirements and fingerprint identities.
  Managed endpoint overrides are rejected; role aliases do not load different GGUFs.
- Term check is integrated after proofreading with offline manual fixes and separate
  explicit network opt-in, bounded suspect count, atomic cache and realignment.
- Verification: 407 passed, 2 skipped before final requirements/prompt fingerprint
  update. Real WAV tests exercise temporary synchronized/edited audio and cleanup;
  tests cover role routing/policy and offline term application. Full legacy settings
  compatibility, raw warp replay and real model/network acceptance remain open.

### Queue, GUI and installed build continuation — 2026-10-09

- Added bounded YouTube channel discovery/acquisition, validated video IDs, project
  resume, independent download failure continuation and atomic acquisition report.
  Channel invokes normal batch pipeline. Batch/channel expose persistent reports
  and structured queue completion events; nightly helper invokes this workflow.
  External notifications and mid-download cooperative cancellation remain open.
- Settings now tracks basic/YAML dirty state, confirms discard at project change,
  close and compute application, preserves edits across navigation, and blocks run
  with unsaved settings. Basic save cannot overwrite edited YAML. Native cut audio
  preview uses timeline master with context, loaded-media seek and stop controls.
  Full recorder/speaker/camera controls and timeline cut preview remain open.
- Added local Linux install script (binary, CLI link, desktop entry/resource check).
  Rebuilt current processing/GUI/channel sources and installed in an isolated
  prefix. Installed resource check, channel help, fixture scan-only batch/report
  and eight-second offscreen GUI startup passed. Final source guard changes after
  that build require another rebuild before distribution.
- Source verification: 410 passed, 2 skipped; Ruff and shell syntax pass. This is
  same-host installation smoke, not clean-machine or Python 3.12 qualification.
  Real ML/GUI listening/NLE, full end-to-end fixture processing and overnight
  channel acceptance remain blocked on model/input/environment qualification.

### Audio-feature continuation — 2026-10-09

- Opt-in candidate mean volume/silence measurement now runs before final ranking.
  Registered timeline master or edited program audio provides domain-aligned input;
  arbitrary original transcript audio is deliberately not used. Missing input or
  failed ffmpeg probe leaves neutral fields with report diagnostics. Model-provided
  audio fields are cleared before scoring; only measurements supply this evidence.
- Positive candidate cap and timeout, finite noise threshold and silence duration
  are validated. Enabled audio input SHA256 participates in stage fingerprint.
- No separate feature cache or placement/warp-aware source-only probing yet; these
  and complete Forge audio parity remain follow-ups. Default is disabled.
- Verification: 401 passed, 2 skipped before final evidence-field reset; real WAV
  regression checks half-silent audio and same-size content replacement. Domain
  selection regression prevents original-audio reuse at edited timestamps.

## Historical baseline inventory (superseded by current snapshot)

- Source snapshots, source provenance, Python 3.12 lock files, CI definitions,
  and local wheel/PyInstaller smoke builds. Remote matrix and clean-machine
  acceptance have not been demonstrated.
- Deterministic two-camera/recorder/drift/pause/retake/GoPro fixture generator.
- Versioned project, artifact manifest, transcript, four-time-domain,
  SourcePlacement, AudioWarpMap, and EditMap contracts.
- Planner and JSON-lines subprocess runner foundations. Effective settings are
  snapshotted per worker; silent cancellation and large stderr are tested with
   real processes. Workers write private results; the runner validates artifact
   checksums and publishes files, project, and manifest with a rollback journal.
   Error/cancellation rollback and killed-publisher recovery are tested. Recovery
   runs under the CLI project lock before planning. This is recoverable multi-file
   publication, not simultaneous filesystem visibility or proven power-loss
   durability. Live event forwarding and Windows tree cancellation remain open.
- Media scan, prepare input declaration, Whisper stage facade and content
  cache, affine text-anchor placement, timeline transcript, deterministic
  pause rough cut, program transcript mapping, microphone energy attribution,
  neutral sequences, FCPXML and XMEML serializers.
- CLI `scan`, `plan`, `run`, and `doctor`; initial PyQt QProcess desktop shell;
  isolated module manager; owned llama-server lifecycle.

## Historical initial gaps (not the current backlog)

- No phase is closed. Phase 0 still requires the private acceptance recording
  and verified CI/installed-artifact checks. Prior task-list completion of
  phase 0 was premature.
- Discovery now requires matching stage identity, current fingerprint, successful
   status, and verified artifacts. Repair optional dependency propagation and include all source identities in
  fingerprints. A completed prototype unit suite is not an end-to-end run.
- Fix source selection/stream decoding and carry over Whisper OOM recovery.
  The current sync implementation fits one affine map even when labelled
  strategy 3; it does not render synchronized recorder audio. Existing NLE
  serializers do not establish synchronized-audio or legacy export parity.
- Rough-cut pause removal still lacks the required energy gate; program
  rendering currently uses one source camera without the required placement,
  synchronized audio, and crossfade integration. These need regression tests
  before use on real material.
- Full characterization/golden migration of WhisperSync matcher, RANSAC,
  acoustic fallback, strategies 1/2/3, piece renderer, Boundary Flex,
   ambience, enhancement, self-check, repair, segments, and verification.
   Initial frozen-source transcript/export, fingerprint, lock, and process
   characterization tests exist; the Hybrid contracts remain to be captured.
- FCPXML behavior parity for connected clips, auditions, roles, synchronized
  rendered audio, relative relinking, multicam, and rough-cut markers.
- Editable Material screen, persistent manual overrides, RecorderTrack model,
  camera/speaker assignment, multicam switching, and edit decision auditioning.
- Forge proofread, article, term check, analysis/reels selection, reel render,
  subtitle burn/editor save contract, YouTube fetch, and unattended batch.
- Model download UI, module settings UI, first-run wizard, auto-tuning, and
  production installers for all three operating systems.

## External acceptance blockers

- A private 20-40 minute two-camera plus recorder sample has not been selected
  in this workspace.
- Final Cut Pro and Premiere Pro versions and machines are not available in
  this environment, so import, relinking, lip-sync, and multicam acceptance is
  open.
- The required overnight channel batch run has not occurred.
- Clean-machine installer acceptance on Linux, Windows, and macOS has not
  occurred.

## Historical continuation audit (2026-10-08)

The earlier inventory above predates the preserved Hybrid integration. Explicit
`sync.mode=complex` now uses migrated matcher, piece planner and renderer, publishes
per-camera voice WAVs and AudioWarpMaps, and has frozen-engine PCM parity tests.
Automatic/simple modes still need integration with this path; multiple recorders,
acoustic fallback and the optional legacy audio steps remain open.

Implemented during this continuation:

- Live validated worker event callback with concurrent stdout/stderr draining;
  silent cancellation, noisy stderr, malformed events and transactional rollback
  remain covered. CLI `run --events`, `--only` and `--skip` expose the runner and
  planner controls. Progress callback tests assert notification before process exit.
- Optional roughcut dependency no longer blocks timeline export.
- Explicit transcription stream decoding; content-cache hits update source paths
  for the current project. OOM batch reduction and unbatched fallback are restored.
  Beam sizes, initial prompt and cache enablement are accepted by typed settings.
- Neutral export includes synchronized voice, applies keep ranges to camera/voice,
  honors requested export formats and FCPXML version, and fingerprints audio/edit
  artifacts. FCPXML connects overlapping media to a timeline gap; replaced camera
  audio is muted. XMEML includes rendered voice audio tracks.

Remaining important gaps found by direct code inspection:

- Program still selects the first camera and does not yet use synchronized voice,
  placement-aware source in-points or audio crossfades.
- Roughcut still treats word gaps as silence without the required energy gate;
  retakes, decision auditioning and camera switching are not integrated.
- XMEML camera-source audio, channel metadata and video/audio links need further
  work. FCPXML multicam, retake auditions, roles, markers and rational media format
  parity remain unverified. Camera clock factors need explicit export retiming.
- CLI failure continuation/reports, capability gating, global settings layers and
  Windows process-tree cancellation remain incomplete.
- Text/reels migration, shared llama session, pyannote facade, usable material/edit
  controls, subtitle save contract and production installers remain incomplete.

Verification uses `/tmp/opencode/studio-py310/bin/python -m pytest -q` on this Linux
host. It exercises the combined package, not the full frozen legacy suites and not
an actual Whisper/model/NLE end-to-end acceptance run.

### Production hardening continuation

- Roughcut execution now has a configurable silence-energy gate for pause and
  head/tail candidates; real generated-audio tests cover silence detection and
  preservation of audible gaps. Thus the earlier energy-gate gap is resolved for
  the currently selected first source stream, not yet arbitrary recorder tracks.
- Program execution now resolves placed camera coverage, selected camera groups,
  and synchronized voice, splits ranges at file boundaries, normalizes media,
  supports silent cameras and preserves output duration. A real ffmpeg integration
  test renders a silent camera with external voice and nonzero timeline placement.
  Audio crossfades, GPU auto-selection and extensive codec/VFR acceptance remain.
- Auto/simple recorder sync now renders through the preserved accepted matching
  facade and strategy selector. Acoustic-first fallback and multi-recorder selection
  are still open. Timeline ordering explicitly includes optional sync.
- Run failures produce a report and continue independent branches; strict settings
  type validation rejects malformed booleans/lists/integers before execution.
- Combined verification: 125 tests passed; Ruff passes on files changed in this
  continuation and git diff --check is clean. This is not full plan acceptance.

### Retakes and review continuation

- Migrated the unchanged frozen token-level retake detector into studio/stages/
  retakes.py with dependency-free local models/settings. Eleven original detector
  regression cases and four direct parity cases pass.
- Roughcut integrates last-take selection, stable cut IDs, saved manual decisions
  and a word-boundary guard. CLI `uvf review SOURCE` lists the edit; `--cut ID
  --accept` / `--reject` transactionally republishes decisions and manifest without
  changing the detector fingerprint. Overrides survive later detector runs by ID.
- FCPXML/XMEML serialize edit reasons and filler markers at edited times. Export
  and program fingerprints now include roughcut settings to invalidate results
  correctly when cutting is toggled while an older edit is still on disk.
- XMEML preserves original camera audio as linked per-channel clipitems and
  disables it when replaced. Sequence dimensions and source channel metadata are
  emitted. Source rates/bit-depth and optimized channel track grouping need work.
- Verification: 143 combined tests pass, changed-file Ruff and diff whitespace
  checks pass. Real NLE marker/link import and GUI auditioning remain unverified.

### Microphone speakers continuation

- SpeakersStage is executable and registered, with typed auto/mics/off settings,
  explicit recorder channel assignments, content fingerprints and transactional
  publication of diarization.json plus provenance report.
- Real ffmpeg channel decoding feeds affine placement-aware RMS bins, preserving
  inpoints/clock factors. Attribution handles bleed, overlap and silence. Missing
  recorder placements are blocked; ordinary stereo is not automatically declared
  to be two separate microphones.
- Tests cover offset stereo attribution through an actual subprocess worker,
  published artifact reuse, channel bounds, silent tracks and clock/inpoint bins.
- Pyannote, short-turn smoothing, RecorderTrack GUI controls and nonlinear warped
  attribution remain open. The current implementation does not establish complete
  speaker/multicam phase acceptance.

### Sequential processing migration: acoustic alignment

- Auto/simple now use migrated waveform-first alignment; complex has waveform
  fallback. GCC output matches the frozen implementation. Real offset audio with
  no transcript is found and camera-only placement retains the measured offset.
- Multiple overlapping recorders are aligned onto the shared primary clock and
  rendered independently. Best mode selects one voice; all mode publishes all
  lanes under sync-tracks keys. Subprocess regression covers two recorder outputs
  and warp maps. Disjoint sessions, lane export and discovery optimization remain.
- Combined suite: 149 tests pass. This does not close the external sync acceptance
  gates or the remaining WhisperSync enhancement/self-check/multicam tasks.

- Boundary Flex is connected to recorder rendering with `sync.boundary_flex=true`.
  Realized waveform verification is migrated and exposed through `sync.verify`
  and CLI `verify`; identical generated audio passes the real measurement/CLI test.
  Enhancement, ambience, self-check/repair and master/segment outputs remain open.

### Voice output continuation

- `sync.voice_segment_minutes` now publishes sample-accurate PCM segments plus
  a rendered-audio index. Full voice outputs remain available. Real PCM tests
  reassemble segments byte-for-byte; worker tests include extensible 24-bit WAV.
- `sync.master_wav` publishes an unedited stereo timeline voice master, chooses
  one replacement on overlapping cameras, retimes camera clock ratios and uses
  silence in uncovered ranges. Audio tests check duration and duplicate-level
  prevention. Quiet seam selection and segment-based NLE links remain open.

### Content self-check continuation

- Preserved content diagnostics and local text/acoustic re-alignment are migrated.
  Nineteen original regression cases pass, including empty-render content loss,
  ASR jitter, repeated occurrences and local repair-alignment rejection.
- `sync.self_check=warn` runs a shared Whisper engine on rendered voices, publishes
  rendered transcripts and content reports, and unloads the model before leaving
  the worker. Three frozen/migrated outcome parity cases and stage tests cover
  report serialization and failed content outcomes. Actual model acceptance is open.
- Automatic repair rendering and atomic warp-map replacement are not implemented;
  repair mode is deliberately not accepted by settings yet. Ambience/enhancement
  integration, master crossfades and remaining multicam tasks are still pending.

### Optional audio backend continuation

- Migrated preserved separation/enhancement and bounded subprocess logging behind
  public-media adapters. Supported denoise/denoise_dereverb/resemble modes are
  configured by sync.voice_enhance; sync.ambience adds separated camera room tone.
- Enhanced voices feed self-check, segments, master and export. Missing modules
  retain original voices with explicit warning reports. Ambience is conformed
  before publication and exported as separate connected audio lanes.
- Seventeen migrated enhancement regression tests pass plus integration dispatch
  tests; combined suite has 193 passing tests. Actual ML environments/models and
  GPU processing are not exercised; separation full regression and real listening
  acceptance remain open. Automatic content repair remains incomplete.

### Verified repair continuation

- `sync.self_check=repair` rerenders failed clips with a freshly derived
  whole-clip linear map. It publishes replacements only after repeated content
  recognition and passed acoustic lag verification. Inconclusive checks do not
  authorize replacement; errors retain the original selected audio and map.
- Selected sync paths, all-recorder selected lanes, warp maps and output report
  update together before segment/master generation. Initial self-check results
  stay intact; separate repair reports record the candidate acceptance evidence.
- Six candidate-policy tests cover acceptance, content failure, lag failure,
  inconclusive lag, skipped clips and backend errors. A stage integration test
  checks selected lane/map/report updates and downstream segment input.
- Repair and enhancement are currently mutually exclusive. Local span repair,
  real Whisper/model acceptance, master crossfades and GPU rendering remain open.
- Full regression run: 201 tests pass; Ruff on changed Python files and
  `git diff --check` pass. Recognition and lag acceptance in repair tests use
  controlled backends; these results do not establish real-model acceptance.

### Program encoder selection

- `program.encoder=auto` probes actual NVENC encoding and selects libx264 when
  unavailable. Explicit cpu/nvenc modes retain their requested behavior; runtime
  auto-mode GPU failures retry the whole job on CPU. `program/render.json` is
  published as `program_report` with encoder selection and fallback diagnostics.
- Local real probe found no h264_nvenc encoder in ffmpeg and selected libx264.
  Real CPU media render and controlled GPU success/failure/timeout tests pass.
  Full suite: 208 passed. Hardware GPU encoding acceptance remains open.

### Voice master crossfades

- Added `sync.master_crossfade_ms` (10 ms default; zero disables) with linear
  complementary fades in real source overlaps, preserving original timeline
  length and silence gaps. No fade is fabricated across non-overlapping sources.
- Fixed timeline-mix limiter latency and sample offset rounding. Real PCM tests
  verify smooth opposite-polarity switches, unchanged correlated-dialogue gain,
  silence gaps and one-sample impulse placement. Full suite: 212 passed; Ruff
  on changed files and `git diff --check` pass.
- Review-program edit joins remain hard cuts. Listening acceptance, VFR/mixed
  source fps acceptance and actual GPU render acceptance remain open.

### Mixed-fps and VFR review master

- Added validated `program.fps` string setting: auto or explicit rational/decimal
  rate from 1 to 240 fps. Review master normalizes source timestamps to CFR with
  cumulative frame rounding and explicit concat durations. Edited transcript
  timing follows the resulting frame padding/trimming, including zero-frame cuts.
- Real fixtures combine 24 fps, 30000/1001 fps and genuinely variable-frame-time
  video. Twelve short retained cuts render at both 25 and 30000/1001 fps with
  verified frame counts, uniform presentation steps and final frame-end duration.
- Fixed CPU encoder alias mismatch between settings and selector; both cpu and
  libx264 work. Full suite: 217 passed; changed-file Ruff and diff checks pass.
- Long-session A/V drift, NLE VFR interpretation and GPU hardware acceptance
  remain open. Next major processing block is speakers/multicam.

### Speaker short-turn smoothing

- Microphone attribution has configurable conservative short-island smoothing,
  preserving uncertain and overlapping speech, pauses and genuine handoffs.
  Raw microphone evidence remains published separately from smoothed turns.
- Eight regressions cover isolated islands, disabled smoothing, unknown/overlap,
  different neighbors, pauses, long turns, short neighbors and overlapping turns.
  Full suite: 225 passed; changed-file Ruff and diff checks pass.
- Pyannote integration, speaker-driven camera switching and true NLE multicam
  export remain open; smoothing does not establish real-world speaker accuracy.

### Speaker-driven review camera selection

- Added `program.speaker_cameras` and `program.min_shot_s` for optional camera
  switching from existing timeline speaker turns. Camera/group mappings are
  coverage-aware; explicit edit choices win, uncertain speech holds the current
  angle and missing preferred coverage falls back. Retained ranges remain intact.
- Program depends optionally on speakers, fingerprints speaker reports/group IDs
  and records its camera plan in render.json. Five regressions cover switching,
  holds, coverage fallback/manual precedence, removed ranges and invalid mappings.
- Full suite: 230 passed; changed-file Ruff and diff checks pass. Pyannote,
  NLE multicam export and real-world speaker/camera acceptance remain open.

### Isolated pyannote integration

- Explicit pyannote method uses the managed diarization environment; missing
  environments block planning. Configure speakers.model/device; model credentials
  use HF_TOKEN in the environment. Primary-source turns map to timeline with
  clipping, explicit overlap resolution and retained raw turns/logs.
- Controlled stage mapping/overlap and missing-module preflight regressions pass.
  Full suite: 232 passed. Real gated model download, inference, GPU usage and
  speaker accuracy have not been exercised. Auto remains microphone-based;
  cross-file speaker identity matching and true NLE multicam export remain open.

### FCPXML multicam writer

- Added optional export.targets entry multicam, producing a separate
  Studio-multicam.fcpxml with native multicam resources, angle lanes and
  available-angle project selection. Synchronized voice is selected independently
  of picture; repeated camera files share their lane and uncovered ranges are gaps.
- Structural regression checks resources, repeated-angle clips, refs, video/audio
  selections and coverage gaps. Full suite: 233 passed; changed-file Ruff and
  diff checks pass. Final Cut/DTD acceptance has not been performed.
- Camera placements with k != 1 are rejected for multicam until retiming support
  exists. Speaker-driven NLE angle selection, ambience selection and actual
  NLE import acceptance remain open; this is a first writer, not completed parity.

### Speaker-selected multicam continuation

- Multicam export now consumes program speaker/camera mappings, minimum shot
  settings and explicit roughcut camera choices. Selection maps from source
  timeline to edited timeline while retaining all alternative angle resources.
- Two new regressions verify alternative-angle preservation and selection across
  removed spans. A generated document passed xmllint validation against Apple's
  FCPXML 1.9 DTD from the CommandPost mirror (external temporary validation asset).
- Final Cut import/playback, camera retiming and ambience selection remain open.
- Full regression run: 235 passed; changed-file Ruff and `git diff --check` pass.

### Multicam ambience continuation

- Selected audio angle now combines synchronized voice and corresponding camera
  ambience via a composite clip with connected background audio. Camera audio
  is the primary component when no replacement voice exists. Source in-points
  are preserved independently of composite-local offsets.
- Structural test covers a nonzero source in-point and selected audio angle;
  generated voice/ambience document passes FCPXML 1.9 DTD validation. Full suite:
  236 passed; changed-file Ruff and diff checks pass. Listening/NLE import and
  camera-retiming acceptance remain open.

### Affine multicam retiming

- Multicam supports camera placements with k != 1 via linear FCPXML timeMaps;
  synchronized voice and ambience follow the same map. Roughcut intersects in
  timeline time and computes source in-points through the inverse affine rate.
- Two regressions cover slow/fast maps, nonzero source in-points, roughcut clipping,
  audio map parity and source resource duration. Generated retimed voice/ambience
  XML passes FCPXML 1.9 DTD validation. Full suite: 238 passed; changed-file Ruff
  and diff checks pass.
- Actual Final Cut playback, frame-rounded lip-sync accuracy and mixed-source
  format interpretation remain acceptance gates. Flat export retiming parity is
  not completed by this multicam-only sequence change.

### Flat export retiming continuation

- Export stage now applies affine camera placement rates to ordinary FCPXML and
  XMEML as well as multicam. FCPXML has linear timeMaps; XMEML has constant speed
  effects on video, voice, ambience and linked camera channels. Roughcut inverse
  source in-points, source out/coverage and timeline duration are consistent.
- Three parameterized stage regressions cover k=0.5, 1.25 and 1 with nonzero
  in-points and roughcut. Retimed flat FCPXML passed xmllint/1.9 DTD validation.
- NLE import/playback acceptance remains open, especially Premiere constant speed
  semantics, source fps differing from sequence fps and sub-frame lip-sync.
- Full suite: 241 passed; changed-file Ruff and `git diff --check` pass.

### Mixed-format export continuation

- Source video fps/dimensions now accompany export clips. FCPXML references
  source format resources; XMEML source in/out and file coverage use source fps,
  including linked camera channels, while sequence start/end retain sequence fps.
- Mixed 25 and 30000/1001 fps regression checks nonzero source in-points, affine
  speed, source dimensions and audio channel indexing. Generated flat FCPXML
  passes 1.9 DTD validation. Full suite: 242 passed; changed-file Ruff passes.
- Actual mixed-format NLE import/playback and speed interpretation remain open.

### XMEML channel/link hardening

- Original camera channels reuse stable lane/channel tracks across cuts; video
  and all channel siblings link to the complete group with resolved media/track/
  clip/group indices. Replaced original sound remains disabled and external voice
  stays separate. Retiming filters are preserved on each channel.
- Regression covers three cuts, three camera channels, replacement voice and
  resolution of every link. Full suite: 243 passed; changed-file Ruff and diff
  checks pass. Premiere link/import and playback acceptance remain open.

### Audio metadata and roles

- Export probes rendered voice/ambience PCM metadata, including extensible WAV,
  and emits actual channels/sample rates/bit depth. FCPXML labels voice as dialogue
  and ambience as effects. Camera audio uses available scan metadata.
- Real mono 24-bit 44.1 kHz voice and stereo 16-bit 48 kHz ambience regression
  passes, along with FCPXML 1.9 DTD validation. Invalid placeholder fixtures were
  replaced with actual WAV headers; invalid media does not silently pass probing.
- Full suite: 244 passed; changed-file Ruff and diff checks pass. Camera bit-depth
  discovery and actual NLE role/audio-layout interpretation remain open.

### Camera audio depth discovery

- Scan stores selected-stream audio bit depth and XMEML exports it when known.
  AAC unknown depth stays absent; packed 24-bit precision takes precedence over
  storage width. Scan/export fingerprint versions invalidate older results.
- Real MOV fixtures verify 24-bit PCM at 44.1 kHz and AAC unknown depth through
  scan and export. Controlled multistream probe checks default-stream selection.
  Full suite: 246 passed; changed-file Ruff and diff checks pass.
- Actual NLE acceptance and explicit multi-audio-stream export selection remain
  open; depth discovery does not establish a complete stream-selection contract.

### Local workflow continuation

- Added opt-in `text` processing. Proofreading batches local LLM requests,
  accepts only conservative corrections, publishes a corrected transcript plus
  SRT and report, and leaves source word timings explicitly marked as ASR.
  Article output publishes Markdown and a faithfulness report; failed requests
  retain the source output instead of silently inventing text. Both features
  default to disabled.
- Material role, group and device assignments are now explicit project overrides:
  `uvf material` persists them transactionally, rescans preserve them while
  refreshing probe metadata, and the desktop Material page uses the same CLI
  path. The desktop Edit page lists rough-cut decisions and accepts or rejects
  them through the existing transactional review command.
- Added `uvf batch` for sequential, isolated project queues and `uvf fetch URL
  DESTINATION` for a single YouTube source in a new/empty material directory.
  Batch deliberately has no arbitrary shell hooks; fetch does not overwrite
  local media and then enters the ordinary scan workflow.
- Combined verification: 255 tests pass; full Ruff and `git diff --check` pass.
  Text uses controlled provider tests only; real model, overnight batch,
  browser subtitle-save, NLE and clean-machine installer acceptance remain open.

### Forge analysis migration continuation

- Migrated robust local-LLM JSON recovery, corrected-text word-timing
  realignment, and a shared typed transcript index. Corrected transcript words
  retain raw ASR words for audit, so quote/subtitle consumers can use the text
  the user actually sees.
- Added typed reels-analysis contracts (`MomentRecord`, prompt chunks),
  deterministic sentence-aware chunking, candidate normalization and heuristic
  scoring. These are isolated from legacy global configuration and are covered
  by Studio regressions.
- Added opt-in `reels` selection stage. It uses the existing local chunk prompt,
  publishes `moments.json`, a Markdown summary and report, and preserves LLM
  request failures as diagnostics. Cleanup/refine/judge, quote validation,
  deduplication/ranking parity and video/subtitle rendering remain to migrate.
- Added the conservative Forge term-check helper, including manual fixes,
  lookup cache and an explicit network verifier. It is not enabled by default
  in proofreading because network use needs its own typed opt-in policy.
- Combined verification: 271 tests pass; full Ruff and `git diff --check` pass.

### Verified reels and render continuation

- Migrated preserved quote validation, cleanup/judge decisions, full heuristic
  scoring and quota/diversity ranking. Candidates are clamped to their scout
  chunk, assigned stable IDs, verified against transcript words and snapped to
  speech boundaries. Unconfirmed quotes are rejected and included in diagnostics.
- Cleanup and judge are bounded batches; responses cannot rewrite evidence.
  Request failures preserve verified input candidates and are recorded in reports.
  Both review stages can be disabled through typed settings.
- Added opt-in `reel_render`: timeline moments render through placed cameras and
  synchronized voice; edited moments cut the review program. Each clip has local
  transcript JSON, SRT and provenance in render.json. Subtitle burning publishes
  a separate captioned MP4 while preserving the clean master. Render fingerprints
  include media content and exclude their own outputs to permit stable reuse.
- Migrated 38 frozen decision/validation/ranking regressions; local real ffmpeg
  test checks clip duration and captioned output. Full suite: 312 passed; Ruff
  passes. Actual LLM quality, listening and external NLE acceptance remain open.
- This completes a local horizontal reel path, not the whole plan. Automatic
  portrait reframing and vision tracking, audio features
  for analysis, episode context/refine parity, shared llama lifecycle, GUI model
  management/settings/first-run flow and production packaging remain implementable
  locally and are not classified as external acceptance blockers.

### Native subtitle editor continuation

- Added a native PyQt Subtitles page with Qt Multimedia playback, seeking,
  cue text/timing editing, add/delete controls, unsaved-change prompts and
  Classic/Large/Top presets. Preview draws text with font, color, outline and
  placement over video. No browser or Qt WebEngine dependency is used.
- Project-local manual edits are separate from ASR transcripts. Saving validates
  ordered non-overlapping cue times and publishes edit JSON plus SRT/ASS through
  the recoverable publication journal under the project lock. Transcript identity
  prevents silently applying old edits to new media/transcript selections.
- Reels render consumes saved edits/styles, fingerprints them and publishes ASS
  plus corrected SRT. Save-and-render invokes the ordinary isolated CLI worker.
  Program cues support editing and subtitle export; program caption rendering
  is not yet connected. Visual playback, font matching and platform multimedia
  behavior still require interactive acceptance.
- Verification: 321 tests pass, including offscreen native GUI editing/save,
  malformed timing rejection, persistence/stale guards and a real ffmpeg rerender
  using manually changed text and style. This does not establish visual acceptance.

### Review captions and portrait framing continuation

- Added independent `program_subtitles` stage and connected the desktop
  save-and-render action for review masters. Manual edits publish SRT, ASS and
  a separate captioned MP4; the original program is preserved. Caption settings
  do not invalidate the expensive clean program render.
- Added source/crop/fit framing, even output dimensions and configurable
  horizontal crop position. Crop fills the frame; fit preserves the whole image
  with padding. Portrait ASS uses a portrait reference canvas; the native overlay
  uses the video's orientation when placing captions. Automatic subject tracking
  and interactive framing controls remain open.
- Real ffmpeg regressions verify both portrait modes, square pixels and duration;
  a worker regression verifies program-caption transactional publication, reuse,
  manual text and preservation of the clean master. Full reel regression covers
  portrait framing and styled caption rerender together.
- Combined verification: 327 tests pass; full Ruff and `git diff --check` pass.
  Interactive playback/appearance and external model/NLE acceptance remain open.

### Desktop framing settings continuation

- The native editor now exposes source/crop/fit, output dimensions and horizontal
  crop position. Choices persist in project settings.yaml and feed the ordinary
  reel worker when Save and render captions is used. These settings apply to all
  reels in the project; program controls are disabled.
- Saving preserves other settings, validates the complete settings layer and
  obeys the project lock. Invalid subtitle timings are checked before updating
  framing settings. Framing changes are shown after rerender, not as an immediate
  crop preview of the already-rendered video.
- Combined verification: 328 tests pass, including GUI persistence and blocked/
  malformed settings saves. Automatic face/active-speaker tracking, per-reel
  framing overrides and interactive crop preview remain open.

### Preserved face tracking continuation

- Migrated YuNet face detection, Light-ASD inference/MFCC, shot-aware face tracks,
  Viterbi speaker selection and smoothed crop-path generation under studio/vision.
  Heavy imports remain inside the optional worker path, not the GUI/stage imports.
- Opt-in tracking feeds the reel framing pass; the desktop exposes a tracking
  checkbox. Model paths use a writable user cache, configurable by UVF_MODELS_DIR.
  No model download is triggered by rendering. Missing models/dependencies/GPU
  retain the configured static crop and record the fallback in render.json.
  Model content participates in render fingerprints.
- Sixteen preserved regressions pass; two optional torch/model checks are skipped
  on this host. Real ffmpeg dispatch/fallback tests pass. Combined suite: 345
  passed, 2 skipped. Actual model inference/GPU tracking and visual speaker accuracy
  remain unverified. Model download UI and managed vision environment remain open.

### Managed vision installation continuation

- Added verified model downloads, status inspection and CLI `models list/install`
  for YuNet/Light-ASD without importing ML packages. Downloads use bounded size,
  checksum verification, a cache lock and atomic replacement; failed verification
  preserves previous files. YuNet fetches the actual LFS media rather than a pointer.
- Added native Modules and models page with asynchronous environment/model install
  commands and output logs. Installed module environments are reused rather than
  cleared by a repeated install request. Closing while installation is active is
  blocked to avoid losing ownership of the process.
- Tracking now runs through the managed vision interpreter with explicit request/
  result files, verified model checks and bounded disk-backed subprocess logs.
  Managed vision includes PyYAML required by the public contract imports.
- Combined verification: 348 passed, 2 skipped. Controlled download tests check
  checksum rollback; subprocess-contract tests verify interpreter dispatch and
  result handling. Actual pip installation and ML inference are not demonstrated;
  frozen packaging of the managed subprocess and detailed progress remain open.

### Installation progress and managed source packaging

- Module/model installation emits JSON-lines phase progress with `--events`;
  model downloads include byte counts and percentage when Content-Length is known.
  Native installation page parses buffered lines and shows determinate/indeterminate
  progress while retaining pip logs. Concurrent module installs are locked.
- PyInstaller specification now includes public core/modules/vision Python sources
  under managed/ for external-interpreter vision workers. Frozen installation
  requires an explicit UVF_MODULE_PYTHON rather than trying to create a venv from
  the application executable. GUI installation commands account for frozen entry.
- This adds packaging contracts, not a verified frozen vision inference run.
  Model-environment end-to-end acceptance, cancellation and production installers
  remain open. Progress reports phases for pip, not fabricated package percentages.

### Cooperative installation cancellation

- Native installation page now offers Cancel installation and waits for cleanup
  before permitting another job. CLI accepts --cancel-file and returns 130 with
  an installation_cancelled event. Owned venv/pip children are cancelled as process
  groups on POSIX and taskkill trees on Windows; Windows behavior remains unverified.
- Cancelled downloads remove private temporary files and preserve previous models.
  Interrupted module environments are not marked installed and can be rebuilt on
  retry. Network read cancellation is bounded by the request timeout, not instant.
- Local tests exercise cancellation of a real sleeping subprocess, model rollback,
  marker suppression and CLI cancellation. Production installer acceptance remains
  open; these changes do not establish clean-machine packaging readiness.

### Frozen launch and local build smoke

- Added a packaged entry dispatcher for --gui, --cli and --worker. Shared launch
  argv now lets desktop commands and stage workers reenter the application binary
  instead of passing Python -m switches to a frozen executable. Qt Multimedia and
  native GUI are included in the specification; binary uses the product name.
- Actual local PyInstaller build succeeded using the available Python 3.10 host.
  Bundled-resource check, model-status command and worker help succeed. A generated
  media fixture was scanned by the packaged CLI through its packaged worker and
  transactionally published four assets. Fixed model recipe lookup discovered by
  this smoke run: frozen status reads bundled managed source files.
- Combined source suite: 355 passed, 2 skipped before the frozen recipe lookup fix;
  regression verification is recorded separately for subsequent changes. This is
  host-local CLI/worker smoke, not GUI visual acceptance, managed ML inference,
  Python 3.12 release qualification or clean-machine installer acceptance.

### Desktop processing settings and execution

- Added project Settings page for sync/Whisper/speakers/encoder choices, local LLM
  endpoint/model and opt-in outputs. Updates merge with advanced settings under
  the project lock and are validated before atomic replacement.
- Plan page now displays actual decisions/reuse and exposes Run processing plan;
  completed runs navigate to Result and errors feed Work logs. Fixed bridge
  handling of multiline JSON documents and buffered JSON-lines worker events.
- Native GUI tests cover settings preservation/rejection and real CLI document
  delivery. A complete interactive/model production workflow remains unaccepted;
  first-run guidance, advanced settings and GPU auto-tuning remain open.

### Native setup and advanced settings

- Added explicit project setup wizard on Start: tool discovery, dependency guidance,
  synchronization and output choices. Existing choices are loaded on reopening;
  finishing validates and saves settings through the normal project lock.
- Added Advanced YAML tab covering all typed settings, with validation before
  atomic full replacement. Basic settings are scrollable; unknown/malformed YAML
  does not replace the project file. Omitted fields use application defaults.
- Combined verification: 358 passed, 2 skipped; native tests cover wizard save,
  advanced replacement and invalid-setting preservation. This is project setup,
  not automatic hardware tuning or a verified first-install model workflow.

### Compute backend inspection

- Added isolated CTranslate2 capability probe and actual NVENC encode check via
  CLI compute. Native Settings page runs inspection asynchronously and explicitly
  applies conservative recommendations through validated project settings.
- Recommendations use supported compute types, CPU batch 1 or initial CUDA batch
  8; they are not VRAM benchmarking or optimal batch auto-tuning. Missing backends
  are reported rather than installed or falsely declared usable.
- Local inspection found no CTranslate2 and no h264_nvenc encoder, recommended CPU.
  Controlled backend tests verify CUDA type selection and missing-backend behavior.
  Combined suite: 360 passed, 2 skipped; full Ruff passes. Hardware acceptance and
  measured auto-tuning remain open.

### Pipeline cancellation ownership

- GUI run cancellation now uses a private cancel-file rather than terminating
  the CLI orchestrator. CLI checks before planning and during worker execution,
  stops the active worker through the existing process-tree cleanup and does not
  continue independent branches after a cancellation request.
- Cancelled runs retain published artifacts, write report status cancelled and
  exit 130. The desktop prevents closing while its processing command is active.
  Scan/plan cancellation still uses the older terminate path; batch cancellation
  and Windows worker-tree cleanup remain open.
- Combined verification: 361 passed, 2 skipped. Added pre-cancel CLI/report/lock
  regression; existing runner tests exercise real-process cancellation and
  publication rollback. Full Ruff and diff whitespace checks pass.

### Discovery and batch cancellation

- Extended cancel-file handling to scan/plan workers; desktop cancellation uses
  cooperative cleanup for discovery as well as full runs. Published discovery
  outputs are retained and locks are released on cancellation.
- Batch propagates cancellation to the current project and stops the queue.
  Unstarted projects are explicitly reported as not_started without creating their
  directories; cancelled batch returns 130 with status cancelled.
- Combined verification: 362 passed, 2 skipped; CLI regression covers pre-cancelled
  scan/plan/batch and untouched unstarted directories. Full Ruff and diff checks pass.

### Persistent local LLM cache

- Text and reels now share project cache/llm outside private disposable worker
  output directories. Previously request caches disappeared on worker cleanup.
  Successful responses persist even when a later stage operation fails; this is
  a request cache, not publication of a complete stage result.
- Cache keys include normalized endpoint, model, prompt, grammar and deterministic
  request version. Writes are atomic; null/empty response content is rejected and
  not cached. Model weights replaced behind an unchanged endpoint/model ID require
  cache invalidation by the user; weights identity is not available from this API.
- Combined verification: 364 passed, 2 skipped. Regressions verify persistent reuse,
  endpoint isolation and invalid-response rejection. Shared managed llama-server
  lifecycle across processing phases remains open.

### Managed llama phase lifecycle

- Added opt-in typed managed llama settings and effective loopback endpoint mapping.
  Pipeline starts its owned server only for non-reused LLM work, waits for health,
  reuses it across adjacent LLM stages and stops it before other processing phases
  and on any exit. Occupied ports are rejected without stopping foreign processes.
- Startup checks model/executable, timeout and cancellation; failed startup cleans
  up the owned server. This shares a process within a pipeline, not across batch
  projects, and is not a real llama/model acceptance run.
- Controlled regressions cover reuse, cleanup, cancellation and endpoint mapping.
  Advanced settings exposes configuration; dedicated model picker and batch-wide
  lifecycle remain open.

### Desktop GGUF selection and model identity

- Processing Settings now exposes owned-server enablement, executable/GGUF file
  pickers and port. Existing typed validation and project save path apply; runtime
  startup still verifies actual files and executable availability.
- Managed GGUF content now participates in text/reels fingerprints and response
  cache keys. Replacing model bytes at the same path no longer reuses an old LLM
  result. Hashing is content-based and may cost time for large models; no metadata
  shortcut is treated as proof of identical weights.
- Combined verification: 367 passed, 2 skipped. GUI tests verify managed settings
  persistence; identity regression verifies replacement detection. Batch-wide
  server lifecycle and real llama acceptance remain open.

### Batch llama ownership

- Batch queue now owns the managed llama session across projects. Matching server
  configuration and GGUF content preserve session ownership; changed identity
  closes the previous process. Project failure, cancellation and queue exit clean
  up the session. Ordinary standalone runs retain their own scoped lifecycle.
- Reuse across projects occurs only for adjacent LLM execution. Non-LLM stages
  still stop the server to release resources; this does not reorder the queue
  into global processing phases or keep GPU weights loaded during Whisper work.
- Combined verification: 368 passed, 2 skipped; owner regression checks reuse,
  replaced-model detection and cleanup. Real llama batch acceptance remains open.

### Reels episode overview and balanced judging

- Added bounded temporal/signal transcript sampling and overview generation using
  the existing RU/EN context prompts. Scout, cleanup and judge receive the overview;
  quote verification and timing continue to use transcript evidence exclusively.
- `reels.episode_context` defaults to true and can disable the additional request.
  Existing provider cache includes the overview prompt and digest. Context prompt
  resources participate in the reels fingerprint; report records digest, validated
  payload, status and failures. Invalid overview falls back to transcript-only work.
- Judge batches now distribute ranked candidates round-robin across quality ranges,
  matching the migrated Forge comparison contract. Cleanup retains sequential batches.
- Verification: 375 passed, 2 skipped; tests cover bounded digest coverage, balanced
  batches, malformed context fallback and propagation through selection/review prompts.
  This is partial episode-context parity: metadata/chapters, diarization-derived
  signal inputs and candidate refine passes remain open, as does real model acceptance.

### Source metadata in episode overview

- Reels overview now consumes registered fetch `.info.json` artifacts, including
  bounded title/channel/description and valid chapter entries. No arbitrary directory
  JSON discovery is used. Both fetch CLI and fetch-stage artifacts are supported.
- Source chapter times are explicitly labeled and remain orientation only; RU/EN
  prompts forbid using them as edited clip bounds or proof of quotes/speaker identity.
- Metadata content participates in reels fingerprint when episode context is enabled.
  Malformed/missing metadata falls back to transcript-only context with diagnostics;
  the bounded metadata preamble is retained in the report.
- Verification: 378 passed, 2 skipped; regressions cover prompt propagation, changed
  metadata invalidation, disabled-context independence, malformed-file fallback,
  bounded descriptions and invalid chapter values. Candidate refine and diarization
  signal integration remain open; real model acceptance is still pending.

### Cleanup/refine migration contract

- Inspected Forge's `cleanup_and_refine_candidates`: refine is its evidence-only
  keep/drop/merge cleanup, not a separate pass rewriting timestamps. Migrated that
  request shape and time-ordered, pre-deduplicated batches into Studio. Existing
  immutable decision application retains source quotes/bounds and handles merges.
- Cleanup now also deduplicates across batch boundaries; failed requests retain
  batch candidates before deterministic deduplication. Judge still uses balanced
  quality batches. Reports include batch IDs, mode, kept/dropped/merged counts or
  IDs, unmentioned/untraceable decisions and failed-batch markers.
- Verification: 380 passed, 2 skipped. Regressions cover evidence-only payload,
  temporal grouping, duplicate removal, attempted evidence rewrite and report
  integration. This closes the sequential cleanup/refine contract; optional Forge
  parallel/retry budgets, candidate caps and diarization digest signals remain open.
  Real LLM acceptance remains pending.

### Bounded reels review workload

- Added positive-integer cleanup/judge candidate caps (100/50 by default), separate
  from final `max_candidates`. Enabled passes deduplicate and select by ranking
  value with deterministic time/ID tie-breaking before constructing LLM batches.
- Reports distinguish budget-excluded IDs, deduplication, input/review counts and
  model decisions. Budget exclusions do not advance to final ranking; disabling
  a review pass also disables that pass's cap. Changed caps invalidate stage reuse.
- Verification: 382 passed, 2 skipped. Regressions cover invalid cap values,
  deterministic exclusion and a real stage request limited to one of three inputs.
  Request retry budgets, diarization digest signals and real LLM acceptance remain open.

### Bounded reels JSON retries

- Added per-request extra attempts and shared per-run budget (defaults 1/5;
  validated integer range 0–100). Applied to overview, scout, cleanup and judge.
  Overview validation also triggers retries; structurally empty candidate/decision
  payloads retain existing semantics rather than being treated as transport failure.
- Extra attempts bypass the provider cache and atomically replace cached responses.
  Exhaustion retains existing overview/scout/review fallback. Process-control
  exceptions are not intercepted by the retry helper. Report records total attempts,
  retries used and remaining shared budget. No parallel requests or backoff added.
- Verification includes exhausted transport failures, shared-budget limits, malformed
  JSON cache replacement and existing stage regressions. Real model acceptance and
  diarization digest signal integration remain open.

### Speaker-change signals in episode sampling

- Reels optionally waits for speakers and consumes registered diarization/report
  artifacts for timeline transcripts. Changes of known speaker prioritize nearby
  sentences in the existing bounded digest; repeated labels are not changes.
- Report-confirmed timeline domain is required. Edited transcripts skip timeline
  speaker signals; no implicit conversion across cuts is attempted. Invalid files
  fall back with diagnostics; absent artifacts are reported as unavailable.
- Speaker artifacts participate in reels fingerprint when episode context is enabled.
  Reports retain signal status and change starts. Tests cover repeated labels,
  identity changes, time-domain mismatch, invalid turns and digest prioritization.
- Verification: 390 passed, 2 skipped before final stage-integration assertion;
  real model acceptance and edited-domain speaker mapping remain open.

### Rendered edited-domain speaker context

- Program now publishes timeline-to-rendered retained intervals and a transcript
  timing fingerprint in render report. Mapping uses actual CFR piece frame counts,
  discards trimmed tails and accounts for padded durations and camera subdivisions.
- Reels maps retained speaker changes to edited time only with a matching duration
  and word-timing fingerprint. Text-only proofreading does not invalidate the map;
  changed timings, absent old-format maps or invalid mapping fall back explicitly.
  Removed-region changes are excluded; this pass does not synthesize new turns at cuts.
- Program fingerprint bumped to regenerate old reports. Program report content is
  included in reels context identity. Tests cover cut exclusion, frame padding,
  frame trimming and stale timing detection; real model acceptance remains pending.

### Speaker transitions at edited cuts

- Added cut-boundary signals when adjacent rendered intervals retain different
  known speakers across a discontinuity in timeline time. This captures changes
  whose original transition was removed by editing, without restoring cut material.
- Requires unambiguous speaker coverage on both boundary sides; unknown/overlap
  labels, simultaneous different speakers and rendered gaps do not synthesize changes.
  Continuous timeline subdivisions do not add artificial cut signals.
- Reels fingerprint bumped. Verification: 396 passed, 2 skipped; regressions cover
  removed transitions, same-speaker cuts, unknown labels and overlapping speakers.
  Real model acceptance remains open.

### Configurable reels request timeout

- Added finite positive `reels.request_timeout_s` (600 seconds default), passed to
  the LlamaProvider HTTP socket timeout for all reels requests and their retries.
  Provider default keeps text-stage behavior compatible. Timeout is operational
  and does not change successful response cache identity; stage settings still
  participate in stage fingerprint. This is not a whole-stage elapsed-time deadline.
- Removed redundant managed-provider reconstruction while passing model identity
  and timeout together. Verification: 397 passed, 2 skipped; tests cover invalid
  timeout settings, timeout retry, configured timeout propagation and cache reuse.

### Transient HTTP retry pacing

- Added bounded exponential transport retry pauses with validated
  `reels.retry_backoff_s` (default 1, range 0–30 seconds). Each pause is capped at
  30 seconds; finite numeric Retry-After values in that range may extend it.
- HTTP 408/429/500/502/503/504 retry within existing budgets. Other HTTP statuses
  fail immediately without consuming extra attempts. Invalid JSON has no network
  delay; report includes cumulative requested pause duration. HTTP-date Retry-After
  is not parsed. Pauses do not constitute an overall stage deadline.
- Verification: 399 passed, 2 skipped; tests cover permanent HTTP failure,
  transient Retry-After handling, exponential cap and settings validation.

### Refreshed frozen Linux smoke verification

- Rebuilt the current tree with PyInstaller 6.22.3 on Python 3.10.20 using
  `packaging/pyinstaller/studio.spec`. Binary is available locally at
  `/tmp/opencode/uvf-dist/ultimate-video-forge`; build completed successfully.
- Verified CLI help, worker dispatcher help, packaged resource check, model listing,
  project planning and scan through the frozen executable. Generated fixture scan
  identified four media assets including the GoPro chapter relationship.
- Offscreen GUI launched with native Qt Multimedia/FFmpeg and remained alive for
  the eight-second smoke window before controlled termination. This checks startup,
  not visual acceptance or interaction with models. Latest source suite remains
  399 passed, 2 skipped; no source changes required by this build.
- This is same-host Linux smoke evidence, not a clean-machine installer test,
  Python 3.12 release qualification, Windows build or real frozen ML inference.
