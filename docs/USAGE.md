# Usage reference

Operational reference for the currently implemented features. Settings keys are
set with `--set section.key=value` on the CLI, in the desktop Settings page, or in
the project `_studio/settings.yaml`. Implementation and acceptance status lives in
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

## Verified reels and local rendering

Reels selection checks candidate quotes against timed transcript evidence, runs
optional cleanup/judge batches, and deduplicates the final diverse selection.
Enable it with `--set reels.enabled=true`. A local llama endpoint must be available
at `reels.base_url`; the default is `http://127.0.0.1:8080`.

Enable video output with `--set reels.render=true`; optionally add
`--set reels.burn_subtitles=true`. The `reel_render` stage publishes MP4, local-time
JSON/SRT and `render.json` in `_studio/stages/reel_render/`. Timeline moments use
placed cameras and synchronized audio; edited moments require `program.mp4`.
`reels.render_fps` defaults to `25`. Rendering currently preserves source framing;
automatic subject tracking remains pending. Set `reels.framing=crop` for portrait
cropping or `reels.framing=fit` to preserve the full image with padding. Output
size defaults to 1080x1920 (`reels.width` / `reels.height`); `reels.crop_x` sets
horizontal crop position from 0 (left) to 1 (right), defaulting to the center.

Rendered reels are loudness-normalized in two passes to `reels.loudness_lufs`
(default −14 LUFS, true peak −1 dBTP; `0` keeps the source level) and get a
`reel-NNN-cover.jpg` taken a third of the way in (`reels.covers`). The measured
input loudness is in `render.json`. For fetched YouTube episodes, the "most
replayed" heatmap from yt-dlp's info.json adds a `replay` ranking factor
(`reels.replay`, weight 1.0); moments are mapped to the downloaded file through
its placement, and through the program mapping for edited transcripts. Without a
heatmap the factor is neutral and the ranking is unchanged.

## Subtitle editor and captions

The desktop **Subtitles** page edits rendered reels and the review program with
Qt video playback, seek controls, editable cue text/times and style presets.
**Save subtitles** writes project-local edits and SRT/ASS under `_studio/subtitles/`.
**Save and render captions** rerenders reels using these edits and styles.
The clean video and source ASR transcript remain available. The action supports
both reels and the review program. Program captions publish a separate
`program-captioned.mp4` through the `program_subtitles` stage; CLI users enable
it with `--set program.burn_subtitles=true`.

The same page exposes project-wide reel framing mode, output dimensions and crop
position. Save the choices and use **Save and render captions** to see the new
framing in the rendered clip. Changes preserve other project settings; framing
controls are disabled for the review program.

## Face / active-speaker tracking

Automatic tracking is opt-in (`reels.tracking=true`, with `reels.framing=crop`)
and available as **Follow face / active speaker** in the desktop editor.
The preserved YuNet/Light-ASD backend uses `reels.tracking_device=cuda` by default.
Install its optional torch/OpenCV dependencies and model files in
`~/.cache/ultimate-video-forge/models/` (override with `UVF_MODELS_DIR`). Rendering
does not automatically download models. When tracking is unavailable, it uses
the configured static crop and records the fallback in `render.json`.

## Modules and models

Use the desktop **Modules and models** page to install the `vision` environment
and download `yunet` / `light-asd` with checksum verification. Installation runs
asynchronously with a visible log. CLI equivalents are `uvf modules install vision`,
`uvf models install yunet`, and `uvf models install light-asd`.
Tracking runs in the managed environment's subprocess, keeping torch/OpenCV out
of the application process. `uvf models list` verifies the installed model files.
Install commands accept `--events` for JSON-lines progress. The GUI shows download
percentage when the server supplies a size, and phase progress for environment
creation, pip installation and checksum verification. Module installs are locked
against concurrent attempts. Packaged builds require `UVF_MODULE_PYTHON` pointing
to a compatible external Python for creating optional environments.
The installation page also supports cancellation. It waits for owned child
processes to stop; cancelled model downloads preserve previous files and cancelled
module installs do not publish a successful installation marker. A network read
may delay cancellation until its timeout. CLI automation can pass `--cancel-file
PATH` and create that file to request cancellation (exit status 130).

### Whisper and GGUF downloads

The same page and `uvf models install NAME` download pinned Whisper models
(`whisper-base`, `whisper-small`, `whisper-medium`, `whisper-large-v3`,
`whisper-large-v3-turbo`; `whisper-tiny` for tests) and Gemma 4 GGUF files
(`gguf-gemma-4-e4b-it-q4`, `gguf-gemma-4-e4b-it-q8`, `gguf-gemma-4-12b-it-q4`,
`gguf-gemma-4-26b-a4b-it-q4`). Each entry is fixed to a Hugging Face commit; files
are checked against the SHA256 (large files) or git blob SHA1 (small files) of that
commit and published only when they match. Downloads continue from `.part` files
after cancellation or a dropped connection. Files live under `UVF_MODELS_DIR`
(default `~/.cache/ultimate-video-forge/models`): Whisper in `whisper/<name>`,
GGUF in `llm/<repo>`. After a download in the GUI, the Whisper model name or GGUF
path is filled into project settings; saving applies it. `transcribe` loads a
downloaded catalog model from disk and otherwise lets faster-whisper fetch it.

Before running, the plan checks what each stage needs (ffmpeg, the Whisper
package, modules, models, a managed llama binary and GGUF). Missing items appear
in `uvf plan` and on the Plan page with how to install them; **Install missing
requirement** opens Modules and models with the item selected. `uvf run` does not
start a stage whose mandatory requirement is missing and reports it as failed.

First-run setup (**Set up processing**) includes a Models page with the same
downloader: pick a Whisper model for the project and optionally download a Gemma
GGUF; Finish stores both in the project settings.

## Packaged executable

The PyInstaller executable exposes `--gui`, `--cli COMMAND ...` and the internal
`--worker` dispatcher. Local build command:
`pyinstaller packaging/pyinstaller/studio.spec`. The binary is named
`ultimate-video-forge`; `--check-resources` checks bundled application resources.

## Desktop settings and running

The native **Settings** page saves project processing choices (sync, Whisper,
speakers, review master, text and reels) while preserving advanced settings.
Use **Build plan** to inspect stage decisions and reuse, then **Run processing
plan** to execute the pipeline. Worker output and errors appear on the Work page.
Start also offers **Set up processing**, a native project wizard with local tool
discovery and output choices. Settings includes an **Advanced** YAML editor for
all typed options; **Validate and save YAML** rejects unknown or invalid settings
before replacing the project file. Omitted fields use application defaults.
Settings also offers **Inspect compute backends** and **Apply compute
recommendations**. Inspection runs outside the GUI process, queries CTranslate2
compute types and tests actual NVENC encoding. The conservative recommendations
set Whisper device/type/batch and program encoder; they do not install missing
backends or claim benchmark-optimal settings. CLI: `uvf compute`.

## Cancellation

Pipeline runs support cooperative cancellation via `uvf run SOURCE --cancel-file
PATH`. The GUI Cancel action requests worker cleanup instead of terminating the
CLI orchestrator. Completed results are retained; report.json records cancellation.
`scan`, `plan` and `batch` also accept `--cancel-file`. A cancelled batch stops
after cleaning up the active worker and reports remaining projects as `not_started`.

## Local LLM: cache and managed server

Text and reel analysis share a persistent project LLM response cache in
`_studio/cache/llm/`. Keys include endpoint, model, prompt and grammar; cache files
are atomically written. When replacing model weights behind the same endpoint,
use a distinct model identifier or clear that cache before regenerating text.
Optional managed llama lifecycle is configured in Advanced settings:
`llm.managed: true`, `llm.executable`, `llm.model_path`, `llm.port` and
`llm.startup_timeout_s`. The orchestrator starts its own server only when an LLM
stage needs execution, waits for health readiness and reuses it across adjacent
LLM stages. It stops that server before other stages and on exit/cancellation.
Occupied ports are rejected; external servers remain untouched. The default
continues to use the independently running endpoint.
Processing settings now provides native Browse buttons for llama-server and GGUF,
an owned-server checkbox and port selection. Managed model content is hashed into
LLM request-cache and stage identities, so replacing GGUF bytes invalidates reuse.
External endpoints still require a distinct model identifier or manual cache reset
when weights change behind the same API identity.
Batch owns a shared llama session across project boundaries. Identical managed
configuration and GGUF content reuse the session when LLM work remains adjacent;
non-LLM stages still stop the server to release resources. Model/configuration
changes, project failures, cancellation and queue completion close the owned session.

## Reels analysis: context, review and retries

Reels selection now generates a sampled episode overview before scouting and
passes it to scout, cleanup and judge. Set `reels.episode_context: false` to skip
this extra LLM request. Its bounded digest, overview and failures are recorded in
the reels report; failed overview generation falls back to transcript-only analysis.
Quotes and timing still require transcript evidence. Judge batches mix candidate
quality ranges rather than reviewing strong and weak candidates separately.
Registered fetch `.info.json` artifacts contribute bounded title/channel/description
and chapter context to the overview. Chapter times are labeled as source times;
they never determine edited clip bounds. Missing or invalid metadata falls back
to transcript evidence and is reported. Metadata changes invalidate reels reuse.
Cleanup/refine now receives evidence-only payloads, deduplicates before review and
groups candidates by time so nearby alternatives can be compared. It applies
keep/drop/merge decisions to source records; quotes and bounds cannot be rewritten.
The reels report records per-batch decision diagnostics and failed-batch fallback.
Review work is bounded by `reels.cleanup_max_candidates` (default 100) and
`reels.judge_max_candidates` (default 50). Each enabled pass deduplicates and takes
the highest-ranked candidates with deterministic tie-breaking before batching.
Budget-excluded IDs are reported separately from model rejections; excluded
candidates do not advance to final selection. Disabled passes do not apply their cap.
Reels JSON requests allow `reels.json_retries` extra attempts per request (default 1),
bounded by shared `reels.retry_budget` per stage run (default 5). Retries bypass
the response cache so invalid cached JSON cannot trap a request. Overview schema,
JSON parsing and transport failures are eligible; exhausted requests use the existing
stage-specific fallback. Attempt/retry counts are recorded in the report.
`reels.request_timeout_s` sets the HTTP socket timeout per LLM attempt (default
600 seconds; finite and positive). Timeout failures consume the same retry budget.
This is a socket timeout, not an overall deadline for a stage or a streaming response.
Changing only the timeout does not invalidate the provider's successful response cache.
Transport retries use exponential pauses starting at `reels.retry_backoff_s`
(default 1 second, configurable 0–30), capped at 30 seconds per pause. HTTP 408,
429, 500, 502, 503 and 504 are retryable; other HTTP errors fail immediately.
Numeric `Retry-After` values within 0–30 seconds can extend a pause. JSON errors
retry without a network backoff. Reports include the total requested pause time.
Episode sampling also prioritizes sentences near known speaker changes from
registered timeline diarization output. It requires the registered speaker report
to confirm the time domain. Edited transcripts use the program report's mapping,
including CFR trim/pad offsets, when its timing fingerprint matches. Missing or
stale maps skip signals rather than reusing timeline times. Failures fall back to ordinary
sampling; speaker-signal status and selected change times are recorded in the report.
At contiguous edited cuts, distinct unambiguous known speakers on the two retained
sides add a change signal even when the original transition was removed. Unknown
labels, overlapping speakers and gaps between rendered intervals do not create
synthetic cut signals.

## Reels audio features

Optional `reels.audio_features: true` measures candidate mean volume and silence
with ffmpeg before final ranking. Timeline transcripts require a registered
`master_wav`; edited transcripts use registered program audio. Original transcript
audio is not assumed to have matching timestamps. Missing audio/probe failures
retain neutral features and report diagnostics. `audio_max_candidates` (50),
`audio_noise_db` (-30), `audio_silence_min_s` (0.35) and `audio_timeout_s` (30)
bound measurement work. Enabled source content SHA256 participates in stage identity.
Successful measurements persist atomically in `_studio/cache/audio_features/`.
Keys include audio SHA256, exact interval, stream and detector thresholds; invalid
cache entries are remeasured. Reports distinguish cache hits and cache write errors.
Without a timeline master, measurement can use the transcript's registered source
asset and SourcePlacement: inverse in-point/offset mapping plus tempo conversion
keeps thresholds in timeline time. Candidates outside that source coverage remain
unmeasured. Source content, placement and selected audio stream enter cache identity.
This fallback does not yet probe piecewise recorder AudioWarpMaps or edited source
spans without a program file.

Reels can now assemble temporary analysis audio from registered `sync:<camera>`
voice outputs across placed camera spans. These outputs already contain piecewise
warp rendering. Edited analysis without program media uses a matching program
report's timeline-to-rendered map, including gaps/padding; stale maps are rejected.
Temporary audio is removed after measurement. Raw recorder warp replay is not used.

## LLM task routing and term check

Configure per-task LLM routing in Advanced YAML with `llm.roles`, for example:

```yaml
llm:
  roles:
    proofread: {model: editor}
    article: {model: writer, base_url: 'http://127.0.0.1:9000'}
    judge: {model: reviewer}
text:
  proofread: true
  term_check: true
  term_check_network: false
  term_fixes: {Oldword: Newword}
  term_max_candidates: 10
```

Supported roles are proofread/article/context/scout/cleanup/judge; omitted roles
inherit section defaults. Managed llama uses one GGUF and owned endpoint: role model
names do not switch weights, and endpoint overrides are rejected in managed mode.
Term check runs after proofreading, then realigns corrected words. Manual fixes
work offline; MediaWiki requests require both term_check and term_check_network.
Lookup cache is atomic; failures preserve text with diagnostics. Maximum suspect
terms is bounded, though each term may require multiple spelling-variant lookups.

## Channel queue and local installation

```bash
uvf channel 'https://www.youtube.com/@CHANNEL/videos' /path/to/channel --limit 10 --events --report /path/to/channel/report.json
bash tools/night_run.sh 'https://www.youtube.com/@CHANNEL/videos' /path/to/channel
bash packaging/install_linux.sh /path/to/ultimate-video-forge
```

Channel discovery is bounded (1–1000 items); projects use validated video IDs.
Existing project directories resume processing, incomplete nonempty downloads are
reported rather than overwritten. Acquisition progress is saved atomically in
`channel-acquisition.json`; batch/channel accept `--report` and emit a structured
`queue_finished` event with `--events`.
The nightly helper accepts extra channel options and `UVF_BIN` for the executable.

Notifications: `--notify-webhook URL` POSTs the JSON summary (`text`/`content`
fields work with Slack, Discord and ntfy), `--notify-desktop` uses `notify-send`
on Linux or `osascript` on macOS, and `--notify-when always|failure|never`
(default `failure`) selects when. Delivery results go into the report; a failed
notification never changes the run status. Arbitrary shell hooks stay disabled.

Defaults live in settings: `batch.notify_when`, `batch.notify_webhook`,
`batch.notify_desktop` (flags override them) and `batch.min_free_disk_gb`, which
stops the queue before a project when the source volume has less free space.
Download options are the `fetch` section: `download` (`video` or `audio`),
`max_height`, `cookies_file`, `retries`, `rate_limit`; channel discovery also
applies `exclude` (video IDs), `skip_live` and `min_duration_s`/`max_duration_s`
before `--limit` (unknown durations are kept). `uvf channel` reads
`<channel folder>/settings.yaml` and `--set`; `uvf fetch` accepts `--set`.

Cancellation (`--cancel-file`) now stops yt-dlp during discovery and during a
download. The partial file stays in the episode folder with a `.fetch-incomplete`
marker; the next run for the same URL resumes it. A cancelled channel run records
the interrupted video, writes the report with status `cancelled` and exits 130.

## Unsaved settings and edit audition

Settings edits survive tab navigation and require confirmation before project
switch/close or applying compute recommendations. Running with unsaved settings
is blocked; processing save cannot silently overwrite changed Advanced YAML.

The Edit page shows the whole timeline above the decision list: kept material,
accepted cuts (solid, coloured by reason), suggestions that are not applied
(hatched) and markers; clicking a cut selects it in the list and hovering shows
its note. The status line gives the resulting program length. **Listen with cut**
plays two seconds before and after with the removed part skipped and a short
crossfade; **Listen to original** plays the same span uncut. Previews come from
the timeline voice master when it exists, otherwise from a synchronized camera
voice or the transcript source through its placement, so no master is required.

`program` joins pieces with `program.crossfade_ms` (default 10 ms) audio
crossfades that keep the exact program length: constant-power on edit joins,
linear where only the camera changes. `0` restores hard joins.

## Rough-cut presets and word check

Pause thresholds come in packaged presets: `roughcut-gentle`, `roughcut-balanced`
(the defaults) and `roughcut-tight`. `uvf preset` lists them,
`uvf preset <folder> --apply roughcut-tight` merges one into the project
settings (other keys stay as they are); the Settings page has the same choice
under **Apply preset**.

Audio that is neither silence nor a recognized word (laughter, music, coughs,
speech Whisper skipped) longer than `roughcut.unknown_sound_min_s` becomes an
`unknown-sound` marker and is never cut automatically. "как бы" is marked as a
filler when spoken together; an LLM filler marker at the same word replaces it.
Every cut, including retakes and LLM findings, is dropped if either edge would
fall inside a word.

`python -m tools.check_program <folder>` re-transcribes the rendered master and
compares its words with the edited transcript in edited time, writing missing,
extra or shifted words. Disagreements need listening: Whisper can mishear the
master, and a clean result is evidence, not proof, of clean joins.

## LLM rough-cut suggestions

`roughcut.llm: true` runs four detectors through the local LLM (role `roughcut`):
`start` (where the conversation really begins), `slips` (corrected slips of the
tongue), `fillers` (semantic fillers) and `delivery` (best take of a retake
group). The model only returns phrase numbers and verbatim quotes; a quote must
match transcript words exactly, and cut times come from those words. Findings
below `roughcut.llm_min_confidence`, longer than `llm_max_cut_s`, beyond
`llm_max_start_s` or not provable are rejected and listed in
`stages/roughcut/llm-report.json`. With `roughcut.llm_apply: suggest` (default)
LLM cuts appear unaccepted in the Edit list and delivery choices as markers;
`cut` applies them. Fillers are always markers.

## Recorder tracks and speaker cameras

Synchronization uses the probed default audio stream by its absolute ffprobe
index. Override it with `--set transcribe.audio_streams.<asset_id>=N`; the same
stream is used for Whisper and synchronization matching/rendering. Camera-only
NLE export writes that stream to `export/camera-audio` as lossless PCM and mutes
embedded camera audio, so Final Cut/Premiere receive an unambiguous audio asset.
Actual import/playback acceptance still requires those applications.

Choose the synchronization reference with
`--set sync.reference_camera=<asset_id_or_group>` (default: `auto`). Recordings
connected through overlapping cameras/recorders share a timeline automatically.
For disconnected sessions, set `sync.session_offsets` for one camera or recorder
in each session, in seconds relative to the reference camera clock. For example,
`--set sync.session_offsets.cam-session-2=120` places that camera at 120 seconds
before global origin normalization. A disconnected session without this choice
produces an actionable error instead of an invented placement.

The Material page lists assets under their groups; drag assets onto another group
to regroup them (several at once with multi-select). Scan proposes one lane per
recorder channel. Select a lane to merge channels (`0,1` for a stereo pair), name
its speaker or disable it. Named lanes drive microphone speaker attribution ahead
of `speakers.tracks`; disabled channels are skipped. Speaker cameras chosen there
are stored in the project and override `program.speaker_cameras` for the same
speaker. CLI:

```bash
uvf tracks SOURCE                                   # list lanes and cameras
uvf tracks SOURCE --asset zoom-h6 --channels 0,1 --speaker Anna
uvf tracks SOURCE --asset zoom-h6 --channels 3 --disable
uvf cameras SOURCE --speaker Anna --camera osmo
uvf material SOURCE --asset cam-a --asset cam-b --group wide
```

### Wide shot

Assign the wide (establishing) camera like a speaker: on the Material page pick
**Wide shot** in the speaker list, or `uvf cameras <folder> --speaker @wide --camera cam-c`
(`program.speaker_cameras: {"@wide": cam-c}` works as a settings default). With a
wide shot, the program and multicam selection go wide when speakers overlap and
when `program.max_switches` speaker changes (default 3) fall inside
`program.switch_window_s` (6 s); `program.max_shot_s` (0 = off) inserts
`wide_insert_s` (3 s) of wide into longer single shots. Switches happen
`program.cut_lead_s` (0.2 s) before the turn. Manual camera choices in the edit
always win. Without a wide shot, overlaps keep the current camera as before.

## Settings migration and legacy comparison

```bash
python -m tools.migrate_settings --whispersync ws.json --forge config.yaml \
  --output /path/to/material/_studio/settings.yaml --report migration.json
python -m tools.legacy_suites run --suite whispersync --python /path/to/ws-env/bin/python
python -m tools.legacy_suites compare old/whispersync.json new/whispersync.json
```

Migration follows plan table 6.3 and copies only keys set in the old files. The
report lists every key as migrated or unmapped with its planned Studio section;
conflicts and dependency fixes (for example tracking without crop framing) are
listed too, and the output passes Studio validation. The legacy runner executes
the frozen suites unmodified in their own environment and compares per-test
outcomes with the 403/620 snapshot and with an earlier report.

## Linux installation

Linux installation copies the standalone binary into `~/.local/bin`, adds the
`uvf` link and desktop entry, and checks packaged resources. Override
`UVF_INSTALL_PREFIX` for another location. It relies on host ffmpeg/system libraries;
this is a local installer scaffold, not a fully bundled cross-platform release.
