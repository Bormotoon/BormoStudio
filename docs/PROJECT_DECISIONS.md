# Project decisions

Confirmed by the project owner on 2026-10-08.

- Product name: Ultimate Video Forge.
- Development repository: `/srv/storage/docs/Ultimate-Video-Forge/`.
- License for the combined project: MIT. Original license notices in preserved
  source snapshots remain intact; third-party assets retain their own licenses.
- Immediate acceptance target: the current Linux workstation. Other platforms
  are deferred until the local workflow can be tested end to end.
- Original WhisperSync and Podcast Reels Forge repositories are left untouched.
  No feature freeze or archival changes are applied to them.
- Subtitle editor (owner correction, 2026-10-09): implement inside the existing
  PyQt GUI with native Qt controls and video preview. Save project settings and
  subtitle edits through the application; no separate browser interface or
  File System Access API dependency. This supersedes the browser-first decision.
- Technology policy: prefer the newest stable releases compatible with the
  actual dependency stack and verified processing workflow. Evaluate Python
  3.14 and current Qt/CTranslate2/torch releases before adopting them; retain
  Python 3.12 as the verified execution baseline until that evaluation passes.
  Optional modules may use separately verified interpreter versions.

## Existing implementation

Implementation and imported source histories currently live in the separate
local repository `/srv/storage/docs/Studio/`, through commit `b95d3c6`.
The destination already contains its own initial documentation commit. Bring
the implementation history into this repository while preserving its existing
documentation and Git remote; do not overwrite either repository or force-push.
The implementation is now merged into the destination with both parent histories
preserved. Existing destination README, MIT license, plan, and remote win conflicts.
The package and command names still need migration from `studio`.

Distribution identity is `ultimate-video-forge`, with `uvf`,
`ultimate-video-forge`, and `uvf-gui` entry points. Keep the internal `studio`
namespace and old commands during migration to avoid invalidating subprocess
entry points and recorded project paths in the same change.

## Autonomous implementation decisions (2026-10-08)

### Text-output migration (2026-10-09)

- Text generation is opt-in (`text.proofread` and `text.article` both default to
  `false`), so a normal local edit never starts an LLM service unexpectedly.
- Corrected transcript segments retain original ASR word timings and record an
  explicit text override. This preserves subtitle/export compatibility without
  falsely claiming word-level realignment; a future alignment pass may replace
  those timings under the same artifact contract.
- A failed proofread batch leaves its source text unchanged. An article chunk
  that fails the faithfulness guardrail is published with its diagnostic report,
  rather than silently presented as a faithful edit.

### Unattended batch safety (2026-10-09)

- `uvf batch` runs project folders sequentially through the normal locked
  pipeline and reports each result. It intentionally exposes no arbitrary
  before/after shell hooks until an allowlist and explicit permission contract
  exist; an unattended helper must not be able to terminate or modify unrelated
  local processes.

### YouTube source safety (2026-10-09)

- `uvf fetch URL DESTINATION` accepts only a new or empty material directory,
  downloads exactly one source plus its metadata without shell execution, then
  runs the standard scan. A remote URL is never mixed into an existing local
  shoot or allowed to overwrite user media.

- Continue in dependency order: reliable execution, source decoding, synchronized
  export, then remaining sync modes, editing/program, speakers, Forge stages,
  desktop controls and distribution. Phase 7 remains the follow-up roadmap after
  phases 1-6, as specified by the plan.
- Preserve frozen legacy snapshots until executable parity checks pass; do not
  delete legacy merely because a stage facade exists.
- Optional stage dependencies still determine ordering but cannot prevent export
  when roughcut is disabled. Required dependencies retain failure propagation.
- Forward validated JSON-lines events while workers are alive. Keep artifact
  publication transactional and separate from progress notifications. Bound the
  retained stderr diagnostic tail to avoid unbounded memory use.
- Explicit transcription streams use ffmpeg's absolute stream index (`0:N`),
  consistent with probe/prepare. Decode selected streams to private mono 16 kHz
  PCM; cache identity remains original content plus stream and decoding settings.
- Whisper OOM retries halve batches, then use unbatched decoding with beam 1.
  Never silently switch a requested GPU job to CPU. Other runtime errors propagate.
- Connect synchronized voice outputs in both export formats. FCPXML uses a gap
  bed with connected camera/audio lanes so overlapping cameras do not become
  overlapping primary-storyline clips; mute original camera audio when replaced.
- Apply the same keep ranges to video and synchronized voice in neutral export.
  Include voice/edit artifact checksums in export fingerprints and honor targets
  and FCPXML version. NLE import acceptance remains unverified.
- A passing combined unit suite is not completion of the entire migration.
  Private recordings, overnight operation, clean installers and NLE import require
  recorded evidence before closing their corresponding acceptance gates.

### Production hardening continuation

- Actual roughcut execution requires decoded-energy evidence: remove a candidate
  only when ffmpeg silencedetect covers its entire interior at the configured
  threshold. Apply this to head/tail as well as gaps. Audible untranscribed sounds
  remain retained; missing energy sources fail instead of silently guessing.
- Render the review master in private per-piece intermediates. Resolve each keep
  range against camera coverage and group selection, convert timeline to source
  time through SourcePlacement, and use the camera's synchronized voice WAV when
  available. Reject uncovered ranges rather than silently render the first camera.
- Normalize video dimensions/fps and audio sample rate/layout before concatenation;
  silent cameras get explicit silence. Current joins are duration-preserving hard
  joins; the required crossfade integration remains a tracked task.
- Auto/simple recorder modes now use the preserved accepted matcher and renderer:
  strategy 1 for simple, drift-based selection for auto, configured strategy for
  complex. Render WAV even near unity until safe original-source NLE references
  are proven. Acoustic-first auto remains outstanding.
- Timeline explicitly orders after sync, with sync optional for a single source.
  Derived timeline transcripts are excluded from sync fingerprint inputs to avoid
  recursive invalidation. Include recorder content identity in the render fingerprint.
- Stage failures preserve previous published results, skip required dependents,
  and let independent/optional branches proceed. CLI writes report.json and exits
  nonzero after processing independent branches. Settings reject wrong types rather
  than treating string booleans as enabled flags.

### Retakes and edit review

- Preserve the frozen WhisperSync token restart detector without changing its
  matching algorithm; adapt only imports, configuration and dataclasses. Run its
  eleven original regression cases against the migrated implementation plus four
  direct frozen/migrated parity cases. The original snapshots remain untouched.
- Roughcut enables deterministic retakes by default, keeps the last attempt and
  removes the preceding attempts only when the keeper boundary intersects no word.
  Retake cuts do not require silence across speech intentionally being removed.
- Give each cut a stable ID derived from boundaries and reason. Persist explicit
  acceptance/rejection in edit-overrides.json; apply matching overrides after future
  detector runs. Unknown IDs are rejected by the review command.
- CLI review transactions update edit.json, override storage and the existing
  roughcut artifact checksum together. Preserve the detector fingerprint so review
  does not rerun recognition or detection. Export/program fingerprints include edit
  checksums and roughcut enablement, so only derived outputs require rebuilding.
- Removed-span markers belong to the edited splice. Retained filler markers map
  through EditMap; markers inside removed ranges are omitted. Export both FCPXML
  and XMEML markers. UI playback/audition controls are still outstanding.
- Preserve original camera audio in XMEML as explicit linked channel items,
  disabled when replaced by synchronized voice. Add sequence dimensions and source
  channel metadata. Mixed-rate/source-format acceptance and NLE import remain open.

### Microphone speaker attribution

- Register the speakers stage in the public execution graph between timeline and
  roughcut. Roughcut orders after it but can run when attribution is unavailable.
- Configure explicit channel assignments with `speakers.tracks`, mapping
  `asset_id:zero_based_channel` to a speaker name. Explicit mics mode may use
  separate channels of one recorder. Auto does not assume ordinary stereo means
  two independent lavaliers; multichannel recordings require explicit assignment.
- Decode microphone channels once to 16 kHz float PCM and integrate RMS energy
  into 50 ms timeline bins using each recorder's SourcePlacement, including in
  points, drift and negative offsets. Require placements for every selected track;
  never compare unaligned microphones. Nonlinear per-camera AudioWarpMap attribution
  and pyannote remain separate future integration tasks.
- Classify silent words as unknown instead of overlap. Retain the 6 dB dominance
  rule for cross-bleed and overlap. Several files assigned to the same speaker
  combine by maximum envelope instead of inventing duplicate speaker identities.
- Persist Forge-compatible turn lists and a separate timeline-domain provenance
  report. Fingerprints include selected source content, placements and assignments.

### Sequential processing migration: acoustic alignment

- Preserve the full frozen acoustic algorithm in sync_acoustic.py, replacing only
  configuration/decoder dependencies. Auto/simple use waveform evidence first;
  complex uses text first with waveform fallback. Both paths use the existing
  acceptance gate; ambiguous/insufficient waveform evidence never becomes placement.
- Place camera-only projects in the first camera's clock, then normalize the
  earliest offset to zero. Retain affine clock factors rather than discard drift.
- Multiple recorders share the first recorder reference clock. Independently align
  each recorder and each camera/recorder pair. Fail when common-clock evidence is
  missing; disjoint recorder sessions are not guessed. Choose best voice by residual
  then inlier count; all mode additionally publishes every rendered recorder lane.
- Keep conservative prepare transcript requirements until acoustic evidence is
  integrated into discovery. Rendering multiple recorders still eagerly renders
  candidate lanes before selection; optimization follows correctness tests.

- Boundary Flex is opt-in during migration and now runs on planned pieces before
  rendering via the preserved acoustic implementation. `sync.verify` publishes
  realized lag/coverage verdicts; `uvf verify CAMERA VOICE` returns nonzero for
  failed or inconclusive measurements. Convert numpy scalars to JSON numbers at
  the report boundary. No unmeasured window is counted as successful sync.

- Voice segments use integer sample boundaries and preserve the original PCM
  codec. Publish a rendered-audio segment index with frame offsets and durations;
  retain the full voice WAV for existing export/program consumers. Use ffprobe
  rather than Python wave for metadata because Python 3.10 cannot read extensible
  24-bit WAV emitted by the renderer. Quiet-seam selection is still outstanding.
- Optional voice-master WAV selects one replacement per timeline interval rather
  than summing duplicated dialogue from overlapping cameras. Retime nonunity
  camera clocks before mixing; preserve gaps with silence and use the existing
  limiter-enabled timeline mixer. This is an unedited voice master.

- Migrate frozen content diagnosis and local re-alignment without algorithm changes.
  `sync.self_check=warn` re-recognizes selected rendered voice and compares it with
  the camera reference, publishes passed/failed/inconclusive outcomes and flagged
  spans, then unloads the shared check model. Keep automatic repair unavailable
  until replacement rendering, warp-map updates and post-repair verification are
  integrated together; local re-alignment alone is not a completed repair feature.

- Preserve separation/enhancement backend algorithms behind a probe facade and
  bounded disk-logged subprocess utility. Resolve `UVF_SEP_VENV`, the managed
  separation module and local .sep-venv in that order. Supported enhancement modes
  are denoise, denoise_dereverb and resemble; other historical modes remain unavailable.
- Enhancement happens before content check, segmentation and master generation.
  Backend-conformed audio replaces only the selected voice reference, preserving
  originals. Optional failures retain originals and publish explicit warning reports.
- Camera ambience is decoded with stable asset filenames, batch-separated and
  conformed to original duration/rate/channels/bit-depth before publication. Export
  attaches ambience as independent audio lanes. Model installation and actual GPU
  listening/quality acceptance are still outstanding.

- `sync.self_check=repair` now offers a conservative whole-clip acoustic-first
  linear rerender for clips whose original content check failed. Passed and
  inconclusive clips are not repaired. Acceptance requires both a passed new
  content check and passed 20 ms acoustic verification; failed or inconclusive
  verification retains the original voice/map. Candidate/backend failures are
  reported, and rejected candidate audio is removed.
- Accepted repair replaces the selected voice, its matching recorder/camera warp
  map and the selected all-recorder lane before segments/master/verification.
  Original assets and initial diagnostic reports remain available. Published
  output.json reflects final selected paths and strategies keyed by warp ID.
- Repair currently requires voice_enhance=off so an accepted rerender cannot
  silently bypass requested enhancement. Local span repair and a real-model
  end-to-end acceptance gate remain open; this is not full legacy repair parity.

- Program encoder auto-selection performs a bounded real one-frame NVENC encode
  probe. Auto falls back to libx264 on unavailable hardware/drivers or timeout;
  explicit nvenc fails with the probe error and explicit cpu skips hardware probes.
  A runtime NVENC failure in auto mode regenerates all pieces on CPU, avoiding
  concatenation of differing hardware/software H.264 parameter sets. The render
  report records selection, fallback reason, format and retained duration.

- Voice master uses configurable linear complementary crossfades
  (`sync.master_crossfade_ms`, default 10 ms) only when switching between
  sources with real shared timeline coverage. Fades occur before the switch;
  the timeline is not shortened and gaps are not filled with invented audio.
  Adjacent intervals belonging to one selected source are merged first.
- PCM tests exposed uncompensated 5 ms lookahead latency in the timeline mix
  limiter. Enable its latency compensation and express input delays in samples
  rather than rounded milliseconds. A one-sample-offset impulse regression
  verifies exact placement and retained output duration.

- Review master has an explicit CFR policy: `program.fps=auto` uses the first
  selected camera's reported rate; an explicit string rate such as `30000/1001`
  overrides it. Supported range is 1-240 fps. Normalize all pieces from actual
  source presentation timestamps, then trim/pad to cumulative rounded frame
  boundaries. Explicit concat durations avoid per-piece timestamp rounding drift.
- Edited transcript positions follow the same frame trim/pad decisions; zero-frame
  pieces contribute no words. The render report records requested/actual duration,
  per-piece frame counts and rational output rate. This changes program rendering,
  not NLE export frame rounding. Preserve libx264 as a CPU encoder alias alongside
  cpu; validation and encoder selection now accept both consistently.

- Microphone speaker attribution now smooths isolated known-speaker A-B-A islands
  shorter than `speakers.min_turn_s` (default 0.3 seconds) only between two stable
  same-speaker neighbors, with gaps within `speakers.max_gap_s` (default 0.3).
  Unknown/overlap, overlapping turns, real handoffs and short neighboring turns
  are retained. Decisions use original neighbors in one pass to avoid cascades.
  Zero min_turn_s disables smoothing. Raw attribution is retained separately in
  diarization.raw.json / speakers_raw; reports record the policy and thresholds.

- Review program supports explicit `program.speaker_cameras` mappings from speaker
  names to camera asset/group IDs. Coverage boundaries and speaker turns split
  retained ranges; unavailable preferred cameras fall back to a covered camera.
  Unknown/overlap/unmapped speech holds the current covered camera. Explicit
  edit camera choices take priority; minimum shot length defaults to one second.
  Camera decisions are recorded in render.json; they do not modify roughcut input
  or NLE exports. Adjacent camera cuts are merged for transcript retention so
  words crossing a camera switch are not removed.

- Explicit `speakers.method=pyannote` runs the existing isolated diarization
  module on one primary source, decoding mono 16 kHz audio. HF_TOKEN is read only
  from the environment. Default model is pyannote/speaker-diarization-3.1; device
  is explicitly cpu or cuda. Source turns map through SourcePlacement and are
  clipped to timeline coverage. Intersecting speaker labels become overlap;
  raw turns and bounded subprocess logs are preserved. Labels are anonymous and
  local to this source, not matched to microphone names. Auto remains mics-only.

- Optional export target multicam creates a separate FCPXML media/multicam
  resource with mc-angle video/audio lanes and project mc-clip/mc-source angle
  selection. Coverage partitions choose available video angles and corresponding
  synchronized voice angles; uncovered picture ranges remain gaps. Existing
  fcpxml/xmeml targets retain their behavior. Camera drift retiming is explicitly
  rejected for this target until multicam time maps are implemented. This writer
  is structurally tested, not yet validated by Final Cut import or an Apple DTD.

- Multicam project selection now reuses program.speaker_cameras/min_shot_s,
  including explicit roughcut camera choices. Source-time decisions are mapped
  across removed ranges into edited multicam time without removing alternative
  angles. Export fingerprints speaker artifacts and program selection settings.
- A generated multicam document passed xmllint against Apple's FCPXML 1.9 DTD
  obtained from the CommandPost mirror. DTD validation checks structure, not
  Final Cut playback, media relinking or actual angle-switch import acceptance.

- Multicam ambience is now connected inside a composite audio angle alongside
  the synchronized voice (or camera audio when no replacement exists). Select
  this single composite angle for audio rather than relying on simultaneous
  independent audio-angle selection. Child source starts retain roughcut in-points;
  composite-local offsets start at zero. Angle children are ordered by offset.
  A voice/ambience fixture passed FCPXML 1.9 DTD validation with xmllint.

- Multicam now carries affine SourcePlacement.k into a dedicated retimed
  sequence: timeline durations multiply by k, roughcut source in-points divide
  elapsed timeline time by k. Camera, replacement voice and ambience use identical
  linear FCPXML timeMap geometry. Adjusted clip start/end are frame-rounded;
  source endpoint values retain microsecond precision. Resource duration describes
  original source coverage rather than retimed duration. Flat exports retain their
  previous policy; NLE semantic playback acceptance remains outstanding.

- Flat export stage now builds the same affine-retimed sequence as multicam.
  FCPXML uses its existing linear timeMap; XMEML writes constant timeremap speed
  effects with percent speed 100/k on video, replacement audio, ambience and
  linked original camera channels. Timeline start/end/duration and source in/out
  stay distinct. Source out and file coverage use duration/k; explicit clip rates
  are emitted. Unity-speed clips have no speed filter. Premiere interpretation of
  constant-speed effects remains an external import/playback acceptance gate.

- Export clips retain source fps and picture dimensions separately from the
  sequence format. XMEML source in/out, file coverage and linked source channels
  use the source clock; sequence start/end use the sequence clock. FCPXML video
  assets reference distinct source format resources, preserving rational rates
  and dimensions. Replacement WAVs use the sequence frame clock for XML indexing.

- XMEML camera-channel tracks are reused per camera lane/channel across roughcut
  pieces. Every video/channel member links to the complete sibling group, with
  media type, track index, clip index and group index resolved after tracks exist.
  This avoids per-cut audio-track proliferation and partial stereo/multichannel
  links. External voice tracks remain separate from the original camera group.

- Rendered voice/ambience export metadata is read from WAV headers, with ffprobe
  fallback for extensible/RF64 PCM. Channels, rate and depth describe the actual
  audio file rather than camera defaults. Invalid audio fails instead of claiming
  a successful export. FCPXML assets expose channels/rate and dialogue/effects
  roles; XMEML emits actual sample rate and depth. Camera metadata comes from scan.

- Media probe now records audio_bits_per_sample for the selected default audio
  stream, preferring bits_per_raw_sample over storage bits_per_sample. Unknown
  depth (e.g. AAC) stays unknown and XMEML omits depth rather than inventing 16-bit
  precision. Scan fingerprint advances to invalidate cached metadata without the
  new field; existing MediaInfo constructors remain compatible via default None.

### Multicam is the Final Cut export (2026-10-12)

- First Final Cut import of sony_run failed: "unexpected value format=r1"
  (custom format name), every clip offline (relative links resolved from the
  document, not the material folder), relink refused ("not long enough": 25 fps
  asset length rounded on the 29.97 grid), 1,783 markers (filler and
  untranscribed-sound hints), no recorder track and a flat timeline.
- Decision: the default FCPXML is a multicam clip. Angles are camera groups
  (`Asset.group_id`), laid out as storylines with explicit gaps, plus one angle
  per placed recorder (the repaired recorder output on its placement). Angles are
  uncut and keep their own sound; the project selects picture from a camera and
  sound from the recorder, one multicam clip per kept range and camera choice.
  Camera plans stay in original time. Without a recorder each angle supplies
  its own sound.
- A placed recorder replaces the per-camera voices rendered from it in every
  export; their warps are single affine pieces matching the recorder placement.
- Markers default to accepted cuts only (`export.markers`).
- Checked: sony_run export resolves all 35 links, no asset is longer than its
  file, and the document validates against Apple's FCPXML 1.9 DTD. Final Cut
  import itself is still to be confirmed.

### Basic and advanced desktop modes (2026-10-12)

- The desktop GUI showed about 25 of some 250 settings; the rest needed YAML, and
  the folder → material → plan → run flow asked for decisions a first-time user
  cannot make.
- Decision: the basic mode is the default. It asks for the folder and four
  choices (language, pause removal, cut markers, review video), saves them into
  the project and runs everything. The advanced mode (per-user preference, not a
  project setting) shows every page and a settings form built from the typed
  settings and `studio/gui/settings_schema.py`, which names and explains every
  field; a test fails when a setting has no place there. Allowed values of text
  settings live once in `settings.CHOICES`, shared by validation and the form.
- The form saves only edited fields, so a value it would display rounded (or a
  list it shows as YAML) is never rewritten by accident.
- `export.marker_min_s` (default 0) is the marker threshold for cut markers in
  the NLE; untranscribed-sound hints keep `roughcut.unknown_sound_min_s`.
