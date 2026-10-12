"""Names, help and order of every project setting in the desktop form.

Every field of :class:`studio.core.settings.Settings` appears here exactly once
(a test checks it), so a new setting cannot be missing from the advanced mode.
"""

from __future__ import annotations

from typing import NamedTuple


class Item(NamedTuple):
    key: str
    label: str
    help: str = ""


class Group(NamedTuple):
    title: str
    items: tuple[Item, ...]


class Section(NamedTuple):
    name: str
    title: str
    groups: tuple[Group, ...]


# Speech languages offered by the basic mode; ``None`` lets Whisper detect it.
LANGUAGES: tuple[tuple[str, str | None], ...] = (
    ("Russian", "ru"), ("English", "en"), ("Detect automatically", None),
)

SECTIONS: tuple[Section, ...] = (
    Section("scan", "Material", (
        Group("Files", (
            Item("min_duration_s", "Ignore clips shorter than, s",
                 "Shorter files are false starts (a camera switched on and off by mistake) "
                 "and are ignored; Final Cut lists them under “Not synchronized”. "
                 "0 keeps everything."),
            Item("channel_analysis", "Compare audio channels",
                 "Finds dual-mono, safety-copy and dead channels in multichannel audio."),
            Item("verify_digests", "File change check",
                 "cached: reuse file digests while size, times, inode and a content sample "
                 "match; always: rehash every file once per run."),
            Item("grouping", "Camera grouping",
                 "Not used yet: files in a subfolder form one camera, loose files are "
                 "grouped by device."),
            Item("probe_timeout_s", "Media probe timeout, s", "Not used yet."),
        )),
    )),
    Section("sync", "Synchronization", (
        Group("Placement", (
            Item("mode", "Mode",
                 "auto: per camera, one stretch of the recorder when the drift is small, "
                 "otherwise a piecewise fit (complex); camera: place cameras without "
                 "rendering recorder sound under them; simple / complex force the fit."),
            Item("placement", "Place cameras by",
                 "text: every camera is transcribed and placed by its words first, "
                 "acoustics refine around that place; acoustic: search the recorder "
                 "acoustically first, text only for the rest."),
            Item("rescue", "Clips nothing placed",
                 "quality: transcribe such a clip again in quality mode and retry; "
                 "off: leave it unplaced."),
            Item("reference_camera", "Reference camera",
                 "auto, or an asset id / camera group whose clock is the timeline origin."),
            Item("locate_search_s", "Search around the predicted place, s",
                 "Other cameras of a group are searched this far around their predicted "
                 "place on the recorder."),
            Item("session_offsets", "Session offsets, s",
                 "For sessions no overlapping recording connects: {asset id: seconds} "
                 "relative to the reference camera clock."),
        )),
        Group("Fitting the recorder", (
            Item("strategy", "Piecewise strategy",
                 "Used by the complex fit. 1: one global stretch; 2: a piece per word "
                 "anchor (tightest); 3: a piece per phrase, cut only in pauses "
                 "(smoothest, recommended)."),
            Item("max_drift_ms", "Drift tolerance, ms",
                 "Below it auto mode keeps one stretch; above it the sync check counts a "
                 "window as bad."),
            Item("boundary_flex", "Boundary flex",
                 "Nudges each piece boundary acoustically so speech lands under the "
                 "picture, independent of Whisper's word timings."),
            Item("recorder_mode", "Recorders per camera",
                 "best: the best-matching recorder under each camera; all: keep every "
                 "synchronized recorder as its own lane."),
        )),
        Group("Acoustic measurement", (
            Item("acoustic_refine", "Acoustic refinement",
                 "auto: correct the plan acoustically when the camera has sound."),
            Item("acoustic_grid_s", "Alignment grid spacing, s",
                 "Acoustic alignment measures a window every this many seconds."),
            Item("acoustic_window_s", "Alignment window, s"),
            Item("acoustic_min_sharpness", "Minimum peak sharpness",
                 "Correlation peaks less sharp than this are not trusted."),
            Item("refine_grid_s", "Check grid spacing, s",
                 "Grid of the sync check, the plan refinement and the drift check."),
            Item("refine_window_s", "Check window, s"),
            Item("refine_max_lag_s", "Largest lag the check sees, s",
                 "Recorder jumps of 0.2–0.8 s are invisible at 0.25 s."),
            Item("step_min_ms", "Smallest step to correct, ms",
                 "A jump inside a clip (a click, lost samples) is modelled from this size."),
        )),
        Group("Recorder repair", (
            Item("recorder_repair", "Repair recorder losses",
                 "auto: a recorder that loses samples (a phone dropping buffers) is "
                 "repaired against the cameras; Final Cut gets it as “Recorder (repaired)”."),
            Item("recorder_fill", "Fill lost time with",
                 "camera: the sound of the camera that heard it; silence."),
        )),
        Group("Checks", (
            Item("verify", "Measure sync after rendering",
                 "Measures every rendered voice against its camera on the check grid."),
            Item("self_check", "Content self-check",
                 "warn: re-transcribe rendered voices and report lost speech; repair: also "
                 "re-render failing clips (needs voice enhancement off)."),
        )),
        Group("Voice outputs", (
            Item("voice_segment_minutes", "Voice segments, min",
                 "0: whole files only; otherwise also publish PCM segments of this length."),
            Item("master_wav", "Voice master WAV",
                 "One unedited stereo voice track of the whole timeline."),
            Item("master_crossfade_ms", "Master crossfade, ms",
                 "Fades where sources overlap in the master; 0 = hard joins."),
            Item("voice_enhance", "Voice enhancement",
                 "denoise, denoise_dereverb or resemble; needs the optional audio module."),
            Item("ambience", "Camera room tone",
                 "Separate the camera's room tone and export it as its own lane."),
            Item("ambience_model", "Room tone separation model"),
        )),
    )),
    Section("transcribe", "Transcription", (
        Group("Whisper", (
            Item("model", "Whisper model", "For example large-v3 or large-v3-turbo."),
            Item("language", "Language",
                 "Language code such as ru or en; empty: detect automatically."),
            Item("mode", "Content transcripts",
                 "fast, or quality (slower, larger beam, fewer gaps)."),
            Item("device", "Device", "auto picks CUDA, Apple MLX or CPU."),
            Item("compute_type", "Compute type",
                 "auto, or a CTranslate2 type such as float16 or int8_float16."),
            Item("batch_size", "Batch size"),
            Item("beam_size", "Beam size (fast)"),
            Item("quality_beam_size", "Beam size (quality)"),
            Item("use_cache", "Reuse earlier transcripts",
                 "Transcripts of unchanged audio with the same options are reused."),
        )),
        Group("What is transcribed", (
            Item("recorder_scope", "Recorder transcription",
                 "auto / cameras: only around located cameras (the whole file when they "
                 "cover most of it); full: always the whole recorder."),
            Item("window_margin_s", "Margin around cameras, s",
                 "Recorder transcription around a camera reaches this far beyond it."),
            Item("ranges", "Manual recorder ranges",
                 "{asset id: [[start, end], ...]} in seconds."),
            Item("camera_transcripts", "Camera transcripts",
                 "auto: only where needed (not located, self-check, camera mode, no "
                 "recorder); always: every camera."),
            Item("camera_mode", "Camera transcripts for placement",
                 "Whisper mode of the transcripts that only place a camera on the recorder."),
            Item("audio_streams", "Audio stream per file",
                 "{asset id: ffprobe stream index}; the stream is used for Whisper and sync."),
        )),
        Group("Vocabulary and decoding", (
            Item("initial_prompt", "Initial prompt", "Text that primes Whisper's style."),
            Item("glossary", "Glossary",
                 "Names and terms Whisper should spell this way; also used as hotwords."),
            Item("hotwords", "Hotwords", "Overrides the glossary as Whisper hotwords."),
            Item("keep_fillers", "Keep fillers",
                 "Ask Whisper to write um / uh, so the rough cut can mark them."),
            Item("repetition_penalty", "Repetition penalty"),
            Item("no_repeat_ngram_size", "No-repeat n-gram size", "0 turns it off."),
            Item("vad_merge_gap_s", "Merge speech across pauses up to, s",
                 "Voice-activity fragments are merged into one Whisper clip only across "
                 "pauses this short."),
            Item("max_word_gap_s", "Split segments at word gaps over, s",
                 "Isolated 1–2 word series beyond such a gap are checked."),
        )),
    )),
    Section("roughcut", "Rough cut", (
        Group("Cuts", (
            Item("enabled", "Remove pauses and retakes"),
            Item("mode", "Cut or only mark",
                 "cut: remove what is accepted; markers: cut nothing, only mark."),
            Item("pause_min_s", "Cut pauses longer than, s"),
            Item("pause_keep_s", "Keep of each cut pause, s",
                 "Must be shorter than the pause threshold."),
            Item("silence_threshold_db", "Silence threshold, dB",
                 "Quieter audio counts as silence; louder sound is never cut as a pause."),
            Item("head_tail_pad_s", "Keep before the first / after the last word, s"),
            Item("head_tail_sound", "Noisy head and tail",
                 "Sound before the first or after the last word: cut, suggest (not "
                 "applied) or keep."),
        )),
        Group("Retakes", (
            Item("detect_retakes", "Detect retakes",
                 "A phrase said again: earlier takes are cut, the last one is kept."),
            Item("retake_min_words", "Retake: minimum repeated words"),
            Item("retake_max_gap_s", "Retake: maximum gap between takes, s"),
            Item("phrase_gap_threshold", "Phrase gap, s",
                 "A word gap this long ends a phrase."),
        )),
        Group("Markers", (
            Item("unknown_sound_min_s", "Mark untranscribed sound longer than, s",
                 "Laughter, music, coughs or speech Whisper skipped become review markers "
                 "and are never cut automatically."),
        )),
        Group("LLM suggestions", (
            Item("llm", "Suggest LLM edits (slips, start, delivery)"),
            Item("llm_detectors", "Detectors",
                 "start: where the conversation begins; slips: corrected slips of the "
                 "tongue; fillers; delivery: the best take of a retake."),
            Item("llm_apply", "Apply LLM findings",
                 "suggest: listed unaccepted in the Edit list; cut: applied."),
            Item("llm_min_confidence", "Minimum confidence",
                 "Findings below it are rejected."),
            Item("llm_max_cut_s", "Longest LLM cut, s"),
            Item("llm_max_start_s", "Latest conversation start, s"),
            Item("llm_chunk_chars", "Characters per request"),
            Item("llm_prompt_language", "Prompt language"),
            Item("llm_json_retries", "JSON retries per request"),
            Item("llm_retry_budget", "Retry budget per run"),
            Item("base_url", "LLM endpoint"),
            Item("model", "LLM model"),
            Item("request_timeout_s", "Request timeout, s"),
        )),
    )),
    Section("export", "Final Cut / Premiere", (
        Group("Project", (
            Item("name", "Final Cut event name",
                 "Event, project and multicam name; empty: the material folder's name."),
            Item("targets", "Formats",
                 "multicam: Final Cut multicam (Studio.fcpxml); xmeml: Premiere / Resolve "
                 "XML; fcpxml: the flat Final Cut timeline; otio: OpenTimelineIO."),
            Item("fcpxml_version", "FCPXML version"),
        )),
        Group("Markers", (
            Item("markers", "Markers",
                 "cuts: one marker per accepted cut; all: also filler and untranscribed "
                 "sound hints (often thousands); none."),
            Item("marker_min_s", "Mark only cuts of at least, s",
                 "Shorter cuts get no marker; 0 marks every accepted cut."),
        )),
    )),
    Section("speakers", "Speakers", (
        Group("Attribution", (
            Item("method", "Method",
                 "auto, mics (who is loudest on their own microphone), pyannote "
                 "(diarization) or off."),
            Item("tracks", "Speaker per microphone",
                 "{asset:channel: speaker}; lanes named on the Material page win."),
            Item("margin_db", "Microphone margin, dB",
                 "A speaker's microphone must be this much louder than the others."),
            Item("silence_floor_db", "Silence floor, dB"),
            Item("step_s", "Analysis step, s"),
            Item("min_turn_s", "Shortest turn, s"),
            Item("max_gap_s", "Merge turns across gaps up to, s"),
        )),
        Group("Diarization", (
            Item("model", "Diarization model"),
            Item("device", "Device"),
            Item("match_files", "One name per person across files",
                 "Matches speaker labels of different files by voice."),
            Item("match_threshold", "Voice signature match",
                 "Cosine similarity of the MFCC voice signatures."),
            Item("embedding_threshold", "Embedding match",
                 "Cosine similarity of pyannote speaker embeddings."),
        )),
    )),
    Section("program", "Review video", (
        Group("Render", (
            Item("enabled", "Render review video"),
            Item("burn_subtitles", "Burn subtitles"),
            Item("encoder", "Video encoder",
                 "auto tries NVENC / VideoToolbox and falls back to the CPU."),
            Item("fps", "Frame rate", "auto, or a rate such as 25 or 30000/1001."),
            Item("crossfade_ms", "Join crossfade, ms", "0 restores hard joins."),
        )),
        Group("Camera switching", (
            Item("speaker_cameras", "Camera per speaker",
                 "{speaker: camera}; @wide names the wide shot."),
            Item("min_shot_s", "Shortest shot, s"),
            Item("cut_lead_s", "Switch before the turn, s"),
            Item("max_switches", "Go wide after this many switches"),
            Item("switch_window_s", "…within, s"),
            Item("max_shot_s", "Insert a wide shot into shots longer than, s", "0 = off."),
            Item("wide_insert_s", "Wide insert length, s"),
        )),
    )),
    Section("text", "Text", (
        Group("Outputs", (
            Item("proofread", "Proofread transcript"),
            Item("article", "Generate article"),
            Item("min_similarity", "Minimum similarity of a correction",
                 "Proofreading that changes a segment more than this is rejected."),
        )),
        Group("Terms", (
            Item("term_check", "Check terms", "Needs proofreading."),
            Item("term_check_network", "Look terms up online (MediaWiki)"),
            Item("term_fixes", "Manual term fixes", "{wrong: right}."),
            Item("term_max_candidates", "Most terms to check"),
        )),
        Group("LLM", (
            Item("base_url", "LLM endpoint",
                 "Rough cut and reels follow it while they use the same endpoint."),
            Item("model", "LLM model"),
            Item("prompt_language", "Prompt language"),
            Item("max_chars_chunk", "Characters per proofreading request"),
            Item("article_max_chars_chunk", "Characters per article request"),
        )),
    )),
    Section("reels", "Reels", (
        Group("How many clips", (
            Item("enabled", "Select reels"),
            Item("clips_per_hour", "Clips per hour of material",
                 "The counts below then set only the mix of types. 0: the counts are exact."),
            Item("story_count", "Stories"),
            Item("story_max_s", "Longest story, s"),
            Item("reel_count", "Reels"),
            Item("reel_max_s", "Longest reel, s"),
            Item("long_reel_count", "Long reels"),
            Item("long_reel_max_s", "Longest long reel, s"),
            Item("highlight_count", "Highlight moments"),
            Item("max_candidates", "Reels when the reel count is 0"),
            Item("target_min_s", "Shortest reel without types, s"),
            Item("target_max_s", "Longest reel without types, s"),
            Item("min_score", "Lowest judge score (1-10)", "0 = no limit."),
            Item("min_duration_s", "Shortest clip, s", "0 = no limit."),
            Item("max_duration_s", "Longest clip, s", "0 = no limit."),
        )),
        Group("Selection", (
            Item("chunk_seconds", "Transcript chunk, s"),
            Item("max_chars_chunk", "Characters per request"),
            Item("chunk_overlap_s", "Chunk overlap, s", "-1: an eighth of the chunk, 20-45 s."),
            Item("chunk_tolerance_s", "Leeway past a chunk, s",
                 "A candidate further out of its chunk has a made-up time and is dropped."),
            Item("prompt_variant", "Prompt variant", "default, a or b."),
            Item("episode_context", "Episode overview",
                 "One extra request that summarizes the episode for every pass."),
            Item("context_digest_chars", "Overview excerpt, characters"),
            Item("cleanup", "Cleanup pass"),
            Item("review_batch_size", "Candidates per cleanup request"),
            Item("cleanup_max_candidates", "Most candidates to clean up", "0 = no extra limit."),
            Item("judge", "Judge pass"),
            Item("judge_batch_size", "Candidates per judge request"),
            Item("judge_max_candidates", "Most candidates to judge", "0 = no extra limit."),
            Item("judge_context", "Show the judge each clip's first and last words"),
            Item("judge_head_s", "Opening shown to the judge, s"),
            Item("judge_tail_s", "Ending shown to the judge, s"),
            Item("judge_excerpt_chars", "Characters per excerpt"),
            Item("require_candidate_overlap", "Drop answers no candidate explains"),
            Item("scout_quote_ratio", "Quote match to review",
                 "Candidates whose quote matches the transcript less are dropped at once."),
            Item("min_quote_ratio", "Quote match to select",
                 "Share of a candidate's quote that must match the transcript."),
            Item("quote_refine", "Widen clips to their quote"),
            Item("snap_boundaries", "Snap clip edges to phrases"),
            Item("snap_max_shift_s", "Furthest snap, s"),
            Item("scoring_weights", "Ranking weights",
                 "{hook: 2.0, ...}; factors: base, hook, readability, completeness, speaker, "
                 "duration, mid_thought, quote, audio, speech_rate, replay."),
            Item("diversity", "Prefer different topics"),
            Item("max_topic_similarity", "Topic similarity limit"),
            Item("mmr_lambda", "Weight of topic similarity"),
            Item("max_overlap_ratio", "Overlap allowed between clips",
                 "Share of the shorter clip; 0 forbids any overlap."),
            Item("replay", "Use YouTube “most replayed”"),
            Item("audio_features", "Measure loudness and silence"),
            Item("audio_max_candidates", "Most candidates to measure"),
            Item("audio_noise_db", "Silence level, dB"),
            Item("audio_silence_min_s", "Shortest silence, s"),
            Item("audio_timeout_s", "Measurement timeout, s"),
        )),
        Group("LLM", (
            Item("base_url", "LLM endpoint"),
            Item("model", "LLM model"),
            Item("prompt_language", "Prompt language"),
            Item("json_retries", "JSON retries per request"),
            Item("retry_budget", "Retry budget per run"),
            Item("request_timeout_s", "Request timeout, s"),
            Item("retry_backoff_s", "First retry pause, s"),
        )),
        Group("Rendering", (
            Item("render", "Render reels"),
            Item("render_fps", "Frame rate"),
            Item("width", "Width"),
            Item("height", "Height"),
            Item("framing", "Framing",
                 "source: keep the picture; crop: portrait crop; fit: whole picture with "
                 "padding."),
            Item("crop_x", "Crop position", "0 = left, 1 = right."),
            Item("tracking", "Follow face / active speaker", "Needs crop framing."),
            Item("tracking_device", "Tracking device"),
            Item("active_speaker", "Prefer the active speaker"),
            Item("face_min_size", "Smallest face, px", "Faces below this height are ignored."),
            Item("face_follow", "Follow the face within a turn"),
            Item("two_speaker_layout", "Two people in frame", "speaker: the one talking; "
                 "split: both, stacked."),
            Item("speaker_switch", "New speaker", "cut, or pan when they sit close."),
            Item("face_min_ratio", "Reject reels with faces in less than",
                 "Share of the reel; with tracking only. 0 keeps every reel."),
            Item("speaker_angles", "Switch cameras by speaker"),
            Item("loudness_lufs", "Loudness, LUFS", "0 keeps the source level."),
            Item("covers", "Cover images"),
            Item("edge_lead_in_s", "Silence before the first word, s"),
            Item("edge_tail_s", "Silence after the last word, s"),
            Item("edge_guard_s", "Distance from neighbouring words, s"),
            Item("edge_max_extend_s", "Grow an edge to its sentence, at most s"),
            Item("edge_max_trim_s", "Drop a sentence fragment up to, s"),
            Item("render_rejected", "Also render rejected moments",
                 "Into rejected/, for review; they are always listed in rejected.json."),
            Item("qa", "Check every reel",
                 "Streams and duration; a broken reel goes to rejected/."),
            Item("qa_blackdetect", "Also check for black picture", "Decodes the whole reel."),
            Item("export_webm", "Also WebM"),
            Item("export_gif", "Also GIF"),
            Item("export_audio", "Also audio (M4A)"),
            Item("preview_reel", "All reels in one preview video"),
            Item("encoder", "Encoder", "auto: NVENC when it works, else libx264."),
            Item("video_bitrate", "Video bitrate", "NVENC: the ceiling; libx264: the target."),
            Item("audio_bitrate", "Audio bitrate"),
            Item("preset", "x264 preset"),
            Item("nvenc_cq", "NVENC quality (lower is better)", "18-24 is sensible."),
            Item("nvenc_preset", "NVENC preset", "p1 (fastest) ... p7 (best)."),
        )),
        Group("Title and layout", (
            Item("layout", "Layout",
                 "video: picture only; title: a title band on top; audiogram: a waveform "
                 "under the title, no camera needed."),
            Item("title_field", "Title text"),
            Item("title_band", "Title band height", "Share of the frame height."),
            Item("title_seconds", "Show the title for, s", "0 = the whole reel."),
            Item("title_font", "Title font"),
            Item("title_color", "Title color"),
            Item("background_color", "Background color"),
            Item("waveform_color", "Waveform color"),
        )),
        Group("Captions", (
            Item("burn_subtitles", "Burn captions"),
            Item("caption_preset", "Caption style", "Empty keeps the editor's plain style."),
            Item("caption_highlight", "Highlight", "Empty: the style's choice."),
            Item("caption_case", "Letter case", "Empty: the style's choice."),
            Item("caption_punctuation", "Punctuation", "Empty: the style's choice."),
            Item("caption_max_words", "Most words per caption", "-1: the style's choice."),
            Item("caption_max_lines", "Most lines", "0: the style's choice."),
            Item("caption_max_chars", "Most characters per line",
                 "0: from the font and the caption width."),
            Item("caption_width_ratio", "Caption width", "Share of the frame width."),
            Item("caption_font_size", "Font size", "Styles are tuned at 96."),
            Item("caption_highlight_color", "Highlight color", "Empty: the style's choice."),
            Item("caption_pause_split_s", "New caption after a pause of, s"),
            Item("caption_max_duration_s", "Longest caption, s"),
            Item("caption_min_gap_s", "Shortest gap between captions, s"),
            Item("caption_fade_min_gap_s", "Fade only across gaps of, s"),
            Item("caption_censor", "Censored words", "Patterns such as бля*."),
            Item("caption_censor_style", "Censor style"),
            Item("caption_split_on_speaker", "New caption at each speaker"),
            Item("caption_speaker_colors", "Speaker colours",
                 "[#FFD400, #00C2FF]; by order of appearance."),
            Item("caption_vertical_align", "Caption row", "style keeps the look's row."),
            Item("caption_vertical_offset", "Caption offset",
                 "Share of the frame height, -0.5 to 0.5."),
            Item("caption_whisper_sync", "Check caption timing with Whisper",
                 "Listens to each reel again; drifted words take the new timings."),
            Item("caption_sync_model", "Whisper model for the check",
                 "Empty: the transcription model."),
            Item("caption_sync_min_match", "Words that must match",
                 "Share of the transcript words; below it the check is not trusted."),
            Item("caption_sync_threshold_s", "Retime from a drift of, s"),
        )),
    )),
    Section("llm", "Local LLM server", (
        Group("Owned llama-server", (
            Item("managed", "Launch owned llama-server for LLM processing"),
            Item("executable", "llama-server executable"),
            Item("model_path", "GGUF model file"),
            Item("port", "Port"),
            Item("startup_timeout_s", "Startup timeout, s"),
            Item("log_file", "Server log", "Empty discards the server output."),
            Item("roles", "Model per task",
                 "{role: {model, base_url}} for proofread, article, context, scout, "
                 "cleanup, judge, roughcut."),
        )),
        Group("Server tuning (“default”: the server's own value)", (
            Item("n_gpu_layers", "GPU layers"),
            Item("ctx_size", "Context size"),
            Item("batch_size", "Batch size"),
            Item("ubatch_size", "Micro-batch size"),
            Item("threads", "Threads"),
            Item("main_gpu", "Main GPU"),
            Item("parallel", "Parallel slots",
                 "Studio sends one request at a time; extra slots only cost VRAM."),
            Item("flash_attn", "Flash attention"),
            Item("cache_type_k", "K cache type", "For example q8_0; needs flash attention."),
            Item("cache_type_v", "V cache type"),
            Item("ctx_checkpoints", "Context checkpoints"),
            Item("cache_ram_mb", "Prompt cache in RAM, MiB",
                 "A number, auto (free memory minus the reserve) or empty."),
            Item("cache_ram_reserve_mb", "RAM reserve for auto, MiB"),
            Item("cache_ram_min_mb", "Smallest auto cache, MiB"),
            Item("cache_ram_max_mb", "Largest auto cache, MiB"),
            Item("extra_args", "Extra arguments"),
        )),
        Group("Requests", (
            Item("temperature", "Temperature"),
            Item("n_predict", "Most tokens per answer"),
            Item("role_options", "Options per task",
                 "{role: {temperature, n_predict, timeout_s}}."),
            Item("watchdog", "Stream and watch responses",
                 "A server that stalls is abandoned and the request retried."),
            Item("first_token_timeout_s", "First token timeout, s"),
            Item("stall_timeout_s", "Stall timeout, s"),
            Item("max_retries", "Retries"),
            Item("retry_backoff_s", "First retry pause, s", "Doubles with each retry."),
        )),
    )),
    Section("compute", "Compute", (
        Group("Processes", (
            Item("cpu_workers", "CPU workers", "0: all cores but one."),
            Item("allow_cpu", "Allow CPU fallback", "Not used yet."),
        )),
        Group("Virtual machines", (
            Item("host_memory", "Free VM memory during runs",
                 "Shrinks running libvirt VMs and shuts down the listed ones; "
                 "all given back at the end (uvf host-memory restore after a crash)."),
            Item("vm_domains", "VMs to shrink", "Empty: every running VM."),
            Item("vm_stop_domains", "VMs to shut down",
                 "VMs with a passed-through GPU: a balloon frees nothing there."),
            Item("vm_shutdown_timeout_s", "Wait for a shutdown, s", "A VM is never forced off."),
            Item("vm_headroom_mb", "Leave each VM above its use, MB"),
            Item("vm_min_mb", "Never shrink a VM below, MB"),
            Item("vm_connect", "libvirt connection"),
        )),
    )),
    Section("fetch", "Download", (
        Group("YouTube and links", (
            Item("url", "Source URL"),
            Item("download", "Download"),
            Item("max_height", "Highest resolution", "0 = best available."),
            Item("cookies_file", "Cookies file"),
            Item("retries", "Retries"),
            Item("rate_limit", "Rate limit", "yt-dlp --limit-rate, e.g. 5M."),
        )),
        Group("Channel discovery", (
            Item("min_duration_s", "Shortest video, s", "0 = no limit."),
            Item("max_duration_s", "Longest video, s", "0 = no limit."),
            Item("skip_live", "Skip live streams"),
            Item("exclude", "Excluded video IDs"),
        )),
    )),
    Section("batch", "Batch", (
        Group("Queue", (
            Item("notify_when", "Notify"),
            Item("notify_webhook", "Webhook URL", "Slack, Discord and ntfy work."),
            Item("notify_desktop", "Desktop notification"),
            Item("min_free_disk_gb", "Stop below free disk, GB", "0 = off."),
        )),
    )),
)

# Multi-choice lists shown as check boxes.
LIST_CHOICES: dict[str, tuple[str, ...]] = {
    "export.targets": ("multicam", "xmeml", "fcpxml", "otio"),
    "roughcut.llm_detectors": ("start", "slips", "fillers", "delivery"),
}

# Bounds of numeric fields that differ from the generic ones.
RANGES: dict[str, tuple[float, float]] = {
    "sync.strategy": (1, 3),
    "reels.chunk_overlap_s": (-1, 600),
    "reels.min_score": (0, 10),
    "reels.nvenc_cq": (0, 51),
    "reels.caption_vertical_offset": (-0.5, 0.5),
    "llm.port": (1, 65535),
}


def items() -> dict[str, Item]:
    """Every described setting by its dotted key."""
    return {f"{section.name}.{item.key}": item
            for section in SECTIONS for group in section.groups for item in group.items}
