# Third-party components and licenses

BormoStudio is `GPL-3.0-or-later` (see `LICENSE`). Every component below is
GPL v3-compatible unless an open issue says otherwise.

Inventory before distribution (2026-10-19). “Bundled” ships inside the wheel or
the PyInstaller build; “on demand” is downloaded by the user's machine through
Modules/Models (or a managed module environment) and is not redistributed by
us; “system” is found on the machine. Entries marked **verify** were written
from memory and must be checked against the upstream license before a release.

## Open issues before a public release

1. ~~PyQt6 is GPL v3~~ — resolved 2026-10-19: BormoStudio itself is now
   `GPL-3.0-or-later`, so binaries bundling PyQt6 are distributed under GPL v3
   with `LICENSE` inside and the source at the tagged commit.
2. **`bignoodletoooblique.ttf` has no license file.** BigNoodleTitling Oblique
   (James Arboghast / Sentinel Type) is distributed as freeware; its terms for
   redistribution inside another product are not recorded here. Obtain and ship
   its license text, or drop the font from the caption presets.
3. **Gemma 4 GGUF terms** (catalog entries `ggml-org/gemma-4-*`): downloaded on
   demand, never bundled, but the setup wizard offers them; link the model's
   own terms in the GGUF picker. **verify** the licence named on the model card.
4. **Light-ASD weights** (`Junhua-Liao/Light-ASD`, downloaded on demand):
   **verify** the repository licence before offering it by default.

## Bundled

| Component | Use | License |
|---|---|---|
| PyQt6 | desktop GUI | GPL v3 (matches the project's licence) |
| Qt 6 (PyQt6-Qt6 wheels) | GUI, Qt Multimedia playback | LGPL v3 (Qt Multimedia's FFmpeg backend: LGPL) |
| NumPy | audio analysis | BSD-3-Clause |
| PyYAML | settings | MIT |
| platformdirs | user folders | MIT |
| Montserrat, Oswald, Unbounded (static instances), Russo One, Press Start 2P, Rubik Mono One | caption looks | SIL OFL 1.1 (`studio/resources/fonts/licenses/`); instanced fonts renamed, no Reserved Font Name |
| BigNoodleTitling Oblique | caption look | unknown (issue 2) |
| Subtitle editor (`subtitle_editor/`), prompts, presets | from Podcast Reels Forge | MIT (Podcast Reels Forge Contributors), notice in `subtitle_editor/LICENSE-Forge.txt` |
| Legacy snapshots (`legacy/`) | frozen comparison suites; not in the wheel or the PyInstaller build | Forge: MIT; WhisperSync: PolyForm Noncommercial 1.0.0 (the owner's own code; the parts ported into `studio/` are released by the owner under the project's GPL) |

## Optional extras and managed modules (installed on the user's machine)

| Component | Where | License |
|---|---|---|
| faster-whisper, CTranslate2 | `[whisper]` extra | MIT |
| mlx-whisper | `[mlx]` extra (Apple Silicon) | MIT |
| PyTorch, torchaudio | `diarization`, `vision` modules | BSD-3-Clause style |
| pyannote.audio 3.x | `diarization` module | MIT |
| opencv-python-headless | `vision` module | Apache 2.0 (wheels carry LGPL FFmpeg) |
| huggingface_hub | `diarization` module | Apache 2.0 |
| yt-dlp | `youtube` module | Unlicense |
| OpenTimelineIO | dev extra (OTIO export tests) | Apache 2.0 |

## Models downloaded on demand

| Model | Source | License |
|---|---|---|
| Whisper tiny/base/small/medium/large-v3 (CTranslate2) | `Systran/faster-whisper-*` | MIT (OpenAI Whisper weights) |
| Whisper large-v3-turbo (CTranslate2) | `deepdml/faster-whisper-large-v3-turbo-ct2` | MIT (OpenAI weights) |
| Gemma 4 E4B / 12B / 26B-A4B GGUF | `ggml-org/gemma-4-*-GGUF` | issue 3 |
| pyannote `speaker-diarization-3.1` (+ `segmentation-3.0`) | Hugging Face, gated (user accepts terms) | MIT |
| WeSpeaker ResNet34 embeddings used by the pipeline | Hugging Face | CC BY 4.0 — attribution in the about/credits text |
| YuNet face detector (`face_detection_yunet_2023mar.onnx`) | OpenCV Zoo | MIT |
| Light-ASD active-speaker model | GitHub `Junhua-Liao/Light-ASD` | issue 4 |

## System tools

| Tool | Use | Note |
|---|---|---|
| FFmpeg / ffprobe | all media work | Not shipped; the user's build decides LGPL/GPL. Shipping one later means choosing an LGPL build or GPL terms for the bundle. |
| llama.cpp `llama-server` | managed LLM | MIT; not shipped |
| notify-send / osascript | desktop notifications | system |
