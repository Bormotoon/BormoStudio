"""Use the same selected audio stream for matching and lossless voice rendering."""

import json
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from studio.core.channels import effective_audio
from studio.core.media import probe as public_probe
from studio.core.project import AssetRole, Project


@dataclass(frozen=True)
class AudioInfo:
    duration: float
    audio_sample_rate: int | None
    audio_channels: int | None
    audio_codec: str | None
    audio_bits_per_sample: int | None


def probe(path: Path) -> AudioInfo:
    info = public_probe(path)
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                             "-show_streams", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True)
    stream = json.loads(result.stdout)["streams"][0]
    bits = int(stream.get("bits_per_raw_sample") or stream.get("bits_per_sample") or 0)
    return AudioInfo(info.duration_s, info.audio_sample_rate, info.audio_channels,
                     info.audio_codec, bits or None)


def pcm_codec_for(info: AudioInfo) -> str:
    bits = info.audio_bits_per_sample or 24
    return "pcm_s16le" if bits <= 16 else "pcm_s24le" if bits <= 24 else "pcm_s32le"


@contextmanager
def selected_media(
    project: Project, directory: Path, streams: dict[str, int] | None = None,
) -> Iterator[Project]:
    with tempfile.TemporaryDirectory(prefix=".selected-audio-", dir=directory) as temporary:
        assets = []
        for asset in project.assets:
            stream = (streams or {}).get(
                asset.id, asset.manual.get("media_info", {}).get("audio_stream_index"),
            )
            if stream is None or asset.role not in {AssetRole.CAMERA, AssetRole.RECORDER}:
                assets.append(asset)
                continue
            source = project.source_dir / asset.path
            effective = effective_audio(asset)
            if effective.pan is None and _plain_wav(source, asset):
                # A single-stream PCM WAV is read directly: decoding hours of
                # recorder into a temporary copy bought nothing.
                assets.append(asset)
                continue
            # Distinct channels only (dual mono → mono); float32 keeps 24-bit PCM exact.
            output = Path(temporary) / f"{asset.id}.wav"
            command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
                       "-map", f"0:{stream}", "-vn"]
            if effective.pan is not None:
                command += ["-af", effective.pan]
            result = subprocess.run([*command, "-c:a", "pcm_f32le", "-rf64", "auto", str(output)],
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"selected stream decode failed for {asset.id}: {result.stderr}")
            info = public_probe(output).to_dict()
            assets.append(replace(asset, path=output, manual={
                **asset.manual, "media_info": {**asset.manual.get("media_info", {}),
                                              **{key: value for key, value in info.items()
                                                 if key.startswith("audio_")}},
            }))
        yield replace(project, assets=assets)


def _plain_wav(path: Path, asset: object) -> bool:
    """Whether ``path`` is a WAV with exactly one (PCM) audio stream."""
    if path.suffix.lower() not in {".wav", ".wave"}:
        return False
    info = getattr(asset, "manual", {}).get("media_info", {})
    codec = str(info.get("audio_codec") or "")
    if not codec.startswith("pcm_"):
        return False
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                                 "-show_entries", "stream=index", "-of", "json", str(path)],
                                capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return len(json.loads(result.stdout).get("streams", [])) == 1
