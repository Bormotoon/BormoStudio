"""Camera-ambience extraction (remove the camera's own voice, keep the room tone).

When the recorder's clean voice and the camera's built-in-mic voice both play on the
final timeline, their ~tens-of-ms offset is heard as a doubled/echoed voice — even
though lip-sync is fine, two near-identical voices comb-filter. Plainly ducking the
camera would also kill the ambience the editor wants. Instead we run a source-
separation model over the camera audio to strip the *vocal* and keep the
*instrumental* (= ambience/room tone), and lay that on its own lane next to the
synced voice. The voice then comes only from the clean recorder track, with no echo.

The separator (audio-separator + a RoFormer model) lives in a SEPARATE Python venv
(``.sep-venv``) because it needs an older Python than the main app; we invoke it as a
subprocess. The model runs on the GPU.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from platformdirs import user_data_dir

from studio.core.proc import run_logged
from studio.stages.sync_media import pcm_codec_for, probe
from studio.stages.sync_render import conform_wav_to

logger = logging.getLogger(__name__)
APP_NAME = "studio"

# audio-separator names its output "<input-stem>_(<Stem>)_<model-stem>.wav".
_STEM = "Instrumental"

# audio-separator's short-audio path is broken for the MDX-C/RoFormer models we
# use: below ~10 s it enables ``override_model_segment_size`` and then dies with
# "The size of tensor a (0) must match the size of tensor b (N)". Field case: a
# 1.75 s camera clip in a 71-clip shoot. Pad anything shorter than this to
# ``MIN_INPUT_SECONDS`` with silence and trim the result back afterwards —
# a couple of seconds of extra GPU work buys a working output for short clips.
MIN_INPUT_SECONDS = 12.0

# Inputs the separator silently dropped are retried one-per-process (a failure
# can be transient — e.g. GPU memory pressure from a neighbouring file in the
# batch). Each retry reloads the 1.5+ GB model, so only retry when the failures
# are a small minority; a batch that mostly failed has a systemic cause that
# retrying won't fix.
_MAX_RETRY_FILES = 10

# audio-separator swallows per-file exceptions (separator.py::separate logs them
# and carries on, so the process still exits 0). This is how we recover the
# reason for a missing output from its stderr.
_FAILED_FILE_RE = re.compile(r"Failed to process file (?P<path>.+?): (?P<reason>.+)")


# Where an optional backend venv may live, in priority order. `repo_root` is
# whatever the caller passes (in a checkout, the repo; after `pip install`,
# site-packages, where no `.sep-venv` will ever exist), so the search must not
# stop there — otherwise an installed WhisperSync can never find a venv the
# user did set up. An explicit path always wins.
SEP_VENV_ENV_VAR = "BSTUDIO_SEP_VENV"


def _venv_bin_names(name: str) -> tuple[str, ...]:
    """Candidate executable paths inside a venv, for this platform.

    POSIX venvs put executables in ``bin/``; Windows uses ``Scripts\\`` with an
    ``.exe`` suffix. Searching only ``bin/python`` meant a Windows user who had
    installed the separator was told it was not set up — the feature was
    unreachable on a platform the project claims to support.
    """
    if sys.platform == "win32":
        return (f"Scripts/{name}.exe", f"Scripts/{name}", f"bin/{name}")
    return (f"bin/{name}", f"Scripts/{name}.exe")


def sep_venv_candidates(repo_root: Path) -> list[Path]:
    """Directories that might hold the separation venv, most specific first."""
    candidates: list[Path] = []
    explicit = os.environ.get(SEP_VENV_ENV_VAR) or os.environ.get("UVF_SEP_VENV")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    from studio.modules.manager import ModuleManager

    candidates.append(ModuleManager().root / "separation")
    candidates.append(repo_root / ".sep-venv")
    # An installed package sits in site-packages, where no `.sep-venv` will
    # ever be; the user data directory is where an installed WhisperSync can
    # legitimately keep one. Deliberately NOT the current directory: making the
    # backend depend on where the user happened to launch from would let an
    # unrelated `.sep-venv` be picked up silently.
    candidates.append(Path(user_data_dir(APP_NAME)) / ".sep-venv")
    seen: set[Path] = set()
    unique: list[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _find_in_venv(repo_root: Path, name: str) -> Path | None:
    for venv in sep_venv_candidates(repo_root):
        for rel in _venv_bin_names(name):
            candidate = venv / rel
            # Existence is not enough: the earlier check accepted an empty,
            # non-executable placeholder as a working backend, so the failure
            # surfaced as a confusing subprocess error much later.
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    return None


def separator_python(repo_root: Path) -> Path | None:
    """The separation venv's python interpreter, or None if it isn't set up."""
    return _find_in_venv(repo_root, "python") or _find_in_venv(repo_root, "python3")


def separator_cli(repo_root: Path) -> Path | None:
    """The isolated venv's ``audio-separator`` console script, or None.

    We invoke this entry point (not ``python -m audio_separator.utils.cli``, which
    has no ``__main__`` guard and would silently no-op)."""
    return _find_in_venv(repo_root, "audio-separator")


def is_available(repo_root: Path) -> bool:
    return separator_cli(repo_root) is not None


def _sanitize_base(name: str) -> str:
    """Mirror audio-separator's own output-name sanitisation.

    It builds output names as "<base>_(<Stem>)_<model>.wav" but runs the base
    through ``CommonSeparator.sanitize_filename`` first: invalid path
    characters become "_", runs of "_" collapse to one, and leading/trailing
    "_. " are stripped. Reproducing that exactly is what makes the predicted
    name reliable — the earlier "strip trailing separators" heuristic missed
    the collapse rule, and a field run showed ``tmpsj40fum_.wav`` landing as
    ``tmpsj40fum_(Instrumental)_...``.
    """
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_. ")


def _expected_output(
    out_dir: Path, input_path: Path, model_filename: str, stem: str = _STEM
) -> Path:
    model_stem = _sanitize_base(Path(model_filename).stem)
    return out_dir / f"{_sanitize_base(input_path.stem)}_({stem})_{model_stem}.wav"


def _find_output(
    out_dir: Path, input_path: Path, model_filename: str, stem: str = _STEM
) -> Path | None:
    """The separator's ``stem`` output for ``input_path``, or None.

    Tries the exact predicted name first, then falls back to scanning the
    output dir for a WAV whose part before "(<stem>)" normalizes to the same
    base as the input — exact normalized equality, so one input's stem being
    a prefix of another's can't cross-match. Multiple survivors (e.g. stale
    files from a previous run) resolve to the newest by mtime.
    """
    exact = _expected_output(out_dir, input_path, model_filename, stem)
    if exact.exists():
        return exact
    marker = f"({stem})"
    want = _sanitize_base(input_path.stem).lower()
    candidates = [
        f
        for f in out_dir.iterdir()
        if f.suffix.lower() == ".wav"
        and marker in f.name
        and _sanitize_base(f.name.split(marker, 1)[0]).lower() == want
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _output_is_usable(path: Path) -> bool:
    """Whether a separator output is a real, readable, non-empty audio file.

    A file EXISTING says nothing about where it came from or whether it is
    complete. This is the last check before an output is accepted as a clip's
    audio, and it is deliberately cheap and total: readable, non-trivial in
    size, and probeable with a positive duration.
    """
    try:
        if not path.is_file() or path.stat().st_size < 1024:
            return False
        info = probe(path)
    except (RuntimeError, OSError, ValueError):
        return False
    return info.duration > 0.0


def _failure_reasons(stderr: str) -> dict[str, str]:
    """Per-file failure reasons recovered from the separator's stderr.

    ``{input basename: reason}``. audio-separator catches every per-file
    exception, logs "Failed to process file <path>: <reason>" and still exits
    0, so this text is the ONLY explanation of a missing output — without it a
    caller can just say "no output was found", which is what a 71-clip field
    run reported instead of "this 1.75 s clip is too short for the model".
    """
    reasons: dict[str, str] = {}
    for match in _FAILED_FILE_RE.finditer(stderr):
        reasons[Path(match.group("path").strip()).name] = match.group("reason").strip()
    return reasons


def _pad_to_minimum(src: Path, work_dir: Path) -> Path | None:
    """A silence-padded copy of ``src`` if it is too short for the separator,
    else None (feed the original). Keeps the same filename so the separator's
    output name stays predictable. See ``MIN_INPUT_SECONDS``."""
    try:
        info = probe(src)
    except (RuntimeError, OSError, ValueError) as e:
        logger.warning("Could not probe %s (%s) — separating it unpadded", src.name, e)
        return None
    if info.duration >= MIN_INPUT_SECONDS:
        return None
    dest = work_dir / src.name
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-af",
        "apad",
        "-t",
        f"{MIN_INPUT_SECONDS:.3f}",
        "-acodec",
        pcm_codec_for(info),
        str(dest),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.warning("Padding %s timed out — separating it unpadded", src.name)
        return None
    if result.returncode != 0:
        logger.warning(
            "Could not pad short input %s (%.2fs) for the separator: %s",
            src.name,
            info.duration,
            result.stderr[-300:],
        )
        return None
    logger.info(
        "Padded %s (%.2fs) to %.1fs — the separator's short-audio path is broken below ~10s",
        src.name,
        info.duration,
        MIN_INPUT_SECONDS,
    )
    return dest


def _restore_original_length(produced: Path, original: Path) -> Path | None:
    """Trim a padded input's output back to the original's exact duration
    (and sample rate/channels/bit depth), in place. ``None`` if it could not
    be done.

    ``None`` is not a formality. A clip padded from 1.75 s to 12 s and then
    NOT trimmed back is a 12-second file standing in for a 1.75-second clip:
    ten seconds of invented material on the timeline, every later clip pushed
    out of sync. The previous code returned the padded file on failure and the
    caller recorded it as a successful output — the one case where "keep what
    we got" is worse than having nothing. A clip with no ambience lane is a
    missing lane; a clip with the WRONG length is a broken edit.
    """
    try:
        info = probe(original)
    except (RuntimeError, OSError, ValueError) as e:
        logger.warning(
            "Could not probe %s (%s) — discarding %s, whose length cannot be verified",
            original.name,
            e,
            produced.name,
        )
        return None
    trimmed = produced.with_name(f"{produced.stem}_trimmed.wav")
    try:
        conform_wav_to(
            produced,
            trimmed,
            info.duration,
            info.audio_sample_rate or 48000,
            info.audio_channels or 2,
            pcm_codec_for(info),
        )
    except (RuntimeError, OSError) as e:
        logger.warning(
            "Could not trim %s back to %.3fs (%s) — discarding it rather than "
            "publishing a padded file as this clip's audio",
            produced.name,
            info.duration,
            e,
        )
        with contextlib.suppress(OSError):
            trimmed.unlink()
        return None
    trimmed.replace(produced)
    return produced


def extract_ambience(
    camera_audio: Path,
    out_dir: Path,
    repo_root: Path,
    model_filename: str,
    model_dir: Path | None = None,
    timeout: int = 1800,
) -> Path:
    """Run the separator on a single ``camera_audio`` file and return the
    ambience-only WAV. For more than one clip, prefer ``extract_ambience_batch``
    — it loads the (1.5+ GB) model once instead of once per call.
    """
    results = extract_ambience_batch(
        [camera_audio], out_dir, repo_root, model_filename, model_dir, timeout
    )
    return results[camera_audio]


def extract_ambience_batch(
    camera_audios: list[Path],
    out_dir: Path,
    repo_root: Path,
    model_filename: str,
    model_dir: Path | None = None,
    timeout: int = 3600,
) -> dict[Path, Path]:
    """Run the separator once over every file in ``camera_audios`` and return a
    ``{input_path: ambience_wav_path}`` map.

    ``audio-separator``'s CLI accepts multiple positional inputs and loads the
    (1.5+ GB) model a single time for the whole batch — calling it once per
    clip (the previous behaviour) reloaded the model from scratch for every
    camera clip in a multi-clip shoot, each load costing tens of seconds. See
    PROJECT_ANALYSIS.md §6.3.

    Clips the separator could not process are simply absent from the returned
    map (with a warning naming each one and why); the caller keeps the
    ambience it did get. See ``run_separator_batch``.
    """
    return run_separator_batch(
        camera_audios,
        out_dir,
        repo_root,
        model_filename,
        _STEM,
        model_dir=model_dir,
        timeout=timeout,
        error_label="Ambience separation",
        unavailable_message=(
            "Ambience separation needs the '.sep-venv' environment "
            "(audio-separator). It is not set up — run setup_sep_venv.sh."
        ),
    )


def run_separator_batch(
    inputs: list[Path],
    out_dir: Path,
    repo_root: Path,
    model_filename: str,
    stem: str,
    model_dir: Path | None = None,
    timeout: int = 3600,
    error_label: str = "Separation",
    unavailable_message: str | None = None,
) -> dict[Path, Path]:
    """Run ``audio-separator`` over ``inputs`` for a given model/stem and return
    a ``{input_path: output_wav_path}`` map. Shared by ambience extraction
    (``stem="Instrumental"``) and voice-enhancement denoise/dereverb
    (``stem="dry"``/``"noreverb"``) — same CLI, same batching rationale, only
    the model and the kept stem differ. See ``extract_ambience_batch`` for why
    the whole batch goes through one process (model load amortized once).

    Partial failures are TOLERATED: the returned map only contains inputs the
    separator actually produced output for, and each missing one is logged
    with the reason recovered from stderr. This used to raise, which meant one
    bad clip out of 71 threw away the ambience for all of them (~30 min of GPU
    work) and left the successful outputs on disk under separator-generated
    names. Only a batch that produced NOTHING raises — that is a broken
    environment/model rather than a bad input file.
    """
    if not inputs:
        return {}
    # The separator names outputs after the input's base name only, so two
    # inputs whose names sanitize alike would overwrite each other's output and
    # both resolve to the same file — one clip silently getting another's
    # audio. Callers must hand over distinct names; say so instead.
    bases = [_sanitize_base(p.stem).lower() for p in inputs]
    if len(set(bases)) != len(bases):
        dupes = sorted({b for b in bases if bases.count(b) > 1})
        raise RuntimeError(
            f"{error_label}: inputs must have distinct filenames — the separator "
            f"names its output after the input, so these would collide: {', '.join(dupes)}"
        )
    cli = separator_cli(repo_root)
    if cli is None:
        raise RuntimeError(
            unavailable_message
            or (
                "Separation needs the '.sep-venv' environment (audio-separator). "
                "It is not set up — run setup_sep_venv.sh."
            )
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    def _invoke(feed: list[Path]) -> str:
        cmd = [
            str(cli),
            *(str(p) for p in feed),
            "--model_filename",
            model_filename,
            "--output_dir",
            str(batch_out),
            "--output_format",
            "WAV",
            "--single_stem",
            stem,
            "--log_level",
            "warning",
        ]
        if model_dir is not None:
            cmd += ["--model_file_dir", str(model_dir)]
        logger.info(
            "%s (%s, stem=%s) on %d file(s) in one batch",
            error_label,
            model_filename,
            stem,
            len(feed),
        )
        try:
            # Streamed to disk, not buffered in memory: this process runs for
            # tens of minutes and prints a progress line per chunk, while the
            # machine is already carrying a 1.5+ GB model.
            result = run_logged(cmd, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            # `TimeoutExpired` inherits from SubprocessError, NOT from
            # RuntimeError/OSError — the pair every caller catches to degrade
            # gracefully. A separator that ran long therefore escaped as an
            # unhandled exception and destroyed the whole run, when the correct
            # outcome for an OPTIONAL stage is a warning and the original
            # audio. Normalising it here means every caller keeps working.
            raise RuntimeError(
                f"{error_label} timed out after {timeout}s on {len(feed)} file(s)"
            ) from e
        if result.returncode != 0:
            raise RuntimeError(
                f"{error_label} failed (exit {result.returncode}): "
                f"{result.stderr[-600:] or result.stdout[-600:]}"
            )
        return result.stderr or ""

    # A PRIVATE output directory for this batch, moved into `out_dir` only
    # once each file is verified.
    #
    # The separator writes names derived from its input, so a partially
    # successful earlier run leaves files in `out_dir` with exactly the names
    # this run would produce. `_find_output` then found the OLD file and
    # reported it as this run's result — confirmed with a mock subprocess where
    # even a "Failed to process file ...: OOM" line still yielded a stale WAV,
    # accepted as the clip's ambience. A file existing has never been evidence
    # of what produced it; a directory nothing else can write to is.
    work_dir = Path(tempfile.mkdtemp(prefix="ws_sep_"))
    batch_out = Path(tempfile.mkdtemp(prefix="ws_sep_out_", dir=out_dir))
    try:
        # Short inputs go in padded (and come back trimmed) — see MIN_INPUT_SECONDS.
        fed: dict[Path, Path] = {}
        padded: set[Path] = set()
        for src in inputs:
            padded_copy = _pad_to_minimum(src, work_dir)
            fed[src] = padded_copy or src
            if padded_copy is not None:
                padded.add(src)

        outputs: dict[Path, Path] = {}
        reasons: dict[str, str] = {}

        def _collect(pending: list[Path]) -> list[Path]:
            still_missing: list[Path] = []
            for src in pending:
                produced = _find_output(batch_out, fed[src], model_filename, stem)
                if produced is None:
                    still_missing.append(src)
                    continue
                if src in padded:
                    restored = _restore_original_length(produced, src)
                    if restored is None:
                        still_missing.append(src)
                        continue
                    produced = restored
                if not _output_is_usable(produced):
                    logger.warning(
                        "%s: output for %s is unreadable or empty — treating it as missing",
                        error_label,
                        src.name,
                    )
                    still_missing.append(src)
                    continue
                outputs[src] = produced
            return still_missing

        reasons.update(_failure_reasons(_invoke([fed[src] for src in inputs])))
        missing = _collect(list(inputs))

        if missing and len(inputs) > 1 and len(missing) <= _MAX_RETRY_FILES:
            logger.warning(
                "%s: %d of %d file(s) produced no output — retrying them individually",
                error_label,
                len(missing),
                len(inputs),
            )
            for src in list(missing):
                with contextlib.suppress(RuntimeError, OSError, subprocess.SubprocessError):
                    reasons.update(_failure_reasons(_invoke([fed[src]])))
            missing = _collect(missing)

        for src in missing:
            logger.warning(
                "%s: no '%s' output for %s — %s",
                error_label,
                stem,
                src.name,
                reasons.get(fed[src].name, "the separator gave no reason"),
            )
        if not outputs:
            detail = "; ".join(f"{name}: {why}" for name, why in list(reasons.items())[:3])
            raise RuntimeError(
                f"{error_label}: the separator reported success but produced no "
                f"'{stem}' output for any of the {len(inputs)} input(s) in {out_dir}"
                + (f" — {detail}" if detail else ".")
            )

        # Publish: move each verified file out of the private batch directory
        # into `out_dir` under a name derived from the INPUT, so what the caller
        # gets back is unambiguously this run's output for that input.
        published_outputs: dict[Path, Path] = {}
        for src, produced in outputs.items():
            final = out_dir / f"{_sanitize_base(src.stem)}_({stem}).wav"
            try:
                produced.replace(final)
            except OSError:
                shutil.copy2(produced, final)
            published_outputs[src] = final
        return published_outputs
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(batch_out, ignore_errors=True)
