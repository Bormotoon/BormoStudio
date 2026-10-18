# Source snapshots

Studio development starts from immutable, tested snapshots of its two source
projects. Feature development in those repositories is frozen while their code
is migrated; only bug fixes may be added and must be recorded here before they
are imported.

## Accepted decisions

Historical capture-time decisions below are preserved for provenance, not current
policy. They are superseded by `docs/PROJECT_DECISIONS.md`: MIT, BormoStudio (then
Ultimate Video Forge), original repositories left untouched, native Qt subtitle editor and Linux
first acceptance. Current-session execution evidence is Python 3.10.20; Python
3.12 capture-time evidence below does not qualify today's installed release.

- Repository: this standalone Studio repository, with both sources imported by
  `git subtree` under `legacy/`.
- License: PolyForm Noncommercial 1.0.0 for Studio until a publication review
  explicitly selects another license. The imported Forge subtree retains its
  MIT license and notices.
- Platforms: Python 3.12 is the primary environment. CI smoke coverage is
  Python 3.10 through 3.13. Linux with NVIDIA is the complete target; macOS and
  Windows initially receive base import/synchronization smoke coverage.
- Source repositories: feature-frozen until migration and acceptance are
  complete. WhisperSync may be archived after phase 2 and Forge after phase 5.
- Name: `Studio` / `studio` remains the temporary product and package name.
- Subtitle editor: open in the system browser first; QWebEngine embedding is a
  later enhancement.
- Minimum NLE targets: Final Cut FCPXML 1.9 and Premiere XMEML v4. Exact
  application versions must be recorded with the manual acceptance results.

## Snapshot manifest

| Source | Imported commit | Branch at capture | Tests | Working tree at capture |
|---|---|---|---:|---|
| WhisperSync | `c3b4b048b3610ad795cac20a3bda45c8fe14f67e` | `audit-remediation-2026-09-08` | 403 | Clean after committing the export/timeline bug fixes |
| Podcast Reels Forge | `846fdc451474e95696d1447edf6bd8cd06a574ea` | `main` | 620 | `docs/UNIFIED_APP_PLAN.md` modified; the imported source snapshot is the committed HEAD |

The audited plan update was subsequently committed in the Forge repository as
`0451f67cc30f27580f54451406124ae346d18ff7`, after the subtree snapshot. The
plan is process documentation, not part of the Forge runtime baseline.

The local validation environment reported Python 3.12.13 for the supported
runtime and Python 3.14.5 as the host default. Python 3.14 is deliberately not
claimed by Studio until Qt, torch, and pyannote are verified together.

## WhisperSync patch provenance

At the beginning of the audit, WhisperSync had uncommitted changes in:

- `CHANGELOG.md`
- `tests/test_export.py`
- `tests/test_pipeline.py`
- `whispersync/engine/export.py`
- `whispersync/engine/media.py`
- `whispersync/engine/pipeline.py`

They were validated with all 403 tests and committed in the source repository
as `c3b4b04` before the subtree import. The exact commit is also preserved as
`0001-fix-export-preserve-measured-timeline-placement.patch`; its SHA-256 is
recorded in `SHA256SUMS`.

## Reproduction

Run each baseline independently from the repository root:

```bash
python3.12 -m pytest -q legacy/whispersync/tests
python3.12 -m pytest -q legacy/forge/tests
```

CI installs each source's dependencies separately so dependency resolution for
one legacy application cannot silently change the other baseline.
