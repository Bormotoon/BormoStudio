# Current-tree CI qualification

- First incomplete phase-0 task: qualify both frozen suites and Python 3.10–3.13 CI.
- Remote CI run 37781511332 succeeded on 2026-10-08, but predates the current
  uncommitted implementation. It is not evidence for the current tree.
- Fixed packaging smoke commands to use the actual `bormostudio` binary,
  and explicitly exercise resource and worker dispatch.
- Status: partial, not accepted. Current implementation must be committed and
  uploaded before GitHub Actions can qualify it. No push was performed.
- Existing uncommitted implementation spans processing, GUI, models and packaging;
  it must be captured deliberately, not silently included in the CI-fix commit.
