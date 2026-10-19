import os
from pathlib import Path

from studio.core.digest import file_digest
from studio.core.project import Project
from studio.core.workspace import published


def test_atomic_writes_get_the_mode_a_plain_write_would(tmp_path: Path) -> None:
    previous = os.umask(0o002)
    try:
        project = Project(tmp_path, tmp_path / "_studio")
        project.save(project.work_dir / "project.json")
        source = tmp_path / "media.bin"
        source.write_bytes(b"x" * 10)
        file_digest(source, project.work_dir)
        with published(tmp_path / "out.txt") as temporary:
            temporary.write_text("done", encoding="utf-8")
        plain = tmp_path / "plain.txt"
        plain.write_text("plain", encoding="utf-8")
        expected = plain.stat().st_mode & 0o777
        assert expected == 0o664
        for path in [project.work_dir / "project.json", tmp_path / "out.txt",
                     *project.work_dir.rglob("digests.json")]:
            assert path.stat().st_mode & 0o777 == expected, path
    finally:
        os.umask(previous)
