from importlib.resources import files

from studio.app import main, packaged_resources


def test_packaged_resources_exist() -> None:
    resources = packaged_resources()
    assert all(resources)


def test_resource_directories_are_importable() -> None:
    root = files("studio.resources")
    assert root.joinpath("prompts", "en", "article_default.txt").is_file()
    assert root.joinpath("fonts", "licenses", "OFL-montserrat.txt").is_file()
    assert root.joinpath("subtitle_editor", "assets", "editor.js").is_file()
    assert root.joinpath("presets", "roughcut-balanced.yaml").is_file()


def test_resource_smoke_command(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["--check-resources"]) == 0
    assert "resources: OK" in capsys.readouterr().out
