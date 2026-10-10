# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the Studio CLI."""

from pathlib import Path

root = Path(SPEC).resolve().parents[2]
resource_root = root / "studio" / "resources"
managed_sources = [(str(path), str(Path("managed") / path.relative_to(root).parent))
                   for folder in ("core", "modules", "vision")
                   for path in (root / "studio" / folder).glob("*.py")]
managed_sources.append((str(root / "studio" / "__init__.py"), "managed/studio"))

a = Analysis(
    [str(root / "studio" / "frozen_entry.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[
        (str(root / "studio" / "gui" / "theme.qss"), "studio/gui"),
        (str(resource_root / "prompts"), "studio/resources/prompts"),
        (str(resource_root / "presets"), "studio/resources/presets"),
        (str(resource_root / "fonts"), "studio/resources/fonts"),
        (str(resource_root / "subtitle_editor"), "studio/resources/subtitle_editor"),
    ] + managed_sources,
    hiddenimports=[
        "studio.resources",
        "studio.stages.worker",
        "studio.gui.main_window",
        "yaml",
        "platformdirs",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "PyQt6.QtMultimedia",
        "PyQt6.QtMultimediaWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ultimate-video-forge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
