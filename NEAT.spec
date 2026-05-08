# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller build spec for NEAT standalone (Windows-friendly defaults).

Build:
    pyinstaller --noconfirm --clean NEAT.spec
"""

from PyInstaller.utils.hooks import collect_all
from pathlib import Path
import sys


block_cipher = None

datas = []
binaries = []
hiddenimports = []
project_root = Path.cwd()

# Pull in scientific/GUI stack resources and dynamic imports used at runtime.
for package_name in ("PyQt5", "matplotlib", "numpy", "scipy", "pandas", "astropy", "PIL"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

# Ensure core CPython extension dependencies are available across Python layouts.
# - python.org layout: <base>/DLLs/*.dll
# - conda-style layout: <base>/Library/bin/*.dll
base_prefix = Path(sys.base_prefix)
required_runtime_dlls = {
    "libexpat.dll": (
        base_prefix / "DLLs" / "libexpat.dll",
        base_prefix / "Library" / "bin" / "libexpat.dll",
    ),
    # _ctypes.pyd on conda-forge depends on ffi-8.dll.
    "ffi-8.dll": (
        base_prefix / "DLLs" / "ffi-8.dll",
        base_prefix / "Library" / "bin" / "ffi-8.dll",
    ),
    # _bz2.pyd on conda-forge depends on libbz2.dll.
    "libbz2.dll": (
        base_prefix / "DLLs" / "libbz2.dll",
        base_prefix / "Library" / "bin" / "libbz2.dll",
    ),
    # _lzma.pyd on conda-forge depends on liblzma.dll.
    "liblzma.dll": (
        base_prefix / "DLLs" / "liblzma.dll",
        base_prefix / "Library" / "bin" / "liblzma.dll",
    ),
    # _ssl.pyd depends on OpenSSL runtime dlls.
    "libcrypto-3-x64.dll": (
        base_prefix / "DLLs" / "libcrypto-3-x64.dll",
        base_prefix / "Library" / "bin" / "libcrypto-3-x64.dll",
    ),
    "libssl-3-x64.dll": (
        base_prefix / "DLLs" / "libssl-3-x64.dll",
        base_prefix / "Library" / "bin" / "libssl-3-x64.dll",
    ),
    # _sqlite3.pyd depends on sqlite3.dll in conda-style layouts.
    "sqlite3.dll": (
        base_prefix / "DLLs" / "sqlite3.dll",
        base_prefix / "Library" / "bin" / "sqlite3.dll",
    ),
}
for _, candidates in required_runtime_dlls.items():
    for candidate in candidates:
        if candidate.exists():
            binaries.append((str(candidate), "."))
            break

# Bundle custom splash asset used by NEAT/app.py.
launch_splash = project_root / "NEAT" / "assets" / "launch_splash.png"
if launch_splash.exists():
    datas.append((str(launch_splash), "NEAT/assets"))

# Use project icon for the standalone executable.
app_icon = project_root / "docs" / "icon" / "NEAT.ico"


a = Analysis(
    ["NEAT/app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NEAT",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(app_icon) if app_icon.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="NEAT",
)
