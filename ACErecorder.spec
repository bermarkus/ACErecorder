# -*- mode: python ; coding: utf-8 -*-
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)
from PyInstaller.utils.hooks import collect_all
import os

# Get absolute paths
base_dir = os.path.abspath(os.path.dirname('ACErecorder.py'))
icon_file = os.path.join(base_dir, 'LogoSquare3.ico')
logo_file = os.path.join(base_dir, 'Head v28 (2).png')

if not os.path.exists(icon_file):
    raise FileNotFoundError(f"Icon file not found: {icon_file}")
if not os.path.exists(logo_file):
    raise FileNotFoundError(f"Logo file not found: {logo_file}")

# Collect data files and dependencies
datas = [(logo_file, '.'), (icon_file, '.')]
binaries = []
hiddenimports = ['serial', 'serial.tools.list_ports', 'brainflow', 'brainflow.board_shim', 'mne', 'mne.io', 'mne_connectivity']

# Collect all required files for brainflow and mne
tmp_ret = collect_all('brainflow')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mne')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    ['ACErecorder.py'],
    pathex=[base_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False
)

# Remove any duplicate data files
a.datas = list(dict.fromkeys(a.datas))

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ACErecorder',
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
    icon=[icon_file],
    version='file_version_info.txt',
    uac_admin=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ACErecorder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['LogoSquare3.ico']
)
