"""Lance Demucs dans un processus séparé (appelé par pipeline/separate.py).

Sous Windows, PyTorch ≥ 2.9 lit et écrit l'audio via torchcodec, qui a besoin des DLL
FFmpeg partagées de l'environnement conda : on les déclare avant d'importer Demucs.
"""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    _dll_ffmpeg = Path(sys.prefix) / "Library" / "bin"
    if _dll_ffmpeg.is_dir():
        os.add_dll_directory(str(_dll_ffmpeg))

from demucs.separate import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
