"""Download and extract MinGW-w64 for Tauri build on Windows.

Downloads the winlibs MinGW-w64 toolchain (includes gcc, dlltool, etc.)
and extracts it to a known location.
"""

import urllib.request
import zipfile
import os
import sys

MINGW_URL = "https://github.com/brechtsanders/winlibs_mingw/releases/download/14.2.0posix-18.0.6-11.0.1-ucrt-r1/winlibs-x86_64-posix-seh-gcc-14.2.0-llvm-18.0.6-mingw-w64ucrt-11.0.1-r1.7z"
INSTALL_DIR = os.path.join(os.path.expanduser("~"), ".mingw-w64")

def main():
    os.makedirs(INSTALL_DIR, exist_ok=True)
    archive_path = os.path.join(INSTALL_DIR, "mingw.7z")

    if not os.path.exists(archive_path):
        print(f"Downloading MinGW-w64 to {archive_path}...")
        print(f"URL: {MINGW_URL}")
        try:
            urllib.request.urlretrieve(MINGW_URL, archive_path)
            print("Download complete.")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please download manually from https://winlibs.com/")
            sys.exit(1)
    else:
        print(f"Archive already exists: {archive_path}")

    print(f"\nTo extract, install 7-Zip and run:")
    print(f'  7z x "{archive_path}" -o"{INSTALL_DIR}" -y')
    print(f"\nThen add to PATH:")
    print(f'  $env:PATH = "{INSTALL_DIR}\\mingw64\\bin;" + $env:PATH')

if __name__ == "__main__":
    main()
