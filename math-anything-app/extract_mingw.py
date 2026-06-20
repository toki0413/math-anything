import py7zr, os

archive = os.path.expanduser("~/.mingw-w64/mingw.7z")
dest = os.path.expanduser("~/.mingw-w64")

print(f"Extracting {archive}...")
with py7zr.SevenZipFile(archive, "r") as z:
    z.extractall(dest)
print("Done!")

mingw_bin = os.path.join(dest, "mingw64", "bin")
if os.path.exists(mingw_bin):
    print(f"MinGW bin: {mingw_bin}")
    print(f"dlltool: {os.path.exists(os.path.join(mingw_bin, 'dlltool.exe'))}")
    print(f"gcc: {os.path.exists(os.path.join(mingw_bin, 'gcc.exe'))}")
else:
    for root, dirs, files in os.walk(dest):
        if "dlltool.exe" in files:
            print(f"Found dlltool at: {root}")
            break
