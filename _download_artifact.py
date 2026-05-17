import urllib.request
import json
import zipfile
import io

artifact_id = 7039772152

url = f"https://api.github.com/repos/toki0413/math-anything/actions/artifacts/{artifact_id}/zip"
req = urllib.request.Request(url)
req.add_header("User-Agent", "Python")
req.add_header("Accept", "application/vnd.github+json")

try:
    response = urllib.request.urlopen(req)
    if response.getcode() in (200, 302):
        data = response.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                print(f"=== {name} ===")
                content = zf.read(name).decode("utf-8", errors="replace")
                print(content)
    else:
        print(f"HTTP {response.getcode()}")
except Exception as e:
    print(f"Error: {e}")
    print("Need authentication to download artifacts")
