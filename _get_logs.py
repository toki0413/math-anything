import urllib.request
import json
import gzip

job_id = 76375798774

logs_url = f"https://api.github.com/repos/toki0413/math-anything/actions/jobs/{job_id}/logs"
req = urllib.request.Request(logs_url)
req.add_header("User-Agent", "Python")
req.add_header("Accept", "application/vnd.github+json")

try:
    response = urllib.request.urlopen(req)
    content_encoding = response.headers.get("Content-Encoding", "")
    data = response.read()
    
    if content_encoding == "gzip":
        data = gzip.decompress(data)
    
    text = data.decode("utf-8", errors="replace")
    
    lines = text.split("\n")
    
    test_section = False
    error_lines = []
    for i, line in enumerate(lines):
        if "Run tests" in line and "step" in line.lower():
            test_section = True
        if test_section:
            error_lines.append(line)
    
    if error_lines:
        for line in error_lines[-100:]:
            print(line.rstrip())
    else:
        print("No test section found. Last 100 lines:")
        for line in lines[-100:]:
            print(line.rstrip())
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.reason}")
    if e.code == 302:
        redirect_url = e.headers.get("Location")
        print(f"Redirect URL: {redirect_url}")
    body = e.read().decode("utf-8", errors="replace")
    print(f"Response body: {body[:500]}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
