import urllib.request
import json
import time

time.sleep(180)

url = "https://api.github.com/repos/toki0413/math-anything/actions/runs?per_page=1&branch=master"
req = urllib.request.Request(url)
req.add_header("User-Agent", "Python")
req.add_header("Accept", "application/vnd.github+json")
data = json.load(urllib.request.urlopen(req))

for r in data.get("workflow_runs", []):
    run_id = r["id"]
    run_num = r["run_number"]
    sha = r["head_sha"][:7]
    status = r["status"]
    conclusion = r["conclusion"]
    print(f"Run #{run_num} ({sha}): {status} - {conclusion}")

    if run_id and status == "completed":
        jobs_url = f"https://api.github.com/repos/toki0413/math-anything/actions/runs/{run_id}/jobs?per_page=30"
        req2 = urllib.request.Request(jobs_url)
        req2.add_header("User-Agent", "Python")
        req2.add_header("Accept", "application/vnd.github+json")
        jobs_data = json.load(urllib.request.urlopen(req2))

        for j in jobs_data.get("jobs", []):
            if "windows" in j["name"] and j["conclusion"] == "failure":
                print(f"\n  {j['name']}:")
                for step in j.get("steps", []):
                    if step["conclusion"] == "failure":
                        print(f"    FAILED: {step['name']} (#{step['number']})")
                break
