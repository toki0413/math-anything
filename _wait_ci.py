import urllib.request
import json
import time

time.sleep(120)

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

        pass_count = 0
        fail_count = 0
        for j in jobs_data.get("jobs", []):
            s = "PASS" if j["conclusion"] == "success" else "FAIL"
            if j["conclusion"] == "success":
                pass_count += 1
            else:
                fail_count += 1
            print(f"  [{s}] {j['name']}")
            if j["conclusion"] not in ("success", "skipped"):
                for step in j.get("steps", []):
                    if step["conclusion"] == "failure":
                        print(f"    FAILED: {step['name']}")

        print(f"\nSummary: {pass_count} passed, {fail_count} failed")
    elif status != "completed":
        print(f"  CI still running... (status: {status})")
