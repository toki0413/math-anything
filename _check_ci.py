import urllib.request, os, json

api_key = os.environ.get("MATON_API_KEY", "")
if not api_key:
    print("MATON_API_KEY not set")
    exit(1)

req = urllib.request.Request(
    "https://gateway.maton.ai/github/repos/toki0413/math-anything/actions/runs?per_page=5"
)
req.add_header("Authorization", f"Bearer {api_key}")

try:
    data = json.load(urllib.request.urlopen(req))
    runs = data.get("workflow_runs", [])
    if not runs:
        print("No workflow runs found")
    for run in runs:
        rid = run["id"]
        name = run["name"]
        status = run["status"]
        conclusion = run.get("conclusion", "N/A")
        event = run["event"]
        branch = run["head_branch"]
        print(f"{rid} | {name} | {status} | {conclusion} | {event} | {branch}")
except Exception as e:
    print(f"Error: {e}")
