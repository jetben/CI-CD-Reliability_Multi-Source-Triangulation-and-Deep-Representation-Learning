import json

RUNS_FILE = "data/runs.json"

# Read first 3 lines and print structure
with open(RUNS_FILE, "r", encoding="utf8") as f:
    for i in range(3):
        item = json.loads(next(f))
        print("\n===== RECORD", i+1, "=====")
        for k,v in item.items():
            print(k, ":", type(v), "→", v)
