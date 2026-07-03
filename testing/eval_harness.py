"""
eval_harness.py
 
Replays labeled trace conversations against your running /chat
endpoint and computes Recall@10 per trace and overall.
 
USAGE
    1. Start your server:  uvicorn app:app --reload
    2. Put your trace files in ./traces/ (one JSON file per trace).
    3. Run:  python eval_harness.py
 
TRACE FILE FORMAT
Each trace file is a JSON object:
{
  "name": "backend_python_developer",
  "turns": [
    "Need Python assessment for Backend Developer with 3 years experience"
  ],
  "expected": [
    "Python (New)",
    "SQL (New)",
    "RESTful Web Services (New)"
  ]
}
 
"turns" is a list of user messages sent one at a time, in order,
building up the conversation history exactly like the real harness
described in the assignment (stateless API, full history resent
each call). The LAST turn's response is what gets scored against
"expected". If your trace has multiple turns (clarify -> answer ->
recommend), list them in order; the harness sends turn 1 alone,
then turns 1+2, then turns 1+2+3, etc.
 
"expected" is the list of assessment NAMES (exact catalog names,
as they appear in the "name" field of your catalog) that should
appear in the final recommendations for full credit.
 
If you don't have the official 10 traces in this exact format yet,
convert them once -- it pays for itself after the first re-run.
"""
 
import json
import os
import sys
import glob
import requests
 
BASE_URL = os.environ.get("SHL_API_BASE_URL", "http://localhost:8000")
TRACES_DIR = os.environ.get("SHL_TRACES_DIR", "./traces")
TIMEOUT_SECONDS = 30
 
 
def load_traces(traces_dir):
    trace_files = sorted(glob.glob(os.path.join(traces_dir, "*.json")))
    if not trace_files:
        print(f"No trace files found in {traces_dir}/. "
              f"Drop your trace JSON files there (see TRACE FILE FORMAT in this script's docstring).")
        sys.exit(1)
    traces = []
    for path in trace_files:
        with open(path, "r", encoding="utf-8") as f:
            trace = json.load(f)
            trace["_source_file"] = path
            traces.append(trace)
    return traces
 
 
def call_chat(messages):
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"messages": messages},
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()
 
 
def check_health():
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        resp.raise_for_status()
        print(f"Health check OK: {resp.json()}")
    except Exception as e:
        print(f"WARNING: health check failed ({e}). Is the server running at {BASE_URL}?")
        sys.exit(1)
 
 
def recall_at_k(returned_names, expected_names, k=10):
    if not expected_names:
        return None  # undefined, skip in averaging
    top_k = set(returned_names[:k])
    hits = sum(1 for name in expected_names if name in top_k)
    return hits / len(expected_names)
 
 
def run_trace(trace):
    messages = []
    last_response = None
 
    for turn_text in trace["turns"]:
        messages.append({"role": "user", "content": turn_text})
        last_response = call_chat(messages)
        # Append assistant reply to history so subsequent turns build
        # on a realistic stateless-API conversation, matching how the
        # real evaluator replays conversations.
        messages.append({"role": "assistant", "content": last_response.get("reply", "")})
 
    returned = last_response.get("recommendations", []) if last_response else []
    returned_names = [r["name"] for r in returned]
    expected_names = trace.get("expected", [])
 
    recall = recall_at_k(returned_names, expected_names, k=10)
 
    missing = [name for name in expected_names if name not in returned_names]
    extra = [name for name in returned_names if name not in expected_names]
 
    return {
        "trace_name": trace.get("name", trace["_source_file"]),
        "recall_at_10": recall,
        "returned": returned_names,
        "expected": expected_names,
        "missing": missing,
        "extra": extra,
        "end_of_conversation": last_response.get("end_of_conversation") if last_response else None,
        "final_reply": last_response.get("reply") if last_response else None,
    }
 
 
def main():
    print(f"Base URL: {BASE_URL}")
    check_health()
 
    traces = load_traces(TRACES_DIR)
    print(f"Loaded {len(traces)} trace(s) from {TRACES_DIR}/\n")
 
    results = []
    for trace in traces:
        try:
            result = run_trace(trace)
        except Exception as e:
            result = {
                "trace_name": trace.get("name", trace["_source_file"]),
                "recall_at_10": None,
                "error": str(e),
            }
        results.append(result)
 
    print("=" * 78)
    print("PER-TRACE RESULTS")
    print("=" * 78)
 
    valid_recalls = []
    for r in results:
        print(f"\n--- {r['trace_name']} ---")
        if r.get("error"):
            print(f"  ERROR: {r['error']}")
            continue
 
        recall = r["recall_at_10"]
        if recall is not None:
            valid_recalls.append(recall)
            print(f"  Recall@10: {recall:.2f}")
        else:
            print("  Recall@10: N/A (no 'expected' list provided)")
 
        print(f"  end_of_conversation: {r['end_of_conversation']}")
        print(f"  Returned ({len(r['returned'])}): {r['returned']}")
        if r["missing"]:
            print(f"  MISSING (expected but not returned): {r['missing']}")
        if r["extra"]:
            print(f"  EXTRA (returned but not expected): {r['extra']}")
 
    print("\n" + "=" * 78)
    if valid_recalls:
        mean_recall = sum(valid_recalls) / len(valid_recalls)
        print(f"MEAN RECALL@10 across {len(valid_recalls)} scored traces: {mean_recall:.3f}")
    else:
        print("No traces had an 'expected' list -- nothing to average.")
    print("=" * 78)
 
 
if __name__ == "__main__":
    main()