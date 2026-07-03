import requests
from openpyxl import Workbook
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------

BACKEND_URL = "http://127.0.0.1:8000/chat"

QUERY_FILE = "manual_queries.txt"

OUTPUT_FILE = "results.xlsx"

# -----------------------------
# Read Queries
# -----------------------------

queries = []

with open(QUERY_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        queries.append(line)

print(f"\nLoaded {len(queries)} queries.\n")

# -----------------------------
# Create Excel
# -----------------------------

wb = Workbook()

ws = wb.active

ws.title = "Benchmark Results"

headers = [
    "Query",
    "Reply",

    "Top1",
    "Top2",
    "Top3",
    "Top4",
    "Top5",
    "Top6",
    "Top7",
    "Top8",
    "Top9",
    "Top10",

    "Total Results",
    "End Conversation",
    "Status"
]

for col, header in enumerate(headers, start=1):

    ws.cell(row=1, column=col).value = header

# -----------------------------
# Execute Queries
# -----------------------------

for idx, query in enumerate(queries, start=2):

    print(f"[{idx-1}/{len(queries)}] {query}")

    payload = {

        "messages": [

            {
                "role": "user",
                "content": query
            }

        ]

    }

    try:

        response = requests.post(
            BACKEND_URL,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        reply = data.get("reply", "")

        recommendations = data.get("recommendations", [])

        names = [r["name"] for r in recommendations]

        row = [
            query,
            reply,
        ]

        # Top 10
        for i in range(10):
            if i < len(names):
                row.append(names[i])
            else:
                row.append("")

        row.append(len(names))
        row.append(data.get("end_of_conversation"))
        row.append("PASS")

        ws.append(row)
        
    except Exception as e:

        ws.append([

            query,

            "",

            "",

            "",

            "",

            "",

            "",

            "",

            f"ERROR: {e}"

        ])

        print(e)

# -----------------------------
# Save
# -----------------------------

wb.save(OUTPUT_FILE)

print("\nFinished.")

print(f"Results saved to {OUTPUT_FILE}")