import requests
import json
import re

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"

response = requests.get(CATALOG_URL)
print("Status:", response.status_code)

raw_text = response.text

# Save raw copy for debugging
with open("raw_catalog.txt", "w", encoding="utf-8") as f:
    f.write(raw_text)

# Remove invalid control characters
clean_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw_text)

data = json.loads(clean_text, strict=False)

with open("catalog.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Catalog saved to catalog.json")

if isinstance(data, list):
    print("Total items:", len(data))
elif isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))