import json
from collections import Counter

with open("catalog.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("TOTAL ITEMS:", len(data))
print("DATA TYPE:", type(data))

print("\nFIRST ITEM:")
print(data[0])

print("\nKEYS FOUND:")
all_keys = Counter()
for item in data:
    all_keys.update(item.keys())
print(all_keys)

print("\nEMPTY FIELD CHECK:")
for key in all_keys:
    empty_count = sum(1 for item in data if not item.get(key))
    print(key, "empty:", empty_count)

print("\nDUPLICATE NAME CHECK:")
name_key = None
for possible in ["name", "assessment_name", "title", "productName", "product_name"]:
    if possible in data[0]:
        name_key = possible
        break

if name_key:
    names = [item.get(name_key) for item in data if item.get(name_key)]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    print("Name key:", name_key)
    print("Duplicate names:", len(duplicates))
    print(duplicates[:10])
else:
    print("Could not auto-detect name key.")

print("\nURL CHECK:")
url_key = None
for possible in ["url", "product_url", "link", "productUrl"]:
    if possible in data[0]:
        url_key = possible
        break

if url_key:
    urls = [item.get(url_key) for item in data if item.get(url_key)]
    print("URL key:", url_key)
    print("URLs present:", len(urls))
    print("First 5 URLs:", urls[:5])
else:
    print("Could not auto-detect URL key.")

print("\nIMPORTANT TERM SEARCH:")
terms = ["OPQ", "GSA", "Java", "Python", "Verify", "Coding", "Personality", "Cognitive", "Situational"]
for term in terms:
    matches = []
    for item in data:
        text = json.dumps(item, ensure_ascii=False).lower()
        if term.lower() in text:
            matches.append(item)
    print(term, "matches:", len(matches))
    if matches:
        print(" sample:", matches[0])