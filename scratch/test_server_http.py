import urllib.request
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

url = "http://localhost:8501"
resp = urllib.request.urlopen(url)
print("Root HTTP Status:", resp.status)
html = resp.read().decode("utf-8")

# Extract all referenced static files
matches = re.findall(r'(?:src|href)="([^"]+)"', html)
print(f"Found {len(matches)} referenced assets in index.html:")

for m in matches:
    if m.startswith("./") or m.startswith("/") or m.startswith("static/"):
        clean_m = m.lstrip("./").lstrip("/")
        full_url = f"{url}/{clean_m}"
        try:
            r = urllib.request.urlopen(full_url)
            print(f"  ✓ {full_url} -> Status {r.status} ({len(r.read())} bytes)")
        except Exception as e:
            print(f"  ✗ {full_url} -> {e}")

# Check health endpoint
health_url = f"{url}/_stcore/health"
try:
    h = urllib.request.urlopen(health_url)
    print("Health Endpoint:", h.status, h.read().decode("utf-8"))
except Exception as e:
    print("Health Endpoint error:", e)
