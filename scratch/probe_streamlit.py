import urllib.request
import json
import sys

try:
    health_url = "http://localhost:8505/_stcore/health"
    req = urllib.request.Request(health_url)
    with urllib.request.urlopen(req, timeout=5) as response:
        status = response.getcode()
        body = response.read().decode('utf-8')
        print(f"Health check status: {status}, body: {body}")
except Exception as e:
    print(f"Error connecting to Streamlit server: {e}")
