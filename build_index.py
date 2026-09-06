import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_vector_store import build_vector_store

def build_vector_index():
    """Alias to build_vector_store for backward compatibility."""
    return build_vector_store()

if __name__ == "__main__":
    build_vector_index()
