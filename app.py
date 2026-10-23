"""
Hybrid Edge-IIoTset + CICIoT2023 Intrusion Detection Framework.
Central Streamlit Application Entry Point.
"""

import sys
from pathlib import Path

# Add project root and app directory to path
root = Path(__file__).resolve().parent
app_dir = root / "app"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

try:
    from app.dashboard import main
except ImportError:
    from dashboard import main

if __name__ == "__main__":
    main()

