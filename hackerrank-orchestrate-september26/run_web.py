"""Launch the Buy or Wait? web dashboard."""
import sys
import os
import threading
import webbrowser

# Put code/ on the import path so "from finance.xxx import yyy" works
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "code"))
os.chdir(ROOT)

import uvicorn


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9000"))

    # Open the dashboard automatically after the server starts
    url = f"http://localhost:{port}"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    print(f"\n🚀 Buy or Wait? dashboard: {url}\n")

    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )