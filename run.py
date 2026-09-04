"""
Entry point for the Universal Job Scraper.
Run this file instead of invoking uvicorn directly:

    python run.py

This ensures the WindowsProactorEventLoopPolicy is set BEFORE uvicorn
creates its event loop — required for Patchright/Playwright to launch
Chrome subprocesses on Windows.
"""
import sys
import asyncio
import os

# The MiniLM chunker model is already cached locally — skip huggingface.co
# version checks so startup doesn't burn ~23s in DNS retries when offline.
os.environ.setdefault("HF_HUB_OFFLINE", "0")

# MUST be set before uvicorn creates any event loop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
