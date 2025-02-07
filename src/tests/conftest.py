"""Global test configuration and fixtures."""
import pytest
import asyncio
import sys

if sys.platform.startswith("win"):
    # Use the ProactorEventLoop on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

@pytest.fixture(scope="session")
def event_loop():
    """Create and provide a session-scoped event loop."""
    if sys.platform.startswith("win"):
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    
    yield loop
    
    # Clean up pending tasks
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    
    if pending:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    
    loop.close()