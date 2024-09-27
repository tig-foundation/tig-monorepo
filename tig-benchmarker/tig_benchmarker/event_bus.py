import asyncio
import logging
import os
import time
from asyncio import Queue
from collections import defaultdict
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

_queue = []

async def emit(event_name: str, **kwargs) -> None:
    global _queue
    _queue.append((event_name, kwargs))

async def _safe_execute(extension_name: str, event_name: str, handler: Callable, kwargs: dict) -> None:
    try:
        logger.debug(f"calling {extension_name}.on_{event_name}")
        await handler(**kwargs)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"'{extension_name}.on_{event_name}': {str(e)}")

async def process_events(extensions: dict):
    global _queue
    num_events = len(_queue)
    event_names = set(
        _queue[i][0]
        for i in range(num_events)
    )
    handlers = {
        e: [
            (ext_name, getattr(ext, f'on_{e}'))
            for ext_name, ext in extensions.items()
            if hasattr(ext, f'on_{e}')
        ]
        for e in event_names
    }
    await asyncio.gather(
        *[
            _safe_execute(ext_name, _queue[i][0], h, _queue[i][1])
            for i in range(num_events)
            for (ext_name, h) in handlers[_queue[i][0]]
        ]
    )
    _queue = _queue[num_events:]