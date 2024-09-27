import asyncio
import logging
import os
import time
from asyncio import Queue
from collections import defaultdict
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

_queue: Queue = Queue()
_listeners: Dict[str, List[Dict[str, Callable | int]]] = defaultdict(list)

async def emit(event_name: str, *args, **kwargs) -> None:
    """Emits an event with arguments."""
    await _queue.put((event_name, args, kwargs))

def on(event: str, /, *, priority: int = 1) -> Callable:
    """Decorator to subscribe a method to an event."""
    def wrapper(callback: Callable) -> Callable:
        subscribe(event, callback, priority=priority)
        return callback
    return wrapper

def subscribe(event: str, callback: Callable, /, *, priority: int = 1) -> None:
    """Subscribes a method to an event."""
    _listeners[event].append({"callback": callback, "priority": priority})
    _listeners[event].sort(key=lambda subscriber: subscriber["priority"], reverse=True)

def unsubscribe(event: str, callback: Callable, /) -> None:
    """Unsubscribes a method from an event."""
    if event in _listeners:
        _listeners[event] = [
            subscriber for subscriber in _listeners[event]
            if subscriber["callback"] != callback
        ]

async def _safe_execute(event_name: str, subscriber: Dict[str, Callable | int], *args, **kwargs) -> None:
    callback = subscriber['callback']
    priority = subscriber['priority']
    try:
        start = time.time()
        await callback(*args, **kwargs)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"callback '{callback}' for event '{event_name}': {str(e)}")

async def process_events():
    while True:
        event_name, args, kwargs = await _queue.get()
        for subscriber in _listeners[event_name]:
            asyncio.create_task(_safe_execute(event_name, subscriber, *args, **kwargs))
        _queue.task_done()