from hashlib import md5
from datetime import datetime

def now() -> int:
    return int(datetime.now().timestamp() * 1000)

def u32_from_str(input_str: str) -> int:
    result = hashlib.md5(input_str.encode('utf-8')).digest()
    return int.from_bytes(result[-4:], byteorder='little', signed=False)