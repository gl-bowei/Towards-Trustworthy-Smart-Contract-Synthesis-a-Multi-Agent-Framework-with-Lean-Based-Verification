from typing import List

def append_unique(items: List[str], value: str):
    if value and value not in items:
        items.append(value)
