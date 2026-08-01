from typing import Dict, List

def build_dataset_targets(dataset_item: Dict) -> List[str]:
    targets = []
    for key in ("prompt", "original_requirement", "strict_interface", "strict_constructor"):
        value = dataset_item.get(key)
        if value:
            targets.append(f"{key}: {value}")
    return targets
