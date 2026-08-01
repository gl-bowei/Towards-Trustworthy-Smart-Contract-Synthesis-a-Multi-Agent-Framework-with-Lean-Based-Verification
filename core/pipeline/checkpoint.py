import json
import os

class CheckpointManager:
    def __init__(self, filepath="checkpoint.json"):
        self.filepath = filepath
        self.data = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    self.data = json.load(f)
            except:
                pass

    def save_stage(self, key, value):
        self.data[key] = value
        self.save_to_disk()

    def load_stage(self, key):
        return self.data.get(key)

    def has_stage(self, key):
        return key in self.data

    def clear_stage(self, key):
        """Clear a key and all downstream state that depends on it."""
        if key == "solidity_code":
            # A code change invalidates derived definitions, proofs, and build results.
            keys_to_remove = [
                "solidity_code",
                "definitions_code",
                "proofs_done",
                "proof_results",
                "formal_failure_feedback",
                "selected_properties",
            ]
            for k in keys_to_remove:
                if k in self.data:
                    del self.data[k]
            self.save_to_disk()
        elif key == "definitions_code":
            keys_to_remove = ["definitions_code", "proofs_done", "proof_results", "formal_failure_feedback"]
            for k in keys_to_remove:
                if k in self.data:
                    del self.data[k]
            self.save_to_disk()

    def save_to_disk(self):
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2)
