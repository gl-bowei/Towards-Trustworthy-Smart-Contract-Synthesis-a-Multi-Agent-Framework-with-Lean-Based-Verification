import os
import datetime

class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            # Initialize a default path so calls made before configuration do not fail.
            cls._instance._configure_default()
        return cls._instance

    def _configure_default(self):
        """Configure a default path for a single quick run."""
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join("logs", f"execution_{timestamp}.log")
        # Do not print here because configuration may not have completed yet.

    def configure(self, log_path: str):
        """
        Allow the entry point to specify the exact log path.
        """
        # Ensure that the parent directory exists.
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = log_path
        print(f"📝 [System] Logger configured to: {self.log_file}")

    def log_agent(self, agent_name: str, prompt: str, response: str):
        # Preserve the original message format.
        max_prompt_len = 1000
        if len(prompt) > max_prompt_len:
            display_prompt = prompt[:max_prompt_len] + f"\n... [TRUNCATED {len(prompt)-max_prompt_len} chars] ..."
        else:
            display_prompt = prompt

        entry = (
            f"\n{'='*60}\n"
            f"🤖 AGENT: {agent_name}\n"
            f"{'-'*60}\n"
            f"[PROMPT (Summary)]:\n{display_prompt.strip()}\n"
            f"{'-'*60}\n"
            f"[RESPONSE]:\n{response.strip()}\n"
            f"{'='*60}\n"
        )
        self._write(entry)

    def log_system(self, event_type: str, content: str):
        # Preserve the original message format.
        entry = (
            f"\n{'#'*60}\n"
            f"⚙️ SYSTEM: {event_type}\n"
            f"{'#'*60}\n"
            f"{content}\n"
        )
        self._write(entry)

    def _write(self, text):
        # Preserve the original message format.
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(text)

# Global accessor
GLOBAL_LOGGER = Logger()
