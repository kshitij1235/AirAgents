# Air/logging/logging.py
import sys
import threading
from Air.logging.variables import Style


class Logging:
    """
    Air Logging: Clean, minimal terminal output inspired by Crew AI.
    Focus on signal, not noise.
    """

    _lock = threading.Lock()

    def __init__(self, use_lock: bool = True):
        self._use_lock = use_lock

    def _write(self, message: str, style: str = ""):
        """Write to stdout with optional styling."""
        code = Style.code(style)
        out = f"{code}{message}\033[0m"
        if self._use_lock:
            with Logging._lock:
                sys.stdout.write(out + "\n")
        else:
            sys.stdout.write(out + "\n")
        sys.stdout.flush()

    # ====================================================================
    # AGENTS
    # ====================================================================
    def init_agent(self, name: str, role: str, goal: str):
        """Agent initialized."""
        self._write(f"  ▸ {name:20} {role:20}", "dim bright_black")

    def agent_thinking(self, name: str):
        """Agent is processing."""
        self._write(f"  → {name} thinking...", "bright_cyan")

    def agent_done(self, name: str, output: str):
        """Agent completed."""
        preview = output[:80].replace("\n", " ")
        self._write(f"  ✓ {name:20} {preview}", "bright_green dim")

    def agent_error(self, name: str, error: str):
        """Agent failed."""
        self._write(f"  ✗ {name:20} {error[:60]}", "bright_red")

    def agent_result(self, name: str, output: str):
        """Agent result in bubble context."""
        preview = output[:70].replace("\n", " ")
        self._write(f"    {name:20} {preview}", "bright_white dim")

    def agent_tool_use(self, name: str, tool: str):
        """Agent is using a tool."""
        self._write(f"  ⚙ {name:20} → {tool}", "bright_yellow")

    def result(self, text: str):
        """Result message."""
        self._write(text, "bright_white bold underline")

    def agent_register(self, name: str, role: str):
        """Agent registered to bubble."""
        self._write(f"    + {name:20} [{role}]", "bright_cyan dim")

    def agent_unregister(self, name: str):
        """Agent unregistered."""
        self._write(f"    - {name}", "dim bright_black")

    # ====================================================================
    # BUBBLE
    # ====================================================================
    def init_bubble(self, name: str, agent_count: int, mode: str):
        """Bubble initialized."""
        self._write("", "")
        self._write(f"◆ {name}", "bright_cyan bold")
        self._write(
            f"  agents: {agent_count} | mode: {mode.upper()}", "dim bright_black"
        )

    def bubble_start(self, name: str, goal: str, user_input: str, mode: str):
        """Bubble execution started."""
        self._write("", "")
        self._write(f"► {name} [{mode}]", "bright_cyan")
        self._write(f"  goal: {goal[:60]}", "dim")
        if user_input:
            self._write(f"  input: {user_input[:60]}", "dim")

    def bubble_done(self, output: str):
        """Bubble execution completed."""
        self._write("", "")
        self._write("FINAL ANSWER:", "bright_green bold")
        # Print output with clean formatting
        for line in output.split("\n"):
            if line.strip():
                self._write(f"  {line}", "bright_white")

    def discussion_start(self):
        """Discussion mode started."""
        self._write(
            "  ► discussion mode: agents sharing perspectives", "bright_cyan dim"
        )

    def hierarchy_start(self):
        """Hierarchy mode started."""
        self._write("  ► hierarchy mode: bottom-up synthesis", "bright_cyan dim")

    def hierarchy_level(self, level: int, count: int):
        """Hierarchy synthesis level."""
        self._write(
            f"  ▲ level {level}: synthesizing {count} outputs", "bright_cyan dim"
        )

    def synthesis_error(self, error: str):
        """Synthesis failed."""
        self._write(f"  ✗ synthesis error: {error[:60]}", "bright_red")

    def bubble_save(self, name: str):
        """Bubble saved to disk."""
        self._write(f"  ⊙ saved: {name}", "dim bright_black")

    def bubble_load(self, name: str):
        """Bubble loaded from disk."""
        self._write(f"  ⊙ loaded: {name}", "dim bright_black")

    # ====================================================================
    # CACHING
    # ====================================================================
    def cache_save(self, name: str):
        """Config saved to cache."""
        self._write(f"  ⊙ cache save: {name}", "dim bright_black")

    def cache_load(self, name: str):
        """Config loaded from cache."""
        self._write(f"  ⊙ cache load: {name}", "dim bright_black")

    def cache_fail(self, name: str, error: str):
        """Cache operation failed."""
        self._write(f"  ⚠ cache error ({name}): {error[:40]}", "yellow dim")

    # ====================================================================
    # GENERAL
    # ====================================================================
    def info(self, text: str):
        """General info message."""
        self._write(f"ℹ {text}", "bright_cyan")

    def success(self, text: str):
        """Success message."""
        self._write(f"✓ {text}", "bright_green")

    def warn(self, text: str):
        """Warning message."""
        self._write(f"⚠ {text}", "yellow bold")

    def error(self, text: str):
        """Error message."""
        self._write(f"✗ {text}", "bright_red bold")

    def dim(self, text: str):
        """Dimmed message."""
        self._write(text, "dim bright_black")

    def fade(self, text: str):
        """Faded/dim message."""
        self._write(text, "dim bright_black italic")

    def banner(self, text: str):
        """Big banner message."""
        self._write(f"╔══ {text} ══╗", "bright_cyan bold")

    def rule(self, title: str = "", style: str = "bright_cyan bold"):
        """Draws a horizontal rule with optional title."""
        width = 80
        text = f" {title} " if title else ""
        line = "─" * max(2, width - len(text) - 2)
        self._write(f"{text}{line}", style)


# global instance
logging_instance = Logging()
