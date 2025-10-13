# Air/logging/logging.py
import sys
import threading
from enum import Enum
from Air.logging.variables import Style


class LogLevel(Enum):
    """Log level hierarchy."""

    DEBUG = 0
    INFO = 1
    SUCCESS = 2
    WARN = 3
    ERROR = 4


class Logging:
    """
    Air Logging: Clean, professional terminal output.
    Purpose-driven, signal-focused, minimal noise.
    """

    _lock = threading.Lock()

    def __init__(self, use_lock: bool = True, min_level: LogLevel = LogLevel.INFO):
        self._use_lock = use_lock
        self._min_level = min_level

    def _write(self, message: str, level: LogLevel = LogLevel.INFO, style: str = ""):
        """Write to stdout with optional styling and level filtering."""
        if level.value < self._min_level.value:
            return

        code = Style.code(style)
        out = f"{code}{message}\033[0m"

        if self._use_lock:
            with Logging._lock:
                sys.stdout.write(out + "\n")
        else:
            sys.stdout.write(out + "\n")

        sys.stdout.flush()

    def _section(self, title: str, style: str = "bright_cyan bold"):
        """Print a clean section divider."""
        self._write(f"\n{title}", LogLevel.INFO, style)

    # ====================================================================
    # INITIALIZATION
    # ====================================================================
    def init_agent(self, name: str, role: str, goal: str):
        """Log agent initialization."""
        self._write(f"  ✓ {name:25} {role}", LogLevel.INFO, "bright_green")

    def init_bubble(self, name: str, agent_count: int, mode: str):
        """Log bubble creation."""
        self._write(f"\n🔮 {name.upper()}", LogLevel.INFO, "bright_cyan bold")
        self._write(
            f"   Agents: {agent_count} | Mode: {mode.upper()}", LogLevel.INFO, "dim"
        )

    # ====================================================================
    # EXECUTION FLOW
    # ====================================================================
    def bubble_start(self, name: str, goal: str, user_input: str, mode: str):
        """Log bubble execution start."""
        self._section(f"▶ {name} [{mode.upper()}]", "bright_cyan bold")
        self._write(f"  Goal: {goal[:70]}", LogLevel.INFO, "dim")
        if user_input:
            self._write(f"  Query: {user_input[:70]}", LogLevel.INFO, "dim")

    def agent_thinking(self, name: str):
        """Log agent processing."""
        self._write(f"  → {name:25} processing...", LogLevel.DEBUG, "bright_cyan")

    def agent_done(self, name: str, output: str):
        """Log agent completion."""
        preview = output[:70].replace("\n", " ")
        self._write(f"  ✓ {name:25} {preview}", LogLevel.INFO, "bright_green dim")

    def agent_error(self, name: str, error: str):
        """Log agent failure."""
        self._write(f"  ✗ {name:25} {error[:60]}", LogLevel.ERROR, "bright_red bold")

    def agent_tool_use(self, name: str, tool: str, params: dict = None):
        """Log tool execution."""
        param_str = f" {params}" if params else ""
        self._write(
            f"  ⚙  {name:25} using {tool}{param_str}", LogLevel.DEBUG, "bright_yellow"
        )

    def tool_result(self, tool: str, success: bool, output: str = ""):
        """Log tool result."""
        status = "✓" if success else "✗"
        style = "bright_green" if success else "bright_red"
        preview = output[:60].replace("\n", " ") if output else "success"
        self._write(f"     {status} {tool:20} → {preview}", LogLevel.DEBUG, style)

    # ====================================================================
    # MODE-SPECIFIC LOGGING
    # ====================================================================
    def discussion_start(self):
        """Log discussion mode start."""
        self._write(
            "  ► Discussion: Agents sharing perspectives",
            LogLevel.INFO,
            "bright_cyan dim",
        )

    def hierarchy_start(self):
        """Log hierarchy mode start."""
        self._write(
            "  ► Hierarchy: Bottom-up synthesis", LogLevel.INFO, "bright_cyan dim"
        )

    def hierarchy_level(self, level: int, count: int):
        """Log hierarchy synthesis level."""
        self._write(
            f"  ▲ Level {level}: Synthesizing {count} outputs",
            LogLevel.DEBUG,
            "bright_cyan dim",
        )

    def chain_progress(self, agent_name: str, step: int, total: int):
        """Log chain mode progress."""
        self._write(
            f"  → Step {step}/{total}: {agent_name}", LogLevel.DEBUG, "bright_cyan"
        )

    # ====================================================================
    # SYNTHESIS & RESULTS
    # ====================================================================
    def synthesis_start(self, synthesis_type: str):
        """Log synthesis start."""
        self._write(
            f"  ⊙ Synthesizing {synthesis_type}...", LogLevel.DEBUG, "bright_yellow"
        )

    def synthesis_error(self, error: str):
        """Log synthesis failure."""
        self._write(
            f"  ✗ Synthesis failed: {error[:60]}", LogLevel.ERROR, "bright_red bold"
        )

    def bubble_done(self, output: str):
        """Log bubble completion and final result."""
        self._section("✨ FINAL ANSWER", "bright_green bold")
        lines = output.split("\n")
        for line in lines:
            if line.strip():
                self._write(f"  {line}", LogLevel.INFO, "bright_white")

    # ====================================================================
    # PERSISTENCE
    # ====================================================================
    def agent_register(self, name: str, role: str):
        """Log agent registration to bubble."""
        self._write(f"  + {name:25} [{role}]", LogLevel.DEBUG, "bright_cyan dim")

    def agent_unregister(self, name: str):
        """Log agent removal."""
        self._write(f"  - {name}", LogLevel.DEBUG, "dim bright_black")

    def bubble_save(self, name: str):
        """Log bubble save."""
        self._write(f"  ⊙ Bubble saved: {name}", LogLevel.INFO, "bright_green dim")

    def bubble_load(self, name: str):
        """Log bubble load."""
        self._write(f"  ⊙ Bubble loaded: {name}", LogLevel.INFO, "bright_green dim")

    def cache_save(self, name: str):
        """Log cache save."""
        self._write(f"  ⊙ Cached: {name}", LogLevel.DEBUG, "dim bright_black")

    def cache_load(self, name: str):
        """Log cache load."""
        self._write(
            f"  ⊙ Loaded from cache: {name}", LogLevel.DEBUG, "dim bright_black"
        )

    def cache_fail(self, name: str, error: str):
        """Log cache failure."""
        self._write(
            f"  ⚠ Cache error ({name}): {error[:40]}", LogLevel.WARN, "yellow bold"
        )

    # ====================================================================
    # GENERAL UTILITIES
    # ====================================================================
    def info(self, text: str):
        """General info message."""
        self._write(f"ℹ {text}", LogLevel.INFO, "bright_cyan")

    def success(self, text: str):
        """Success message."""
        self._write(f"✓ {text}", LogLevel.SUCCESS, "bright_green bold")

    def warn(self, text: str):
        """Warning message."""
        self._write(f"⚠ {text}", LogLevel.WARN, "yellow bold")

    def error(self, text: str):
        """Error message."""
        self._write(f"✗ {text}", LogLevel.ERROR, "bright_red bold")

    def debug(self, text: str):
        """Debug message."""
        self._write(f"🔍 {text}", LogLevel.DEBUG, "bright_black dim")

    def divider(self, title: str = ""):
        """Print a horizontal divider."""
        width = 80
        line = "─" * width
        if title:
            self._write(f"─ {title} {line[len(title) + 3:]}", LogLevel.INFO, "dim")
        else:
            self._write(line, LogLevel.INFO, "dim")


# global instance
logging_instance = Logging()
