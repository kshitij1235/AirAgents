class Style:
    """
    Minimal ANSI style resolver.
    Supports: bold, dim, italic, underline, blink, invert, and 256-color fg.
    """

    STYLES = {
        "bold": "\033[1m",
        "dim": "\033[2m",
        "italic": "\033[3m",
        "underline": "\033[4m",
        "blink": "\033[5m",
        "invert": "\033[7m",
    }

    COLORS = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 37,
        "bright_black": 90,
        "bright_red": 91,
        "bright_green": 92,
        "bright_yellow": 93,
        "bright_blue": 94,
        "bright_magenta": 95,
        "bright_cyan": 96,
        "bright_white": 97,
    }

    @staticmethod
    def code(style_string: str) -> str:
        """Compose multiple styles like 'dim cyan italic'."""
        parts = style_string.split()
        seq = []
        for p in parts:
            if p in Style.STYLES:
                seq.append(Style.STYLES[p])
            elif p in Style.COLORS:
                seq.append(f"\033[{Style.COLORS[p]}m")
        return "".join(seq)

    @staticmethod
    def apply(text: str, style_string: str) -> str:
        return f"{Style.code(style_string)}{text}\033[0m"
