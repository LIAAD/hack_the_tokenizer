# ASCII Color Codes
from dataclasses import dataclass

@dataclass
class Color:
    """Represents an ASCII color with its corresponding ANSI escape code.

    Attributes:
        name (str): The human-readable name of the color (e.g., 'RED', 'BLUE').
        code (str): The ANSI escape code sequence for the color (e.g., '31m', '34m').
        _RESET (str): The ANSI escape code to reset text formatting to default.
    """
    name: str
    code: str
    _RESET: str = '\x1b[0m'  # Default reset code

    def __str__(self) -> str:
        """Returns the ANSI escape sequence to apply the color.

        Example:
            >>> print(Colors.RED)
            \x1b[0m\x1b[31m
        """
        return f'{self._RESET}\x1b[{self.code}'

    @property
    def bold(self) -> str:
        """Returns the ANSI escape sequence to apply the color in bold.

        Example:
            >>> print(Colors.BLUE.bold)
            \x1b[0m\x1b[1;34m
        """
        return f'{self._RESET}\x1b[1;{self.code}'

class Colors:
    """
    Overview
    --------
    A collection of predefined Color objects for common ANSI colors.

    This class provides easy access to various colors and a reset option,
    allowing for simple application of color to terminal output.

    Usage Examples:
    ---------------
        Apply color directly in an f-string:
        >>> print(f'{Colors.BLACK}This text is black.{Colors.RESET}')
        \x1b[0m\x1b[30mThis text is black.\x1b[0m

        Apply bold color directly in an f-string:
        >>> print(f'{Colors.GREEN.bold}This text is bold green.{Colors.RESET}')
        \x1b[0m\x1b[1;32mThis text is bold green.\x1b[0m

        Combine colors:
        >>> print(f'{Colors.YELLOW}Warning:{Colors.RESET} {Colors.CYAN}System status is normal.')
        \x1b[0m\x1b[33mWarning:\x1b[0m \x1b[0m\x1b[36mSystem status is normal.
    """
    BLACK   = Color('Black',    '30m')
    RED     = Color('RED',      '31m')
    GREEN   = Color('GREEN',    '32m')
    YELLOW  = Color('YELLOW',   '33m')
    BLUE    = Color('BLUE',     '34m')
    PURPLE  = Color('PURPLE',   '35m')
    CYAN    = Color('CYAN',     '36m')
    WHITE   = Color('WHITE',    '37m')
    RESET   = Color('NO_COLOR', '0m')
