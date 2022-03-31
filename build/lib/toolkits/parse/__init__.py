"""
The `toolkits.parse` module implements log parser.
"""

from .between_lines import between_lines, between_lines_on_file, between_lines_on_dir

__all__ = [
    "between_lines",
    "between_lines_on_file",
    "between_lines_on_dir",
]
