from .base import CodePreprocessor
from typing import List, Dict, Any, Optional
import polars as pl

class CodeTruncationPreprocessor(CodePreprocessor):
    """
    A preprocessor that truncates codes containing '//' to their first two parts.
    E.g., 'MEDICAL//A//1' becomes 'MEDICAL//A'.
    Codes without '//' are left unchanged.
    """
    def _transform_code(self, code: str) -> str:
        """
        Transforms the code by truncating it.

        Args:
            code (str): The original code string.

        Returns:
            str: The transformed (truncated) code.
        """
        if "//" in code:
            parts = code.split("//")
            # Take the first two parts and join them back
            return "//".join(parts[:2])
        return code

    def fit(self, event_files: List[str]) -> None:
        """
        This preprocessor doesn't need training data as it uses a deterministic algorithm.
        The truncation is based on the '//' delimiter and not data distribution.
        
        Args:
            event_files (List[str]): Not used for this preprocessor
        """
        print("CodeTruncationPreprocessor fit complete (no training required - uses deterministic algorithm)")