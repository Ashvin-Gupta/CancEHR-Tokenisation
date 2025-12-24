from .base import ValuePreprocessor

class RoundNumericPreprocessor(ValuePreprocessor):
    """
    A preprocessor that rounds numeric values to a specified number of decimal places.
    This helps prevent vocabulary explosion when using raw numeric tokens.

    Args:
        matching_type (str): the type of matching to use.
        matching_value (str): the value to match.
        value_column (str): the column containing the numeric values.
        decimals (int): number of decimal places to round to. Default is 1.
    """
    def __init__(self, matching_type: str, matching_value: str, value_column: str, decimals: int = 1):
        super().__init__(matching_type, matching_value, value_column)
        self.decimals = decimals

    def _fit(self) -> None:
        """
        No fitting required for rounding.
        """
        pass

    def _encode(self, code: str, value: float) -> str:
        """
        Round the value to the specified number of decimal places.
        """
        return f"{value:.{self.decimals}f}"