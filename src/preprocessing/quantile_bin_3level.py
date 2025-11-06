import numpy as np
from typing import List
from .base import ValuePreprocessor

class QuantileBin3LevelPreprocessor(ValuePreprocessor):
    """
    A preprocessor that bins values into 3 levels: low, normal, or high.
    - Below Q1 (25th percentile): "low"
    - Between Q1 and Q3 (inclusive): "normal"
    - Above Q3 (75th percentile): "high"

    Args:
        matching_type (str): the type of matching to use. Must be one of "starts_with", "ends_with", "contains", "equals", "regex"
        matching_value (str): the value to match.
        value_column (str): the column containing the numeric values to bin.
    """
    def __init__(self, matching_type: str, matching_value: str, value_column: str):
        # Initialize base class
        super().__init__(matching_type, matching_value, value_column)
        
        print(f"QuantileBin3LevelPreprocessor initialized: matching_type='{matching_type}', matching_value='{matching_value}', value_column='{value_column}'")

    def _fit(self) -> None:
        """
        Compute Q1 (25th percentile), Q2 (median), and Q3 (75th percentile) for each code.
        Store the quantiles in self.fits.
        """

        if self.data is None:
            raise ValueError("Preprocessor must be fitted before encoding. Call fit() first.")
        
        print(f"\n--- DEBUG: Fitting QuantileBin3LevelPreprocessor for '{self.matching_value}' ---")
        
        for code, values in self.data.items():
            values = np.asarray(values)
            q1 = np.quantile(values, 0.25)
            q2 = np.quantile(values, 0.50)  # median
            q3 = np.quantile(values, 0.75)
            self.fits[code] = {'q1': q1, 'q2': q2, 'q3': q3}

            print(f"  - Learned quantiles for code '{code}': Q1={q1:.4f}, Q2={q2:.4f}, Q3={q3:.4f}")
    
    def _encode(self, code: str, value: float) -> str:
        """
        Given a code and a value, return the bin level (low, normal, or high)

        Args:
            code (str): the code to encode
            value (float): the value to encode

        Returns:
            str: the bin label ("low", "normal", or "high")
        """
        if self.fits is None:
            raise ValueError("Preprocessor must be fitted before encoding. Call fit() first.")
        
        if self._match(code):
            if code not in self.fits:
                # This is a code matches the criteria but was not present in the training data.
                # In this case we return the value as is without any encoding.
                return value
            else:
                quantiles = self.fits[code]
                q1 = quantiles['q1']
                q3 = quantiles['q3']
                
                if value < q1:
                    bin_label = "low"
                elif value <= q3:
                    bin_label = "normal"
                else:
                    bin_label = "high"

                return bin_label
        else:
            raise ValueError(f"_encode() called with code {code} that does not match the matching criteria.")
    
if __name__ == "__main__":
    # Test the preprocessor
    import numpy as np
    
    # Create test data
    train_values = np.random.normal(100, 15, 1000)  # Normal distribution
    
    QBP3 = QuantileBin3LevelPreprocessor(matching_type="starts_with", matching_value="LAB", value_column="numeric_value")
    
    # Note: In real usage, fit() would be called with event files
    # This is just for testing the logic
    print("QuantileBin3LevelPreprocessor test completed!")

