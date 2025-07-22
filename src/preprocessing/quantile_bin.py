import numpy as np
from typing import List
from .base import Preprocessor

class QuantileBinPreprocessor(Preprocessor):
    def __init__(self, matching_type: str, matching_value: str, k: int):
        # Initialize base class
        super().__init__(matching_type, matching_value)
        
        # Store quantile-specific parameters
        self.k = k
        self.edges = None

    def _fit(self) -> np.ndarray:
        """
        Compute self.k equal frequency (quantile) bin edges.
        Returns a NumPy array of length self.k+1.
        """

        if self.data is None:
            raise ValueError("Preprocessor must be fitted before encoding. Call fit() first.")
        
        for code, values in self.data.items():
            values = np.asarray(values)
            probs = np.linspace(0, 1, self.k + 1) # e.g. [0 , 0.25, 0.5, 0.75, 1]
            edges = np.quantile(values, probs) # float edges, monotonic
            self.fits[code] = edges
    
    def _encode(self, code: str, value: float) -> str:
        """
        Given a code and a value, return the bin index

        Args:
            code (str): the code to encode
            value (float): the value to encode
        """
        if self.fits is None:
            raise ValueError("Preprocessor must be fitted before encoding. Call fit() first.")
        
        if self._match(code):
            if code not in self.fits:
                # This is a code matches the criteria but was not present in the training data.
                # In this case we return the value as is without any encoding.
                return value
            else:
                # np.digitize handles scalars & vectors
                bin_index = np.digitize(value, self.fits[code][1:-1])

                # map bin indices to bin labels
                bin_label = f"Q{bin_index}"

                # return the bin labels
                return bin_label
        else:
            raise ValueError(f"_encode() called with code {code} that does not match the matching criteria.")
    
if __name__ == "__main__":

    train_values = np.random.randint(0, 100, 10)

    QBP = QuantileBinPreprocessor(matching_type="starts_with", matching_value="LAB", k=10)
    QBP.fit(train_values)
    print(QBP.edges)

    test_values = np.random.randint(0, 100, 5)
    print(test_values)
    print(QBP.encode(test_values))