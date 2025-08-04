from typing import List, Dict, Any
import polars as pl
import numpy as np
from abc import ABC, abstractmethod
import os
from tqdm import tqdm

class Postprocessor(ABC):
    """
    Base class for all postprocessors operating on token sequences
    """
    def __init__(self):
        pass

    def encode(self, subject_data: List[Dict[str, Any]]) -> List[str]:
        """
        Postprocess the subject data, calls _encode (implemented by subclasses) for each datapoint

        Args:
            subject_data: list of dictionaries containing the subject data

        Returns:
            list of encoded data
        """
        encoded_data = []
        for datapoint in tqdm(subject_data, desc="Postprocessing", leave=False):
            encoded_data.append(self._encode(datapoint))

        return encoded_data
            
    @abstractmethod
    def _encode(self, datapoint: Dict[str, Any]) -> List[str]:
        """
        Encode a single datapoint, implemented by subclasses

        Args:
            datapoint: dictionary containing the datapoint
        """
        raise NotImplementedError("Subclasses must implement this method")
