"""Abstract base processor defining the interface for all data processors."""

from abc import ABC, abstractmethod
from typing import Any


class BaseProcessor(ABC):
    """Abstract base class for all data processors.

    All processors (Tabular, Text, Image) must inherit from this class
    and implement the required interface methods.
    """

    @abstractmethod
    def load_data(self, data: bytes) -> None:
        """Load raw data from bytes into the processor.

        Args:
            data: Raw file content as bytes.
        """
        ...

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """Return statistical information about the loaded data.

        Returns:
            Dictionary containing relevant statistics for the data type.
        """
        ...

    @abstractmethod
    def get_processed_data(self) -> bytes:
        """Return processed data as bytes.

        Returns:
            Processed data serialized as bytes.
        """
        ...
