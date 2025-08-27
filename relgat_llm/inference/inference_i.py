import torch

from typing import Optional
from abc import ABC, abstractmethod


class InferenceI(ABC):
    """
    Common interface for graph embedding propagation/inference methods.

    Each implementation must accept at least:
      - edge_index: torch.LongTensor of shape [2, E]
      - edge_type:  Optional[torch.LongTensor] of shape [E]
      - device:     Optional[torch.device]

    And must implement:
      - run(base_emb: torch.Tensor [N, D]) -> torch.Tensor [N, D]
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if edge_index is None or edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must be a [2, E] LongTensor")
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.device = device or edge_index.device

    @abstractmethod
    def run(self, base_emb: torch.Tensor) -> torch.Tensor:
        """
        Execute propagation/inference on the base node embeddings.

        Parameters
        ----------
        base_emb : torch.Tensor [N, D]
            Base node embeddings. Unknown nodes can be zero or small noise.

        Returns
        -------
        torch.Tensor [N, D]
            Stabilized/propagated embeddings.
        """
        ...
