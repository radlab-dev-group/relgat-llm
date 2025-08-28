"""
Inference utilities for RelGAT models.

The module provides a high‑level :class:`Inference` class that hides all
boiler‑plate required to load a trained RelGAT checkpoint, construct the
graph tensors (``edge_index``/``edge_type``) and run any of the supported
inference methods (currently **RD‑GAT** and **Iterative GAT**).

Typical usage
--------------
```
python
from relgat_llm.inference import Inference

# paths produced by the trainer (see ``RelGATTrainer._save_checkpoint``)
model_path = "checkpoints/relgat_distmult_ratio90/best_checkpoint_1000/relgat_model.pt"
nodes_path = "checkpoints/relgat_distmult_ratio90/best_checkpoint_1000/nodes.json"
rels_path  = "checkpoints/relgat_distmult_ratio90/best_checkpoint_1000/rel_to_idx.json"
edges_path = "checkpoints/relgat_distmult_ratio90/best_checkpoint_1000/edges.json"

# optional hyper‑parameters for the chosen method
hyperparams = {
    "alpha": 0.5,
    "beta": 0.5,
    "T": 1.0,
    "steps": 10,
    "gamma": 0.1,
    "clip_norm": None,
}

inf = Inference(
    model_path=model_path,
    nodes_path=nodes_path,
    rels_path=rels_path,
    edges_path=edges_path,
    device="cuda",
    hyperparams=hyperparams,
)

# base embeddings – e.g. known node vectors + zeros for the rest
base_emb = inf.default_base_embeddings()

# run the RD‑GAT propagation
embeddings = inf.infer("rdgnode", base_emb=base_emb)
```
The class can be called with any supported method name
(`"rdgnode"` or `"iterative"`).  Hyper‑parameters that are not
explicitly supplied fall back to the defaults defined for the method.
"""

import json
import torch

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from relgat_llm.trainer.relation.model import RelGATModel
from relgat_llm.trainer.main.part.constants import ConstantsRelGATTrainer

from relgat_llm.inference.methods.rdgnode import RDGNODEInference
from relgat_llm.inference.methods.iterative import IterativeRelGATInference


# --------------------------------------------------------------------------- #
# Helper functions – loading graph data and model
# --------------------------------------------------------------------------- #
def _load_embeddings_and_mappings(
    nodes_path: str,
    rels_path: str,
    edges_path: str,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, int], List[Tuple[int, int, str]]]:
    """
    Load node embeddings, relation‑to‑index mapping and raw edge list.
    The files are JSON‑encoded exactly as written by
    :class:`RelGATTrainer._save_checkpoint`.
    """
    # node embeddings – list of vectors stored as JSON
    with open(nodes_path, "r", encoding="utf-8") as f:
        node2emb_raw = json.load(f)  # {str(node_id): [float, ...]}
    node2emb = {
        int(k): torch.tensor(v, dtype=torch.float)
        for k, v in node2emb_raw.items()
    }

    # relation mapping
    with open(rels_path, "r", encoding="utf-8") as f:
        rel2idx = json.load(f)  # {rel_str: idx}
    # edge list – each entry is (src_id, dst_id, rel_str)
    with open(edges_path, "r", encoding="utf-8") as f:
        edge_index_raw = json.load(f)  # List[Tuple[int, int, str]]

    return node2emb, rel2idx, edge_index_raw


def _build_edge_tensors(
    edge_index_raw: List[Tuple[int, int, str]],
    rel2idx: Dict[str, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert the raw edge list to ``edge_index`` (2×E) and ``edge_type`` (E)
    tensors that are ready for the model.
    """
    src, dst, rel = zip(*edge_index_raw)  # type: ignore[arg-type]
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_type = torch.tensor(
        [rel2idx[r] if isinstance(r, str) else int(r) for r in rel],
        dtype=torch.long,
        device=device,
    )
    return edge_index, edge_type


def _instantiate_model(
    node2emb: Dict[int, torch.Tensor],
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    rel2idx: Dict[str, int],
    device: torch.device,
) -> RelGATModel:
    """
    Build a :class:`RelGATModel` instance exactly as the trainer does.
    The function mirrors the relevant part of ``RelGATTrainer.__init__``.
    """
    # Prepare the embedding matrix
    all_node_ids = sorted(node2emb.keys())
    node_emb_matrix = torch.stack(
        [node2emb[nid].to(dtype=torch.float) for nid in all_node_ids],
        dim=0,
    ).to(device)

    # number of relations
    num_rel = len(rel2idx)

    # Hyper‑parameters that are *not* stored in the checkpoint.
    # They are kept at their trainer defaults – they can be overridden later.
    scorer_type = "distmult"
    gat_out_dim = 200
    gat_heads = 6
    gat_num_layers = 1
    dropout = 0.2
    relation_attn_dropout = 0.0

    model = RelGATModel(
        node_emb=node_emb_matrix,
        edge_index=edge_index,
        edge_type=edge_type,
        num_rel=num_rel,
        scorer_type=scorer_type,
        gat_out_dim=gat_out_dim,
        gat_heads=gat_heads,
        dropout=dropout,
        relation_attn_dropout=relation_attn_dropout,
        gat_num_layers=gat_num_layers,
    ).to(device)

    return model


def load_checkpoint_state(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """
    Load a checkpoint saved by ``RelGATTrainer._save_checkpoint``.
    Returns a dict with the raw state_dict and the auxiliary config files.
    """
    # the checkpoint is a *state_dict*; the surrounding config files are
    # stored alongside it (see trainer._save_checkpoint)
    checkpoint_dir = Path(checkpoint_path).parent
    state_dict_path = Path(checkpoint_path)

    state_dict = torch.load(state_dict_path, map_location=device)

    # auxiliary files
    config_path = checkpoint_dir / ConstantsRelGATTrainer.Default.TRAINING_CONFIG_FILE_NAME
    rel2idx_path = checkpoint_dir / ConstantsRelGATTrainer.Default.TRAINING_CONFIG_REL_TO_IDX

    with open(config_path, "r", encoding="utf-8") as f:
        run_config = json.load(f)

    with open(rel2idx_path, "r", encoding="utf-8") as f:
        rel2idx = json.load(f)

    return {
        "state_dict": state_dict,
        "run_config": run_config,
        "rel2idx": rel2idx,
    }


# --------------------------------------------------------------------------- #
# Public Inference class
# --------------------------------------------------------------------------- #
class Inference:
    """
    High‑level wrapper that loads a trained RelGAT model together with the
    graph structure and offers a unified interface for the supported
    inference procedures.

    Parameters
    ----------
    model_path : str
        Path to the ``*.pt`` checkpoint containing the model ``state_dict``.
    nodes_path : str
        JSON file with node embeddings (produced by the trainer).
    rels_path : str
        JSON file with ``relation → idx`` mapping (produced by the trainer).
    edges_path : str
        JSON file with the raw edge list – each entry is
        ``[src_id, dst_id, relation_str]``.
    device : Optional[str] or :class:`torch.device`, default ``cpu``.
    hyperparams : Optional[Dict[str, Any]]
        Dictionary with *method‑specific* hyper‑parameters.  The keys are the
        exact argument names of the inference methods (e.g. ``alpha``,
        ``beta`` for ``RDGNODE``).  Missing values are filled with the
        defaults defined in the corresponding inference class.
    """

    def __init__(
        self,
        model_path: str,
        nodes_path: str,
        rels_path: str,
        edges_path: str,
        device: Optional[torch.device] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = torch.device(device) if device else torch.device("cpu")
        self.hyperparams = hyperparams or {}

        # ------------------------------------------------------------------- #
        # Load graph data (node embeddings, relation mapping, raw edges)
        # ------------------------------------------------------------------- #
        node2emb, rel2idx, edge_index_raw = self._load_graph_data(
            nodes_path, rels_path, edges_path
        )

        # Build tensors required by the inference methods
        self.edge_index, self.edge_type = self._build_edge_tensors(
            edge_index_raw, rel2idx
        )

        # ------------------------------------------------------------------- #
        # Load the model (RelGATModel) from checkpoint
        # ------------------------------------------------------------------- #
        self.model = self._load_model_from_checkpoint(
            model_path, node2emb, rel2idx, self.edge_index, self.edge_type
        )

        # Store the original node embeddings – useful for creating a default
        # ``base_emb`` where known nodes keep their trained vectors.
        self.node2emb = node2emb

    # ------------------------------------------------------------------- #
    # Internal helpers
    # ------------------------------------------------------------------- #
    @staticmethod
    def _load_graph_data(
        nodes_path: str,
        rels_path: str,
        edges_path: str,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[str, int], List[Tuple[int, int, str]]]:
        """
        Load node embeddings, relation mapping and the raw edge list.
        The file format matches the one produced by
        ``RelGATMainTrainerHandler.load_embeddings_and_edges``.
        """
        # node embeddings – stored as {str(node_id): [float, ...]}
        with open(nodes_path, "r", encoding="utf-8") as f:
            raw_node2emb = json.load(f)
        node2emb = {
            int(k): torch.tensor(v, dtype=torch.float)
            for k, v in raw_node2emb.items()
        }

        # relations mapping
        with open(rels_path, "r", encoding="utf-8") as f:
            rel2idx = json.load(f)

        # raw edges – list of [src, dst, rel_str]
        with open(edges_path, "r", encoding="utf-8") as f:
            edge_index_raw = json.load(f)

        return node2emb, rel2idx, edge_index_raw

    def _build_edge_tensors(
        self,
        edge_index_raw: List[Tuple[int, int, str]],
        rel2idx: Dict[str, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw edges to ``edge_index`` and ``edge_type`` tensors.
        """
        # Map original node IDs to a compact index space (0 … N‑1)
        all_node_ids = sorted({s for s, _, _ in edge_index_raw} |
                           {d for _, d, _ in edge_index_raw})
        id2idx = {nid: i for i, nid in enumerate(all_node_ids)}

        mapped = [
            (id2idx[s], id2idx[d], r) for s, d, r in edge_index_raw
        ]
        src, dst, rel = zip(*mapped)  # type: ignore[arg-type]

        edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        edge_type = torch.tensor(
            [rel2idx[r] if isinstance(r, str) else int(r) for r in rel],
            dtype=torch.long,
            device=self.device,
        )
        return edge_index, edge_type

    def _load_model_from_checkpoint(
        self,
        checkpoint_path: str,
        node2emb: Dict[int, torch.Tensor],
        rel2idx: Dict[str, int],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> RelGATModel:
        """
        Re‑create the :class:`RelGATModel` that was used during training.
        The checkpoint only stores the *state_dict*; the architecture
        hyper‑parameters are taken from the trainer defaults (they are
        also saved in the config files but re‑creating them here is simpler).
        """
        num_rel = len(rel2idx)

        # The trainer uses the same defaults for the model architecture;
        # they are reproduced here.
        scorer_type = "distmult"
        gat_out_dim = 200
        gat_heads = 6
        gat_num_layers = 1
        dropout = 0.2
        relation_attn_dropout = 0.0

        model = RelGATModel(
            node_emb=torch.stack(
                [node2emb[nid] for nid in sorted(node2emb.keys())],
                dim=0,
            ).to(self.device),
            edge_index=edge_index,
            edge_type=edge_type,
            num_rel=num_rel,
            scorer_type=scorer_type,
            gat_out_dim=gat_out_dim,
            gat_heads=gat_heads,
            dropout=dropout,
            relation_attn_dropout=relation_attn_dropout,
            gat_num_layers=gat_num_layers,
        ).to(self.device)

        # Load parameters
        state = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state, strict=False)

        return model

    def default_base_embeddings(self) -> torch.Tensor:
        """
        Construct a ``base_emb`` tensor that contains the trained node
        vectors for all *known* nodes and zeros for the rest.
        """
        N = self.model.node_emb.shape[0]
        base = torch.zeros((N, self.model.node_emb.shape[1]), dtype=torch.float, device=self.device)

        # Fill with the original node embeddings (if any)
        for idx, vec in enumerate(self.node2emb.values()):
            base[idx] = vec.to(self.device)

        return base

    # ------------------------------------------------------------------- #
    # Public inference methods
    # ------------------------------------------------------------------- #
    def _prepare_hyperparams(
        self,
        method: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Merge the user supplied ``overrides`` with the class‑level
        ``hyperparams`` dictionary (if any).  Method‑specific defaults are
        added automatically.
        """
        overrides = overrides or {}
        merged = dict(self.hyperparams)  # copy

        # Method‑specific defaults
        if method == "rdgnode":
            defaults = {
                "alpha": 0.5,
                "beta": 0.5,
                "T": 1.0,
                "steps": 10,
                "gamma": 0.1,
                "clip_norm": None,
            }
        elif method == "iterative":
            defaults = {
                "n_iters": 5,
                "step_size": 0.1,
            }
        else:
            raise ValueError(f"Unsupported inference method: {method}")

        merged.update(defaults)
        merged.update(overrides)  # user supplied values win
        return merged

    def run_rdgnode(
        self,
        base_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the RD‑GAT propagation (continuous ODE solver).

        Parameters
        ----------
        base_emb : Optional[torch.Tensor]
            Initial node features.  If ``None`` a default tensor with
            zero‑vectors (filled with the trained node embeddings) is used.
        **kwargs
            Any of the RD‑GAT hyper‑parameters
            (``alpha``, ``beta``, ``T``, ``steps``, ``gamma``,
            ``clip_norm``).  Missing values are taken from the defaults.

        Returns
        -------
        torch.Tensor
            Propagated node embeddings of shape ``(N, D)``.
        """
        if base_emb is None:
            base_emb = self.default_base_embeddings()

        params = self._prepare_hyperparams("rdgnode", kwargs)

        # Initialise the RD‑GAT inference object
        rdgnode = RDGNODEInference(
            model=self.model,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            device=self.device,
        )
        # Pass the hyper‑parameters to the underlying class
        rdgnode.alpha = params["alpha"]
        rdgnode.beta = params["beta"]
        rdgnode.T = params["T"]
        rdgnode.steps = params["steps"]
        rdgnode.gamma = params["gamma"]
        rdgnode.clip_norm = params["clip_norm"]

        return rdgnode.run(base_emb)

    def run_iterative(
        self,
        base_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the iterative RelGAT inference method.

        Parameters
        ----------
        base_emb : Optional[torch.Tensor]
            Same semantics as :meth:`run_rdgnode`.
        **kwargs
            Hyper‑parameters specific to the iterative method
            (e.g. ``n_iters``, ``step_size``).

        Returns
        -------
        torch.Tensor
            Updated node embeddings.
        """
        if base_emb is None:
            base_emb = self.default_base_embeddings()

        params = self._prepare_hyperparams("iterative", kwargs)

        iterative = IterativeRelGATInference(
            model=self.model,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            device=self.device,
        )
        # Apply user‑provided hyper‑parameters
        for name, value in params.items():
            setattr(iterative, name, value)

        return iterative.run(base_emb)

    # ------------------------------------------------------------------- #
    # Unified entry point
    # ------------------------------------------------------------------- #
    def infer(
        self,
        method: str,
        base_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Dispatch inference to the requested ``method``.

        Supported methods
        -----------------
        * ``"rdgnode"`` – continuous ODE propagation.
        * ``"iterative"`` – discrete iterative propagation.

        Any additional keyword arguments are forwarded to the concrete
        method implementation.
        """
        method = method.lower()
        if method == "rdgnode":
            return self.run_rdgnode(base_emb=base_emb, **kwargs)
        elif method == "iterative":
            return self.run_iterative(base_emb=base_emb, **kwargs)
        else:
            raise ValueError(f"Unknown inference method: {method}")

    # ------------------------------------------------------------------- #
    # Convenience ``__call__`` – mirrors ``infer`` for a more pythonic API
    # ------------------------------------------------------------------- #
    def __call__(
        self,
        method: str,
        base_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.infer(method, base_emb=base_emb, **kwargs)


# --------------------------------------------------------------------------- #
# Backward compatibility – simple script entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run inference (RD‑GAT or Iterative) on a saved RelGAT model."
    )
    parser.add_argument("--model", required=True, help="Path to model checkpoint (*.pt)")
    parser.add_argument(
        "--nodes", required=True, help="Path to node embeddings JSON file"
    )
    parser.add_argument(
        "--rels", required=True, help="Path to relation‑to‑idx JSON file"
    )
    parser.add_argument(
        "--edges", required=True, help="Path to raw edge list JSON file"
    )
    parser.add_argument(
        "--method",
        choices=["rdgnode", "iterative"],
        default="rdgnode",
        help="Inference method to run",
    )
    parser.add_argument(
        "--device", default=None, help="Torch device (e.g. cuda, cpu)."
    )
    args = parser.parse_args()

    # Minimal hyper‑parameter handling – users can extend this as needed.
    hyperparams = {}

    inf = Inference(
        model_path=args.model,
        nodes_path=args.nodes,
        rels_path=args.rels,
        edges_path=args.edges,
        device=args.device,
        hyperparams=hyperparams,
    )

    # Use the stored node embeddings as the initial known vectors.
    base = inf.default_base_embeddings()
    out = inf(method=args.method, base_emb=base)
    print("Inference result shape:", out.shape)
