
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "malecns"


@dataclass
class Connectome:
    body_ids: np.ndarray
    offsets: np.ndarray
    sources: np.ndarray
    weights: np.ndarray
    retina: np.ndarray
    descending: np.ndarray

    @property
    def num_neurons(self) -> int:
        return len(self.body_ids)

    @property
    def num_connections(self) -> int:
        return len(self.sources)

    def sparse_matrix(self) -> csr_matrix:
        """W[dst, src] = signed connection weight."""
        return csr_matrix(
            (self.weights, self.sources, self.offsets),
            shape=(self.num_neurons, self.num_neurons),
            copy=False,
        )

    def incoming(self, neuron_index: int):
        """Return the incoming edges of a neuron."""
        if not 0 <= neuron_index < self.num_neurons:
            raise IndexError("Neuron index out of range")

        start = int(self.offsets[neuron_index])
        end = int(self.offsets[neuron_index + 1])

        return self.sources[start:end], self.weights[start:end]


def load_connectome(
    data_dir: Path = DATA_DIR,
) -> Connectome:
    """Memory-map the MaleCNS graph without copying all arrays."""

    data_dir = Path(data_dir)

    if not (data_dir / "manifest.json").exists():
        raise FileNotFoundError(
            "Dataset missing. Run: python -m connectome.download"
        )

    graph = data_dir / "graph"
    neurons = data_dir / "neurons"
    populations = data_dir / "populations"

    body_ids = np.memmap(
        neurons / "body_id.i64", dtype=np.int64, mode="r"
    )

    offsets = np.memmap(
        graph / "edges_offsets.i32", dtype=np.int32, mode="r"
    )

    sources = np.memmap(
        graph / "edges_source.i32", dtype=np.int32, mode="r"
    )

    weights = np.memmap(
        graph / "edges_weight.f32", dtype=np.float32, mode="r"
    )

    retina = np.fromfile(
        populations / "retina.i32", dtype=np.int32
    )

    descending = np.fromfile(
        populations / "descending.i32", dtype=np.int32
    )

    return Connectome(
        body_ids=body_ids,
        offsets=offsets,
        sources=sources,
        weights=weights,
        retina=retina,
        descending=descending,
    )
