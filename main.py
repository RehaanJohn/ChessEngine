
import numpy as np

from connectome.loader import load_connectome


def main():
    print("Loading fruit fly brain...\n")

    fly = load_connectome()

    print(f"Neurons:     {fly.num_neurons:,}")
    print(f"Connections: {fly.num_connections:,}")

    # Construct the complete sparse connectivity matrix.
    W = fly.sparse_matrix()

    print("\nSparse matrix:")
    print(f"Shape:       {W.shape}")
    print(f"Stored edges:{W.nnz:,}")
    print(f"Data type:   {W.dtype}")

    # Retrieve a specific neuron and its inputs.
    neuron = 12345

    sources, weights = fly.incoming(neuron)

    print(f"\nNeuron {neuron}")
    print(f"Biological body ID: {fly.body_ids[neuron]}")
    print(f"Incoming connections: {len(sources)}")

    # Test a single sparse matrix-vector multiplication.
    # This is NOT a biological simulation yet.
    rng = np.random.default_rng(42)

    activity = rng.uniform(
        -1.0, 1.0, fly.num_neurons
    ).astype(np.float32)

    response = W @ activity

    print("\nNeural computation test:")
    print(f"Input shape:  {activity.shape}")
    print(f"Output shape: {response.shape}")
    print(f"Finite output: {np.all(np.isfinite(response))}")

    assert response.shape == (fly.num_neurons,)
    assert np.all(np.isfinite(response))

    print("\nSUCCESS: Complete graph is ready.")


if __name__ == "__main__":
    main()
