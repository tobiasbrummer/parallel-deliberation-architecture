import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import svd, orth
from typing import List, Dict, Any, Callable, Optional

# --- 1a: Subspace Generation ---

def generate_orthogonal_subspaces(
    n_subspaces: int,
    dim: int,
    subspace_dim: int,
    orthogonality: float = 1.0
) -> List[np.ndarray]:
    """
    Generates n_subspaces projection matrices (dim x subspace_dim).
    
    Args:
        n_subspaces: Number of subspaces to create.
        dim: Total dimension of the embedding space.
        subspace_dim: Dimension of each individual subspace.
        orthogonality: Interpolation factor between exactly orthogonal (1.0) and random (0.0).
        
    Returns:
        List of projection matrices (bases), each of shape (dim, subspace_dim).
    """
    if n_subspaces * subspace_dim > dim:
        raise ValueError(
            f"Cannot fit {n_subspaces} orthogonal subspaces of dimension "
            f"{subspace_dim} into total dimension {dim}"
        )

    # Generate a random orthogonal basis for the entire space
    # scipy.linalg.orth provides an orthonormal basis
    full_basis = orth(np.random.randn(dim, dim))

    subspaces = []
    for i in range(n_subspaces):
        # Exactly orthogonal part
        start = i * subspace_dim
        ortho_basis = full_basis[:, start:start + subspace_dim]

        if orthogonality < 1.0:
            # Random subspace basis
            random_basis = orth(np.random.randn(dim, subspace_dim))
            # Linear interpolation followed by re-orthogonalization
            # (Note: This is a simplified SLERP on the Grassmann manifold)
            mixed = orthogonality * ortho_basis + (1 - orthogonality) * random_basis
            mixed = orth(mixed)
            subspaces.append(mixed)
        else:
            subspaces.append(ortho_basis)

    return subspaces

# --- 1b: Semantic Vectors ---

def load_semantic_vectors(n_vectors: int, dim: int) -> np.ndarray:
    """
    Loads pre-trained embeddings or falls back to cluster-structured vectors.
    
    Returns:
        Array of shape (n_vectors, dim).
    """
    try:
        # Placeholder for external embedding loaders (e.g. gensim)
        raise ImportError("Pre-trained embedding loader not implemented yet.")
    except ImportError:
        # Fallback: Cluster-structured vectors (non-random, structured semantic space)
        n_clusters = max(3, n_vectors // 10)
        centers = np.random.randn(n_clusters, dim)
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)

        vectors = []
        for i in range(n_vectors):
            cluster = i % n_clusters
            # Add noise to simulate cluster spread
            noise = np.random.randn(dim) * 0.3
            vec = centers[cluster] + noise
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            vectors.append(vec)

        return np.array(vectors)

# --- 1c: Merge Strategies ---

def merge_average(worker_states: List[np.ndarray]) -> np.ndarray:
    """Baseline: Weighted average of worker states."""
    return np.mean(worker_states, axis=0)

def merge_phase_alignment(worker_states: List[np.ndarray]) -> np.ndarray:
    """
    Reinforces in-phase components and dampens out-of-phase components.
    Uses cosine similarity to the mean as a 'phase' weight.
    """
    mean = np.mean(worker_states, axis=0)
    mean_flat = mean.flatten()
    mean_norm = np.linalg.norm(mean_flat) + 1e-8
    
    aligned = np.zeros_like(mean)
    total_weight = 0.0

    for state in worker_states:
        state_flat = state.flatten()
        # Cosine similarity as phase alignment weight
        cos_sim = np.dot(state_flat, mean_flat) / (np.linalg.norm(state_flat) * mean_norm + 1e-8)
        
        # Only use constructive interference (weight > 0)
        weight = max(0.0, float(cos_sim))
        aligned += weight * state
        total_weight += weight

    return aligned / (total_weight + 1e-8)

def merge_frequency_selective(worker_states: List[np.ndarray]) -> np.ndarray:
    """
    Uses SVD decomposition to selectively weight components.
    Dominant components (low frequency) are averaged; fine details (high frequency) 
    are boosted.
    """
    stacked = np.stack([s.flatten() for s in worker_states])
    U, S, Vt = svd(stacked, full_matrices=False)

    n_dominant = max(1, len(S) // 2)
    
    result_flat = np.zeros(stacked.shape[1])
    for i, (s, v) in enumerate(zip(S, Vt)):
        # Heuristic: Boost 'higher frequency' (fine detail) components
        weight = 1.0 if i < n_dominant else 2.0
        result_flat += (s * v * weight) / len(S)

    return result_flat.reshape(worker_states[0].shape)

def merge_sidechain(worker_states: List[np.ndarray]) -> np.ndarray:
    """
    One worker provides the content ('what'), while others modulate the intensity ('how strong').
    Uses the sigmoid of the mean of non-primary workers as a gate.
    """
    if len(worker_states) < 2:
        return worker_states[0]

    primary = worker_states[0]
    # Modulator is the mean of all other workers
    modulator = np.mean(worker_states[1:], axis=0)

    # Element-wise modulation via gating
    gate = 1.0 / (1.0 + np.exp(-modulator))
    return primary * gate

MERGE_STRATEGIES = {
    "average": merge_average,
    "phase_alignment": merge_phase_alignment,
    "frequency_selective": merge_frequency_selective,
    "sidechain": merge_sidechain,
}

# --- 1f: Diversity Enforcement ---

def apply_diversity_repulsion(
    states: List[np.ndarray],
    subspaces: Optional[List[np.ndarray]] = None,
    strength: float = 0.1,
) -> List[np.ndarray]:
    """
    Applies contrastive repulsion to prevent worker collapse.
    Forces workers apart if they become too similar.
    """
    new_states = []
    for i in range(len(states)):
        repulsion = np.zeros_like(states[i]).astype(np.float64)
        s_i = states[i].flatten()
        
        for j in range(len(states)):
            if i == j:
                continue
            
            diff = s_i - states[j].flatten()
            dist_sq = np.sum(diff**2) + 1e-8
            # Repulsive force inversely proportional to distance squared
            repulsion += strength * diff / dist_sq
            
        new_states.append((s_i + repulsion).reshape(states[i].shape))
        
    return new_states

# --- 1d: Convergence Mechanisms ---

def iterate_fixpoint(
    worker_states: List[np.ndarray],
    subspaces: List[np.ndarray],
    merge_fn: Callable[[List[np.ndarray]], np.ndarray],
    max_iter: int = 50,
    epsilon: float = 1e-6,
    diversity_strength: float = 0.0,
) -> Dict[str, Any]:
    """
    Fixed-point iteration: Merge -> Re-project -> Repeat.
    """
    history = {"states": [], "deltas": [], "merged": []}
    states = [s.copy() for s in worker_states]

    for t in range(max_iter):
        history["states"].append([s.copy() for s in states])

        # 1. Merge all worker states
        merged = merge_fn(states)
        history["merged"].append(merged.copy())
        merged_flat = merged.flatten()

        # 2. Re-project and update
        new_states = []
        for i, (state, P) in enumerate(zip(states, subspaces)):
            # Project merged result into worker's subspace
            projected = P @ (P.T @ merged_flat)
            # Smooth update: mix old state with projection
            new_state = 0.7 * state.flatten() + 0.3 * projected
            new_states.append(new_state.reshape(state.shape))

        # 3. Optional Diversity Enforcement
        if diversity_strength > 0:
            new_states = apply_diversity_repulsion(
                new_states, subspaces, diversity_strength
            )

        # 4. Convergence check
        deltas = [np.linalg.norm(n - o) for n, o in zip(new_states, states)]
        max_delta = max(deltas)
        history["deltas"].append(max_delta)

        states = new_states

        if max_delta < epsilon:
            break

    history["states"].append([s.copy() for s in states])
    history["n_iterations"] = t + 1
    history["converged"] = (max_delta < epsilon) if 'max_delta' in locals() else False
    return history

def iterate_energy(
    worker_states: List[np.ndarray],
    subspaces: List[np.ndarray],
    merge_fn: Callable[[List[np.ndarray]], np.ndarray],
    max_iter: int = 50,
    epsilon: float = 1e-6,
    lr: float = 0.1,
    diversity_strength: float = 0.0,
) -> Dict[str, Any]:
    """
    Energy minimization: Gradient descent on worker states to reach consensus.
    """
    history = {"states": [], "deltas": [], "energies": [], "merged": []}
    
    # Convert to torch tensors for autograd
    states_t = [torch.tensor(s, dtype=torch.float32, requires_grad=True)
                for s in worker_states]
    subspaces_t = [torch.tensor(P, dtype=torch.float32) for P in subspaces]

    max_delta = 0.0
    for t in range(max_iter):
        history["states"].append([s.detach().cpu().numpy().copy() for s in states_t])

        # 1. Calculate Energy
        energy = torch.tensor(0.0)

        # Pairwise distance (Consensus force)
        for i in range(len(states_t)):
            for j in range(i + 1, len(states_t)):
                dist = torch.norm(states_t[i] - states_t[j])
                energy = energy + dist ** 2

        # Diversity repulsion (prevents collapse)
        if diversity_strength > 0:
            for i in range(len(states_t)):
                for j in range(i + 1, len(states_t)):
                    cos_sim = F.cosine_similarity(
                        states_t[i].view(1, -1), states_t[j].view(1, -1)
                    )
                    # Penalize high similarity
                    energy = energy - diversity_strength * (1.0 - cos_sim.squeeze())

        history["energies"].append(energy.item())

        # 2. Gradient Descent step
        if energy.requires_grad:
            energy.backward()

        new_states = []
        for i, (s, P) in enumerate(zip(states_t, subspaces_t)):
            with torch.no_grad():
                grad = s.grad if s.grad is not None else torch.zeros_like(s)
                new_s = s - lr * grad
                # Re-project into subspace to maintain architectural constraint
                projected = P @ (P.T @ new_s.flatten())
                new_states.append(projected.reshape(s.shape))
            
            s.grad = None

        # 3. Measurement
        deltas = [torch.norm(n - o).item() for n, o in zip(new_states, states_t)]
        max_delta = max(deltas)
        history["deltas"].append(max_delta)

        # Merge for diagnostics
        merged = merge_fn([s.detach().cpu().numpy() for s in new_states])
        history["merged"].append(merged.copy())

        # Prepare for next iteration
        states_t = [s.clone().detach().requires_grad_(True) for s in new_states]

        if max_delta < epsilon:
            break

    history["states"].append([s.detach().cpu().numpy().copy() for s in states_t])
    history["n_iterations"] = t + 1
    history["converged"] = max_delta < epsilon
    return history

# --- Test Block ---

if __name__ == "__main__":
    print("Testing sim1_helpers.py...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configuration
    dim = 64
    n_subspaces = 2
    subspace_dim = 16
    
    # 1. Generate subspaces
    print(f"Generating {n_subspaces} orthogonal subspaces in {dim}d...")
    subspaces = generate_orthogonal_subspaces(n_subspaces, dim, subspace_dim, orthogonality=1.0)
    
    # Verify orthogonality
    dot_prod = np.abs(np.dot(subspaces[0].T, subspaces[1]))
    print(f"Orthogonality check (max dot product): {np.max(dot_prod):.2e}")
    
    # 2. Initialize workers with random vectors in their subspaces
    workers = [P @ np.random.randn(subspace_dim) for P in subspaces]
    
    # 3. Run Fixpoint Iteration
    print("Running fixpoint iteration...")
    history_fp = iterate_fixpoint(workers, subspaces, merge_average, max_iter=20)
    print(f"Fixpoint converged: {history_fp['converged']} in {history_fp['n_iterations']} iterations.")
    
    # 4. Run Energy Iteration
    print("Running energy minimization...")
    history_en = iterate_energy(workers, subspaces, merge_average, max_iter=20)
    print(f"Energy converged: {history_en['converged']} in {history_en['n_iterations']} iterations.")
