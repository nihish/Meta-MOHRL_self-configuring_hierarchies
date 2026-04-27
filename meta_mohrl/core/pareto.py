"""
Pareto front management for multi-objective optimization in MOHRL-ci.

Implements:
- HQ operator: Pareto dominance filter (retains non-dominated Q-vectors)
- BQ operator: Vectorized Bellman update with Pareto filtering
- Hypervolume computation for Pareto front quality evaluation
- Preference vector sampling from approximated Pareto front
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ParetoSolution:
    """A solution in the Pareto archive."""
    q_values: np.ndarray     # K-dimensional objective values
    omega: np.ndarray        # preference vector that produced this solution
    policy_params: Optional[dict] = None  # optional snapshot of policy


class ParetoFront:
    """Maintains and manages an archive of Pareto-optimal solutions.
    
    From MOHRL-ci Algorithm 2:
    - HQ(s, ω) = {Q(s,a; ω) | ∄ Q' ≻ Q}  (non-dominated filter)
    - Q(s,a; ω) = Σᵢ ωᵢ Qⁱ(s,a)          (scalarized aggregation)
    """

    def __init__(
        self,
        num_objectives: int = 3,
        archive_size: int = 100,
        reference_point: Optional[np.ndarray] = None
    ):
        self.num_objectives = num_objectives
        self.archive_size = archive_size
        self.archive: List[ParetoSolution] = []

        # Reference point for hypervolume (dominated by all solutions of interest)
        if reference_point is None:
            self.reference_point = np.zeros(num_objectives)
        else:
            self.reference_point = reference_point

    def add_solution(self, q_values: np.ndarray, omega: np.ndarray):
        """Add a candidate solution and filter dominated ones."""
        # Check if new solution is dominated by any existing
        if self._is_dominated(q_values):
            return

        # Remove solutions dominated by the new one
        self.archive = [
            sol for sol in self.archive
            if not self._dominates(q_values, sol.q_values)
        ]

        # Add new solution
        self.archive.append(ParetoSolution(
            q_values=q_values.copy(),
            omega=omega.copy()
        ))

        # Trim archive if over capacity (keep most spread solutions)
        if len(self.archive) > self.archive_size:
            self._trim_archive()

    def HQ(self, q_vectors: np.ndarray) -> np.ndarray:
        """Pareto dominance filter (HQ operator).
        
        Retains only non-dominated Q-vectors from a set.
        
        Args:
            q_vectors: [N, K] array of Q-value vectors
        Returns:
            non_dominated: [M, K] array of non-dominated vectors (M <= N)
        """
        if len(q_vectors) == 0:
            return q_vectors

        n = len(q_vectors)
        is_dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                if self._dominates(q_vectors[j], q_vectors[i]):
                    is_dominated[i] = True
                    break

        return q_vectors[~is_dominated]

    def BQ(
        self,
        reward: np.ndarray,
        next_q_vectors: np.ndarray,
        gamma: float,
        omega: np.ndarray
    ) -> np.ndarray:
        """Vectorized Bellman update with Pareto filtering.
        
        BQ(s,a,ω,τ) = r(s,a,τ) + γ E_{s'}[HQ(s',ω,τ)]
        
        Args:
            reward: immediate reward vector [K]
            next_q_vectors: Q-vectors at next state [N, K]
            gamma: discount factor
            omega: preference vector [K]
        Returns:
            updated Q-vector [K]
        """
        # Apply Pareto filter to next-state Q-vectors
        non_dominated = self.HQ(next_q_vectors)

        if len(non_dominated) == 0:
            return reward

        # Select best non-dominated vector under current preference
        scalarized = non_dominated @ omega  # [M]
        best_idx = np.argmax(scalarized)
        best_next_q = non_dominated[best_idx]

        return reward + gamma * best_next_q

    def sample_preference(self) -> np.ndarray:
        """Sample a preference vector ω from the current Pareto front.
        
        If archive is empty, return uniform preference.
        Otherwise, sample from archive or generate a random one.
        """
        if len(self.archive) == 0 or np.random.random() < 0.3:
            # Random preference on simplex
            omega = np.random.dirichlet(np.ones(self.num_objectives))
        else:
            # Sample from archive preferences
            idx = np.random.randint(len(self.archive))
            omega = self.archive[idx].omega
            # Add small noise for exploration
            omega = omega + np.random.normal(0, 0.05, size=omega.shape)
            omega = np.clip(omega, 0.01, None)
            omega = omega / omega.sum()

        return omega

    def compute_hypervolume(self) -> float:
        """Compute hypervolume indicator of the current Pareto front.
        
        HV = λ(∪_{v ∈ P̂} [r_ref, v])
        Uses a simple Monte Carlo approximation for arbitrary dimensions.
        """
        if len(self.archive) == 0:
            return 0.0

        points = np.array([sol.q_values for sol in self.archive])

        # Normalize to [0, 1] for computation
        ref = self.reference_point
        # Only consider points that dominate the reference
        valid = np.all(points > ref, axis=1)
        if not valid.any():
            return 0.0

        valid_points = points[valid]

        if self.num_objectives == 2:
            return self._hv_2d(valid_points, ref)
        else:
            return self._hv_mc(valid_points, ref, n_samples=10000)

    def _hv_2d(self, points: np.ndarray, ref: np.ndarray) -> float:
        """Exact 2D hypervolume computation."""
        # Sort by first objective descending
        sorted_idx = np.argsort(-points[:, 0])
        sorted_points = points[sorted_idx]

        hv = 0.0
        prev_y = ref[1]
        for p in sorted_points:
            if p[1] > prev_y:
                hv += (p[0] - ref[0]) * (p[1] - prev_y)
                prev_y = p[1]
        return hv

    def _hv_mc(
        self,
        points: np.ndarray,
        ref: np.ndarray,
        n_samples: int = 10000
    ) -> float:
        """Monte Carlo hypervolume estimation for K >= 3."""
        # Bounding box
        upper = points.max(axis=0)
        lower = ref

        # Generate random samples in bounding box
        samples = np.random.uniform(lower, upper, size=(n_samples, self.num_objectives))

        # Count samples dominated by at least one Pareto point
        dominated_count = 0
        for s in samples:
            for p in points:
                if np.all(p >= s):
                    dominated_count += 1
                    break

        # HV = volume of bounding box × fraction dominated
        box_volume = np.prod(upper - lower)
        return box_volume * dominated_count / n_samples

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check if a Pareto-dominates b (a ≻ b)."""
        return np.all(a >= b) and np.any(a > b)

    def _is_dominated(self, q_values: np.ndarray) -> bool:
        """Check if q_values is dominated by any archive solution."""
        for sol in self.archive:
            if self._dominates(sol.q_values, q_values):
                return True
        return False

    def _trim_archive(self):
        """Trim archive to max size using crowding distance."""
        if len(self.archive) <= self.archive_size:
            return

        points = np.array([sol.q_values for sol in self.archive])
        distances = self._crowding_distance(points)

        # Keep solutions with highest crowding distance (most spread)
        keep_idx = np.argsort(-distances)[:self.archive_size]
        self.archive = [self.archive[i] for i in keep_idx]

    def _crowding_distance(self, points: np.ndarray) -> np.ndarray:
        """Compute crowding distance for archive trimming."""
        n = len(points)
        if n <= 2:
            return np.full(n, np.inf)

        distances = np.zeros(n)
        for k in range(self.num_objectives):
            sorted_idx = np.argsort(points[:, k])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf

            obj_range = points[sorted_idx[-1], k] - points[sorted_idx[0], k]
            if obj_range == 0:
                continue

            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (
                    (points[sorted_idx[i + 1], k] - points[sorted_idx[i - 1], k])
                    / obj_range
                )
        return distances

    def get_front(self) -> np.ndarray:
        """Return current Pareto front as numpy array [M, K]."""
        if len(self.archive) == 0:
            return np.empty((0, self.num_objectives))
        return np.array([sol.q_values for sol in self.archive])

    def cardinality(self) -> int:
        """Number of solutions in the Pareto front."""
        return len(self.archive)
