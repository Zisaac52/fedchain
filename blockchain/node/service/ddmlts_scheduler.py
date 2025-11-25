import math
import random
from typing import Dict, List, Tuple

import numpy as np


class DDMLTSScheduler:
    """
    Implements a lightweight variant of the DDMLTS task scheduler described in the paper.
    It consumes client state vectors (CPU/GPU/network indicators) and assigns per-client
    workloads so that faster clients receive proportionally more batches/local epochs.
    """

    def __init__(self, conf: Dict):
        self.conf = conf or {}
        # Cache of the last computed plan for debugging/visualisation.
        self._last_plan: Dict[str, Dict] = {}

    # ------------------------------------------------------------------ Public API
    def build_plan(self, state_vectors: Dict[str, Tuple[float, ...]]) -> Dict[str, Dict]:
        """
        Args:
            state_vectors: mapping client_id -> tuple describing its state vector.
                           The last element is interpreted as throughput (samples/sec).
        Returns:
            dict client_id -> assignment metadata
        """
        scheduler_mode = self._get_conf('scheduler_mode', 'fedavg').lower()
        if scheduler_mode not in {'ddmlts_a', 'ddmlts_b'}:
            self._last_plan = {}
            return {}

        prepared = self._prepare_vectors(state_vectors)
        if not prepared:
            self._last_plan = {}
            return {}

        if scheduler_mode == 'ddmlts_b':
            cluster_labels = self._run_kmeans(prepared)
        else:
            cluster_labels = {entry['client_id']: 0 for entry in prepared}

        plan = {}
        for entry in prepared:
            cid = entry['client_id']
            cluster_id = cluster_labels.get(cid, 0)
            assignment = self._compute_assignment(entry, cluster_id, prepared, cluster_labels)
            plan[cid] = assignment
        self._last_plan = plan
        return plan

    def last_plan(self) -> Dict[str, Dict]:
        return self._last_plan

    # ------------------------------------------------------------------ Helpers
    def _prepare_vectors(self, state_vectors: Dict[str, Tuple[float, ...]]):
        data = []
        for client_id, vec in state_vectors.items():
            if vec is None:
                continue
            if not isinstance(vec, (list, tuple)):
                continue
            # Throughput / processing capability is taken as the last value.
            perf = float(vec[-1]) if len(vec) else 1.0
            if perf <= 0:
                perf = 1.0
            padded = list(vec)
            # Expand to a fixed dimension for clustering
            while len(padded) < 4:
                padded.append(0.0)
            data.append({
                'client_id': client_id,
                'vector': np.array(padded, dtype=np.float32),
                'speed': perf
            })
        return data

    def _run_kmeans(self, prepared: List[Dict]) -> Dict[str, int]:
        points = np.stack([entry['vector'] for entry in prepared])
        num_points = points.shape[0]
        k = max(1, min(self._get_conf('ddmlts_cluster_count', 2), num_points))
        centroids = self._init_kmeans_plus_plus(points, k)
        max_iter = 25
        labels = np.zeros(num_points, dtype=np.int32)
        for _ in range(max_iter):
            # Assign step
            distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            # Update step
            for idx in range(k):
                members = points[labels == idx]
                if len(members) == 0:
                    centroids[idx] = points[random.randint(0, num_points - 1)]
                else:
                    centroids[idx] = np.mean(members, axis=0)
        cluster_map = {}
        for entry, label in zip(prepared, labels.tolist()):
            cluster_map[entry['client_id']] = int(label)
        return cluster_map

    def _init_kmeans_plus_plus(self, points: np.ndarray, k: int) -> np.ndarray:
        centroids = []
        # pick first centroid randomly
        centroids.append(points[random.randint(0, points.shape[0] - 1)])
        while len(centroids) < k:
            dist_sq = np.min(
                np.linalg.norm(points[:, None, :] - np.array(centroids)[None, :, :], axis=2) ** 2,
                axis=1
            )
            probs = dist_sq / np.sum(dist_sq)
            cumulative = np.cumsum(probs)
            r = random.random()
            next_idx = np.searchsorted(cumulative, r)
            centroids.append(points[next_idx])
        return np.stack(centroids)

    def _compute_assignment(self, entry, cluster_id: int, prepared, cluster_labels):
        base_epoch = max(1, int(self._get_conf('local_epoch', 1)))
        alpha = float(self._get_conf('ddmlts_alpha', 1.0))
        tau_ratio = float(self._get_conf('ddmlts_tau_ratio', 0.25))
        tau_ratio = min(1.0, max(0.0, tau_ratio))

        speed = entry['speed']
        cluster_speeds = [
            item['speed']
            for item in prepared
            if cluster_labels.get(item['client_id'], 0) == cluster_id
        ]
        cluster_avg = np.mean(cluster_speeds) if cluster_speeds else speed
        relative_speed = speed / cluster_avg if cluster_avg else 1.0
        workload_scale = relative_speed ** alpha
        local_epoch = max(1, int(round(base_epoch * workload_scale)))

        # Smaller tau -> more fine-grained batches for stragglers.
        micro_batches = max(1, int(math.ceil(local_epoch * tau_ratio)))
        estimated_time = local_epoch / max(1e-6, speed)

        return {
            'cluster_id': cluster_id,
            'relative_speed': relative_speed,
            'local_epoch': local_epoch,
            'micro_batches': micro_batches,
            'estimated_time': estimated_time,
            'speed': speed,
        }

    def _get_conf(self, key, default):
        value = self.conf.get(key, default)
        return value
