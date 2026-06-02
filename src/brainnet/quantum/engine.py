"""Backend Qiskit 2.2 e motore parameter-shift per il layer quanvoluzionale.

Espone:
  * BackendManager  — sceglie l'Estimator (StatevectorEstimator esatto per il
                       dev, AerEstimator V2 per la velocita').
  * QuantumEngine   — forward e forward+gradiente con la regola parameter-shift
                       (due valutazioni per parametro), con batching PUB e
                       splitting opzionale in K chunk per il parallelismo.
"""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit.primitives import StatevectorEstimator

from .circuit import QuanvCircuitBuilder

QISKIT_VERSION = tuple(int(x) for x in qiskit.__version__.split(".")[:2])
assert QISKIT_VERSION >= (2, 0), f"Richiesto Qiskit >= 2.0, trovato {qiskit.__version__}"

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator
    HAS_AER = True
except ImportError:
    HAS_AER = False


class BackendManager:
    """Gestisce l'Estimator (primitives V2) per Qiskit 2.x."""

    def __init__(self, backend_type: str = "statevector", seed: int = 42,
                 aer_parallel: int = 0):
        self.backend_type = backend_type
        self.aer_parallel = aer_parallel
        self.rng = np.random.default_rng(seed)
        self.estimator = None
        self.backend_name = "unknown"

    def initialize(self) -> "BackendManager":
        if self.backend_type == "aer" and HAS_AER:
            backend = AerSimulator(method="statevector")
            if self.aer_parallel > 0:
                backend.set_options(max_parallel_experiments=self.aer_parallel,
                                    max_parallel_threads=self.aer_parallel)
            try:
                self.estimator = AerEstimator(backend)
            except TypeError:
                self.estimator = AerEstimator()
            self.backend_name = f"AerSimulator(statevector, parallel={self.aer_parallel})"
        else:
            self.estimator = StatevectorEstimator()
            self.backend_name = "StatevectorEstimator"
        return self


class QuantumEngine:
    """Motore di esecuzione con batching PUB e splitting in K chunk."""

    SHIFT = np.pi / 2

    def __init__(self, num_qubits: int, backend_manager: BackendManager,
                 ansatz: str = "rzrx", measure_qubit: int = 0,
                 n_parallel_chunks: int = 1):
        self.bm = backend_manager
        self.n = num_qubits
        self.K = max(1, n_parallel_chunks)

        self.builder = QuanvCircuitBuilder(num_qubits, ansatz, measure_qubit)
        self.circuit = self.builder.circuit
        self.observables = self.builder.observables
        self.num_weights = self.builder.num_weights
        self.num_observables = self.builder.num_observables

        self.total_estimator_calls = 0
        self.total_pub_count = 0

    # ── PUB helpers ──
    def _run_pubs(self, pubs):
        res = self.bm.estimator.run(pubs).result()
        self.total_estimator_calls += 1
        self.total_pub_count += len(pubs)
        return res

    def _effective_K(self, N):
        return self.K if N >= self.K else 1

    def _make_chunked_pubs(self, pv, K_eff):
        if K_eff == 1:
            return [(self.circuit, self.observables, pv[:, np.newaxis, :])]
        chunks = np.array_split(pv, K_eff, axis=0)
        return [(self.circuit, self.observables, c[:, np.newaxis, :]) for c in chunks]

    def _gather(self, results, start, K_eff, N):
        if K_eff == 1:
            return np.array(results[start].data.evs)[:N]
        parts = [np.array(results[start + k].data.evs) for k in range(K_eff)]
        return np.concatenate(parts, axis=0)[:N]

    # ── API ──
    def forward(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        pv, N = self.builder.build_param_array(inputs, weights)
        K_eff = self._effective_K(N)
        results = self._run_pubs(self._make_chunked_pubs(pv, K_eff))
        return self.builder.parse_output(self._gather(results, 0, K_eff, N), N)

    def forward_and_gradient(self, inputs, weights, skip_input_grad=False):
        """Forward + jacobiani parameter-shift rispetto a pesi e (opz.) input."""
        N = inputs.shape[0]
        nw, n, n_obs, shift = self.num_weights, self.n, self.num_observables, self.SHIFT
        K_eff = self._effective_K(N)

        pubs = []
        pv_fwd, _ = self.builder.build_param_array(inputs, weights)
        pubs.extend(self._make_chunked_pubs(pv_fwd, K_eff))

        for j in range(nw):
            wp = weights.copy(); wp[j] += shift
            wm = weights.copy(); wm[j] -= shift
            pubs.extend(self._make_chunked_pubs(self.builder.build_param_array(inputs, wp)[0], K_eff))
            pubs.extend(self._make_chunked_pubs(self.builder.build_param_array(inputs, wm)[0], K_eff))

        if not skip_input_grad:
            for i in range(n):
                xp = inputs.copy(); xp[:, i] += shift
                xm = inputs.copy(); xm[:, i] -= shift
                pubs.extend(self._make_chunked_pubs(self.builder.build_param_array(xp, weights)[0], K_eff))
                pubs.extend(self._make_chunked_pubs(self.builder.build_param_array(xm, weights)[0], K_eff))

        results = self._run_pubs(pubs)
        fwd = self._gather(results, 0, K_eff, N)

        grad_w = np.zeros((nw, N, n_obs))
        for j in range(nw):
            ev_p = self._gather(results, K_eff * (1 + 2 * j), K_eff, N)
            ev_m = self._gather(results, K_eff * (2 + 2 * j), K_eff, N)
            grad_w[j] = (ev_p - ev_m) / 2.0

        if skip_input_grad:
            grad_x = None
        else:
            grad_x = np.zeros((n, N, n_obs))
            base = 1 + 2 * nw
            for i in range(n):
                ev_p = self._gather(results, K_eff * (base + 2 * i), K_eff, N)
                ev_m = self._gather(results, K_eff * (base + 2 * i + 1), K_eff, N)
                grad_x[i] = (ev_p - ev_m) / 2.0

        return fwd, grad_w, grad_x
