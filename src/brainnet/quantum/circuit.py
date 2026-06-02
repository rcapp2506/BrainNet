"""Circuito quanvoluzionale parametrico (Qiskit 2.2, primitives V2).

Per ogni patch k x k (qui 3x3 -> n=9 qubit):
    RY(x_i)            angle encoding dei pixel normalizzati in [0, pi]
    H(i)               superposizione
    CX(i, i+1)         catena entangling in avanti
    ANSATZ(w)          layer variazionale addestrabile
    CX(i+1, i) inversa catena entangling indietro
    <Z> sul qubit 0    -> uno scalare per patch (la feature quanvoluzionale)

Scelta dell'ansatz variazionale (parametro `ansatz`):
  "rzrx"  RZ(w)+RX(w) per filo (2n pesi) — fedele all'Eq. hardware-efficient
          della tesi; pesi ADDESTRABILI (RX non commuta con l'osservabile Z).
  "ry"    RY(w) per filo (n pesi) — variante "lean", pesi addestrabili, minima
          capacita': adatta al regime small-n (poche decine di pazienti).
  "rz"    RZ(w) per filo (n pesi) — RIPRODUCE il codice originale, ma con
          osservabile Z i gradienti dei pesi sono NULLI: feature map FISSA,
          pesi non addestrabili. Mantenuto solo per riproducibilita'.

Ansatz locale e shallow: tiene i gradienti lontani dai barren plateau.
"""
from __future__ import annotations

import warnings
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuanvCircuitBuilder:
    def __init__(self, num_qubits: int, ansatz: str = "rzrx", measure_qubit: int = 0):
        self.n = num_qubits
        self.ansatz = ansatz
        self.measure_qubit = measure_qubit
        self.num_weights = 2 * num_qubits if ansatz == "rzrx" else num_qubits

        if ansatz == "rz":
            warnings.warn("ansatz='rz': con osservabile Z i pesi hanno gradiente "
                          "nullo (feature map fissa, non addestrabile).")

        self.input_params = ParameterVector("x", self.n)
        self.weight_params = ParameterVector("w", self.num_weights)
        self.circuit = self._build()

        self.observables = [
            SparsePauliOp.from_sparse_list(
                [("Z", [self.measure_qubit], 1.0)], num_qubits=self.n)
        ]
        self.num_observables = 1

        self.param_list = list(self.circuit.parameters)
        self.num_params = len(self.param_list)
        idx = {p: i for i, p in enumerate(self.param_list)}
        self.input_indices = [idx[self.input_params[i]] for i in range(self.n)]
        self.weight_indices = [idx[self.weight_params[i]] for i in range(self.num_weights)]

    def _build(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n)
        for i in range(self.n):
            qc.ry(self.input_params[i], i)
        for i in range(self.n):
            qc.h(i)
        for i in range(self.n - 1):
            qc.cx(i, i + 1)
        self._variational(qc)
        for i in range(self.n - 2, -1, -1):
            qc.cx(i, i + 1)
        return qc

    def _variational(self, qc: QuantumCircuit) -> None:
        n = self.n
        if self.ansatz == "rzrx":
            for i in range(n):
                qc.rz(self.weight_params[i], i)
                qc.rx(self.weight_params[n + i], i)
        elif self.ansatz == "ry":
            for i in range(n):
                qc.ry(self.weight_params[i], i)
        elif self.ansatz == "rz":
            for i in range(n):
                qc.rz(self.weight_params[i], i)
        else:
            raise ValueError(f"ansatz sconosciuto: {self.ansatz!r}")

    def build_param_array(self, inputs_2d: np.ndarray, weights_1d: np.ndarray):
        N = inputs_2d.shape[0]
        params = np.zeros((N, self.num_params), dtype=np.float64)
        for j, k in enumerate(self.weight_indices):
            params[:, k] = weights_1d[j]
        for j, k in enumerate(self.input_indices):
            params[:, k] = inputs_2d[:, j]
        return params, N

    @staticmethod
    def parse_output(evs: np.ndarray, N: int) -> np.ndarray:
        return evs[:N]
