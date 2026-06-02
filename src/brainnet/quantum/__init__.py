"""Rete ibrida quanvoluzionale (late fusion) per BrainNet, su Qiskit 2.2.

Adattata dall'architettura della tesi di dottorato: stesso circuito
quanvoluzionale (angle encoding + ansatz hardware-efficient + misura Z,
gradienti via parameter-shift), riusata sul dataset PET C-DOPA con le slice
peri-striatali come canali di ingresso.
"""
