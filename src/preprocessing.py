import awkward as ak
import numpy as np

def clean_photons(t, chx, chy, t_min=5, t_max=90):
    mask = (t >= t_min) & (t <= t_max)
    return t[mask], chx[mask], chy[mask]

def compute_labels(pdg):
    labels = np.full(len(pdg), -1)
    labels[pdg == 321] = 1
    labels[pdg == 211] = 0
    return labels
