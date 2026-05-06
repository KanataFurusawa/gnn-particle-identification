import numpy as np
import awkward as ak

def generate_dummy_data(n_events=100):
    t = []
    chx = []
    chy = []
    trkp = []
    eHx = []
    eHz = []
    pdg = []
    numPhot = []

    for _ in range(n_events):
        n_hits = np.random.randint(10, 30)

        t.append(np.random.uniform(0, 100, n_hits))
        chx.append(np.random.randint(0, 64, n_hits))
        chy.append(np.random.randint(0, 8, n_hits))

        trkp.append(np.random.uniform(0.5, 5.0))
        eHx.append(np.random.uniform(-50, 50))
        eHz.append(np.random.uniform(-50, 50))

        pdg.append(np.random.choice([211, 321]))
        numPhot.append(n_hits)

    return (
        ak.Array(t),
        ak.Array(chx),
        ak.Array(chy),
        ak.Array(trkp),
        ak.Array(eHx),
        ak.Array(eHz),
        np.array(pdg),
        ak.Array(numPhot),
    )
