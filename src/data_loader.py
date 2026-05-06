import uproot
import awkward as ak
import numpy as np

def load_top_data(filename):
    file = uproot.open(filename)
    tree = file["topML"]

    t      = tree["dgtTime"].array(library="ak")
    ch_x   = tree["dgtPixelCol"].array(library="ak")
    ch_y   = tree["dgtPixelRow"].array(library="ak")
    trkp   = tree["track_p"].array(library="ak")
    pdg    = tree["PDG"].array(library="np")
    numPhot= tree["numPhot"].array(library="ak")

    if "extHit" in tree.keys():
        ext_hit = tree["extHit"].array(library="ak")
        eHit_x  = ext_hit["eHit_x"]
        eHit_z  = ext_hit["eHit_z"]
    else:
        raise KeyError("extHit branch not found")

    return t, ch_x, ch_y, trkp, eHit_x, eHit_z, pdg, numPhot
