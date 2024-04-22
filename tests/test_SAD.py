#!/usr/bin/env python3

import os
import numpy as np
from qstack import compound
from qstack.spahm.rho import atom


def test_water():
    path = os.path.dirname(os.path.realpath(__file__))
    mol = compound.xyz_to_mol(path+'/data/H2O.xyz', 'minao', charge=0, spin=None)
    print(mol.spin)

    Xcore = atom.get_repr(mol, ["H", "O"], 0, None, dm=None,
                      guess='core', model='lowdin-long-x', auxbasis='ccpvdzjkfit')
    
    Xsad = atom.get_repr(mol, ["H", "O"], 0, None, dm=None,
                      guess='sad', model='lowdin-long-x', auxbasis='ccpvdzjkfit')

    print(Xsad.shape, Xcore.shape)
    #assert(Xsad.shape == Xcore.shape)
    for a, a_true in zip(Xcore, Xsad):
        #assert(a[0] == a_true[0])                        # atom type
        print(a[0] == a_true[0])                        # atom type
        print("{:.3e}".format(np.abs(a[1]-a_true[1]).mean()))   # atom representations
        #assert(np.linalg.norm(a[1]-a_true[1]) < 1e-08)   # atom representations


if __name__ == '__main__':
    test_water()
