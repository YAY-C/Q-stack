#!/usr/bin/env python3

import os
import argparse
import numpy as np
from qstack.tools import correct_num_threads
from . import utils, dmb_rep_bond as dmbb
from .utils import defaults

def bond(mols, dms,
         bpath=defaults.bpath, cutoff=defaults.cutoff, omods=defaults.omod,
         spin=None, elements=None, only_m0=False, zeros=False, split=False, printlevel=0,
         pairfile=None, dump_and_exit=False, no_oriented=False, no_lowdin=False, all_same=False, only_z=[]):

    elements, mybasis, qqs0, qqs4q, idx, M = dmbb.read_basis_wrapper(mols, bpath, only_m0, printlevel,
                                                                     elements=elements, cutoff=cutoff,
                                                                     pairfile=pairfile, dump_and_exit=dump_and_exit, all_same=all_same)
    if spin is None:
        omods = [None]
    qqs = qqs0 if zeros else qqs4q
    maxlen = max([dmbb.bonds_dict_init(qqs[q0], M)[1] for q0 in elements])
    if len(only_z) > 0:
        print(f"Selecting atom-types in {only_z}")
        zinmols = []
        for mol in mols:
            zinmol = [sum(z == np.array(mol.elements)) for z in only_z]
            zinmols.append(sum(zinmol))
        natm  = max(zinmols)
    else:
        natm   = max([mol.natm for mol in mols])
        zinmols = [mol.natm for mol in mols]
    allvec = np.zeros((len(omods), len(mols), natm, maxlen))

    for imol, (mol, dm) in enumerate(zip(mols,dms)):
        if printlevel>0: print('mol', imol, flush=True)
        for iomod, omod in enumerate(omods):
            DM  = utils.dm_open_mod(dm, omod) if spin else dm
            vec = dmbb.repr_for_mol(mol, DM, qqs, M, mybasis, idx, maxlen, cutoff, single=no_oriented, no_lowdin=no_lowdin, only_z=only_z)
            allvec[iomod,imol,:len(vec)] = vec

    if split is False:
        shape  = (len(omods), -1, maxlen)
        atidx  = np.where(np.array([[1]*zin + [0]*(natm-zin) for zin in zinmols]).flatten())
        allvec = allvec.reshape(shape)[:,atidx,:].reshape(shape)
    return allvec


def main():
    parser = argparse.ArgumentParser(description='This program computes the chosen initial guess for a given molecular system.')
    parser.add_argument('--mol',      type=str,            dest='filename',  required=True,                    help='file containing a list of molecular structures in xyz format (single xyz file also accepted)')
    parser.add_argument('--guess',    type=str,            dest='guess',     default=defaults.guess,           help='initial guess type')
    parser.add_argument('--units',     dest='units',     default='Angstrom',               type=str, help=f"The units of the given coordinates files (default: Angstrom")
    parser.add_argument('--basis',    type=str,            dest='basis'  ,   default=defaults.basis,           help='AO basis set (default=MINAO)')
    parser.add_argument('--charge',   type=str,            dest='charge',    default=None,                     help='file with a list of charges')
    parser.add_argument('--spin',     type=str,            dest='spin',      default=None,                     help='file with a list of numbers of unpaired electrons')
    parser.add_argument('--xc',       type=str,            dest='xc',        default=defaults.xc,              help='DFT functional for the SAD guess (default=HF)')
    parser.add_argument('--dir',      type=str,            dest='dir',       default='./',                     help='directory to save the output in (default=current dir)')
    parser.add_argument('--cutoff',   type=float,          dest='cutoff',    default=defaults.cutoff,          help='bond length cutoff (A)')
    parser.add_argument('--bpath',    type=str,            dest='bpath',     default=defaults.bpath,           help='dir with basis sets')
    parser.add_argument('--omod',     type=str,            dest='omod',      default=defaults.omod, nargs='+', help='model for open-shell systems (alpha, beta, sum, diff')
    parser.add_argument('--print',    type=int,            dest='print',     default=0,                        help='printlevel')
    parser.add_argument('--zeros',    action='store_true', dest='zeros',     default=False,                    help='if use a version with more padding zeros')
    parser.add_argument('--split',    action='store_true', dest='split',     default=False,                    help='if split into molecules')
    parser.add_argument('--merge',    action='store_true', dest='merge',     default=True,                     help='if merge different omods')
    parser.add_argument('--onlym0',   action='store_true', dest='only_m0',   default=False,                    help='if use only fns with m=0')
    parser.add_argument('--single',   action='store_true', dest='single',   default=False,                    help='for generating non-oriented representations')
    parser.add_argument('--no-lowdin',   action='store_true', dest='no_lowdin',   default=False,                    help='for generating non-lowdin splitted representations')
    parser.add_argument('--savedm',   action='store_true', dest='savedm',    default=False,                    help='if save dms')
    parser.add_argument('--symbols',  action='store_true', dest='with_symbols',    default=False,             help='if save tuples with (symbol, vec) for all atoms')
    parser.add_argument('--readdm',   type=str,            dest='readdm',    default=None,                     help='dir to read dms from')
    parser.add_argument('--elements', type=str,            dest='elements',  default=None,  nargs='+',         help="the elements contained in the database")
    parser.add_argument('--only-z', type=str,            dest='only_z',  default=[],  nargs='+',         help="restrict the representation to one or several atom types")
    parser.add_argument('--name',       dest='name_out',   required=True,                         type=str, help='name of the output file.')
    parser.add_argument('--pairfile',      type=str,            dest='pairfile',         default=None,                     help='file with atom pairs')
    parser.add_argument('--dump_and_exit', action='store_true', dest='dump_and_exit',    default=False,                    help='if write the pairfile (and exit)')
    parser.add_argument('--same_basis', action='store_true', dest='same_basis',    default=False,                    help='if write the pairfile (and exit)')
    args = parser.parse_args()
    if args.cutoff <= 0: args.cutoff = None
    if args.print>0: print(vars(args))
    correct_num_threads()

    if args.name_out is None:
        args.name_out = os.path.splitext(args.filename)[0]

    xyzlistfile = [args.filename] if args.filename.split('.')[-1] == 'xyz' else args.filename
    if type(xyzlistfile) == list:
        xyzlist = xyzlistfile
        charge  = utils.get_chsp(args.charge, len(xyzlist))
        spin    = utils.get_chsp(args.spin,   len(xyzlist))
    else:
        xyzlist = np.loadtxt(xyzlistfile, dtype=str, usecols=0)
        charge = np.loadtxt(args.filename, usecols=1, dtype=int)
        try :
            spin = np.loadtxt(args.filename, usecols=2, dtype=int)
        except:
            spin = np.array([None]*len(xyzlist))
    mols    = utils.load_mols(xyzlist, charge, spin, args.basis, args.print, units=args.units)
    if args.with_symbols:
        if len(args.only_z) > 0:
            all_atoms   = np.array([z for mol in mols for z in mol.elements if z in args.only_z]).flatten()
        else:
            all_atoms   = np.array([mol.elements for mol in mols]).flatten()
    dms     = utils.mols_guess(mols, xyzlist, args.guess,
                               xc=defaults.xc, spin=args.spin, readdm=args.readdm, printlevel=args.print)
    allvec  = bond(mols, dms, args.bpath, args.cutoff, args.omod,
                   spin=args.spin, elements=args.elements,
                   only_m0=args.only_m0, zeros=args.zeros, split=args.split, printlevel=args.print,
                   pairfile=args.pairfile, dump_and_exit=args.dump_and_exit, no_oriented=args.single, no_lowdin=args.no_lowdin, all_same=args.same_basis, only_z=args.only_z)

    if args.print>1: print(allvec.shape)

    if args.single :
        import sys
        sys.path.insert(0, "/home/calvino/yannick/SPAHM-RHO/rxn/")
        from bond_bagging import bagged_atomic_bonds
        bagged = [bagged_atomic_bonds(allvec[i], xyzlist[i], args.pairfile, bpath=args.bpath, omod=args.omod, same_basis=args.same_basis) for i in range(len(allvec))]
        allvec = bagged
    allvec = np.squeeze(allvec)
    allvec = np.array(allvec, ndmin=2)
    if args.spin:
        if args.merge is False:
            for omod, vec in zip(args.omod, allvec):
                if args.with_symbols: allvec = np.array([(z, v) for v,z in zip(allvec, all_atoms)], dtype=object)
                np.save(args.name_out+'_'+omod, vec)
        else:
            allvec = np.hstack(allvec)
            if args.with_symbols: allvec = np.array([(z, v) for v,z in zip(allvec, all_atoms)], dtype=object)
            np.save(args.name_out+'_'+'_'.join(args.omod), allvec)
    if args.with_symbols: allvec = np.array([(z, v) for v,z in zip(allvec, all_atoms)], dtype=object)
    np.save(args.name_out+'_'+'_'.join(args.omod), allvec)

if __name__ == "__main__":
    main()

