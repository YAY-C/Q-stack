import sys, os
import numpy as np
from types import SimpleNamespace
import qstack.spahm.compute_spahm as spahm
import qstack.spahm.guesses as guesses
from qstack import compound

defaults = SimpleNamespace(
    guess='LB',
    model='Lowdin-long-x',
    basis='minao',
    auxbasis='ccpvdzjkfit',
    omod=['alpha', 'beta'],
    elements=["H", "C", "N", "O", "S"],
    cutoff=5.0,
    xc='hf',
    bpath=os.path.dirname(__file__)+'/basis_opt',
  )



def get_chsp(f, n):
    if os.path.isfile(f):
      chsp = np.loadtxt(f, dtype=int).reshape(-1)
      if(len(chsp)!=n):
          print('Wrong lengh of the file', f, file=sys.stderr);
          exit(1)
    else:
        chsp = np.ones(n, dtype=int) * int(f)
    return chsp

def load_mols(xyzlist, charge, spin, basis, printlevel=0, units='ANG'):
    mols = []
    for xyzfile, ch, sp in zip(xyzlist, charge, spin):
        if printlevel>0: print(xyzfile, flush=True)
        mols.append(compound.xyz_to_mol(xyzfile, basis, charge=0 if ch is None else ch, spin=0 if ch is None else sp, unit=units)) #TODO
    if printlevel>0: print()
    return mols

def mols_guess(mols, xyzlist, guess, xc=defaults.xc, spin=None, readdm=False, printlevel=0):
    dms = []
    guess = guesses.get_guess(guess)
    for xyzfile, mol in zip(xyzlist, mols):
        if printlevel>0: print(xyzfile, flush=True)
        if not readdm:
            e, v = spahm.get_guess_orbitals(mol, guess, xc=xc)
            dm   = guesses.get_dm(v, mol.nelec, mol.spin if spin != None else None)
        else:
            dm = np.load(readdm+'/'+os.path.basename(xyzfile)+'.npy')
            if spin and dm.ndim==2:
                dm = np.array((dm/2,dm/2))
        dms.append(dm)
        if printlevel>0: print()
    return dms


def dm_open_mod(dm, omod):
    dmmod = {'sum':   lambda dm: dm[0]+dm[1],
             'diff':  lambda dm: dm[0]-dm[1],
             'alpha': lambda dm: dm[0],
             'beta':  lambda dm: dm[1]}
    return dmmod[omod](dm)


def get_xyzlist(xyzlistfile):
  return np.loadtxt(xyzlistfile, dtype=str, ndmin=1)

def load_reps(f_in, from_list=True, with_labels=False, local=True, summ=False):
    if from_list:
        X_list = get_xyzlist(f_in)
        Xs = [np.load(f_X, allow_pickle=True) for f_X in X_list]
    else:
        Xs = [np.load(f_in, allow_pickle=True)]
    reps = []
    for x in Xs:
        labels = []
        if local == True:
            if  type(x[0,0]) == str:
                if summ:
                    reps.append(x[:,1].sum(axis=0))
                else:
                    reps.extend(x[:,1])
                    labels.extend(x[:,0])
            else:
                reps.extend(x)
        else:
           if type(x[0]) == str:
                reps.append(x[1])
                labels.extend(x[0])
           else:
                reps.extend(x) 
    try:
        reps = np.array(reps, dtype=float)
    except:
        print(reps[0])
        reps = np.array(reps, dtype=float)
        print("Error while loading representations, verify you parameters !")
        exit()
    if with_labels:
        return reps, labels
    else:
        return reps
def add_progressbar(legend='', max_value=100):
    import progressbar
    import time
    widgets=[\
    ' [', progressbar.Timer(), '] ',\
    progressbar.Bar(),\
    ' (', progressbar.ETA(), ') ',]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=max_value).start()
    return bar

def build_reaction(reacts_file, prods_file, local=False, print_level=0):
    reacts = []
    with open(reacts_file, 'r') as r_in:
        lines = r_in.readlines()
        for line in lines:
            line = line.rstrip('\n')
            structs = line.split(' ')
            reacts.append(structs)
    prods = []
    with open(prods_file, 'r') as p_in:
        lines = p_in.readlines()
        for line in lines:
            line = line.rstrip('\n')
            structs = line.split(' ')
            prods.append(structs)
    tot = len(reacts)+len(prods)
    if print_level > 0 : progress = add_progressbar(max_value=tot)
    i = 0
    XR = []
    for rxn in reacts:
        xr = []
        for r in rxn:
            xr.append(load_reps(r, from_list=False, with_labels=False, local=local, summ=True))
        xr = np.array(xr)
        if xr.ndim > 1:
            xr = xr.sum(axis=0)
        XR.append(xr)
        i+=1
        if print_level > 0 : progress.update(i)
    XP = []
    for rxn in prods:
        xp=[]
        for p in rxn:
            xp.append(load_reps(p, from_list=False, with_labels=False, local=local, summ=True))
        xp = np.array(xp)
        if xp.ndim > 1:
            xp = xp.sum(axis=0)
        XP.append(xp)
        i+=1
        if print_level > 0 : progress.update(i)
    XR = np.array(XR)
    XP = np.array(XP)
    rxn = XP - XR
    return rxn