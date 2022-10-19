#!/usr/bin/env python3
import numpy as np
from qstack.regression.kernel_utils import get_kernel, defaults
from qstack.tools import correct_num_threads

def kernel(X, Y=[], sigma=defaults.sigma, akernel=defaults.kernel, gkernel=defaults.gkernel):
    if len(Y) == 0 :
        Y = X
    kernel = get_kernel(akernel)
    K = kernel(X, Y, 1.0/sigma)
    return K

def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='This program computes kernel.')
    parser.add_argument('--x',      type=str,   dest='repr',      required=True,           help='path to the representations file')
    parser.add_argument('--sigma',  type=float, dest='sigma',     default=defaults.sigma,  help='sigma hyperparameter (default='+str(defaults.sigma)+')')
    parser.add_argument('--akernel', type=str,   dest='akernel',    default=defaults.kernel, help='kernel type (G for Gaussian, L for Laplacian, myL for Laplacian for open-shell systems) (default '+defaults.kernel+')')
    parser.add_argument('--gkernel', type=str,   dest='gkernel',    default=None, help='global kernel type (agv for average, rem for REMatch kernel, None for local kernels) (default None)')
    parser.add_argument('--dir',    type=str,   dest='dir',       default='./',            help='directory to save the output in (default=current dir)')
    parser.add_argument('--ll',     action='store_true', dest='ll', default=False,  help='if correct for the numper of threads')
    args = parser.parse_args()
    print(vars(args))
    if(args.ll): correct_num_threads()
    X = np.load(args.repr)
    K = kernel(X, args.sigma, args.kernel, args.gkernel)
    np.save(args.dir+'/K_'+os.path.splitext(os.path.basename(args.repr))[0]+'_'+args.kernel+'_'+"%e"%args.sigma, K)

if __name__ == "__main__":
    main()
