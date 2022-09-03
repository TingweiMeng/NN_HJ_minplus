These are the codes for our paper "Neural network architectures using min-plus algebra for solving certain high dimensional optimal control problems and Hamilton-Jacobi PDEs" by Jerome Darbon, Peter M. Dower, and Tingwei Meng. 

There are four files corresponding to the four experimental results. See the filenames and the first line comment in each file.

To run the code, use the following command lines:

- To plot the solution V, run "python3 filename.py --t0 0.25"
- To plot trajectories, run "python3 filename.py --T 5.0 --plot xu --Lx 200"
- To plot errors, run "python3 filename.py --t0 0.75 --plot err --Lx 40"

The meaning of the arguments are in the comments of the files.
