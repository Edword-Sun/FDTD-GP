# run:  mpiexec -np 8 python comm.py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print('My rank is ', rank)