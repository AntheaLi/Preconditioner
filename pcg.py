from cgitb import enable
from marshal import load
import sys
import math
import time
import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import DataLoader
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate
from tqdm import tqdm

from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve_triangular, aslinearoperator

from sys import path
path.append('../../python/example/phys_gnn/')
path.append('../../python/py_phys_sim/')
path.append('../../python/')

torch.set_num_threads(64)

def ic( A ):
    mat = np.copy( A )
    n = mat.shape[1]
    
    for k in range( n ):
        mat[k,k] = math.sqrt( mat[k,k] )
        for i in range(k+1, n):
            if mat[i,k] != 0:
                mat[i,k] = mat[i,k] / mat[k,k]
        for j in range(k+1, n):
            for i in range(j, n):
                if mat[i,j] != 0:
                    mat[i,j] = mat[i,j] - mat[i,k] * mat[j,k]
    for i in range(n):
        for j in range(i+1, n):
            mat[i,j] = 0
    
    return mat


# implementing IC based on TAO's c++ code
def ic_torch_st( A , device='cuda:0'):
    A = A.to(device)
    ic = A.clone()
    A_copy = A.clone()
    n = A.shape[0]
    for j in tqdm(range(n)): 
        ic[j,j] = torch.sqrt(ic[j,j])
        i_ = ic[j+1:,j].nonzero()
        assert i_.shape[-1] == 1
        i_ = i_.reshape(-1)
        i_ += (j+1)
        for i in i_:
            ic[i, j]  = ic[i, j] - (ic[i, :]*ic[j, :]).reshape(-1)[:j].sum()
            ic[i, j] = ic[i, j] / ic[j, j]

    ic = torch.tril(ic)

    return ic


# IC implementation with tensorized parallel speedup: --> should use this for comparison (this is faster)
def ic_torch_optimize_st( A , device='cuda:0'):
    A = A.to(device)
    ic = A.clone()
    A_copy = A.clone()
    n = A.shape[0]
    
    for j in tqdm(range(n)): 
        ic[j,j] = torch.sqrt(ic[j,j])
        i_ = ic[j+1:,j].nonzero()
        if i_.shape[0] > 0:
            assert i_.shape[-1] == 1
            i_ = i_.reshape(-1)
            i_ += (j+1)
            inner_sum = (ic[i_, :]*ic[j, :])
            inner_sum = inner_sum[:, :j].sum(dim=-1)            
            ic[i_, j]  = ic[i_, j] - inner_sum
            ic[i_, j] = ic[i_, j] / ic[j, j]

    ic = torch.tril(ic)

    return ic


def ic_np( A ):
    A = A.astype(np.float128)
    ic = A.copy()
    print(A)
    n = A.shape[0]
    for j in range(n): 
        assert ic[j, j] > 0, f"{j} {ic[j, j]} "

        ic[j,j] = np.sqrt(ic[j,j])
        # print(A[j, j])
        i_ = (ic[j+1:,j].nonzero())[0]
        i_ = i_.reshape(-1)
        i_ += (j+1)
        for i in i_:
            print((ic[i, :].reshape(-1)).shape)
            inner_sum = np.multiply(ic[i, :].reshape(-1), ic[j, :].reshape(-1)).reshape(-1)[:j].sum()
            # if ic[i, j] < inner_sum: print(i, j, ic[i, j], inner_sum)
            ic[i, j]  = ic[i, j] - inner_sum
            assert ic[j, j] > 0.0, f"diagonal negative"
            ic[i, j] = ic[i, j] / ic[j, j]
            ic[i, i] = ic[i, i] - (ic[i, j] ** 2)
            assert np.all(np.diag(A) > 0.0), f" after: number at {i} {j} is {A[i, i]}, {ic[i, j]} {ic[j, j]} {inner_sum} "

    ic = torch.tril(ic)

    return ic




def ic(A):
    n = A.shape[0]
    A_copy = A.clone()
    ic = A.copy()
    for j in range(n):
        ic[j, j] = np.sqrt(A[j,j])
        for i in range(j+1, n):
            for k in range(j):
                ic[i, j] = ic[i, j] - ic[i, k] * ic[j, k]
            ic[i, j] = ic[i, j] / ic[j, j]
            A[i, i] = A[i, i] - ic[i, j] * ic[i, j]
    
    ic = np.tril(ic)
    return ic


def ic_torch_old( A , device='cuda:0'):
    A = A.to(device)
    ic = A.copy()
    n = A.shape[0]
    for k in range(n): 
        ic[k,k] = torch.sqrt(A[k,k])
        i_ = ic[k+1:,k].nonzero() 
        assert i_.shape[-1] == 1
        i_ = i_.view(-1)
        if i_.shape[0] > 0:
            i_ = i_ + (k+1)
            ic[i_,k] = ic[i_,k]/ic[k,k]
            for j in i_:
                i2_ = ic[j:n,j].nonzero()
                assert i2_.shape[-1] == 1
                i2_ = i2_.view(-1)
                # if len(i2_) > 0:
                if i2_.shape[0] < 0:
                    i2_ = i2_ + j
                    factor = ic[j,k]
                    ic[i2_, j]  = ic[i2_, j] - ic[i2_,k]*factor
            # for j in 
    
    ic = torch.tril(ic)

    return ic


# jacobi based on C++ code
def jacobi(A):
    mat = np.zeros(A.shape)
    for i in range(A.shape[0]):
        mat[i, i] = 1.0/A[i, i]
    return mat


# faster jacobi (tensorized) --> should use this for comparison this is faster
def jacobi_torch(A, device='cuda:0'):
    mat = torch.eye(A.shape[0], device=device)
    mat = mat * (1.0 / torch.diagonal(A).to(device) )
    return mat



def gs_torch(A, device='cuda:0'):
    L = torch.tril(A)
    U = torch.triu(A, diagonal=1)
    return L, U


def cholesky_eigen(A):
    ''' this is complete cholesky decomposition
    '''
    import eigenpy
    L = eigenpy.LLT(A.numpy())
    L = L.matrixL()
    L = np.array(L)
    return L

def cg( A, b, options, model=None, device='cuda:0'):
    A = A.to(device)
    b = b.to(device)
    start_cg_time = time.time()
    rel_tol = options['abs_tol']
    abs_tol = options['rel_tol']
    rel_tols = [1e-4, 1e-6, 1e-8, 1e-10]
    abs_tols = [ 1e-4, 1e-6, 1e-8, 1e-10]
    max_iter = options['max_iter']
    first_flag = True
    second_flag = True
    preconditioner = options['precondition_matrix']

    if 'x' not in options:
        x = torch.zeros((A.shape[0], 1), device=A.device).double()
    else:
        x = options['x']

    r = A @ x - b
    y = torch.mm( preconditioner, r )
    p = -y
    convergent_iterations = {}
    for i in range(max_iter):
       
        Ap       = torch.mm( A , p )
        alpha    = torch.mm(r.T, y)/torch.mm( p.T, Ap )
        x        = x + alpha * p
        r_next   = r + alpha * Ap


        if first_flag and torch.abs(r_next).max() <= torch.abs(b).max() * rel_tols[0] + abs_tols[0]:
            end_cg_time = time.time()
            print(f'1e-4 Pcg Converged in {i} steps time ')
            convergent_iterations['1e-4'] = i
            first_flag = False
        if second_flag and torch.abs(r_next).max() <= torch.abs(b).max() * rel_tols[1] + abs_tols[1]:
            end_cg_time = time.time()
            print(f'1e-6 Pcg Converged in {i} steps time ')
            convergent_iterations['1e-6'] = i
            second_flag = False
        if  torch.abs(r_next).max() <= torch.abs(b).max() * rel_tols[2] + abs_tols[2]:
            end_cg_time = time.time()
            print(f'1e-8 Pcg Converged in {i} steps time {end_cg_time - start_cg_time}')
            convergent_iterations['1e-8'] = i
            return i, convergent_iterations
        
        y_next   = torch.mm( preconditioner, r_next )
        beta     = torch.mm(y_next.T, (r_next - r))/ r.T.mm(y) # Polak-Ribiere
        p        = -y_next + beta * p
        y = y_next
        r = r_next

    if i >= (max_iter-1):
        print('Convergence failed.')
        
    end_cg_time = time.time()


    return 1000000, convergent_iterations



def cg_np(A, b, options):
    ''' single threaded numpy conjugate gradient that supports sparse triangular solver using the naive scipy sparse implementation
    '''
    A = A.cpu().numpy()
    b = b.cpu().numpy()
    start_cg_time = time.time()
    solve_triag = options['sptriangular']
    solve_sparse = options['solve_sparse']

    residual = []
    abs_tols = options['abs_tols']
    rel_tols = options['rel_tols']
    tols_dict = options['tol_dict']
    max_iter = options['max_iter']
    flags = [True for x in rel_tols]
    max_iter = options['max_iter']
    preconditioner = options['precondition_matrix']

    if 'x' not in options:
        x = np.zeros((A.shape[0], 1))
    else:
        x = options['x'].cpu().numpy()
        

    r =  A.dot(x) - b
    if solve_triag:
        r_new = r.reshape(-1, 1)
        r_new = np.concatenate([r_new, np.zeros_like(r_new)], axis=-1)
        y0 = spsolve_triangular(preconditioner.transpose(), r_new, lower=False)
        y = spsolve_triangular(preconditioner, y0, lower=True)
        y = y[:, 0].reshape(-1,1)
    elif solve_sparse:
        r_new = r
        y = preconditioner.matvec(r_new)
        y = y.reshape(-1)
    else:
        y = np.dot( preconditioner, r )
    p = -y

    convergent_iterations = {}
    convergent_time = {}
    for i in range(max_iter):
        Ap       = np.dot( A , p )
        alpha    = np.dot(r.T, y)/np.dot( p.T, Ap )
        x        = x + alpha * p
        r_next   = r.reshape(-1) + alpha.reshape(-1) * Ap.reshape(-1)

        r = r.reshape(-1)
        y = y.reshape(-1)
        
        for j in range(len(rel_tols)-1):
            if flags[j] and np.abs(r_next).max() <= np.abs(b).max() * rel_tols[j] + abs_tols[j]:
                # delta_time =  time.time() - iter_start_time
                delta_time =  time.time() - start_cg_time
                iter_start_time = time.time()
                print(f'{tols_dict[j]} Pcg Converged in {i} steps ')
                convergent_iterations[tols_dict[j]] = i
                convergent_time[tols_dict[j]] = delta_time
                flags[j] = False
                
        if  np.abs(r_next).max() <= np.abs(b).max() * rel_tols[-1] + abs_tols[-1]:
            delta_time =  time.time() - start_cg_time
            iter_start_time = time.time()
            print(f'{tols_dict[-1]} Pcg Converged in {i} steps {iter_start_time - start_cg_time}')
            convergent_iterations[tols_dict[-1]] = i
            convergent_time[tols_dict[-1]] = delta_time
            return i, convergent_iterations, convergent_time, residual
        
        if solve_triag:
            r_next_new = r_next.reshape(-1, 1)
            r_next_new = np.concatenate([r_next, np.zeros_like(r_next_new)], axis=-1)
            y0 = spsolve_triangular(preconditioner.transpose(), r_next_new, lower=False)
            y_next = spsolve_triangular(preconditioner, y0, lower=True)
            y_next = y_next[:, 0].reshape(-1, 1)
        elif solve_sparse:
            r_new = r_next.reshape(-1)#.reshape(-1, 1)
            y_next = preconditioner.matvec(r_new)
            y_next = y_next#.reshape(-1,1)
        else:
            y_next = np.dot( preconditioner, r_next )
    
        beta     = np.dot(y_next.T, (r_next - r))/ np.dot(r.T, y) # Polak-Ribiere
        p        = -y_next + beta * p
        y = y_next
        r = r_next

    if i >= (max_iter-1):
        print('Convergence failed.')
        
    end_cg_time = time.time()


    return 1000000, convergent_iterations, convergent_time, residual




def cg_torch(A, b, options, model=None, device='cuda:0', num_threads=64, plot=False):
    ''' pytorch implementation of conjugate gradient that supports gpu/cpu and multithreading 
        use max as stopping criteria
    '''
    torch.set_num_threads(num_threads)

    start_cg_time = time.time()
    iter_start_time = time.time()

    abs_tols = options['abs_tols']
    rel_tols = options['rel_tols']
    tol_dict = options['tol_dict']
    max_iter = options['max_iter']
    flags = [True for x in rel_tols]
    preconditioner = options['precondition_matrix']
    x = options['x']

    residual=[]

    A = A.to(device).float()
    b = b.to(device).float()
    x = x.to(device).float()
    preconditioner = preconditioner.to(device).float()
    

    r = A @ x - b
    y = torch.mm( preconditioner, r )
    p = -y
    convergent_iterations = {}
    convergent_time = {}
    for i in range(max_iter):
        Ap       = torch.mm( A , p )
        alpha    = torch.mm(r.T, y)/torch.mm( p.T, Ap )
        x        = x + alpha * p
        r_next   = r + alpha * Ap

        if plot:
            res = torch.abs(r_next).max()
            delta_time =  time.time() - iter_start_time
            iter_start_time = time.time()
            residual.append((res.item(), delta_time))

        for j in range(len(rel_tols)-1):
            if flags[j] and torch.abs(r_next).max() <= torch.abs(b).max() * rel_tols[j] + abs_tols[j]:
                delta_time =  time.time() - start_cg_time
                iter_start_time = time.time()
                print(f'{tol_dict[j]} Pcg Converged in {i} steps ')
                convergent_iterations[tol_dict[j]] = i
                convergent_time[tol_dict[j]] = delta_time
                flags[j] = False
                
        if  torch.abs(r_next).max() <= torch.abs(b).max() * rel_tols[-1] + abs_tols[-1]:
            delta_time =  time.time() - start_cg_time
            iter_start_time = time.time()
            print(f'{tol_dict[-1]} Pcg Converged in {i} steps {iter_start_time - start_cg_time}')
            convergent_iterations[tol_dict[-1]] = i
            convergent_time[tol_dict[-1]] = delta_time
            return i, convergent_iterations, convergent_time, residual
        
        y_next   = torch.mm( preconditioner, r_next)
        beta     = torch.mm(y_next.T, (r_next - r))/ r.T.mm(y) # Polak-Ribiere
        p        = -y_next + beta * p
        y = y_next
        r = r_next

    if i >= (max_iter-1):
        print('Convergence failed.')
        
    end_cg_time = time.time()


    return 1000000, convergent_iterations, convergent_time, residual


def cg_torch_mean(A, b, options, model=None, device='cuda:0', num_threads=64, plot=False):
    ''' pytorch implementation of conjugate gradient that supports gpu/cpu and multithreading 
        use mean as stopping criteria
    '''
    torch.set_num_threads(num_threads)

    start_cg_time = time.time()
    iter_start_time = time.time()

    abs_tols = options['abs_tols']
    rel_tols = options['rel_tols']
    tol_dict = options['tol_dict']
    max_iter = options['max_iter']
    flags = [True for x in rel_tols]
    preconditioner = options['precondition_matrix']
    x = options['x']

    residual=[]

    A = A.to(device)
    b = b.to(device)
    x = x.to(device)
    preconditioner = preconditioner.to(device)

    r = A @ x - b
    y = torch.mm( preconditioner, r )
    p = -y

    convergent_iterations = {}
    convergent_time = {}
    for i in range(max_iter):

        Ap       = torch.mm( A , p )
        alpha    = torch.mm(r.T, y)/torch.mm( p.T, Ap )
        x        = x + alpha * p
        r_next   = r + alpha * Ap

        if plot:
            res = torch.abs(r_next).max()
            delta_time =  time.time() - iter_start_time
            iter_start_time = time.time()
            residual.append((res.item(), delta_time))

        for j in range(len(rel_tols)-1):
            if flags[j] and torch.abs(r_next).mean() <= torch.abs(b).max() * rel_tols[j] + abs_tols[j]:
                delta_time =  time.time() - start_cg_time
                iter_start_time = time.time()
                print(f'{tol_dict[j]} Pcg Converged in {i} steps ')
                convergent_iterations[tol_dict[j]] = i
                convergent_time[tol_dict[j]] = delta_time
                flags[j] = False
                
        if  torch.abs(r_next).mean() <= torch.abs(b).max() * rel_tols[-1] + abs_tols[-1]:
            delta_time =  time.time() - start_cg_time
            iter_start_time = time.time()
            print(f'{tol_dict[-1]} Pcg Converged in {i} steps {iter_start_time - start_cg_time}')
            convergent_iterations[tol_dict[-1]] = i
            convergent_time[tol_dict[-1]] = delta_time
            return i, convergent_iterations, convergent_time, residual
        
        y_next   = torch.mm( preconditioner, r_next)
        beta     = torch.mm(y_next.T, (r_next - r))/ r.T.mm(y) # Polak-Ribiere
        p        = -y_next + beta * p
        y = y_next
        r = r_next

    if i >= (max_iter-1):
        print('Convergence failed.')
        
    end_cg_time = time.time()


    return 1000000, convergent_iterations, convergent_time, residual


