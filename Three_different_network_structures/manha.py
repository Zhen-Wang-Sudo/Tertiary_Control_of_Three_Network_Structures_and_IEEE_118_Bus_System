import numpy as np
from scipy import sparse
from scipy.linalg import solve_continuous_lyapunov 
from scipy.optimize import minimize, Bounds, LinearConstraint, SR1
from scipy.stats import norm
B_inc = np.loadtxt('incidence_manha.csv', delimiter=',') 


n_bus, n_lines = B_inc.shape

B_inc_csr = sparse.csr_matrix(B_inc)
W = 30 * np.ones(n_lines)
gen_bus_idx = np.arange(0, 4)
load_bus_idx = np.arange(4, 25)
M = 0.05 * np.ones(25)
M[:4] = 10
D = 0.01 * np.ones(25)
D[:4] = 4
M_csr = sparse.csr_matrix(np.diag(M))
D_csr = sparse.csr_matrix(np.diag(D))
W_csr =  sparse.csr_matrix(np.diag(W))

noise_intensity = np.zeros(n_bus)
np.random.seed(42)
# generator side; disturbance set as large due to renewable sources
sigma_gen_base = 0.15 * 10
# sigma_gen_base = 0.45 * 10  
# generator side; disturbance set as small due to randomness of customers
sigma_load_base = 0.03 * 10
# sigma_load_base = 0.09 * 10
# browse all the nodes, assign different strength according to the category
for i in range(n_bus):
    
    is_gen = (i in gen_bus_idx)
    is_load = (i in load_bus_idx)
    

        
    # Category 2: Pure Gen or Gen+Load
    if is_gen:
        # Adding ±20% stochastic difference
        noise_intensity[i] += np.random.uniform(0.8, 1.2) * sigma_gen_base
    
    # Category 2: (Pure Load or Gen+Load)
    if is_load:
        # Adding ±20% stochastic difference
        noise_intensity[i] += np.random.uniform(0.8, 1.2) * sigma_load_base
# print(noise_intensity)
K = np.diag(noise_intensity)
P_initial = np.array([30,15,20])
P_upper = 50 * np.ones(4)
P_lower = np.zeros(4)

P_d = 90 * np.random.dirichlet(np.ones(21))


while np.any(P_d > 50):
  P_d = 90 * np.random.dirichlet(np.ones(21))

M_inv_sqrt_sparse =sparse.diags(M ** (-0.5),offsets=0, format='csr')
M_inv_sparse =sparse.diags(M ** (-1),offsets=0, format='csr')

L = (B_inc_csr @ W_csr @ B_inc_csr.T).toarray() 

eigvals, U1 = np.linalg.eigh(L)

# Check the smallest eigenvalue
tol = 1e-10
inv_eigvals = np.zeros_like(eigvals)

# addictive inverse of the non-zero elements
non_zero_mask = eigvals > tol
inv_eigvals[non_zero_mask] = 1.0 / eigvals[non_zero_mask]

# L_dagger = U @ Lambda_pinv @ U.T
# accelerate: U * inv_eigvals is equivalent to U @ diag(inv_eigvals)
L_pinv = (U1 * inv_eigvals) @ U1.T
U = U1.T
# Construct the control objective function
# Formulate the expectation
U_s = np.zeros([n_bus, len(gen_bus_idx) - 1])
U_d_list = []
for i in range(len(gen_bus_idx)-1):
    U_s[:,i] = U[:,gen_bus_idx[i]] - U[:,gen_bus_idx[-1]]
for i in range(n_bus):
    if i not in gen_bus_idx:
        U_d_list.append(U[:,gen_bus_idx[-1]] - U[:,i])
U_d_1 = np.array(U_d_list)
U_d = U_d_1.T
# Computing b
b = B_inc.T @ (U1 * inv_eigvals) @ U_d @ P_d
A = B_inc.T @ (U1 * inv_eigvals) @ U_s

# Check the algebraic multiplicity of the eigenvalue
F = sparse.diags(np.cos(np.asin(A @ P_initial + b)),offsets=0, format='csr')
L_1 = (M_inv_sqrt_sparse @B_inc_csr @ W_csr @ F @ B_inc_csr.T@M_inv_sqrt_sparse )
L_1_dense = L_1.toarray()
# Verify the connectivity
eigvals, eigvecs = np.linalg.eigh(L_1_dense)
tolerance = 1e-6

diffs = np.diff(eigvals)


if np.any(diffs < tolerance):
    print("find the eigenvalue with an algebric multiplicity more than 1")
    
    # Check the location of that eigenvalue
    close_indices = np.where(diffs < tolerance)[0]
    for idx in close_indices:
        val1 = eigvals[idx]
        val2 = eigvals[idx+1]
        print(f" - the eigenvalue with index {idx} approximates {idx+1}: {val1:.8f} and {val2:.8f}, the difference is {val2 - val1:.2e}")
else:
    print("All eigenvalues are with an algebraic multiplicity one")
    
print(f"The three least eigenvalues of Laplacian: {eigvals[:3]}")
assert np.abs(eigvals[0]) < 1e-8, "The smallest Laplacian eigenvalue should approximate 0"
assert eigvals[1] > 1e-4, " (Fiedler value) should be greater than 0"



r = 3.08
def diff_con_obj(P_s):
    F = sparse.diags(np.cos(np.asin(A@P_s +b)),offsets=0, format='csr')
    cos_angles = np.cos(np.asin(A@P_s +b))

    L_1 = (M_inv_sqrt_sparse @B_inc_csr @ W_csr @ F @ B_inc_csr.T@M_inv_sqrt_sparse )# (118, 118)
    L_1_dense = L_1.toarray()

    eigenvalue, eigvector = np.linalg.eigh(L_1_dense)
    A0_1=np.hstack([np.zeros([n_bus-1,n_bus-1]),np.zeros([n_bus-1,1]),np.eye(n_bus-1)])
    A0_2 = np.hstack([-eigvector.T @M_inv_sqrt_sparse@B_inc_csr @ W_csr @ F @ B_inc_csr.T@M_inv_sqrt_sparse@eigvector[:,1:n_bus], -eigvector.T @ M_inv_sparse @ D_csr @ eigvector])
    A0 = np.vstack([A0_1,A0_2])


    B0=np.vstack([np.zeros([n_bus-1,n_bus]), eigvector.T @ M_inv_sqrt_sparse @ K])

    C0= np.hstack([B_inc_csr.T @ M_inv_sqrt_sparse @ eigvector[:,1:n_bus], np.zeros([n_lines, n_bus])])

    Q0 = solve_continuous_lyapunov(A0, -B0 @ B0.T)

    variance = np.diag(C0 @ Q0 @ C0.T)

    standard_deviation = np.sqrt(np.diag(C0 @ Q0 @ C0.T))
    gradient_V = np.zeros([n_lines, len(P_s)])
    gradient_sigma = np.zeros([n_lines, len(P_s)])
    Unit = np.ones(n_lines)
    for k in range(len(P_s)):
        F_diff_1 = sparse.diags(-(A@P_s +b) * (Unit - (A@P_s +b) ** 2) ** (-1/2) * A[:,k],offsets=0, format='csr')
        L_diff_1 = M_inv_sqrt_sparse @B_inc_csr @ W_csr@ F_diff_1 @ B_inc_csr.T@M_inv_sqrt_sparse
        # U_diff_1 = np.zeros([n_bus,n_bus])
        # for i in range(n_bus):
        #     for j in range(n_bus):
        #         if j != i:
        #             Coefficient = (eigvector[:,j] @ L_diff_1 @ eigvector[:,i]) / (eigenvalue[i] - eigenvalue[j])
        #             U_diff_1[:,i] += Coefficient * eigvector[:,j]
        D_eigen = eigenvalue[None, :] - eigenvalue[:, None]
        np.fill_diagonal(D_eigen, np.inf)  
        Coe = (eigvector.T @ L_diff_1 @ eigvector) / D_eigen
        U_diff_1 = eigvector @ Coe
        A1_1=np.hstack([np.zeros([n_bus-1,n_bus-1]),np.zeros([n_bus-1,n_bus])])
        # A1_2 = np.hstack([-U_diff_1.T @M_inv_sqrt_sparse@B_inc_csr @ W_b@ F @ B_inc_csr.T@M_inv_sqrt_sparse@eigvector[:,1:n_bus]
        #                 -eigvector.T @M_inv_sqrt_sparse@B_inc_csr @ W_b@ F_diff_1 @ B_inc_csr.T@M_inv_sqrt_sparse@eigvector[:,1:n_bus]
        #                 -eigvector.T @M_inv_sqrt_sparse@B_inc_csr @ W_b@ F @ B_inc_csr.T@M_inv_sqrt_sparse@U_diff_1[:,1:n_bus], 
        #                 -U_diff_1.T @ M_inv_sparse @ D_mat @ eigvector - eigvector.T  @ M_inv_sparse @ D_mat @ U_diff_1])
        S_full1 = (
            M_inv_sqrt_sparse @ B_inc_csr @ W_csr @ F @ B_inc_csr.T @ M_inv_sqrt_sparse
        )
        S_mat2 = M_inv_sparse @ D_csr
        
        S_right = B_inc_csr.T @ M_inv_sqrt_sparse
        S_left = M_inv_sqrt_sparse @ B_inc_csr @ W_csr
        term1 = U_diff_1.T @ (S_full1 @ eigvector[:, 1:n_bus])


        y = S_right @ eigvector[:, 1:n_bus]  # Sparse @ Dense -> Dense
        y = F_diff_1 @ y  # Sparse @ Dense -> Dense
        y = W_csr @ y  # Sparse @ Dense -> Dense
        y = B_inc_csr @ y  # Sparse @ Dense -> Dense
        y = M_inv_sqrt_sparse @ y  # Sparse @ Dense -> Dense
        term2 = eigvector.T @ y  # Dense.T @ Dense -> Dense

        # --- Term 3 ---
        term3 = eigvector.T @ (S_full1 @ U_diff_1[:, 1:n_bus])

        # --- Term 4 ---
        term4 = U_diff_1.T @ (S_mat2 @ eigvector)


        term5 = eigvector.T @ (S_mat2 @ U_diff_1)

        A1_2 = np.hstack([-term1 - term2 - term3, -term4 - term5])

        A1 = np.vstack([A1_1,A1_2])

        B1=np.vstack([np.zeros([n_bus-1,n_bus]), U_diff_1.T @ M_inv_sqrt_sparse @ K])

        C1= np.hstack([B_inc_csr.T @ M_inv_sqrt_sparse @ U_diff_1[:,1:n_bus], np.zeros([n_lines, n_bus])])

        Q1 = solve_continuous_lyapunov(A0, -(Q0 @ A1.T + A1 @ Q0 + B0 @ B1.T +  B1 @ B0.T))
        gradient_V[:,k] = np.diag(C1 @ Q0 @ C0.T + C0 @ Q1 @ C0.T + C0 @ Q0 @ C1.T)
    for i in range(len(P_s)):
        gradient_sigma[:,i] = gradient_V[:,i] / (2 * standard_deviation)
        
    degree = np.asin(A@P_s +b)
    control_objective_function = abs(degree) + 3.08 * standard_deviation
    idx_con = np.argsort(control_objective_function)[-5:][::-1]
    
    #Print(gradient_sigma[:,0])
    #Compute the directional derivative of the max layer given the direction 
    max_val = np.max(control_objective_function)

    f_obj_diff = np.zeros(len(P_s))
    max_indices = np.where(control_objective_function == max_val)[0]
    f_diff = np.zeros(len(max_indices))
    e_k = np.zeros(len(P_s))
    for k in range(len(P_s)):
        e_k = np.zeros(len(P_s))
        e_k[k] = 1.0
        v = e_k
        idxx = 0
        for i in max_indices:
            if np.abs(A[i,:] @ P_s + b[i]) <= 10 ** (-2):
                f_diff[idxx] = np.abs(A[i,:] @ v) + r * gradient_sigma[i,:] @ v
            if A[i,:] @ P_s + b[i] > 10 ** (-2):
                f_diff[idxx] = (1 - (A[i,:] @ P_s + b[i]) ** 2) ** (-1/2) * A[i,:] @ v + r * gradient_sigma[i,:] @ v
            if A[i,:] @ P_s + b[i] < -10 ** (-2):
                f_diff[idxx] = -(1 - (A[i,:] @ P_s + b[i]) ** 2) ** (-1/2) * A[i,:] @ v + r * gradient_sigma[i,:] @ v
            idxx +=1
        f_obj_diff[k] = np.max(f_diff) 
    return max_val, f_obj_diff

print(diff_con_obj(P_initial))

#return the degree, standard deviation, and the control objective function
def mean_stand_div_con_obj(P_s):
    F = sparse.diags(np.cos(np.asin(A@P_s +b)),offsets=0, format='csr')
    cos_angles = np.cos(np.asin(A@P_s +b))

    L_1 = (M_inv_sqrt_sparse @B_inc_csr @ W_csr @ F @ B_inc_csr.T@M_inv_sqrt_sparse )# (118, 118)
    L_1_dense = L_1.toarray()

    eigenvalue, eigvector = np.linalg.eigh(L_1_dense)
    A0_1=np.hstack([np.zeros([n_bus-1,n_bus-1]),np.zeros([n_bus-1,1]),np.eye(n_bus-1)])
    A0_2 = np.hstack([-eigvector.T @M_inv_sqrt_sparse@B_inc_csr @ W_csr @ F @ B_inc_csr.T@M_inv_sqrt_sparse@eigvector[:,1:n_bus], -eigvector.T @ M_inv_sparse @ D_csr @ eigvector])
    A0 = np.vstack([A0_1,A0_2])


    B0=np.vstack([np.zeros([n_bus-1,n_bus]), eigvector.T @ M_inv_sqrt_sparse @ K])

    C0= np.hstack([B_inc_csr.T @ M_inv_sqrt_sparse @ eigvector[:,1:n_bus], np.zeros([n_lines, n_bus])])

    Q0 = solve_continuous_lyapunov(A0, -B0 @ B0.T)

    variance = np.diag(C0 @ Q0 @ C0.T)

    standard_deviation = np.sqrt(np.diag(C0 @ Q0 @ C0.T))
    degree = np.asin(A@P_s +b)
    control_objective_function = abs(degree) + 3.08 * standard_deviation
    idx_con = np.argsort(control_objective_function)[-5:][::-1]
    
    #Print(gradient_sigma[:,0])
    #Compute the directional derivative of the max layer given the direction 
    max_val = np.max(control_objective_function)
    max_indices = np.where(control_objective_function == max_val)[0]
    indices_list = list(max_indices)
    degrees_list = [degree[idx] for idx in max_indices]
    stds_list = [standard_deviation[idx] for idx in max_indices]
    return max_val, indices_list, degrees_list, stds_list


bounds = Bounds(P_lower[:-1], P_upper[:-1])


A_ineq = np.ones((1, len(P_upper) - 1))


lb = np.sum(P_d) - P_upper[-1]
ub = np.sum(P_d) - P_lower[-1]

slack_ineq_constraint = LinearConstraint(A_ineq, lb=lb, ub=ub)

res = minimize(
    fun = diff_con_obj,  
    x0 = P_initial,
    method='trust-constr',
    jac=True,
    hess=SR1(),
    bounds=bounds,
    constraints=[slack_ineq_constraint],
    options={'verbose': 3, 'maxiter': 200}
)
print("The optimal value）:", res.fun)
optimal_p_red = res.x
optimal_p_slack = np.sum(P_d)- np.sum(optimal_p_red)
optimal_p = np.hstack([optimal_p_red, optimal_p_slack])



print(mean_stand_div_con_obj(optimal_p_red))
print(mean_stand_div_con_obj(P_initial))
P_pro_con = np.sum(P_d)/np.sum(P_upper) * P_upper[:-1]
print(mean_stand_div_con_obj(P_pro_con))


def sequential_output_obj(P_s):
    F = sparse.diags(np.cos(np.asin(A@P_s +b)),offsets=0, format='csr')
    cos_angles = np.cos(np.asin(A@P_s +b))

    L_1 = (M_inv_sqrt_sparse @B_inc_csr @ W_csr @ F @ B_inc_csr.T@M_inv_sqrt_sparse )# (118, 118)
    L_1_dense = L_1.toarray()

    eigenvalue, eigvector = np.linalg.eigh(L_1_dense)
    A0_1=np.hstack([np.zeros([n_bus-1,n_bus-1]),np.zeros([n_bus-1,1]),np.eye(n_bus-1)])
    A0_2 = np.hstack([-eigvector.T @M_inv_sqrt_sparse@B_inc_csr @ W_csr @ F @ B_inc_csr.T@M_inv_sqrt_sparse@eigvector[:,1:n_bus], -eigvector.T @ M_inv_sparse @ D_csr @ eigvector])
    A0 = np.vstack([A0_1,A0_2])


    B0=np.vstack([np.zeros([n_bus-1,n_bus]), eigvector.T @ M_inv_sqrt_sparse @ K])

    C0= np.hstack([B_inc_csr.T @ M_inv_sqrt_sparse @ eigvector[:,1:n_bus], np.zeros([n_lines, n_bus])])

    Q0 = solve_continuous_lyapunov(A0, -B0 @ B0.T)

    variance = np.diag(C0 @ Q0 @ C0.T)

    standard_deviation = np.sqrt(np.diag(C0 @ Q0 @ C0.T))
    degree = np.asin(A@P_s +b)
    control_objective_function = abs(degree) + 3.08 * standard_deviation
    max_val = np.max(control_objective_function)
    idx_seq = np.argsort(control_objective_function)[-n_lines:][::-1]
    line_number_list = []
    for i in range(n_lines):
        indices = np.nonzero(B_inc[:,idx_seq[i]])[0]
        indices = indices + 1
        indices = indices.tolist()
        line_number_list.append(indices)
    return max_val, idx_seq, line_number_list, control_objective_function[idx_seq], degree[idx_seq], standard_deviation[idx_seq]

print(sequential_output_obj(optimal_p_red))


def compare_exit_prob(P_s_1, P_s):    
    F = sparse.diags(np.cos(np.asin(A@P_s +b)),offsets=0, format='csr')
    cos_angles = np.cos(np.asin(A@P_s +b))

    L_1 = (M_inv_sqrt_sparse @B_inc_csr @ W_csr @ F @ B_inc_csr.T@M_inv_sqrt_sparse )# (118, 118)
    L_1_dense = L_1.toarray()

    eigenvalue, eigvector = np.linalg.eigh(L_1_dense)
    A0_1=np.hstack([np.zeros([n_bus-1,n_bus-1]),np.zeros([n_bus-1,1]),np.eye(n_bus-1)])
    A0_2 = np.hstack([-eigvector.T @M_inv_sqrt_sparse@B_inc_csr @ W_csr @ F @ B_inc_csr.T@M_inv_sqrt_sparse@eigvector[:,1:n_bus], -eigvector.T @ M_inv_sparse @ D_csr @ eigvector])
    A0 = np.vstack([A0_1,A0_2])


    B0=np.vstack([np.zeros([n_bus-1,n_bus]), eigvector.T @ M_inv_sqrt_sparse @ K])

    C0= np.hstack([B_inc_csr.T @ M_inv_sqrt_sparse @ eigvector[:,1:n_bus], np.zeros([n_lines, n_bus])])

    Q0 = solve_continuous_lyapunov(A0, -B0 @ B0.T)

    variance = np.diag(C0 @ Q0 @ C0.T)

    standard_deviation = np.sqrt(np.diag(C0 @ Q0 @ C0.T))
    degree = np.asin(A@P_s +b)
    control_objective_function = abs(degree) + 3.08 * standard_deviation
    idx_seq = sequential_output_obj(P_s_1)[1]
    line_number_list = []
    for i in range(n_lines):
        indices = np.nonzero(B_inc[:,idx_seq[i]])[0]
        indices = indices + 1
        indices = indices.tolist()
        line_number_list.append(indices) 
    degree_P_C  = sequential_output_obj(P_s_1)[4]
    standard_deviation_P_C = sequential_output_obj(P_s_1)[5]  
    r_a_P_C = (-np.pi/2 - degree_P_C)/ standard_deviation_P_C
    r_b_P_C = (-np.pi/2 + degree_P_C)/ standard_deviation_P_C
    r_a_P_S_C = (-np.pi/2 - degree)/ standard_deviation
    r_b_P_S_C = (-np.pi/2 + degree)/ standard_deviation
    P_a_P_C = norm.cdf(r_a_P_C)
    P_b_P_C = norm.cdf(r_b_P_C)
    P_a_P_S_C = norm.cdf(r_a_P_S_C)
    P_b_P_S_C = norm.cdf(r_b_P_S_C)
    return idx_seq, line_number_list, sequential_output_obj(P_s_1)[3], control_objective_function, degree_P_C, degree, standard_deviation_P_C, standard_deviation, P_a_P_C,P_a_P_S_C,P_b_P_C,P_b_P_S_C
print(compare_exit_prob(P_pro_con, optimal_p_red))

r_epsilon = (-np.pi/2 + sequential_output_obj(P_pro_con)[0])/sequential_output_obj(P_pro_con)[5][0] -3.08
r_new_epsilon = (-np.pi/2 + sequential_output_obj(optimal_p_red)[0])/sequential_output_obj(optimal_p_red)[5][0]-3.08
print(norm.cdf(r_epsilon),norm.cdf(r_new_epsilon))
