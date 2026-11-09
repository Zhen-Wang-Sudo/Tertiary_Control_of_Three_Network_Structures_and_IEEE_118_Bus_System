#The following is the code of constructing the control objective function
import numpy as np
from scipy import sparse
from scipy.linalg import solve_continuous_lyapunov 
from scipy.optimize import minimize, Bounds, LinearConstraint, SR1
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
# 1. import the data
B_inc = np.loadtxt('incidence_matrix_118.csv', delimiter=',')          # (118, 186)
b_vec = np.loadtxt('branch_susceptance_vector.csv', delimiter=',')      # (186,)
M_mat = np.loadtxt('M_diag_matrix.csv', delimiter=',')                      # (118,)
D_mat = np.loadtxt('D_diag_matrix.csv', delimiter=',')                      # (118,)
gen_limits = np.loadtxt('gen_pmax_limits.csv', delimiter=',')          # generator data
load_data  = np.loadtxt('load_15min_demand_7.csv', delimiter=',')        # load data

p_max_pu = gen_limits[:, 3]
n_bus, n_lines = B_inc.shape

# 2. Construct the Laplacian matrix: L = B_inc * W_b * B_inc^T
W_b = sparse.diags(b_vec, offsets=0, format='csr')
B_inc_csr = sparse.csr_matrix(B_inc)
L = (B_inc_csr @ W_b @ B_inc_csr.T).toarray()  # (118, 118)

# 3. verify the connectivity
eigvals = np.linalg.eigvalsh(L)
tolerance = 1e-6

diffs = np.diff(eigvals)

# 3. check
if np.any(diffs < tolerance):
    print("find the eigenvalue with an algebric multiplicity more than 1")
    
    # 4. check the location of that eigenvalue
    close_indices = np.where(diffs < tolerance)[0]
    for idx in close_indices:
        val1 = eigvals[idx]
        val2 = eigvals[idx+1]
        print(f" - the eigenvalue with index {idx} approximates {idx+1}: {val1:.8f} and {val2:.8f}, the difference is {val2 - val1:.2e}")
else:
    print("All eigenvalues are with an algebraic multiplicity one")
    
# print(f"The three least eigenvalues of Laplacian: {eigvals[:3]}")
assert np.abs(eigvals[0]) < 1e-8, "The smallest Laplacian eigenvalue should approximate 0"
assert eigvals[1] > 1e-4, " (Fiedler value) should be greater than 0"

# 4. Box Constraints (0 <= P <= P_max)
gen_bus_idx = gen_limits[:, 1].astype(int) - 1  # 0-based generator index

# print(p_max)

# 5. Construct P_d (118,)

load_bus_idx   = load_data[:, 0].astype(int) - 1  # load bus index
p_load_15m_pu  = load_data[:, 3]                  # demand in 15-minute (p.u.)


n_bus = 118
P_d_vector = np.zeros(n_bus)
P_d_vector[load_bus_idx] = p_load_15m_pu

np.random.seed(32)
P_test = np.random.uniform(p_max_pu- 0.04, p_max_pu)
print(np.sum(P_test) -P_test[0] , np.sum(P_d_vector))
print(np.sum(P_d_vector) - (np.sum(P_test) -P_test[0]), p_max_pu[0])
# print(P_test, p_max_pu)
P_test[0] = np.sum(P_d_vector) - (np.sum(P_test) -P_test[0])
# print(np.sum(p_max_pu), np.sum(P_d_vector))
P_test_new = P_test[:len(P_test)-1]

P_s_vector = np.zeros(n_bus)
P_s_vector[gen_bus_idx[:len(gen_bus_idx)-1]] = P_test_new




bus_gen_and_load_list = []
for i in load_bus_idx:
    if i in gen_bus_idx:
        bus_gen_and_load_list.append((i, P_d_vector[i]))
bus_gen_and_load =  np.array(bus_gen_and_load_list)  
bus_gen_and_load_idx = bus_gen_and_load[:,0].astype(int)
# print(bus_gen_and_load_idx)

P_s_vector = np.zeros(n_bus)
P_upper_bound = np.zeros(n_bus)
P_lower_bound = np.zeros(n_bus)
P_s_vector[gen_bus_idx] = P_test
P_upper_bound[gen_bus_idx] = p_max_pu

P_s_vector[bus_gen_and_load_idx] -=  P_d_vector[bus_gen_and_load_idx]
P_upper_bound[bus_gen_and_load_idx] -= P_d_vector[bus_gen_and_load_idx]
P_lower_bound[bus_gen_and_load_idx] -= P_d_vector[bus_gen_and_load_idx]


P_s_list= []
P_upper_list = []
P_lower_list = []
for i in range(118):
    if i in gen_bus_idx:
        P_s_list.append(P_s_vector[i])
        P_upper_list.append(P_upper_bound[i])
        P_lower_list.append(P_lower_bound[i])
P_s_ori = np.array(P_s_list)
P_upper = np.array(P_upper_list)
P_lower = np.array(P_lower_list)

P_s_red = P_s_ori[:len(P_s_ori)-1]
P_upper_red = P_upper[:len(P_upper)-1]
P_lower_red = P_lower[:len(P_lower)-1]

# print(P_d.shape)
#Compute p_sum_demand and shift the bounds of the nodes containing both generator and load


# print(len(bus_gen_and_load_idx))
# print(bus_gen_and_load.shape)
# print(bus_gen_and_load[:,0])

P_d_list= []
for i in range(118):
    if i not in gen_bus_idx:
        P_d_list.append(P_d_vector[i])
P_d = np.array(P_d_list)
# print(P_d.shape)


        

# 6. Eigenvalue decomposition of the Laplacian matrix
eigvals, U1 = np.linalg.eigh(L)

# 2. check the smallest eigenvalue
tol = 1e-10
inv_eigvals = np.zeros_like(eigvals)

# addictive inverse of the non-zero elements
non_zero_mask = eigvals > tol
inv_eigvals[non_zero_mask] = 1.0 / eigvals[non_zero_mask]

# L_dagger = U @ Lambda_pinv @ U.T
# accelerate: U * inv_eigvals is equivalent to U @ diag(inv_eigvals)
L_pinv = (U1 * inv_eigvals) @ U1.T
U = U1.T
# 7. Construct the control objective function
#(1) Formulate the expectation
U_s = np.zeros([118, len(gen_bus_idx) - 1])
U_d_list = []
for i in range(len(gen_bus_idx)-1):
    U_s[:,i] = U[:,gen_bus_idx[i]] - U[:,gen_bus_idx[-1]]
for i in range(118):
    if i not in gen_bus_idx:
        U_d_list.append(U[:,gen_bus_idx[-1]] - U[:,i])
U_d_1 = np.array(U_d_list)
U_d = U_d_1.T
# print(U_s.shape, U_d.shape)
# print(len(gen_bus_idx))
# Computing b
b = B_inc.T @ (U1 * inv_eigvals) @ U_d @ P_d
A = B_inc.T @ (U1 * inv_eigvals) @ U_s

#(2.1) Formulate the strength of the disturbances 
# Initialize the disturbance intensity 
noise_intensity = np.zeros(n_bus)
np.random.seed(42)
# 1. generator side; disturbance set as large due to renewable sources
sigma_gen_base = 0.15 * 10 
# 2. generator side; disturbance set as small due to randomness of customers
sigma_load_base = 0.03 * 10
# browse all the nodes, assign different strength according to the category
for i in range(n_bus):
    
    is_gen = (i in gen_bus_idx)
    has_load = (P_d_vector[i] > 0)
    
    # Category 1: transit node
    if not is_gen and not has_load:
        noise_intensity[i] = 0.0
        
    # Category 2: Pure Gen or Gen+Load
    if is_gen:
        # Adding ±20% stochastic difference
        noise_intensity[i] += np.random.uniform(0.8, 1.2) * sigma_gen_base
    
    # Category 2: (Pure Load or Gen+Load)
    if has_load:
        # Adding ±20% stochastic difference
        noise_intensity[i] += np.random.uniform(0.8, 1.2) * sigma_load_base

#Validation
pure_gen_noise = [noise_intensity[i] for i in range(n_bus) if (i in gen_bus_idx) and (P_d_vector[i] == 0)]
pure_load_noise = [noise_intensity[i] for i in range(n_bus) if (i not in gen_bus_idx) and (P_d_vector[i] > 0)]
transit_noise = [noise_intensity[i] for i in range(n_bus) if (i not in gen_bus_idx) and (P_d_vector[i] == 0)]
gen_load_noise = [noise_intensity[i] for i in range(n_bus) if (i in gen_bus_idx) and (P_d_vector[i] > 0)]

K = np.diag(noise_intensity)




#The differential direction
v_test = np.ones(len(P_s_red))
# The parameter in the control objective function
r = 3.08
F = sparse.diags(np.cos(np.asin(A@P_s_red +b)),offsets=0, format='csr')
cos_angles = np.cos(np.asin(A@P_s_red +b))

M_vec = np.diag(M_mat)
M_inv_sqrt_sparse =sparse.diags(M_vec ** (-0.5),offsets=0, format='csr')
M_inv_sparse =sparse.diags(M_vec ** (-1),offsets=0, format='csr')


def diff_con_obj(P_s):
    F = sparse.diags(np.cos(np.asin(A@P_s +b)),offsets=0, format='csr')
    cos_angles = np.cos(np.asin(A@P_s +b))

    L_1 = (M_inv_sqrt_sparse @B_inc_csr @ W_b@ F @ B_inc_csr.T@M_inv_sqrt_sparse )# (118, 118)
    L_1_dense = L_1.toarray()

    eigenvalue, eigvector = np.linalg.eigh(L_1_dense)
    A0_1=np.hstack([np.zeros([n_bus-1,n_bus-1]),np.zeros([n_bus-1,1]),np.eye(n_bus-1)])
    A0_2 = np.hstack([-(eigvector.T @M_inv_sqrt_sparse@B_inc_csr @ W_b@ F @ B_inc_csr.T@M_inv_sqrt_sparse@eigvector)[:,1:n_bus], -eigvector.T @ M_inv_sparse @ D_mat @ eigvector])
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
        L_diff_1 = M_inv_sqrt_sparse @B_inc_csr @ W_b@ F_diff_1 @ B_inc_csr.T@M_inv_sqrt_sparse
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
            M_inv_sqrt_sparse @ B_inc_csr @ W_b @ F @ B_inc_csr.T @ M_inv_sqrt_sparse
        )
        S_mat2 = M_inv_sparse @ D_mat
        
        S_right = B_inc_csr.T @ M_inv_sqrt_sparse
        S_left = M_inv_sqrt_sparse @ B_inc_csr @ W_b
        term1 = U_diff_1.T @ (S_full1 @ eigvector[:, 1:n_bus])


        y = S_right @ eigvector[:, 1:n_bus]  # Sparse @ Dense -> Dense
        y = F_diff_1 @ y  # Sparse @ Dense -> Dense
        y = W_b @ y  # Sparse @ Dense -> Dense
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
# print(diff_con_obj(P_s_red)) 

def mean_stand_div_con_obj(P_s):
    F = sparse.diags(np.cos(np.asin(A@P_s +b)),offsets=0, format='csr')
    cos_angles = np.cos(np.asin(A@P_s +b))

    L_1 = (M_inv_sqrt_sparse @B_inc_csr @ W_b@ F @ B_inc_csr.T@M_inv_sqrt_sparse )# (118, 118)
    L_1_dense = L_1.toarray()

    eigenvalue, eigvector = np.linalg.eigh(L_1_dense)
    A0_1=np.hstack([np.zeros([n_bus-1,n_bus-1]),np.zeros([n_bus-1,1]),np.eye(n_bus-1)])
    A0_2 = np.hstack([-(eigvector.T @M_inv_sqrt_sparse@B_inc_csr @ W_b@ F @ B_inc_csr.T@M_inv_sqrt_sparse@eigvector)[:,1:n_bus], -eigvector.T @ M_inv_sparse @ D_mat @ eigvector])
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


def sequential_output_obj(P_s):
    F = sparse.diags(np.cos(np.asin(A@P_s +b)),offsets=0, format='csr')
    cos_angles = np.cos(np.asin(A@P_s +b))

    L_1 = (M_inv_sqrt_sparse @B_inc_csr @ W_b@ F @ B_inc_csr.T@M_inv_sqrt_sparse )# (118, 118)
    L_1_dense = L_1.toarray()

    eigenvalue, eigvector = np.linalg.eigh(L_1_dense)
    A0_1=np.hstack([np.zeros([n_bus-1,n_bus-1]),np.zeros([n_bus-1,1]),np.eye(n_bus-1)])
    A0_2 = np.hstack([-(eigvector.T @M_inv_sqrt_sparse@B_inc_csr @ W_b@ F @ B_inc_csr.T@M_inv_sqrt_sparse@eigvector)[:,1:n_bus], -eigvector.T @ M_inv_sparse @ D_mat @ eigvector])
    A0 = np.vstack([A0_1,A0_2])


    B0=np.vstack([np.zeros([n_bus-1,n_bus]), eigvector.T @ M_inv_sqrt_sparse @ K])

    C0= np.hstack([B_inc_csr.T @ M_inv_sqrt_sparse @ eigvector[:,1:n_bus], np.zeros([n_lines, n_bus])])

    Q0 = solve_continuous_lyapunov(A0, -B0 @ B0.T)

    variance = np.diag(C0 @ Q0 @ C0.T)

    standard_deviation = np.sqrt(np.diag(C0 @ Q0 @ C0.T))
    degree = np.asin(A@P_s +b)
    control_objective_function = abs(degree) + 3.08 * standard_deviation
    max_val = np.max(control_objective_function)
    idx_seq_full = np.argsort(control_objective_function)
    idx_seq = np.argsort(control_objective_function)[-10:][::-1]
    line_number_list = []
    for i in range(10):
        indices = np.nonzero(B_inc[:,idx_seq[i]])[0].tolist()
        line_number_list.append(indices)
    return max_val, idx_seq, line_number_list, control_objective_function[idx_seq], degree[idx_seq], standard_deviation[idx_seq]

history_time = [0.0]
history_obj = []  
t_start = time.perf_counter()


# The proportional control

np.sum(p_load_15m_pu)/np.sum(p_max_pu) * p_max_pu

P_s_pc = np.zeros(n_bus)

P_s_pc[gen_bus_idx] = np.sum(p_load_15m_pu)/np.sum(p_max_pu) * p_max_pu

P_s_pc[bus_gen_and_load_idx] -=  P_d_vector[bus_gen_and_load_idx]

P_s_pc_list= []
for i in range(118):
    if i in gen_bus_idx:
        P_s_pc_list.append(P_s_pc[i])
P_s_pc_array = np.array(P_s_pc_list)

def record_callback(xk, state):
  elapsed = time.perf_counter() - t_start
  history_time.append(elapsed)
  history_obj.append(state.fun)
  
f0, _ = diff_con_obj(P_s_red)
history_obj.append(f0)

bounds = Bounds(P_lower[:-1], P_upper[:-1])


A_ineq = np.ones((1, len(P_upper) - 1))


lb = np.sum(P_d) - P_upper[-1]
ub = np.sum(P_d) - P_lower[-1]

slack_ineq_constraint = LinearConstraint(A_ineq, lb=lb, ub=ub)

res = minimize(
    fun = diff_con_obj,  
    # x0 = P_s_red,
    x0 = P_s_pc_array[:-1],
    method='trust-constr',
    hess=SR1(),
    jac=True,
    bounds=bounds,
    constraints=[slack_ineq_constraint],
    callback=record_callback,
    options={'verbose': 3, 'maxiter': 70}
)

iters = np.arange(len(history_obj))

fig, ax1 = plt.subplots(figsize=(6.5, 4.2), dpi=300)

# 
ax1.plot(
    history_time,
    history_obj,
    marker='o',
    markersize=4,
    linewidth=1.8,
    color='#0072BD',
    label='Max Stability Risk',
)
ax1.set_xlabel('Cumulative CPU Time (s)', fontsize=11, fontweight='bold')
ax1.set_ylabel(
    r'Objective $\max_k (|\theta_{i_k} - \theta_{j_k}| + \bar{r}_\epsilon\sigma_k)$',
    fontsize=11,
    fontweight='bold',
)
ax1.grid(True, linestyle='--', alpha=0.6)

# 
ax2 = ax1.twiny()
# 
tick_indices = np.linspace(0, len(iters) - 1, min(6, len(iters)), dtype=int)
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks([history_time[i] for i in tick_indices])
ax2.set_xticklabels([f'{iters[i]}' for i in tick_indices])
ax2.set_xlabel('Iteration Number', fontsize=11)

# 
total_time = history_time[-1]
final_obj = history_obj[-1]
ax1.annotate(
    f'Converged in {total_time:.1f} s\n({len(iters)-1} iterations)',
    xy=(total_time, final_obj),
    xytext=(total_time * 0.55, final_obj + (history_obj[0] - final_obj) * 0.25),
    arrowprops=dict(arrowstyle='->', color='black', lw=1),
    fontsize=10,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA', edgecolor='gray'),
)

plt.tight_layout()
plt.savefig('ieee118_cpu_convergence.pdf', bbox_inches='tight')
plt.show()
print("The optimal value）:", res.fun)

print(mean_stand_div_con_obj(res.x))

print(mean_stand_div_con_obj(P_s_red))


print(mean_stand_div_con_obj(P_s_pc_array[:-1]))


# Recover the optimal power dispatch
optimal_p_red = res.x
optimal_p_slack = np.sum(P_d)- np.sum(optimal_p_red)
optimal_p = np.hstack([optimal_p_red, optimal_p_slack])

optimal_P_s = np.zeros(n_bus)
optimal_P_s[gen_bus_idx] = optimal_p 
optimal_P_s[bus_gen_and_load_idx] += P_d_vector[bus_gen_and_load_idx]
optimal_P_s_list = []
for i in range(118):
    if i in gen_bus_idx:
        optimal_P_s_list.append(optimal_P_s[i])
optimal_P_s_final = np.array(optimal_P_s_list)
print(optimal_P_s_final)
# print(optimal_P_s_final[i], p_max_pu[i])
print(sequential_output_obj(P_s_pc_array[:-1]))
print(sequential_output_obj(optimal_p_red))

def compare_exit_prob(P_s_1, P_s):  
    F = sparse.diags(np.cos(np.asin(A@P_s +b)),offsets=0, format='csr')
    cos_angles = np.cos(np.asin(A@P_s +b))

    L_1 = (M_inv_sqrt_sparse @B_inc_csr @ W_b@ F @ B_inc_csr.T@M_inv_sqrt_sparse )# (118, 118)
    L_1_dense = L_1.toarray()

    eigenvalue, eigvector = np.linalg.eigh(L_1_dense)
    A0_1=np.hstack([np.zeros([n_bus-1,n_bus-1]),np.zeros([n_bus-1,1]),np.eye(n_bus-1)])
    A0_2 = np.hstack([-(eigvector.T @M_inv_sqrt_sparse@B_inc_csr @ W_b@ F @ B_inc_csr.T@M_inv_sqrt_sparse@eigvector)[:,1:n_bus], -eigvector.T @ M_inv_sparse @ D_mat @ eigvector])
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
    for i in range(10):
        indices = np.nonzero(B_inc[:,idx_seq[i]])[0].tolist()
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
    return idx_seq, line_number_list, degree_P_C[:10], degree[:10], standard_deviation_P_C[:10], standard_deviation[:10], P_a_P_C[:10],P_a_P_S_C[:10],P_b_P_C[:10],P_b_P_S_C[:10]
print(compare_exit_prob(P_s_pc_array[:-1], optimal_p_red))     

print(sequential_output_obj(P_s_pc_array[:-1])[0], sequential_output_obj(P_s_pc_array[:-1])[4][0],sequential_output_obj(P_s_pc_array[:-1])[5][0])
print(sequential_output_obj(optimal_p_red)[0], sequential_output_obj(optimal_p_red)[4][0], sequential_output_obj(optimal_p_red)[5][0])

r_epsilon = (-np.pi/2 + sequential_output_obj(P_s_pc_array[:-1])[0])/sequential_output_obj(P_s_pc_array[:-1])[5][0] -3.08
r_new_epsilon = (-np.pi/2 + sequential_output_obj(optimal_p_red)[0])/sequential_output_obj(optimal_p_red)[5][0]-3.08
print(norm.cdf(r_epsilon),norm.cdf(r_new_epsilon))



    












        
    



