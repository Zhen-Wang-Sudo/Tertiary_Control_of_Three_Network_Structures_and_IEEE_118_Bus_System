import numpy as np
from scipy import sparse

# 1. random number generator
n = 118
np.random.seed(42)

# 2. generator index
# 
gen_buses_1based = np.loadtxt('gen_buses.csv', delimiter=',').astype(int)
gen_idx = gen_buses_1based - 1  

# load
load_buses_1based = np.loadtxt('load_buses.csv', delimiter=',').astype(int)
load_idx = load_buses_1based - 1  

all_nodes = np.arange(n)

# 3. Four kinds of nodes
is_gen = np.isin(all_nodes, gen_idx)
is_load = np.isin(all_nodes, load_idx)

pure_gen_idx = all_nodes[is_gen & ~is_load]     # 纯发电机
gen_load_idx = all_nodes[is_gen & is_load]      # 发电机+负荷复合节点
pure_load_idx = all_nodes[~is_gen & is_load]    # 纯负荷节点
transit_idx = all_nodes[~is_gen & ~is_load]     # 联络节点 (零注入)

# 4. Initialize M_vec and D_vec
M_vec = np.zeros(n)
D_vec = np.zeros(n)

# (1) Pure generator indices: large inertia and damping
M_vec[pure_gen_idx] = np.random.uniform(3.0, 6.0, size=len(pure_gen_idx))
D_vec[pure_gen_idx] = np.random.uniform(1.0, 3.0, size=len(pure_gen_idx))

# (2) Mixed generator and load indices: large inertia and damping
M_vec[gen_load_idx] = np.random.uniform(6.0, 9.0, size=len(gen_load_idx))
D_vec[gen_load_idx] = np.random.uniform(3.0, 5.0, size=len(gen_load_idx))

# (3)  Pure load indices: lower inertia and damping
M_vec[pure_load_idx] = 0.05
D_vec[pure_load_idx] = np.random.uniform(0.1, 0.3, size=len(pure_load_idx))

# (4) Transit node indices: the lowest inertia and the zero damping coefficients 
M_vec[transit_idx] = 1e-3
D_vec[transit_idx] = 0.0

# 5. Construct the matrix
M_diag = np.diag(M_vec)
D_diag = np.diag(D_vec)

# 6. save to csv
np.savetxt('M_diag_matrix.csv', M_diag, delimiter=',')
np.savetxt('D_diag_matrix.csv', D_diag, delimiter=',')

# =================Print =================
print(f"1. Pure generator ({len(pure_gen_idx):2d} )  | M mean: {M_vec[pure_gen_idx].mean():.2f} | D mean: {D_vec[pure_gen_idx].mean():.2f}")
print(f"2. Mixed generator and load ({len(gen_load_idx):2d} )  | M mean: {M_vec[gen_load_idx].mean():.2f} | D mean: {D_vec[gen_load_idx].mean():.2f} (the largest)")
print(f"3. Pure load ({len(pure_load_idx):2d} )| M mean: {M_vec[pure_load_idx][0]:.3f} | D mean: {D_vec[pure_load_idx].mean():.2f} (Minor)")
print(f"4. Transit node ({len(transit_idx):2d} )  | M mean: {M_vec[transit_idx][0]:.0e} | D mean: {D_vec[transit_idx][0]:.1f} (0)")
print("-" * 35)
print(f"M : {M_diag.shape}, D : {D_diag.shape}")
print(D_vec)