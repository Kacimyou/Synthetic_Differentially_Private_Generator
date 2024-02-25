
#%%
import numpy as np
import matplotlib.pyplot as plt

def get_i_k(j):
    
    i = int(np.log2(j))
    k = j - 2**i
    
    if k == 0:
        i -= 1
        k = 2 ** i

    return (i,k-1)


#%%

for i in range(2,17):
    print(i ,get_i_k(i))

# %%
def phi_bar_bis(i,k, t):
    """
    Compute the j-th Schauder basis function at time points t.
    
    Parameters:
        j (int): Index of the basis function.
        k (int): Index of the basis function.
        t (float or array-like): Time points at which to evaluate the function.
    
    Returns:
        array-like: Values of the j-th Schauder basis function evaluated at time points t.
    """
    t = np.asarray(t)
    
    phi_values = 1 - 2**i*np.abs(t - k/2**i)
    phi_values[phi_values < 0] = 0


    return phi_values


def phi_bar(j,t):
    
    if j ==1:
        return t
    
    i,k = get_i_k(j)
    
    return phi_bar_bis(i,k,t)


#%%
# Plot the Schauder basis functions
num_functions = 4
t = np.linspace(0, 3, 1000)

plt.figure(figsize=(10, 6))
for j in range(1, num_functions + 1):
    basis_function_j = phi_bar(j, t)
    plt.plot(t, basis_function_j, label=f'φ_{j}(t)')

plt.title('Schauder Basis Functions')
plt.xlabel('t')
plt.ylabel('φ_j(t)')
plt.grid(True)
plt.legend()
plt.show()

#%%
# Plot the Schauder basis functions
i_max = 4

t = np.linspace(0, 1, 1000)

plt.figure(figsize=(10, 6))
for i in range(i_max):
    for k in range(1, 2**i+1,2):
        basis_function_j = phi_bar_bis(i,k, t)
        plt.plot(t, basis_function_j, label=f'φ_{i}{k}(t)')

plt.title('Schauder Basis Functions')
plt.xlabel('t')
plt.ylabel('φ_j(t)')
plt.grid(True)
plt.legend()
plt.show()