import numpy as np
import matplotlib.pyplot as plt
#%%
def get_i_k(j):
    """
    Calculate the values of i and k for the given j.
    
    Parameters:
        j (int): Index of the basis function.
    
    Returns:
        tuple: A tuple containing the values of i and k.
    """
    if j == 1:
        return (0, 1)
    elif j == 2:
        return (1, 1)
    else:
        j -= 1
        i = int(np.log2(j))
        k = j - 2**i
        return (i+1, 2*k+1)

# %%
def phi_bar_i_k(i, k, t):
    """
    Compute the j-th Schauder basis function at time points t.
    
    Parameters:
        j (int): Index of the basis function.
        k (int): Index of the basis function.
        t (float or array-like): Time points at which to evaluate the function.
    
    Returns:
        array-like: Values of the j-k-th Schauder basis function evaluated at time points t.
    """
    t = np.asarray(t)  # Ensure t is a numpy array
    
    # Compute the values of the j-k-th Schauder basis function
    phi_values = 1 - 2**i * np.abs(t - k / 2**i)
    
    # Set negative values to zero
    phi_values[phi_values < 0] = 0

    return phi_values


def phi_bar(j, t):
    """
    Compute the j-th Schauder basis function at time points t.
    
    Parameters:
        j (int): Index of the basis function.
        t (float or array-like): Time points at which to evaluate the function.
    
    Returns:
        array-like: Values of the j-th Schauder basis function evaluated at time points t.
    """
    # Get the values of i and k for the given j
    i, k = get_i_k(j)
    
    # Call phi_bar_i_k function to compute the Schauder basis function
    return phi_bar_i_k(i, k, t)


def psi_bar_i_k(i, k, t):
    """
    Compute the derivative of the Schauder basis function at time points t.
    
    Parameters:
        i (int): Index i.
        k (int): Index k.
        t (array-like): Time points at which to evaluate the derivative.
    
    Returns:
        array-like: Values of the derivative of the Schauder basis function evaluated at time points t.
    """
    phi_values = phi_bar_i_k(i,k,t)
    if (i,k) == (0,1):
        return np.ones(len(t))
    psi_values = np.zeros(len(t))
    psi_values[t - k / 2**i>= 0] = -2**(i-1)
    psi_values[t - k / 2**i < 0] = 2**(i-1)
    psi_values[phi_values == 0 ] = np.nan
    
    return psi_values

def psi_bar_j(j, t):
    """
    Compute the derivative of the j-th Schauder basis function at time points t.
    
    Parameters:
        j (int): Index of the basis function.
        t (array-like): Time points at which to evaluate the derivative.
    
    Returns:
        array-like: Values of the derivative of the j-th Schauder basis function evaluated at time points t.
    """
    # Get the values of i and k for the given j
    i, k = get_i_k(j)
    
    # Call psi_bar_i_k function to compute the Schauder basis function
    return psi_bar_i_k(i, k, t)


#%%
# Plot the Schauder basis functions
i_max = 5

t = np.linspace(0, 1, 1000)

plt.figure(figsize=(10, 6))
for i in range(0,i_max):
    for k in range(1, 2**i+1,2):
        basis_function_j = phi_bar_i_k(i,k, t)
        plt.plot(t, basis_function_j, label=f'φ_{i}_{k}(t)')

plt.title('Schauder Basis Functions')
plt.xlabel('t')
plt.ylabel('φ_j(t)')
plt.grid(True)
plt.legend()
plt.show()
#%%
# Plot the Schauder basis functions
num_functions = 8
t = np.linspace(0, 1, 1000)

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
# Plot the Haar basis functions
i_max = 4

t = np.linspace(0, 1, 1000)

plt.figure(figsize=(10, 6))
for i in range(0,i_max):
    for k in range(1, 2**i+1,2):
        basis_function_j = psi_bar_i_k(i,k, t)
        plt.plot(t, basis_function_j, label=f'φ_{i}_{k}(t)')

plt.title('Schauder Basis Functions')
plt.xlabel('t')
plt.ylabel('φ_j(t)')
plt.grid(True)
plt.legend()
plt.show()
