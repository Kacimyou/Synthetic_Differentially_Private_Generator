
#%%
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
    psi_values[phi_values == 0 ] = 0
    
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

def random_walk_laplace(n):
    
    L = int(np.log2(n))
    
    laplace_noise = np.array([ np.random.laplace(loc = 0, scale = L + 2) for _ in range(n)])
    
    return laplace_noise
    

def super_regular_random_walk(laplace_noise, n, i ):
    
    psi_bar = np.array([psi_bar_j(j, np.array([i]))/n for j in range(1,n+1)])
    psi_bar = psi_bar.flatten()
    
    return np.dot(laplace_noise, psi_bar)

def super_regular_noise(n, epsilon):
    
    privacy_factor = 2/(epsilon*n)
    
    laplace_noise = random_walk_laplace(n)
    
    super_regular_noise = np.zeros(n)
    
    for i in range(1, n+1):
        
        super_regular_noise[i-1] = super_regular_random_walk(laplace_noise, n, i/n)
        
    private_super_regular_noise = privacy_factor * super_regular_noise
        
    return private_super_regular_noise



        
        
        


#%%
n = 47
super_regular_noise(n, epsilon= 0.9)

# %%
