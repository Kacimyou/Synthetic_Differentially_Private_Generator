#%%
import numpy as np
import matplotlib.pyplot as plt


def haar_wavelet(n, k, t):
    """
    Generate Haar wavelet function for given parameters n and k.
    
    Parameters:
        n (int): Scale parameter.
        k (int): Shift parameter.
        t (float or array-like): Time points at which to evaluate the wavelet function.
    
    Returns:
        array-like: Haar wavelet function evaluated at time points t.
    """
    t = np.asarray(t)
    result = np.zeros_like(t)
    for i, ti in enumerate(t):
        ti_scaled = ti / (2 ** n) - k
        if 0 <= ti_scaled < 0.5:
            result[i] = 1
        elif 0.5 <= ti_scaled < 1:
            result[i] = -1
    return result

# Example usage:
n = 1
k = 1
t = np.linspace(0, 10, 1000)  # Time points
wavelet = haar_wavelet(n, k, t)


# Plot the Haar wavelet
plt.figure(figsize=(10, 6))
plt.plot(t, wavelet, label=f'Haar Wavelet (n={n}, k={k})')
plt.title('Haar Wavelet')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

#%%
def phi_bar_derivative(j, t):
    """
    Compute the derivative of the j-th Schauder basis function at time points t.
    
    Parameters:
        j (int): Index of the basis function.
        t (float or array-like): Time points at which to evaluate the function.
    
    Returns:
        array-like: Values of the derivative of the j-th Schauder basis function evaluated at time points t.
    """
    t = np.asarray(t)
    qj = int(np.log2(j))
    pj = j - 2**qj
    pj_max = 2**(qj - 1) - 1
    
    derivative_values = np.zeros_like(t)
    
    for idx, ti in enumerate(t):
        if 0 <= ti < 1:
            if j == 1:
                derivative_values[idx] = 1
            else:
                interval_idx = pj * 2**(qj - 1) <= ti < (pj + 1) * 2**(qj - 1)
                if interval_idx:
                    derivative_values[idx] = 2**qj / 2**(qj - 1)
                else:
                    derivative_values[idx] = -2**qj / 2**(qj - 1)
    
    return derivative_values

# Plot the derivative of the Schauder basis functions
num_functions = 2
t = np.linspace(0, 1, 1000)

plt.figure(figsize=(10, 6))
for j in range(1, num_functions + 1):
    derivative_function = phi_bar_derivative(j, t)
    plt.plot(t, derivative_function, label=f'd(φ_{j}(t))/dt')

plt.title('Derivative of Schauder Basis Functions')
plt.xlabel('t')
plt.ylabel('d(φ_j(t))/dt')
plt.grid(True)
plt.legend()
plt.show()