import numpy as np 
import matplotlib as plt


prob = np.array([0.006, 0.05,  0.198, 0.36,  0.272, 0.108, 0.004, 0.002])


print(np.sum(prob))

delta = 0.1
m = len(prob)


smoothed = (1 - delta)*prob + delta/m

#print(smoothed, np.sum(smoothed))



def modified_histogram_estimator(X, h=0.1, adaptative = True, delta = 0.1):
    
    # Calculate the dimensions of the data
    n, d = X.shape

    """if np.min(X, axis=0) < 0 or np.max(X, axis=0) > 1:
        # Scale the data to [0, 1] range
        X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    
    #elif np.min(X, axis=0) >= 0 and np.max(X, axis=0) <= 1:
        
        X_scaled = X
    """
    
    X_scaled = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    
    if adaptative == True:
        # Calculate the binwidth based on the given parameters
        h = n**(-1 / (2 + d))
    
    assert h<1 and h>0 , "Error: h must be between 0 and 1"
    
    
    # Check if 1/h is an integer, and convert if necessary
    m_inverse = 1 / h
    if not m_inverse.is_integer():
        m_per_axis = int(np.round(m_inverse))
        print(f"Warning: 1/h is not an integer. Converting to the closest integer: {m_per_axis}")
    else:
        m_per_axis = int(m_inverse)
    

    # Initialize the histogram estimator
    fbm_shape = (m_per_axis,) * d
    fbm = np.zeros(fbm_shape)

    # Iterate through each row as a separate data point
    for i in range(n):
        # Determine the bin index for the current data point in each dimension
        bin_indices = tuple(int(np.floor(X_scaled[i, j] / h)) for j in range(d))
        
        # Update the histogram estimator
        fbm[bin_indices] += 1


    # Normalize the histogram estimator
    fbm /= n
    #TODO CONTINUE the implementation of privacy options
    print(np.sum(fbm),fbm, np.full(fbm_shape, delta))
    
    fbm = (1-delta)*fbm + delta/(m_per_axis**d)

    print(fbm, np.sum(fbm))
    # Check if the sum of elements is equal to 1
    
    assert np.isclose(np.sum(fbm), 1.0), "Error: Sum of histogram elements is not equal to 1."
    
    
    rescaling_factors = np.array([np.min(X, axis=0), np.max(X, axis=0)])


    return fbm, rescaling_factors

X = np.random.normal(loc=0, scale=1, size=(1000, 2))

modified_histogram_estimator(X)
