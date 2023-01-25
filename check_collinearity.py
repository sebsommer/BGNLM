def check_collinearity(val, values):
    return np.any(np.any(val == values, axis=0))