import numpy as np

def standard_softmax(scores):
    """Traditional 2-pass softmax"""
    max_val = np.max(scores)
    exp_scores = np.exp(scores - max_val)
    return exp_scores / np.sum(exp_scores)


def online_softmax(scores):
    """Single-pass online softmax"""
    m = -np.inf  # Running max
    d = 0.0      # Running sum of exp(s - m)
    
    # Single pass: update max and sum together
    for s in scores:
        m_new = max(m, s)
        
        # Rescale old sum + add new term
        d = d * np.exp(m - m_new) + np.exp(s - m_new)
        
        m = m_new
    
    # Now compute final softmax values
    return np.exp(scores - m) / d

