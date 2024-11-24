from dtw import *

def DTW_function(reference_data, target_data, parameter, start_idx, end_idx):
    """
    Perform Dynamic Time Warping (DTW) on the specified parameter.
    """
    query = reference_data.bx.values[start_idx:end_idx]
    template = target_data.bx.values[start_idx:end_idx] #the reference for DTW

    alignment = dtw(query, template, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c"))
    #indices = alignment.index2  # Warping indices
    indices = warp(alignment, index_reference=False) 
    
    return query, template, alignment, indices

def warp_function(data, indices, parameter):
    """
    Warp the specified parameter using DTW indices.
    """
    return data[parameter].values[indices]

