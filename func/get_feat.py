def get_feat(jump, feat, jIdx):
    """
    Extracts the specified feature from a jump object.
    
    Parameters:
    jump (object): The jump object containing the features.
    feat (str): The feature to extract from the jump object.
    
    Returns:
    list: A list of values for the specified feature.
    """
    if feat in ["jh", "ft", "gct", "rsi", "rsi_adj"]:
        val = getattr(jump, feat)
        return "{:.2f}".format(val)
    elif feat == "start_time":
        t = getattr(jump, feat) - jump.subject.jumps[jump.subject.trial_indices[0]].start_time
        return "{:.2f}".format(t)
    elif feat == "idx":
        return jIdx+1 # Truly awful, idx property should be set properly in data processing
    elif feat in ["peak", "peak_loc", "vel_change", "avg_force", "avg_ecc", "avg_conc"]:
        val = jump.force[feat]
        return "{:.2f}".format(val)
    elif hasattr(jump, feat):
        return getattr(jump, feat)
    else:
        return ""