import numpy as np

def get_feat(jump, feat, jIdx):
    """
    Extracts the specified feature from a jump object.
    
    Parameters:
    jump (object): The jump object containing the features.
    feat (str): The feature to extract from the jump object.
    
    Returns:
    list: A list of values for the specified feature.
    """
    j2seg = {
        "ank": "ankle",
        "kne": "knee",
        "hip": "hip",
    }
    power2feat = {
        "pc": "peak_conc",
        "pe": "peak_ecc",
        "nw": "net_work",
        "cw": "work_conc",
        "ew": "work_ecc",
    }
    rel2feat = {
        "cp": "conc_prop",
        "cc": "concentric",
        "ec": "eccentric",
    }
    feat_arr = feat.split("-")
    if feat_arr[0] in ["jh", "ft", "gct", "rsi", "rsi_adj"]:
        val = getattr(jump, feat)
        return val
    elif feat_arr[0] == "start_time":
        t = getattr(jump, feat) - jump.subject.jumps[jump.subject.trial_indices[0]].start_time
        return t
    elif feat_arr[0] == "idx":
        return jIdx+1 # Truly awful, idx property should be set properly in data processing
    elif feat_arr[0] in ["peak", "peak_loc", "vel_change", "avg_force", "avg_ecc", "avg_conc"]:
        val = jump.force[feat]
        return val
    elif feat_arr[0] == "pow":
        power = jump.kinetics["power"][j2seg[feat_arr[2]]]
        if feat_arr[1] == "both":
            power = power["total"]
            if feat_arr[3] in power2feat.keys():
                return power[power2feat[feat_arr[3]]]
            elif feat_arr[3] in rel2feat.keys():
                return power["relative"][rel2feat[feat_arr[3]]]
            else:
                print("Missed")
                return 0
    elif feat_arr[0] == "mom":
        #mom-side-joint-feat
        moments = jump.kinetics["moments"][j2seg[feat_arr[2]]]
        if feat_arr[1] == "both":
            moments = moments["total"]
        else:
            moments = moments[feat_arr[1]]
        feature = feat_arr[3]
        if feature == "avg_mom":
            feature = "avg"
        return moments[feature]
    elif feat_arr[0] == "kin":
        side = feat_arr[1]
        joint = j2seg[feat_arr[2]]

        if side == "both":
            side = "total"
        
        return jump.kinematics[joint][side][feat_arr[3]] * 180 / np.pi
        

        
        
    elif hasattr(jump, feat):
        return getattr(jump, feat)
    else:
        return 0