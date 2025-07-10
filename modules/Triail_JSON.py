class Triail_JSON:
    def __init__(self, subj):
        # Import the trial
        self.path = subj.path
        # Can't contain trial data
        self.id = subj.id
        self.date = subj.date
        self.pro = subj.pro
        self.output_str = subj.outputStr
        self.time = subj.time

        #self.force = subj.force # Too Many Samples

        for jump in subj.jumps:
            if hasattr(jump, "forces"):
                del jump.forces
            if hasattr(jump, "anat_locs"):
                del jump.anat_locs
            jump.subject = self
        self.jumps = subj.jumps
        self.trial_indices = subj.trial_indices


        # Plan an anthro dict in the future
        self.mass = subj.mass
        self.snapshot = subj.snapshot
        if self.pro == "30":
            self.start_inds, self.end_inds = subj.start_inds, subj.end_inds
        self.stats = subj.stats