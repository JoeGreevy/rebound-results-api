class Subject_JSON:
    def __init__(self, subj):
        # Import the trial
        self.path = subj.path
        # Can't contain trial data
        self.id = subj.id
        self.output_str = subj.outPutStr

        self.track_len = subj.n_samples_tracks

        # Cut down to more dictionaries
        self.agg_f_z = subj.agg_f_z
        self.ts_force = subj.ts_force
        self.fs_force = 2000
        self.force = subj.force

        for jump in subj.jumps:
            jump.subject = self
        self.jumps = subj.jumps
        self.trial_indices = subj.trial_indices

        # Integrate in the future
        self.ten_jump = 0

        # Marker Velocities being held out until a marker v dict is created

        # Plan an anthro dict in the future
        self.mass = subj.mass

        self.snapshot = subj.snapshot

        self.basic_stats = subj.basic_stats

        self.markers = subj.markers

        #self.angles = subj.angle_dict
        self.sagittal = subj.sagittal
        self.seg_angles = subj.seg_angles
        #temp
        self.thigh_coda = subj.thigh_coda
        self.angles_arr = subj.angles_arr

        self.anat_locs = subj.locs

        self.velocities = subj.velocities

        self.accelerations = subj.accelerations

        self.link_segment = subj.link_segment
        self.standard =subj.standard

        #self.emg_dict = subj.emg_dict

