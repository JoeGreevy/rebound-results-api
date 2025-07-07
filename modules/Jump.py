import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.resetwarnings()

warnings.filterwarnings("error", category=RuntimeWarning)


class Jump:
    # Properties
    # gc_i - initial ground contact for this jump and the next
    # t_i  - takeoff index
    # gct - ground-contact time
    # ft  - flight time
    # j_t  - gct + ft
    # h_calc - ft**2*9.81/8 - calculated jump height
    # rsi  - ft / gct
    # rsi_m - h_calc / gct
    # start_time - trial time of initial ground contact
    # agg_f_series - aggregated forces across jump
    # agg_f_peak - agg f peak force
    # agg_f_peak_norm - agg_f_peak / subj weight
    # peak_loc_ind - index of peak across jump
    # peak_loc_time - seconds after ground contact of peak
    # peak_loc_rel - rel position of peak in force series
    # time - the jump time-array
    # integ - integral of force time curve
    # integ_ecc - integral for eccentric portion
    # integ_conc - integral for concentric portion

    def __init__(self, idx, inds, subject):
        self.idx = idx # Jump Object Idx - crap
        
        self.subject = subject # Dependecy Injection
        
        # Landing and Take-Off Indices
        self.gc_i = [inds[0], inds[2]] # Landing Indices
        self.t_i = inds[1] # Take_Off Inds

        self.agg_f_series = self.set_agg_f_series()
        self.res_f = self.agg_f_series - (9.81* self.subject.mass)
        self.ft = self.set_ft()
        self.gct = self.set_gct()
        self.peak_loc_ind = self.set_peak_loc_ind()
        
        self.time = self.set_time() # array of time-points - bit silly to be storing in memory
        self.temp = self.set_temp() # indices and lengths [t] t refers to kinetic sample rate
        self.start_time = self.gc_i[0] / self.subject.time["force"]["fs"] # Start time of jump in seconds

        
        self.rsi = self.ft / self.gct
        self.jh = 1/2*9.81*(self.ft/2)**2
        self.rsi_adj = self.jh / self.gct
        


        self.force_dict = self.set_force_dict() # Lots of force features


        self.total_impulse = np.trapz(self.res_f[:self.temp["f"]["gc_l"]+1], dx=1/2000) # Ns
        self.vel_change = self.total_impulse / self.subject.mass # m/s
        self.avg_res_force = self.total_impulse / self.gct # N

        ## Should just be using self.temp["t"]["j_inds"]
        # self.forces = self.set_forces()
        # self.anat_locs = self.set_anat_locs(subject)
        self.active_plates = self.set_active_plates()
        
        #self.integ = self.set_integ()

        ## Functions called after link_segment is defined in the trial
        # get_kinetic_features()
        # get_kinematic_features()
        # get_force_features()

        self.weight_dist = -1 # -1 unset, 0 muddled, 1 split
        self.angle_dict = {
            "foot" : {
                side : {
                    # Everything will relate to x-z plane for the moment
                } for side in ["left", "right"]
            }
        }


    
    def get_kinetic_features(self):
        sides = ["left", "right"]
        joints = ["ankle", "knee", "hip"]
        segments = ["foot", "shank", "thigh"]
        ls = self.subject.link_segment
        
        t_inds = np.arange(self.temp["t"]["gc_i"][0], self.temp["t"]["t_i"]+1)
        N = len(t_inds)

        # Kinetic Features for both sides
        # Moment, Powers, RF
        kinetics = {
            "moments" : {
                "ankle" : {
                    side : {
                        "A1" : np.max(ls["foot"][side]["prox"]["M"][t_inds]) / self.subject.mass,
                        "A1_loc" : np.argmax(ls["foot"][side]["prox"]["M"][t_inds]) / N,
                        "avg" : np.mean(ls["foot"][side]["prox"]["M"][t_inds]) / self.subject.mass,
                    } for side in sides
                },
                "knee" : {
                    side : {
                        "K1" : np.max(ls["shank"][side]["prox"]["M"][t_inds[:N//4]]) / self.subject.mass,
                        "K1_loc" : np.argmax(ls["shank"][side]["prox"]["M"][t_inds[:N//4]]) / N,
                        "K2" : np.min(ls["shank"][side]["prox"]["M"][t_inds[N//10:N//3]]) / self.subject.mass,
                        "K2_loc" : (np.argmin(ls["shank"][side]["prox"]["M"][t_inds[N//10:N//3]]) + N//10) / N,
                        "K3" : np.max(ls["shank"][side]["prox"]["M"][t_inds[N//4:]]) / self.subject.mass,
                        "K3_loc" : (np.argmax(ls["shank"][side]["prox"]["M"][t_inds[N//4:]]) + N//4) / N,
                        "K4": np.min(ls["shank"][side]["prox"]["M"][t_inds[N*4//5:]]) / self.subject.mass,
                        "K4_loc": (np.argmin(ls["shank"][side]["prox"]["M"][t_inds[N*4//5:]]) + N*4//5) / N,
                        "avg" : np.mean(ls["shank"][side]["prox"]["M"][t_inds]) / self.subject.mass,
                    } for side in sides
                },
                "hip": {
                    side : {
                        "H1" : np.min(ls["thigh"][side]["prox"]["M"][t_inds[:N//4]]) / self.subject.mass,
                        "H1_loc" : np.argmin(ls["thigh"][side]["prox"]["M"][t_inds[:N//4]]) / N,
                        "H2" : np.max(ls["thigh"][side]["prox"]["M"][t_inds] / self.subject.mass),
                        "H2_loc" : np.argmax(ls["thigh"][side]["prox"]["M"][t_inds]) / N,
                        "avg" : np.mean(ls["thigh"][side]["prox"]["M"][t_inds]) / self.subject.mass,
                    } for side in sides
                }
            },
            "power" : {
                seg : {
                    side : {
                        "peak_conc" : np.max(ls[seg][side]["power"][t_inds] / self.subject.mass),
                        "peak_ecc" : np.min(ls[seg][side]["power"][t_inds] / self.subject.mass),
                        "net_work" : np.trapz(ls[seg][side]["power"][t_inds] / self.subject.mass, dx=1/1000),
                        "work_conc" : np.trapz(ls[seg][side]["power"][t_inds][ls[seg][side]["power"][t_inds] > 0] / self.subject.mass, dx=1/1000),
                        "work_ecc" : np.trapz(ls[seg][side]["power"][t_inds][ls[seg][side]["power"][t_inds] < 0] / self.subject.mass, dx=1/1000),
                        "work_ecc_late" : np.trapz(ls[seg][side]["power"][t_inds[N//2:]][ls[seg][side]["power"][t_inds[N//2:]] < 0] / self.subject.mass, dx=1/1000),
                    } for side in sides
                } for seg in joints
            },
            "stiffness" : {
                joint: {
                    side : {
                        "k" : [],
                        "mom_loc" : [], # Peak Moment Location
                        "ang_loc" : [], # Peak Angular Displacement
                    } for side in sides
                } for joint in joints
            },
            "rf" : {
                side : {
                    "peak" : np.max(ls["foot"][side]["dist"]["RF_z"][t_inds]),
                    "loc" : np.argmax(ls["foot"][side]["dist"]["RF_z"][t_inds]) / N,
                } for side in sides
            } 
        }
        
        for joint, seg in zip(joints, segments):
            for side in sides:
                m = ls[seg][side]["prox"]["M"][t_inds]
                m_peak, m_loc = np.max(m), np.argmax(m)
                a = ls[joint][side]["ang"][t_inds] - ls[joint][side]["ang"][t_inds[0]]
                a_peak, a_loc = np.min(a), np.argmin(a)
                
                kinetics["stiffness"][joint][side]["mom_loc"] = m_loc / N
                kinetics["stiffness"][joint][side]["ang_loc"] = a_loc / N
                kinetics["stiffness"][joint][side]["M"] = m_peak
                kinetics["stiffness"][joint][side]["a"] = a_peak

                if m_loc/N < 0.3 or a_loc/N < 0.3 or m_loc/N > 0.66 or a_loc/N > 0.66 or abs(m_loc - a_loc) / N > 0.15:
                    # Spring Model is bad
                    kinetics["stiffness"][joint][side]["k"] = np.nan
                else:   
                    kinetics["stiffness"][joint][side]["k"] = m_peak / a_peak

        thigh_inds = np.arange(self.temp["t"]["gc_i"][0], self.temp["t"]["t_i"]+1)
        sides2 = ["left", "right"]
        if self.subject.id[-3:] == "105":
            sides2 = ["left", "left"]
        elif self.subject.id[-1] == "106":
            sides2 = ["right", "right"]

        # Leg Stiffness Measures
        thigh_dict = {
            side : np.mean(self.subject.markers["thigh"][side]["z"][thigh_inds, :], axis=1) for side in sides2
        }
        th_height_abs = np.mean([thigh_dict[side] for side in sides2], axis=0) / 1000
        th_height = th_height_abs - th_height_abs[0]
        kinetics["stiffness"]["leg"] = {
            "k" : np.max(self.res_f) / abs(np.min(th_height)),
            "F" : np.max(self.res_f),
            "L" : np.min(th_height),
            "f_loc" : np.argmax(self.res_f) / (N*2),
            "h_loc" : np.argmin(th_height) / len(thigh_inds),
            "th_land" : th_height_abs[0],
            "th_land_off": th_height[-1],
            # phase_shift = "f_loc" - "h_loc" how much later is the force peak than the height peak
        }

        ## Combined Moments and Powers across legs
        for joint in joints:
            kinetics["moments"][joint]["total"] = {}
            kinetics["moments"][joint]["difference"] = {}
            valid_sides = ["left", "right"]
            if self.subject.id[-3:] in ["105", "106"]:
                if self.subject.id[-1] == "5":
                    valid_sides = ["left", "left"]
                else:
                    valid_sides = ["right", "right"]

            for key in kinetics["moments"][joint]["left"]:
                stat_arr = [kinetics["moments"][joint][s][key] for s in valid_sides]
                kinetics["moments"][joint]["total"][key] = np.mean(stat_arr)
                kinetics["moments"][joint]["difference"][key] = np.diff(stat_arr)[0]

            # Powers is just moments repeated
            kinetics["power"][joint]["total"] = {}
            kinetics["power"][joint]["difference"] = {}
            for key in kinetics["power"][joint]["left"]:
                stat_arr = [kinetics["power"][joint][s][key] for s in valid_sides]

                kinetics["power"][joint]["total"][key] = np.mean(stat_arr)
                kinetics["power"][joint]["difference"][key] = np.diff(stat_arr)[0]

            # kinetics["stiffness"][joint]["total"] = {}
            # kinetics["stiffness"][joint]["difference"] = {}
            # for key in ["k", "M"]:
            #     stat_arr = [kinetics["stiffness"][joint][s][key] for s in valid_sides]

            #     kinetics["stiffness"][joint]["total"][key] = np.mean(stat_arr)
            #     kinetics["stiffness"][joint]["difference"][key] = np.diff(stat_arr)[0]
            # for key in ["mom_loc", "ang_loc", "a"]:
            #     stat_arr = [kinetics["stiffness"][joint][s][key] for s in valid_sides]
            #     kinetics["stiffness"][joint]["total"][key] = np.sum(stat_arr) / 2

        kinetics["rf"]["total"] = {
            "peak" : kinetics["rf"]["left"]["peak"] + kinetics["rf"]["right"]["peak"],
        }
        kinetics["rf"]["difference"] = {
            "peak" : kinetics["rf"]["left"]["peak"] - kinetics["rf"]["right"]["peak"],
            "loc" : kinetics["rf"]["left"]["loc"] - kinetics["rf"]["right"]["loc"],
        }

        # The relative features but now moved into "moments" and "power"
        for side in ["total", "left", "right"]:
            total_support = np.sum([kinetics["moments"][joint][side]["avg"] for joint in joints])
            total_concentric = np.sum([kinetics["power"][joint][side]["work_conc"] for joint in joints])
            total_eccentric = np.sum([kinetics["power"][joint][side]["work_ecc"] for joint in joints])
            for joint in joints:
                kinetics["power"][joint][side]["relative"] = {
                    "conc_prop" : kinetics["power"][joint][side]["work_conc"] / (kinetics["power"][joint][side]["work_conc"] + abs(kinetics["power"][joint][side]["work_ecc"])),
                    "concentric" :  kinetics["power"][joint][side]["work_conc"] / total_concentric,
                    "eccentric" :  kinetics["power"][joint][side]["work_ecc"] / total_eccentric 
                }

                kinetics["moments"][joint][side]["relative"] = {
                    "support" :  kinetics["moments"][joint][side]["avg"] / total_support, 
                }

        self.kinetics = kinetics

    def get_kinematic_features(self):
        sides = ["left", "right"]
        joints = ["ankle", "knee", "hip"]
        segments = ["foot", "shank", "thigh"]
        ls = self.subject.link_segment
        t_inds = np.arange(self.temp["t"]["gc_i"][0], self.temp["t"]["t_i"]+1)
        kin = {
            joint : {
                side : {
                    "td" : ls[joint][side]["ang"][t_inds[0]],
                    "to" : ls[joint][side]["ang"][t_inds[-1]],
                    "disp" : np.min(ls[joint][side]["ang"][t_inds]) - ls[joint][side]["ang"][t_inds[0]],
                    "flex" : np.min(ls[joint][side]["ang"][t_inds]),
                } for side in sides
            } for joint in joints
        }
        segs = {
            "foot" : {
                side : self.subject.seg_angles["foot"][side]["xz"].copy()*-1 for side in sides
            },
            "shank" : {
                side : self.subject.seg_angles["shank"][side]["xz"].copy() for side in sides
            },
            "thigh" : {
                side : np.pi - self.subject.seg_angles["thigh"][side]["xz"].copy() for side in sides
            }
        }
        for seg in segments:
            kin[seg] = {
                side : {
                    "td": segs[seg][side][t_inds[0]],
                    "to": segs[seg][side][t_inds[-1]],
                    "flex": np.min(segs[seg][side][t_inds]),
                    "disp": np.min(segs[seg][side][t_inds]) - segs[seg][side][t_inds[0]],
                } for side in sides
            }
        for j in kin.keys():
            kin[j]["total"] = {}
            kin[j]["diff"] = {}
            for feat in kin[j]["left"].keys():
                kin[j]["total"][feat] = (kin[j]["left"][feat] + kin[j]["right"][feat]) / 2
                kin[j]["diff"][feat] = kin[j]["left"][feat] - kin[j]["right"][feat]
        
        self.kinematics =  kin
    
    def get_force_features(self):
        t_inds = np.arange(self.temp["f"]["gc_i"][0], self.temp["f"]["t_i"]+1)
        N = len(t_inds) # Force Sampling Frequency
        time = N / 2000
        F = self.res_f[:N]
        peak, peak_loc = np.max(F), np.argmax(F)
        vel_change = np.cumsum(F) * (1/2000) / self.subject.mass
        v_avg = np.sum(vel_change)/len(vel_change) # average velocity change
        com_change = self.kinetics["stiffness"]["leg"]["th_land_off"]
        # notation is aotp
        v_off = com_change / time # velocity at takeoff
        v = vel_change - v_avg + v_off # v = v(0) + vel_change ||| v(0) = v_off - v_avg
        self.v = v
        peak_disp_ind = np.flatnonzero(v > 0)[0]
        peak_disp_loc = peak_disp_ind / N
        peak_disp = np.trapz(v[:peak_disp_ind], dx=1/2000) # m

        # Eccentric Proportion was a stupid methric, just capturing that 
        # subjects tend to take off with less kinetic energy that they land with.
        # ecc_imp = np.trapz(F[:peak_disp_ind], dx=1/2000) # N*m
        # conc_imp = np.trapz(F[peak_disp_ind:], dx=1/2000) # N*m
        if len(F[:peak_disp_ind]) == 0 or len(F[peak_disp_ind:]) == 0:
            avg_ecc = 0
            avg_conc = 0
            loc_p_rfd, loc_min_rfd, rates = 0, 0, [0, 1]

        else:
            avg_ecc = np.sum(F[:peak_disp_ind]) / len(F[:peak_disp_ind])
            avg_conc = np.sum(F[peak_disp_ind:]) / len(F[peak_disp_ind:])

            # Rate of Force Development
            f_10_ms = F[::20] # 10 ms intervals
            self.rates = ((f_10_ms[3:] - f_10_ms[:-3]) / 0.03) # N/s
            self.rates /= (self.subject.mass * 9.81) # BW/s
            rates= self.rates

            loc_p_rfd = (np.argmax(rates) * 0.01 + 0.01) / self.gct
            loc_min_rfd = (np.argmin(rates) * 0.01 + 0.01) / self.gct


        self.force = {
            "peak" : peak / (self.subject.mass * 9.81),
            "peak_loc" : peak_loc / N,
            "vel_change": vel_change[-1],
            "peak_disp_val" : peak_disp,
            "peak_disp" : peak_disp_loc,
            "avg_force": np.mean(F) / (self.subject.mass * 9.81),
            "avg_ecc" : avg_ecc/ (self.subject.mass * 9.81),
            "avg_conc" : avg_conc / (self.subject.mass * 9.81),
            "peak_rfd": np.max(rates),
            "min_rfd": np.min(rates),
            "peak_rfd_loc": loc_p_rfd,
            "min_rfd_loc": loc_min_rfd,
        }



    def set_agg_f_series(self):
        inds = self.gc_i
        return self.subject.force["z"]["agg"][inds[0]:inds[1]]
    
    def set_ft(self):
        return (self.gc_i[1] - self.t_i) / self.subject.time["force"]["fs"] # Flight Time
    
    def set_peak_loc_ind(self):
        return np.argmax(self.agg_f_series)

    def set_temp(self):
        # 
        ds = 2 # downsample factor, previously 10
        t_gc_i = np.round(np.array(self.gc_i) / ds).astype(int)
        t_t_i = round(self.t_i / ds)
        t_peak_ind = round(self.peak_loc_ind / ds)
        temp_dict = {
            "f" : {
                "gc_i": self.gc_i,
                "t_i": self.t_i,
                "peak_i": self.peak_loc_ind,
                "j_l": self.gc_i[1] - self.gc_i[0],
                "gc_l": self.t_i - self.gc_i[0],
                "f_l": self.gc_i[1] - self.t_i,
                "ecc_l": self.peak_loc_ind,
                "conc_l": (self.t_i - self.gc_i[0] - self.peak_loc_ind),
                "j_inds": np.arange(self.gc_i[0], self.t_i+1),
            },
            "t" : {
                "gc_i": t_gc_i,
                "t_i": t_t_i ,
                "peak_i": t_peak_ind,
                "j_l": t_gc_i[1] - t_gc_i[0], 
                "gc_l": t_t_i - t_gc_i[0],
                "f_l": t_gc_i[1] - t_t_i,
                "conc_l": t_t_i - t_peak_ind,# wrong i think
                "ecc_l": t_peak_ind - t_gc_i[0],
                "gc_inds": np.arange(t_gc_i[0], t_t_i),
                "j_inds": np.arange(t_gc_i[0], t_gc_i[1]),
            }
        }
        return temp_dict

    def set_force_dict(self):
        """
        Aggregated Force Features: integ, avg, peak
        """
        peak_ind = np.argmax(self.agg_f_series)
        peak = np.max(self.agg_f_series)
        ecc_t = peak_ind / self.subject.time["force"]["fs"] 
        conc_t = (self.temp["f"]["gc_l"] - peak_ind) / self.subject.time["force"]["fs"] 
        half_max = peak / 2
        fwhm_inds = np.flatnonzero(self.agg_f_series > half_max)
        weight = self.subject.mass * 9.81
        ecc_int = np.trapz(self.agg_f_series[:peak_ind], self.time[:peak_ind])
        conc_int = np.trapz(self.agg_f_series[peak_ind:], self.time[peak_ind:])
        fwhm_int = np.trapz(self.agg_f_series[fwhm_inds[0]:fwhm_inds[-1]+1], self.time[fwhm_inds[0]:fwhm_inds[-1]+1])
        full_int = ecc_int + conc_int
        force_dict = {
            "integ" : full_int,
            "avg": full_int / self.time[-1],
            "peak": {
                "f" : self.agg_f_peak,
                "ind" : peak_ind,
                "ttp" : peak_ind / self.subject.time["force"]["fs"],
                "rel_ttp": peak_ind / self.temp["f"]["gc_l"],
            },
            "ecc": {
                "int" : ecc_int,
                "t" : ecc_t,
                "afd": peak / (ecc_t) # average force dev
            },
            "concentric": {
                "int" : conc_int,
                "t" : conc_t,
                "afd" : peak / conc_t
            },
            "fwhm": {
                "init": fwhm_inds[0],
                "end": fwhm_inds[-1],
                "len": fwhm_inds[-1] - fwhm_inds[0],
                "int" : fwhm_int,
                "efd" : half_max / (peak_ind - fwhm_inds[0]),
                "cfd" : half_max / (fwhm_inds[-1]+1 - peak_ind)
            },
            "norm": {
                "peak": self.agg_f_peak_norm,
                "int" : full_int / weight,
                "ecc" : {
                    "int": ecc_int/weight,
                    "afd": peak / ecc_t / weight
                },
                "conc" : {
                    "int": conc_int/weight,
                    "afd": peak / conc_t / weight
                },
                "fwhm" : {
                    "int": fwhm_int / weight,
                    "efd" : half_max / (peak_ind - fwhm_inds[0]) / weight,
                    "cfd" : half_max / (fwhm_inds[-1]+1 - peak_ind) / weight
                }
            }

        }
        return force_dict

    # def set_forces(self):
    #     ds = 2 # downsample factor previously 10
    #     forces = {
    #         "cop" : {
    #             ax : {} for ax in ["x", "y"]
    #         }, 
    #         "downsampled" : {
    #             "cop" : {
    #                 ax : {} for ax in ["x", "y"]
    #             }
    #         }      
    #     }
    #     gc = self.temp["f"]["gc_i"][0]
    #     gc1 = self.temp["f"]["gc_i"][1]
    #     t_off = self.temp["f"]["t_i"]
    #     for ax in ["x", "y", "z"]:
    #         forces[ax] = {}
    #         for plate in np.arange(4):
    #             forces[ax][plate] = self.subject.force[ax][plate][gc:gc1]

        
    #     jump_inds = np.arange(gc, t_off+1)
    #     plates = [[3, 2], [1, 0]]
    #     for plate in np.arange(4):
    #         v_force = self.subject.force["z"][plate][jump_inds] 
    #         thresh = np.flatnonzero(v_force > 85)
            
    #         if len(thresh) > 2:
    #             t_inds = thresh[0], thresh[-1]
    #             conf_int = np.arange(t_inds[0], t_inds[1])
    #             for ax in ["x", "y"]:
    #                 ts = self.subject.force["cop"][ax][plate][jump_inds]
    #                 forces["cop"][ax][plate] = np.pad(ts[conf_int], (thresh[0], len(ts)- thresh[-1]-1), mode="edge")
    #                 forces["downsampled"]["cop"][ax][plate] = forces["cop"][ax][plate][::ds]
    #         else:
    #             for ax in ["x", "y"]:
    #                 forces["cop"][ax][plate] = np.zeros(len(v_force))
    #                 forces["downsampled"]["cop"][ax][plate] = np.zeros(int(len(v_force) // ds))
                
        
                

    #     return forces
    
    def set_active_plates(self):
        inds = self.temp["f"]["j_inds"]
        forces = np.vstack([self.subject.force["z"][pl][inds] for pl in np.arange(4)])
        active = np.flatnonzero(np.max(forces, axis=1) > 50)
        return active
    
    def set_gct(self):
        return (self.t_i - self.gc_i[0]) / self.subject.time["force"]["fs"] # Ground Contact Time
    
    
    @property
    def j_t(self):
        return self.gct + self.ft
    @property
    def h_calc(self):
        return self.ft**2*9.81/8
    

    # def set_rsi(self):
    #     return self.ft / self.gct
    @property
    def rsi_m(self):
        return self.h_calc / self.gct


    @property
    def agg_f_peak(self):
        return np.max(self.agg_f_series)    
    @property
    def agg_f_peak_norm(self):
        return self.agg_f_peak / (self.subject.mass*9.81)
    
    @property
    def peak_loc_time(self):
        return self.peak_loc_ind / self.time["force"]["fs"]
    @property
    def peak_loc_rel(self):
        return self.peak_loc_time / self.gct
    
    def set_time(self):
        nPts = len(self.agg_f_series)
        return np.linspace(0, (self.gc_i[1] - self.gc_i[0]) / self.subject.time["force"]["fs"], nPts)
    
    # def set_integ(self):
    #     return np.trapz(self.agg_f_series, self.time)
    # @property
    # def integ_ecc(self):
    #     return np.trapz(self.agg_f_series[:self.peak_loc_ind], self.time[:self.peak_loc_ind])
    # @property
    # def integ_conc(self):
    #     return np.trapz(self.agg_f_series[self.peak_loc_ind:], self.time[self.peak_loc_ind:])

        
    def foot_angle(self):
        # Landing Angle, Peak_Force Angle or Mid GC_Point Angle, Take_off Angle, Max_Height Angle -> Back to landing angle
        gc_i = self.temp["t"]["gc_i"]
        t_i = self.temp["t"]["t_i"]
        for side in ["left", "right"]:
            ts_j = self.subject.seg_angles["foot"][side]["xz"][gc_i[0]:gc_i[1]+1]

            self.angle_dict["foot"][side]["xz"] = ts_j
            self.angle_dict["foot"][side]["landing"] = ts_j[0]
            self.angle_dict["foot"][side]["peak_force"] = ts_j[self.temp["t"]["peak_i"]]
            self.angle_dict["foot"][side]["takeoff"] = ts_j[self.temp["t"]["gc_l"]]
            self.angle_dict["foot"][side]["max_h"] = ts_j[self.temp["t"]["gc_l"] + (self.temp["t"]["f_l"]//2)]
            self.angle_dict["foot"][side]["landing2"] = ts_j[self.temp["t"]["j_l"]]

        

    
