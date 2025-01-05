import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class corticalParams:
    def __init__(self):
        # Fixed operations
        self.Operation_Niflheim = 1  # Importing realistic head turning
        self.Operation_Noatun = 1  # Extending HD trajectory with linearity
        self.Operation_Nord = 0  # Incoming proximal cues (waiting for renewing...)
        self.Operation_Nidhogg = 1  # Scaling on visual input signals
        self.Operation_Urdar_brunnr = 0  # Centralization on visual input signals
        self.Operation_Ratatoskr = 0  # 'Quite' perfect HD weights
        self.Operation_Cyaegha = 0  # Random initial HD in each environment
        self.Operation_Loki = 0  # Visual STM instead of visual PM
        self.Operation_Valhalla = 0  # Restricted visual field
        self.Operation_Bifrost = 0  # Featural attention
        self.Operation_Odin = 0  # Multi-environment wandering
        self.duration_Odin = 10  # Single-environment wandering time (seconds)

        # Tune the following operations for different simulations (Figs 1-4, S1-S5, S10)
        self.Operation_Fimbulvetr = 1  # Containing feedback transmission (1 for OSA, 0 for others)

        # Specific moving cue (1 to N_cue, 0 for disabled)
        self.Operation_Jormungand = 0  # Blue cue (e.g., 2 for Figs 3A-B)
        self.velocity_Jormungand = 0  # Moving velocity (degrees/second; 90 for Fig 3A)
        self.jumping_Jormungand = 0  # Duration after randomly jumping (seconds; 10 for Fig 3B)

        # Single-environment simulation (e.g., Fig 2)
        self.Operation_Zwerge = 0  # Duration for taking aLB testing snapshots at beginning (seconds; 300 for S2 Fig)
        self.snapshot_interval = 0  # Temporal interval of snapshots (seconds; 15 for S2 Fig)
    