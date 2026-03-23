import os
import sys
from pathlib import Path

import numpy as np

from motion_pred.utils.dataset import Dataset
from motion_pred.utils.skeleton import Skeleton

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from common.preprocessing import select_most_active_hand, split_train_val_test  # type: ignore


class DatasetAssembly(Dataset):
    def __init__(
        self,
        mode,
        t_his=70,
        t_pred=30,
        actions="all",
        data_dir=None,
        action_filter="",
        stride=5,
        seed=0,
        time_interp=None,
        window_norm=None,
        use_vel=False,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.action_filter = action_filter or ""
        self.stride = max(1, int(stride))
        self.seed = int(seed)
        self.time_interp = time_interp
        self.window_norm = int(window_norm) if window_norm is not None else int(t_his)
        self.use_vel = use_vel
        super().__init__(mode, t_his, t_pred, actions)
        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        if not self.data_dir:
            self.data_dir = "/mnt/turing-datasets/AssemblyHands/assembly101-download-scripts/data_our/"

        self.subjects_split = {}
        self.subjects = [self.mode]
        self.kept_joints = np.arange(21)
        # Root at wrist followed by the remaining local hand joints.
        self.skeleton = Skeleton(
            parents=[-1, 7, 10, 13, 16, 19, 0, 6, 0, 8, 9, 0, 11, 12, 0, 14, 15, 0, 17, 18, 0],
            joints_left=[],
            joints_right=[],
        )
        self.process_data()

    def process_data(self):
        train_files, val_files, test_files = split_train_val_test(
            data_dir=self.data_dir,
            action_filter=self.action_filter,
            seed=self.seed,
        )
        split_files = {"train": train_files, "val": val_files, "test": test_files}
        if self.mode not in split_files:
            raise ValueError(f"Unknown Assembly split '{self.mode}'. Expected one of train/val/test.")

        sequences = {}
        self.norm_factors = {}
        for file_path in split_files[self.mode]:
            selected = select_most_active_hand(
                file_path,
                time_interp=self.time_interp,
                window_norm=self.window_norm,
            )
            if selected is None:
                continue

            seq, norm_factor = selected
            if seq.shape[0] < self.t_total:
                continue

            if self.use_vel:
                velocity = np.diff(seq[:, :1], axis=0)
                velocity = np.append(velocity, velocity[[-1]], axis=0)
            seq = seq.copy()
            seq[:, 1:] -= seq[:, :1]
            if self.use_vel:
                seq = np.concatenate((seq, velocity), axis=1)
            seq_name = os.path.basename(file_path)
            sequences[seq_name] = seq.astype(np.float32)
            self.norm_factors[seq_name] = float(norm_factor)

        if not sequences:
            raise RuntimeError(f"No valid Assembly sequences remained for split '{self.mode}'.")
        self.data = {self.mode: sequences}

    def iter_generator(self, step=None):
        for data_s in self.data.values():
            for seq in data_s.values():
                max_start = seq.shape[0] - self.t_total
                if max_start <= 0:
                    continue
                for start in range(0, max_start, self.stride):
                    yield seq[None, start : start + self.t_total]

    def iter_generator_with_scale(self):
        for data_s in self.data.values():
            for seq_name, seq in data_s.items():
                max_start = seq.shape[0] - self.t_total
                if max_start <= 0:
                    continue
                norm_factor = self.norm_factors.get(seq_name, 1.0)
                for start in range(0, max_start, self.stride):
                    yield seq[None, start : start + self.t_total], float(norm_factor)
