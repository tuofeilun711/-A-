"""HDF5 data loader for bridge flow data."""
import h5py
import numpy as np
from pathlib import Path

class BridgeFlowData:
    def __init__(self, h5_path: str, pylibs_path: str = None):
        if pylibs_path:
            import sys
            sys.path.insert(0, pylibs_path)
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 not found: {h5_path}")
        self.trajectories = {}

    def load(self):
        with h5py.File(self.h5_path, 'r') as f:
            for role in ['train', 'validation', 'test', 'transition']:
                if role not in f:
                    continue
                self.trajectories[role] = {}
                for traj_id in f[role]:
                    grp = f[role][traj_id]
                    self.trajectories[role][traj_id] = {
                        'field': grp['field'][:],
                        'time': grp['time'][:],
                        'mask': grp['mask'][:],
                        'coordinates': grp['coordinates'][:],
                        'x': grp['x'][:],
                        'z': grp['z'][:],
                        'source_indices': grp['source_indices'][:],
                        'n_frames': grp['field'].shape[0],
                    }
        self._compute_global_stats()
        return self

    def _compute_global_stats(self):
        all_fields = []
        for traj_id, data in self.trajectories.get('train', {}).items():
            all_fields.append(data['field'])
        if all_fields:
            cat = np.concatenate(all_fields, axis=0)
            self.global_mean = cat.mean(axis=(0,), keepdims=True)
            self.global_std = cat.std(axis=(0,), keepdims=True) + 1e-8
        else:
            self.global_mean = 0.0
            self.global_std = 1.0

    def get_fluid_mask(self):
        for role in ['train', 'validation', 'test']:
            if role in self.trajectories:
                first_id = list(self.trajectories[role].keys())[0]
                return self.trajectories[role][first_id]['mask'].astype(bool)
        raise ValueError("No trajectories loaded")

    def normalize(self, field):
        return (field - self.global_mean) / self.global_std

    def denormalize(self, field_norm):
        return field_norm * self.global_std + self.global_mean

    def summary(self):
        lines = ["=== BridgeFlowData ==="]
        for role in ['train', 'validation', 'test', 'transition']:
            if role in self.trajectories:
                lines.append(f"  {role}:")
                for tid, d in self.trajectories[role].items():
                    lines.append(f"    {tid}: {d['n_frames']} frames, field={d['field'].shape}")
        return "\n".join(lines)


def build_snapshot_matrix(fields, fluid_indices=None):
    """Flatten fields to snapshot matrix for POD.
    fields: (..., 2, H, W) -> (..., 2*N_fluid) using fluid_indices.
    If fluid_indices is None, return all pixels.
    """
    fields = np.asarray(fields)
    shape = fields.shape
    batch_dims = shape[:-3]
    C, H, W = shape[-3:]
    flat = fields.reshape(*batch_dims, C * H * W)
    if fluid_indices is not None:
        return flat[..., fluid_indices]
    return flat


def create_sequences(data, input_len, pred_len, stride=1):
    T = data.shape[0]
    inputs, targets = [], []
    for i in range(0, T - input_len - pred_len + 1, stride):
        inputs.append(data[i:i+input_len])
        targets.append(data[i+input_len:i+input_len+pred_len])
    if not inputs:
        return np.array([]), np.array([])
    return np.stack(inputs), np.stack(targets)


def create_train_val_data(bridge_data, input_len=10, pred_len=20, stride=1):
    train_inputs, train_targets = [], []
    for tid, d in bridge_data.trajectories.get('train', {}).items():
        field_norm = bridge_data.normalize(d['field'])
        inp, tgt = create_sequences(field_norm, input_len, pred_len, stride)
        if len(inp) > 0:
            train_inputs.append(inp)
            train_targets.append(tgt)

    val_inputs, val_targets = [], []
    for tid, d in bridge_data.trajectories.get('validation', {}).items():
        field_norm = bridge_data.normalize(d['field'])
        inp, tgt = create_sequences(field_norm, input_len, pred_len, stride)
        if len(inp) > 0:
            val_inputs.append(inp)
            val_targets.append(tgt)

    return {
        'train': (np.concatenate(train_inputs) if train_inputs else np.array([]),
                   np.concatenate(train_targets) if train_targets else np.array([])),
        'val': (np.concatenate(val_inputs) if val_inputs else np.array([]),
                 np.concatenate(val_targets) if val_targets else np.array([])),
    }
