import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

AuxFeatureType = Union[torch.Tensor, Dict[str, torch.Tensor]]


def _slice_aux_feature(features: AuxFeatureType, idx: int):
    if torch.is_tensor(features):
        return features[idx]
    if isinstance(features, dict):
        out: Dict[str, torch.Tensor] = {}
        for key, tensor in features.items():
            if not torch.is_tensor(tensor):
                raise TypeError(f"aux_features[{key}] must be a tensor, got {type(tensor)}")
            out[key] = tensor[idx]
        return out
    raise TypeError(f"Unsupported aux_features type: {type(features)}")


def _normalize_aux_feature(features: AuxFeatureType, expected_len: int) -> AuxFeatureType:
    if torch.is_tensor(features):
        if int(features.shape[0]) != expected_len:
            raise ValueError(f"aux_features length mismatch: expected {expected_len}, got {features.shape[0]}")
        return features.detach().cpu().float().contiguous()
    if isinstance(features, dict):
        normalized: Dict[str, torch.Tensor] = {}
        for key, tensor in features.items():
            if not torch.is_tensor(tensor):
                raise TypeError(f"aux_features[{key}] must be a tensor, got {type(tensor)}")
            if int(tensor.shape[0]) != expected_len:
                raise ValueError(
                    f"aux_features[{key}] length mismatch: expected {expected_len}, got {tensor.shape[0]}"
                )
            normalized[key] = tensor.detach().cpu().float().contiguous()
        return normalized
    raise TypeError(f"Unsupported aux_features type: {type(features)}")


def _normalize_action_metadata(
    sequence_action_ids: Optional[List[int]],
    sequence_count: int,
    action_to_id: Optional[Dict[str, int]],
) -> Tuple[Optional[List[int]], Dict[str, int], List[str]]:
    mapping = dict(action_to_id or {})
    id_to_action = ["unknown" for _ in range(max(mapping.values()) + 1)] if mapping else []
    for name, idx in mapping.items():
        if idx < 0:
            raise ValueError(f"action id must be non-negative, got {idx} for {name!r}")
        while len(id_to_action) <= idx:
            id_to_action.append("unknown")
        id_to_action[idx] = str(name)
    if sequence_action_ids is None:
        return None, mapping, id_to_action
    if len(sequence_action_ids) != sequence_count:
        raise ValueError(
            f"sequence_action_ids length mismatch: expected {sequence_count}, got {len(sequence_action_ids)}"
        )
    return [int(action_id) for action_id in sequence_action_ids], mapping, id_to_action


def _build_window_action_ids(
    index: List[Tuple[int, int]],
    sequence_action_ids: Optional[List[int]],
) -> Optional[torch.Tensor]:
    if sequence_action_ids is None:
        return None
    return torch.tensor([sequence_action_ids[si] for si, _st in index], dtype=torch.long)


def _merge_window_aux(
    aux_features: Optional[AuxFeatureType],
    action_ids: Optional[torch.Tensor],
    sequence_ids: torch.Tensor,
) -> AuxFeatureType:
    if aux_features is None:
        merged: Dict[str, torch.Tensor] = {}
    if torch.is_tensor(aux_features):
        merged = {"history": aux_features}
    elif isinstance(aux_features, dict):
        merged = dict(aux_features)
    if action_ids is not None:
        merged["action_id"] = action_ids
    merged["sequence_id"] = sequence_ids
    return merged


class AssemblyDataset(Dataset):
    """Sliding-window dataset over multiple (T, N, 7) sequences."""

    def __init__(
        self,
        sequences: List[Union[np.ndarray, Tuple[np.ndarray, float]]],
        input_n: int,
        output_n: int,
        stride: int = 5,
        sequence_action_ids: Optional[List[int]] = None,
        action_to_id: Optional[Dict[str, int]] = None,
    ):
        self.sequences: List[torch.Tensor] = []
        self.norm_factors: List[torch.Tensor] = []
        for item in sequences:
            if isinstance(item, tuple) and len(item) == 2:
                seq_arr, factor = item
            else:
                seq_arr, factor = item, 1.0
            self.sequences.append(torch.from_numpy(seq_arr))
            self.norm_factors.append(torch.tensor(float(factor), dtype=torch.float32))
        self.input_n = int(input_n)
        self.output_n = int(output_n)
        self.stride = max(1, int(stride))
        self.aux_features: Optional[AuxFeatureType] = None
        self.sequence_action_ids, self.action_to_id, self.id_to_action = _normalize_action_metadata(
            sequence_action_ids,
            len(self.sequences),
            action_to_id,
        )
        self.num_actions = len(self.id_to_action)
        self.index: List[Tuple[int, int]] = []  # (seq_idx, start_t)
        for si, seq in enumerate(self.sequences):
            T = seq.shape[0]
            max_start = T - (self.input_n + self.output_n)
            if max_start <= 0:
                continue
            for st in range(0, max_start, self.stride):
                self.index.append((si, st))
        self.window_action_ids = _build_window_action_ids(self.index, self.sequence_action_ids)
        self.window_sequence_ids = torch.tensor([si for si, _st in self.index], dtype=torch.long)

        print(
            "AssemblyDataset: "
            f"{len(self.sequences)} sequences, {len(self.index)} samples, "
            f"input_n={self.input_n}, output_n={self.output_n}, stride={self.stride}"
        )

    def __len__(self) -> int:  # pragma: no cover - trivial getter
        return len(self.index)

    def __getitem__(self, i: int):
        si, st = self.index[i]
        seq = self.sequences[si]
        a = st
        b = st + self.input_n
        c = b + self.output_n
        norm_factor = self.norm_factors[si].to(dtype=seq.dtype)
        all_aux = _merge_window_aux(
            self.aux_features, self.window_action_ids, self.window_sequence_ids
        )
        return seq[a:b], seq[b:c], norm_factor, _slice_aux_feature(all_aux, i)

    def set_aux_features(self, features: Optional[AuxFeatureType]) -> None:
        if features is None:
            self.aux_features = None
            return
        self.aux_features = _normalize_aux_feature(features, len(self))

class h2oDataset(Dataset):
    """Sliding-window dataset over multiple (T, N, 7) sequences."""

    def __init__(
        self,
        sequences: List[Union[np.ndarray, Tuple[np.ndarray, float]]],
        input_n: int,
        output_n: int,
        stride: int = 5,
        sequence_action_ids: Optional[List[int]] = None,
        action_to_id: Optional[Dict[str, int]] = None,
    ):
        self.sequences: List[torch.Tensor] = []
        self.norm_factors: List[torch.Tensor] = []
        for item in sequences:
            if isinstance(item, tuple) and len(item) == 2:
                seq_arr, factor = item
            else:
                seq_arr, factor = item, 1.0
            self.sequences.append(torch.from_numpy(seq_arr))
            self.norm_factors.append(torch.tensor(float(factor), dtype=torch.float32))
        self.input_n = int(input_n)
        self.output_n = int(output_n)
        self.stride = max(1, int(stride))
        self.aux_features: Optional[AuxFeatureType] = None
        self.sequence_action_ids, self.action_to_id, self.id_to_action = _normalize_action_metadata(
            sequence_action_ids,
            len(self.sequences),
            action_to_id,
        )
        self.num_actions = len(self.id_to_action)
        self.index: List[Tuple[int, int]] = []  # (seq_idx, start_t)
        for si, seq in enumerate(self.sequences):
            T = seq.shape[0]
            max_start = T - (self.input_n + self.output_n)
            if max_start <= 0:
                continue
            for st in range(0, max_start, self.stride):
                self.index.append((si, st))
        self.window_action_ids = _build_window_action_ids(self.index, self.sequence_action_ids)
        self.window_sequence_ids = torch.tensor([si for si, _st in self.index], dtype=torch.long)

        print(
            "h2oDataset: "
            f"{len(self.sequences)} sequences, {len(self.index)} samples, "
            f"input_n={self.input_n}, output_n={self.output_n}, stride={self.stride}"
        )

    def __len__(self) -> int:  # pragma: no cover - trivial getter
        return len(self.index)

    def __getitem__(self, i: int):
        si, st = self.index[i]
        seq = self.sequences[si]
        a = st
        b = st + self.input_n
        c = b + self.output_n
        norm_factor = self.norm_factors[si].to(dtype=seq.dtype)
        all_aux = _merge_window_aux(
            self.aux_features, self.window_action_ids, self.window_sequence_ids
        )
        return seq[a:b], seq[b:c], norm_factor, _slice_aux_feature(all_aux, i)

    def set_aux_features(self, features: Optional[AuxFeatureType]) -> None:
        if features is None:
            self.aux_features = None
            return
        self.aux_features = _normalize_aux_feature(features, len(self))

class BigHandsDataset(Dataset):
    """Sliding-window dataset over multiple (T, N, 7) sequences."""

    def __init__(
        self,
        sequences: List[Union[np.ndarray, Tuple[np.ndarray, float]]],
        input_n: int,
        output_n: int,
        stride: int = 5,
        sequence_action_ids: Optional[List[int]] = None,
        action_to_id: Optional[Dict[str, int]] = None,
    ):
        self.sequences: List[torch.Tensor] = []
        self.norm_factors: List[torch.Tensor] = []
        for item in sequences:
            if isinstance(item, tuple) and len(item) == 2:
                seq_arr, factor = item
            else:
                seq_arr, factor = item, 1.0
            self.sequences.append(torch.from_numpy(seq_arr))
            self.norm_factors.append(torch.tensor(float(factor), dtype=torch.float32))
        self.input_n = int(input_n)
        self.output_n = int(output_n)
        self.stride = max(1, int(stride))
        self.aux_features: Optional[AuxFeatureType] = None
        self.sequence_action_ids, self.action_to_id, self.id_to_action = _normalize_action_metadata(
            sequence_action_ids,
            len(self.sequences),
            action_to_id,
        )
        self.num_actions = len(self.id_to_action)
        self.index: List[Tuple[int, int]] = []  # (seq_idx, start_t)
        for si, seq in enumerate(self.sequences):
            T = seq.shape[0]
            max_start = T - (self.input_n + self.output_n)
            if max_start <= 0:
                continue
            for st in range(0, max_start, self.stride):
                self.index.append((si, st))
        self.window_action_ids = _build_window_action_ids(self.index, self.sequence_action_ids)
        self.window_sequence_ids = torch.tensor([si for si, _st in self.index], dtype=torch.long)

        print(
            "BigHandsDataset: "
            f"{len(self.sequences)} sequences, {len(self.index)} samples, "
            f"input_n={self.input_n}, output_n={self.output_n}, stride={self.stride}"
        )

    def __len__(self) -> int:  # pragma: no cover - trivial getter
        return len(self.index)

    def __getitem__(self, i: int):
        si, st = self.index[i]
        seq = self.sequences[si]
        a = st
        b = st + self.input_n
        c = b + self.output_n
        norm_factor = self.norm_factors[si].to(dtype=seq.dtype)
        all_aux = _merge_window_aux(
            self.aux_features, self.window_action_ids, self.window_sequence_ids
        )
        return seq[a:b], seq[b:c], norm_factor, _slice_aux_feature(all_aux, i)

    def set_aux_features(self, features: Optional[AuxFeatureType]) -> None:
        if features is None:
            self.aux_features = None
            return
        self.aux_features = _normalize_aux_feature(features, len(self))

class FPHADataset(Dataset):
    """Sliding-window dataset over multiple (T, N, 7) sequences."""

    def __init__(
        self,
        sequences: List[Union[np.ndarray, Tuple[np.ndarray, float]]],
        input_n: int,
        output_n: int,
        stride: int = 5,
        sequence_action_ids: Optional[List[int]] = None,
        action_to_id: Optional[Dict[str, int]] = None,
    ):
        self.sequences: List[torch.Tensor] = []
        self.norm_factors: List[torch.Tensor] = []
        for item in sequences:
            if isinstance(item, tuple) and len(item) == 2:
                seq_arr, factor = item
            else:
                seq_arr, factor = item, 1.0
            self.sequences.append(torch.from_numpy(seq_arr))
            self.norm_factors.append(torch.tensor(float(factor), dtype=torch.float32))
        self.input_n = int(input_n)
        self.output_n = int(output_n)
        self.stride = max(1, int(stride))
        self.aux_features: Optional[AuxFeatureType] = None
        self.sequence_action_ids, self.action_to_id, self.id_to_action = _normalize_action_metadata(
            sequence_action_ids,
            len(self.sequences),
            action_to_id,
        )
        self.num_actions = len(self.id_to_action)
        self.index: List[Tuple[int, int]] = []  # (seq_idx, start_t)
        for si, seq in enumerate(self.sequences):
            T = seq.shape[0]
            max_start = T - (self.input_n + self.output_n)
            if max_start <= 0:
                continue
            for st in range(0, max_start, self.stride):
                self.index.append((si, st))
        self.window_action_ids = _build_window_action_ids(self.index, self.sequence_action_ids)
        self.window_sequence_ids = torch.tensor([si for si, _st in self.index], dtype=torch.long)

        print(
            "FPHADataset: "
            f"{len(self.sequences)} sequences, {len(self.index)} samples, "
            f"input_n={self.input_n}, output_n={self.output_n}, stride={self.stride}"
        )

    def __len__(self) -> int:  # pragma: no cover - trivial getter
        return len(self.index)

    def __getitem__(self, i: int):
        si, st = self.index[i]
        seq = self.sequences[si]
        a = st
        b = st + self.input_n
        c = b + self.output_n
        norm_factor = self.norm_factors[si].to(dtype=seq.dtype)
        all_aux = _merge_window_aux(
            self.aux_features, self.window_action_ids, self.window_sequence_ids
        )
        return seq[a:b], seq[b:c], norm_factor, _slice_aux_feature(all_aux, i)

    def set_aux_features(self, features: Optional[AuxFeatureType]) -> None:
        if features is None:
            self.aux_features = None
            return
        self.aux_features = _normalize_aux_feature(features, len(self))
