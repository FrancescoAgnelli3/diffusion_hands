import torch
import os
from functools import partial
import numpy as np

from .metrics.fid import get_classifier, MetricStorerFID
from .metrics.multimodal import apd, ade, mae, fde, mmade, mmfde, cmd, mpjpe, lat_apd
from .metrics.apde import MetricStorerAPDE
from .metrics.body_realism import limb_length_variance, limb_length_jitter, limb_length_variation_difference_wrtGT, limb_length_error, limb_stretching_normed_rmse, limb_stretching_normed_mean, limb_jitter_normed_rmse, limb_jitter_normed_mean 
from .metrics.utils import draw_table, get_best_sample_idx
from .metrics.metric_storer import MetricStorer
from .metrics.cmd import CMDMetricStorer, resolve_cmd, motion_for_cmd
from .metrics.ignite_mpjpe import MeanPerJointPositionError
from .metrics.ignite_fde import FinalDisplacementError
from .metrics.utils import choose_best_sample


def get_stats_funcs(stats_mode, skeleton, **kwargs):
    limbseq = skeleton.get_limbseq()
    limbseq_mae = limbseq.copy()
    limb_angles_idx = skeleton.limb_angles_idx.copy()

    # Values are in cm in the paper
    limb_stretching_normed_mean_scaled = lambda *args, **kwargs: limb_stretching_normed_mean(*args, **kwargs)*100
    limb_jitter_normed_mean_scaled = lambda *args, **kwargs: limb_jitter_normed_mean(*args, **kwargs)*100
    limb_stretching_normed_rmse_scaled = lambda *args, **kwargs: limb_stretching_normed_rmse(*args, **kwargs)*100
    limb_jitter_normed_rmse_scaled = lambda *args, **kwargs: limb_jitter_normed_rmse(*args, **kwargs)*100

    assert not kwargs["if_consider_hip"]
    if "deterministic" in stats_mode.lower():
        stats_func = { 'ADE': ade, 'FDE': fde , 'MAE': partial(mae, limbseq=limbseq_mae, limb_angles_idx=limb_angles_idx),
                      'APD':  apd, 
                    'StretchMean': partial(limb_stretching_normed_mean_scaled, limbseq=limbseq),
                    'JitterMean': partial(limb_jitter_normed_mean_scaled, limbseq=limbseq),
                    'StretchRMSE': partial(limb_stretching_normed_rmse_scaled, limbseq=limbseq),
                    'JitterRMSE': partial(limb_jitter_normed_rmse_scaled, limbseq=limbseq),
                    }
    elif "probabilistic_orig" == stats_mode.lower():
        stats_func = {  'APD': apd, 'ADE': ade, 'FDE': fde, 
                        'MMADE': mmade, 'MMFDE': mmfde,
                    }
    elif "probabilistic" == stats_mode.lower():
        stats_func = {  'ADE': ade, 'FDE': fde , 'MAE': partial(mae, limbseq=limbseq_mae, limb_angles_idx=limb_angles_idx),
                        'MMADE': mmade, 'MMFDE': mmfde,
                        'APD':  apd,
                        'StretchMean': partial(limb_stretching_normed_mean_scaled, limbseq=limbseq),
                        'JitterMean': partial(limb_jitter_normed_mean_scaled, limbseq=limbseq),
                        'StretchRMSE': partial(limb_stretching_normed_rmse_scaled, limbseq=limbseq),
                        'JitterRMSE': partial(limb_jitter_normed_rmse_scaled, limbseq=limbseq),
                    }
    else :
        raise NotImplementedError(f"stats_mode not implemented: {stats_mode}")
    return stats_func

def get_apde_storer(**kwargs):
    return partial(MetricStorerAPDE, mmapd_gt_path= os.path.join(kwargs['annotations_folder'], "mmapd_GT.csv")), lambda pred, **kwargs: apd(pred) 
    
def get_fid_storer(**kwargs):
    storer = partial(MetricStorerFID, classifier_path=kwargs['precomputed_folder'], **kwargs)
    return storer , lambda pred, target,  **kwgs: (pred, target)

def get_cmd_storer(dataset, if_consider_hip=False, **kwargs):
    assert not if_consider_hip
    cmdstorer = {"CMD": ( partial(CMDMetricStorer, final_funct=partial(resolve_cmd, idx_to_class=dataset.idx_to_class,  mean_motion_per_class=dataset.mean_motion_per_class)),
                              lambda pred, extra, **kwgs: (motion_for_cmd(pred.clone()), 
                                np.array([dataset.class_to_idx[c] for c in extra["metadata"][dataset.metadata_class_idx]]))
                            ),
                }
    return cmdstorer

def attach_engine_to_metrics(engine, dataset_split, stats_mode, dataset, skeleton, if_compute_cmd=False, if_compute_fid=False, if_compute_apde=False, **config):
    stats_func = get_stats_funcs(stats_mode, skeleton=skeleton, **config)
    
    stats_metrics = {k: MetricStorer(output_transform=lambda dict, funct=funct: funct(**(dict.copy())),
                                                    return_op='max' if '_max' in k else 'avg') for k, funct in stats_func.items()}
    
    for name, metric in stats_metrics.items():
        metric.attach(engine, name)

    if dataset_split == 'test' and if_compute_fid:
        fid_storer, funct = get_fid_storer(**config)
        apde_metrics = {'FID': fid_storer(output_transform=funct)}
        for name, metric in apde_metrics.items():
            metric.attach(engine, name)

    if dataset_split=='test' and if_compute_cmd:
        cmd_storer = get_cmd_storer(dataset, **config)
        cmd_metrics = {cmd_name: cmdclass(output_transform=lambda x_dict: funct(**x_dict.copy())) 
                            for cmd_name, (cmdclass, funct) in cmd_storer.items()}
        for name, metric in cmd_metrics.items():
            metric.attach(engine, name)
    if if_compute_apde:
        storer, funct = get_apde_storer(**config)
        apde_metrics = {'APDE': storer(output_transform=lambda x_dict: funct(**x_dict.copy()))}
        for name, metric in apde_metrics.items():
            metric.attach(engine, name)
