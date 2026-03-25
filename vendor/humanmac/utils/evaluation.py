import csv
import os
import sys
from pathlib import Path

import pandas as pd
from utils.metrics import *
from tqdm import tqdm
from utils import *
from utils.script import sample_preprocessing

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from common.metrics import distributional_motion_metrics  # type: ignore

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros

EVAL_SAMPLES_PER_SEQUENCE = 10


def compute_stats(diffusion, multimodal_dict, model, logger, cfg):
    """
    The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
    several (K=50) times.
    """

    # TODO reduce computation complexity
    def _predict_once(data, model_select):
        traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
        traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
        traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)
        # traj.shape: [*, t_his + t_pre, 3 * joints_num]

        mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(traj, cfg, mode='metrics')
        sampled_motion = diffusion.sample_ddim(model_select,
                                               traj_dct,
                                               traj_dct_cond,
                                               mode_dict)

        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        # traj_est.shape (K, 125, 48)
        traj_est = traj_est.cpu().numpy()
        return traj_est

    def get_prediction(data, model_select):
        eval_bs = max(1, int(getattr(cfg, 'batch_size', 64)))
        preds = []
        for st in range(0, data.shape[0], eval_bs):
            ed = min(data.shape[0], st + eval_bs)
            preds.append(_predict_once(data[st:ed], model_select))
        pred = np.concatenate(preds, axis=0)
        return pred[None, ...]

    gt_group = multimodal_dict['gt_group']
    data_group = multimodal_dict['data_group']
    traj_gt_arr = multimodal_dict['traj_gt_arr']
    num_samples = multimodal_dict['num_samples']

    stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE', 'CMD', 'FID']
    stats_meter = {x: {y: AverageMeter() for y in ['HumanMAC']} for x in stats_names}

    K = EVAL_SAMPLES_PER_SEQUENCE
    pred = []
    for i in tqdm(range(0, K), position=0):
        # It generates a prediction for all samples in the test set
        # So we need loop for K times
        pred_i_nd = get_prediction(data_group, model)
        pred.append(pred_i_nd)
        if i == K - 1:  # in last iteration, concatenate all candidate pred
            pred = np.concatenate(pred, axis=0)
            # pred [K, N, T, D]
            pred = pred[:, :, cfg.t_his:, :]
            # Use GPU to accelerate
            try:
                gt_group = torch.from_numpy(gt_group).to('cuda')
            except:
                pass
            try:
                pred = torch.from_numpy(pred).to('cuda')
            except:
                pass
            # pred [50, 5187, 100, 48]
            for j in range(0, num_samples):
                apd, ade, fde, mmade, mmfde = compute_all_metrics(pred[:, j, :, :],
                                                                        gt_group[j][np.newaxis, ...],
                                                                        traj_gt_arr[j])
                stats_meter['APD']['HumanMAC'].update(apd)
                stats_meter['ADE']['HumanMAC'].update(ade)
                stats_meter['FDE']['HumanMAC'].update(fde)
                stats_meter['MMADE']['HumanMAC'].update(mmade)
                stats_meter['MMFDE']['HumanMAC'].update(mmfde)

            motion_dist_metrics = distributional_motion_metrics(pred, gt_group)
            stats_meter['CMD']['HumanMAC'].update(float(motion_dist_metrics['CMD']))
            stats_meter['FID']['HumanMAC'].update(float(motion_dist_metrics['FID']))

            for stats in stats_names:
                str_stats = f'{stats}: ' + ' '.join(
                    [f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()]
                )
                logger.info(str_stats)
            pred = []

    # save stats in csv
    file_latest = '%s/stats_latest.csv'
    file_stat = '%s/stats.csv'
    with open(file_latest % cfg.result_dir, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + ['HumanMAC'])
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            val = new_meter['HumanMAC']
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    val = float(val.detach().cpu().item())
                else:
                    val = float(val.detach().cpu().mean().item())
            else:
                val = float(val)
            new_meter['HumanMAC'] = val
            new_meter['Metric'] = stats
            writer.writerow(new_meter)
    df1 = pd.read_csv(file_latest % cfg.result_dir)

    if os.path.exists(file_stat % cfg.result_dir) is False:
        df1.to_csv(file_stat % cfg.result_dir, index=False)
    else:
        df2 = pd.read_csv(file_stat % cfg.result_dir)
        df = pd.concat([df2, df1['HumanMAC']], axis=1, ignore_index=True)
        df.to_csv(file_stat % cfg.result_dir, index=False)


def compute_mpjpe_stats(diffusion, dataset_multi_test, model, logger, cfg):
    """
    Compute MPJPE in normalized space and MPJPE after rescaling by per-sequence norm factors.
    This mirrors SplineEqNet's (mpjpe, mpjpe_norm) reporting.
    """

    def _predict_once(data):
        # data: (B, T, J, 3)
        traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
        traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
        traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)
        mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(traj, cfg, mode='metrics')
        sampled_motion = diffusion.sample_ddim(model, traj_dct, traj_dct_cond, mode_dict)
        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        traj_est = traj_est.reshape(traj_est.shape[0], traj_est.shape[1], -1, 3)
        return traj_est

    def _predict_batched(data):
        eval_bs = max(1, int(getattr(cfg, 'batch_size', 64)))
        out = []
        for st in range(0, data.shape[0], eval_bs):
            ed = min(data.shape[0], st + eval_bs)
            out.append(_predict_once(data[st:ed]))
        return torch.cat(out, dim=0)

    logger.info('preparing MPJPE evaluation dataset...')
    data_group = []
    norm_group = []
    data_gen_multi_test = dataset_multi_test.iter_generator(step=cfg.t_his)
    for batch in data_gen_multi_test:
        if isinstance(batch, (tuple, list)):
            if len(batch) >= 2:
                data, norm_factor = batch[0], batch[1]
            else:
                data, norm_factor = batch[0], None
        else:
            data, norm_factor = batch, None
        data_group.append(data)
        if norm_factor is None:
            norm_group.append(np.ones(data.shape[0], dtype=np.float32))
        else:
            norm_arr = np.asarray(norm_factor, dtype=np.float32).reshape(-1)
            if norm_arr.size == 1 and data.shape[0] > 1:
                norm_arr = np.full((data.shape[0],), float(norm_arr[0]), dtype=np.float32)
            norm_group.append(norm_arr)

    if not data_group:
        logger.warning('MPJPE evaluation skipped: no samples from iterator.')
        return {'mpjpe': float('nan'), 'mpjpe_norm': float('nan'), 'samples': 0}

    data_group = np.concatenate(data_group, axis=0)
    norm_factor = np.concatenate(norm_group, axis=0)
    gt = torch.from_numpy(data_group[..., 1:, :]).to(cfg.device, dtype=torch.float32)
    gt_future = gt[:, cfg.t_his:, :, :]
    norm_factor_t = torch.from_numpy(norm_factor).to(cfg.device, dtype=torch.float32).view(-1)

    best_of_k = EVAL_SAMPLES_PER_SEQUENCE
    if best_of_k == 1:
        pred = _predict_batched(data_group)
        pred_future = pred[:, cfg.t_his:, :, :]
        mpjpe_terms = torch.norm(pred_future - gt_future, dim=-1)
        per_sample_mpjpe = mpjpe_terms.mean(dim=(1, 2))
    else:
        all_preds = []
        for _ in range(best_of_k):
            pred = _predict_batched(data_group)
            all_preds.append(pred[:, cfg.t_his:, :, :])
        stacked = torch.stack(all_preds, dim=0)  # (K, B, T, J, 3)
        mpjpe_terms = torch.norm(stacked - gt_future.unsqueeze(0), dim=-1)
        mpjpe_k = mpjpe_terms.mean(dim=(2, 3))  # (K, B)
        per_sample_mpjpe = mpjpe_k.min(dim=0)[0]

    per_sample_mpjpe_norm = per_sample_mpjpe * norm_factor_t
    mpjpe = float(per_sample_mpjpe.mean().item())
    mpjpe_norm = float(per_sample_mpjpe_norm.mean().item())
    samples = int(per_sample_mpjpe.shape[0])

    logger.info(f"MPJPE: {mpjpe:.6f} | MPJPE_NORM: {mpjpe_norm:.6f} | samples: {samples} | best_of_k: {best_of_k}")

    file_mpjpe_latest = '%s/mpjpe_latest.csv'
    file_mpjpe = '%s/mpjpe.csv'
    row = pd.DataFrame([{
        'Metric': 'MPJPE',
        'HumanMAC': mpjpe,
        'HumanMAC_Norm': mpjpe_norm,
        'Samples': samples,
        'BestOfK': best_of_k,
    }])
    row.to_csv(file_mpjpe_latest % cfg.result_dir, index=False)
    if os.path.exists(file_mpjpe % cfg.result_dir) is False:
        row.to_csv(file_mpjpe % cfg.result_dir, index=False)
    else:
        prev = pd.read_csv(file_mpjpe % cfg.result_dir)
        pd.concat([prev, row], axis=0, ignore_index=True).to_csv(file_mpjpe % cfg.result_dir, index=False)

    return {'mpjpe': mpjpe, 'mpjpe_norm': mpjpe_norm, 'samples': samples}
