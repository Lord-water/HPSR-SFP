"""Train-only readout fitting; development cross-fitting for model selection."""
import itertools

import numpy as np

from .hierarchical import hcal, hfit, hpredict, oof
from .readouts import L2, apply_cal, calibrate, fit, predict, prepare


def fit_selected(train, dev, y, dy, folds, method='hpsr', progress=None):
    if set(train['record_ids']) & set(dev['record_ids']):
        raise ValueError('Train and development IDs overlap')
    if str(train['encoder_id']) != str(dev['encoder_id']):
        raise ValueError('Train/dev encoder identities differ')
    if set(folds.tolist()) != set(range(5)):
        raise ValueError('Expected fixed context-level folds 0..4')
    if any(not np.any((dy >= 0)[folds != fold], axis=0).all() for fold in range(5)):
        raise ValueError('Every calibration-fitting fold needs resolved labels for each candidate')
    if not np.any(y >= 0, axis=0).all() or not np.any(dy >= 0, axis=0).all():
        raise ValueError('Every candidate needs resolved training and development labels')
    arm = {'hpsr': 'raw', 'raw': 'raw', 'centered': 'center_linear',
           'scalar': 'scalar_mono', 'spline': 'scalar_flex'}[method]
    x, stats = prepare(train['h'], train['u'], arm)
    xd, _ = prepare(dev['h'], dev['u'], arm, stats=stats)
    mean = float(train['u'].astype(float).mean())
    sd = max(float(train['u'].astype(float).std()), 1e-5)
    raw_mean = train['u'].astype(float).mean(1)
    bm, bs = float(raw_mean.mean()), max(float(raw_mean.std()), 1e-5)
    u, ud = (train['u'].astype(float)-mean)/sd, (dev['u'].astype(float)-mean)/sd
    ub, ubd = (raw_mean-bm)/bs, (dev['u'].astype(float).mean(1)-bm)/bs
    grid = itertools.product([.1, .3, 1., 3., 10., 30.], [1., 5., 10., 50., float('inf')], [False, True]) if method == 'hpsr' else ((value, None, None) for value in L2)
    selected, trace = None, []
    for number, (lam, rho, cues) in enumerate(grid, 1):
        head = hfit(x, u, ub, y, lam, rho, cues) if method == 'hpsr' else fit(x, y, lam, 0., mono=method == 'scalar')
        z = hpredict(head, xd, ud, ubd) if method == 'hpsr' else predict(head, xd)
        for mode in (['temp', 'candidate'] if method == 'hpsr' else ['raw', 'temp', 'affine', 'candidate']):
            score = oof(z, dy, folds, mode, hier=method == 'hpsr')
            trace.append(dict(lam=lam, rho=rho, cues=cues, calibration=mode,
                              dev_oof_brier=score, head_converged=head['success'],
                              nit=head['nit'], grad_max=head['grad_max']))
            if selected is None or score < selected[0]:
                selected = (score, head, mode, z)
        if progress:
            progress(dict(completed_readout_fits=number, method=method))
    score, head, mode, z = selected
    calibrator = hcal(z, dy, mode) if method == 'hpsr' else calibrate(z, dy, mode)
    bundle = dict(format='hpsr-sfp-readout-v1', method=method, arm=arm, head=head,
                  stats=stats, u_mean=mean, u_sd=sd, ubar_mean=bm, ubar_sd=bs,
                  calibrator=calibrator, dev_oof_brier=score,
                  encoder_id=str(train['encoder_id']), feature_dimension=int(train['h'].shape[-1]))
    return bundle, trace


def restore_bundle(bundle):
    if bundle['format'] != 'hpsr-sfp-readout-v1':
        raise ValueError('Unknown readout format')
    for field in ['stats', 'head', 'calibrator']:
        bundle[field] = {key: np.asarray(value, float) if isinstance(value, list) else value
                         for key, value in bundle[field].items()}
    if bundle['head'].get('rho') == 'infinity':
        bundle['head']['rho'] = float('inf')
    return bundle


def predict_bundle(bundle, cache):
    if str(cache['encoder_id']) != bundle['encoder_id']:
        raise ValueError('Features were extracted from a different encoder/adapter')
    if cache['h'].shape[-1] != bundle['feature_dimension']:
        raise ValueError('Feature dimension mismatch')
    x, _ = prepare(cache['h'], cache['u'], bundle['arm'], stats=bundle['stats'])
    if bundle['method'] == 'hpsr':
        u = (cache['u'].astype(float)-bundle['u_mean'])/bundle['u_sd']
        ub = (cache['u'].astype(float).mean(1)-bundle['ubar_mean'])/bundle['ubar_sd']
        z = hpredict(bundle['head'], x, u, ub)
    else:
        z = predict(bundle['head'], x)
    return apply_cal(bundle['calibrator'], z)
