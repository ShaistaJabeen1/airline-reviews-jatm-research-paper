import numpy as np, pandas as pd
from scipy import stats
def dm(d, col, ref, pre):
    x = pd.get_dummies(d[col].astype(str), prefix=pre, prefix_sep=': ').astype(float); return x.drop(columns=f'{pre}: {ref}')
def controls(d):
    return pd.concat([(d.airline == 'Turkish').astype(float).rename('Carrier: Turkish Airlines (ref. Pegasus)'),
                      (d.source == 'Trustpilot').astype(float).rename('Platform: Trustpilot (ref. TripAdvisor)'),
                      dm(d, 'year_grp', '2024', 'Period'), np.log1p(d.words).rename('Review length (log words)')], axis=1)
def blocks(d, jref='Interactional', kref='Performance', jcol='J', kcol='K', scol='S', zcol='sent'):
    C = controls(d); S = dm(d, scol, 'General/Other', 'Service')
    Jd = dm(d, jcol, jref, 'Justice'); Kd = dm(d, kcol, kref, 'Kano')
    Z = ((d[zcol] - d[zcol].mean()) / d[zcol].std()).rename('Text sentiment (z)')
    return C, S, Jd, Kd, Z
def tidy(r, cols):
    z = r['b'] / r['se']; p = 2 * stats.norm.sf(abs(z))
    return pd.DataFrame({'B': r['b'], 'SE': r['se'], 'OR': np.exp(r['b']), 'lo': np.exp(r['b'] - 1.96 * r['se']), 'hi': np.exp(r['b'] + 1.96 * r['se']), 'p': p}, index=cols)
def wald(r, cols, names, Lmat=None):
    idx = [cols.index(n) for n in names]; b = r['b'][idx]; V = r['cov'][np.ix_(idx, idx)]
    if Lmat is None: Lmat = np.eye(len(idx))
    Lb = Lmat @ b; LV = Lmat @ V @ Lmat.T; W = float(Lb @ np.linalg.solve(LV, Lb)); return W, Lmat.shape[0], stats.chi2.sf(W, Lmat.shape[0]), Lb, np.sqrt(np.diag(LV))

