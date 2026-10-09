import numpy as np, pandas as pd, json
from scipy import stats
from scipy.special import expit
from ologit import fit
df = pd.read_pickle('df.pkl')
lab = pd.read_csv('llm_labels_all.csv')
df = df.merge(lab[['index', 'llm_justice', 'llm_kano', 'llm_service', 'llm_sentiment']], on='index', how='left')
assert df.llm_justice.notna().all()
df['J'] = df.llm_justice.str.replace(' Justice', ''); df['K'] = df.llm_kano; df['S'] = df.llm_service
L = df.text_length_chars; df['excerpt'] = (df.source == 'TripAdvisor') & L.between(270, 280)
df.to_pickle('df_llm.pkl')
w = df[df.star.notna()].dropna(subset=['year_grp']).copy(); w['y'] = w.star.astype(int) - 1
print('N rated with date', len(w))

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

y = w.y.values; C, S, Jd, Kd, Z = blocks(w)
null = fit(np.zeros((len(w), 0)), y, 5)
MB = (w.K == 'Must-be').astype(float)
I5 = pd.concat([(MB * (w.J == j)).rename(f'Must-be × {j}') for j in ['Distributive', 'Procedural', 'General/Other']], axis=1)
M = {'M1': C, 'M2': pd.concat([C, S], axis=1), 'M2b': pd.concat([C, S, Z], axis=1), 'M3': pd.concat([C, S, Jd, Kd], axis=1),
     'M4': pd.concat([C, S, Jd, Kd, Z], axis=1), 'M5': pd.concat([C, S, Jd, Kd, Z, I5], axis=1)}
# M6: carrier x justice with General/Other as justice reference (H5)
Jg = dm(w, 'J', 'General/Other', 'Justice'); TK = (w.airline == 'Turkish').astype(float)
I6 = pd.concat([(TK * Jg[c]).rename('Turkish Airlines × ' + c.split(': ')[1]) for c in Jg.columns], axis=1)
M['M4g'] = pd.concat([C, S, Jg, Kd, Z], axis=1); M['M6'] = pd.concat([C, S, Jg, Kd, Z, I6], axis=1)
out, fits = {}, []
for nm, X in M.items():
    r = fit(X.values, y, 5); out[nm] = (r, list(X.columns)); tidy(r, list(X.columns)).to_csv(f'hyp_{nm}.csv')
    fits.append(dict(model=nm, ll=r['ll'], k=r['k'], aic=2 * r['k'] - 2 * r['ll'], r2=1 - r['ll'] / null['ll'], conv=r['conv']))
F = pd.DataFrame(fits).set_index('model')
def lr(a, b):
    s = 2 * (F.loc[b, 'll'] - F.loc[a, 'll']); dfree = F.loc[b, 'k'] - F.loc[a, 'k']; return s, dfree, stats.chi2.sf(s, dfree)
LR = {}
for a, b in [('M1', 'M2'), ('M2', 'M3'), ('M2', 'M2b'), ('M2b', 'M4'), ('M3', 'M4'), ('M4', 'M5'), ('M4g', 'M6')]:
    LR[f'{b} vs {a}'] = lr(a, b)
# block-specific LR: justice only / kano only added to M2b
for nm, X in [('M2b+K', pd.concat([C, S, Z, Kd], axis=1)), ('M2b+J', pd.concat([C, S, Z, Jd], axis=1)), ('M2+K', pd.concat([C, S, Kd], axis=1)), ('M2+J', pd.concat([C, S, Jd], axis=1))]:
    r = fit(X.values, y, 5); F.loc[nm] = dict(ll=r['ll'], k=r['k'], aic=2 * r['k'] - 2 * r['ll'], r2=1 - r['ll'] / null['ll'], conv=r['conv']); out[nm] = (r, list(X.columns))
for a, b in [('M2', 'M2+K'), ('M2', 'M2+J'), ('M2b', 'M2b+K'), ('M2b', 'M2b+J')]:
    LR[f'{b} vs {a}'] = lr(a, b)
print(F.round(4).to_string()); print({k: (round(v[0], 1), int(v[1]), v[2]) for k, v in LR.items()})
T = {nm: pd.read_csv(f'hyp_{nm}.csv', index_col=0) for nm in M}
for nm in ['M3', 'M4', 'M5', 'M6']: print('\n', nm); print(T[nm].round(3).to_string())
# planned contrasts
res = {}
for mod in ['M3', 'M4']:
    r, cols = out[mod]
    for h, n in [('H1a procedural vs interactional', 'Justice: Procedural'), ('H1b distributive vs interactional', 'Justice: Distributive'), ('H2 must-be vs performance', 'Kano: Must-be')]:
        i = cols.index(n); b, se = r['b'][i], r['se'][i]
        res[(mod, h)] = dict(B=b, OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p_two=2 * stats.norm.sf(abs(b / se)), p_one=stats.norm.cdf(b / se))
r5, c5 = out['M5']
W = wald(r5, c5, ['Must-be × Distributive', 'Must-be × Procedural'])
res[('M5', 'H4 joint (MB×Dist, MB×Proc)')] = dict(W=W[0], df=W[1], p=W[2])
for n in ['Must-be × Distributive', 'Must-be × Procedural']:
    i = c5.index(n); b, se = r5['b'][i], r5['se'][i]; res[('M5', 'H4 ' + n)] = dict(B=b, OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p_two=2 * stats.norm.sf(abs(b / se)), p_one=stats.norm.cdf(b / se))
r6, c6 = out['M6']
for n in ['Turkish Airlines × Distributive', 'Turkish Airlines × Interactional', 'Turkish Airlines × Procedural']:
    i = c6.index(n); b, se = r6['b'][i], r6['se'][i]; res[('M6', 'H5 ' + n)] = dict(B=b, OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p_two=2 * stats.norm.sf(abs(b / se)), p_one=stats.norm.sf(b / se))
# carrier-specific simple effects (vs General/Other) from M6
for j in ['Distributive', 'Interactional', 'Procedural']:
    for car, Lv in [('Pegasus', [1, 0]), ('Turkish Airlines', [1, 1])]:
        names = [f'Justice: {j}', f'Turkish Airlines × {j}']
        idx = [c6.index(n) for n in names]; b = np.array(Lv) @ r6['b'][idx]; se = np.sqrt(np.array(Lv) @ r6['cov'][np.ix_(idx, idx)] @ np.array(Lv))
        res[('M6', f'simple {j} | {car}')] = dict(B=b, OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p_two=2 * stats.norm.sf(abs(b / se)))
R = pd.DataFrame(res).T; print(R.round(4).to_string())
# Holm across confirmatory/planned p-values (one-sided for directional hypotheses)
plan = {'H1a': res[('M3', 'H1a procedural vs interactional')]['p_one'], 'H1b': res[('M3', 'H1b distributive vs interactional')]['p_one'],
        'H2': res[('M3', 'H2 must-be vs performance')]['p_one'], 'H4': res[('M5', 'H4 joint (MB×Dist, MB×Proc)')]['p'], 'H5': LR['M6 vs M4g'][2]}
order = sorted(plan, key=plan.get); m = len(plan); holm = {}; run = 0
for i, kk in enumerate(order): run = max(run, min(1, (m - i) * plan[kk])); holm[kk] = run
print('Holm', {k: round(v, 4) for k, v in holm.items()})
# cell sizes
ct = pd.crosstab(w.J, w.K); print(ct)
pd.to_pickle(dict(F=F, LR=LR, T=T, R=R, plan=plan, holm=holm, ct=ct, out={k: (v[0]['b'], v[1]) for k, v in out.items()}), 'hyp.pkl')
