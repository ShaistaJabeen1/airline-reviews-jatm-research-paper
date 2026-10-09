import numpy as np, pandas as pd
from scipy import stats
from scipy.special import expit
from ologit import fit
import hyp_helpers as H
df = pd.read_pickle('df_llm.pkl')
w = df[df.star.notna()].dropna(subset=['year_grp']).copy(); w['y'] = w.star.astype(int) - 1
foc = ['Justice: Procedural', 'Justice: Distributive', 'Justice: General/Other', 'Kano: Must-be', 'Kano: Attractive', 'Kano: General/Other']
def m4(d, zcol='sent', service=True, jcol='J', kcol='K'):
    C, S, Jd, Kd, Z = H.blocks(d, zcol=zcol, jcol=jcol, kcol=kcol)
    X = pd.concat([C] + ([S] if service else []) + [Jd, Kd, Z], axis=1)
    r = fit(X.values, d.y.values, 5); return H.tidy(r, list(X.columns)), r, X
rows = {}
specs = {'Main (Model 4)': (w, {}), 'Excluding excerpts': (w[~w.excerpt], {}), 'LLM sentiment as control': (w.assign(llm_s=w.llm_sentiment.astype(float)), dict(zcol='llm_s')),
         'Without service dimension': (w, dict(service=False))}
for nm, (d, kw) in specs.items():
    t, r, X = m4(d, **kw); rows[nm] = {f: f"{t.loc[f,'OR']:.2f} [{t.loc[f,'lo']:.2f}, {t.loc[f,'hi']:.2f}]{'*' if t.loc[f,'p']<.05 else ''}" for f in foc}; rows[nm]['n'] = len(d)
# zero-shot illustration: justice ref Interactional, kano ref Performance using zero-shot labels
z = w.copy(); z['Jz'] = z.justice_dimension.str.split(' - ').str[0].str.replace(' Justice', ''); z['Kz'] = z.kano_category.str.replace(' Quality', '')
z['Sz'] = z.service_dimension
t, r, X = m4(z, jcol='Jz', kcol='Kz')
C, S, Jd, Kd, Zs = H.blocks(z, jcol='Jz', kcol='Kz', scol='Sz'); X = pd.concat([C, S, Jd, Kd, Zs], axis=1); r = fit(X.values, z.y.values, 5); t = H.tidy(r, list(X.columns))
rows['Superseded zero-shot labels'] = {f: (f"{t.loc[f,'OR']:.2f} [{t.loc[f,'lo']:.2f}, {t.loc[f,'hi']:.2f}]{'*' if t.loc[f,'p']<.05 else ''}" if f in t.index else '–') for f in foc}; rows['Superseded zero-shot labels']['n'] = len(z)
RB = pd.DataFrame(rows).T[['n'] + foc]; print(RB.to_string())

# ---- Brant test for Model 4
_, r4, X4 = m4(w)
def logit_irls(X, yb):
    X1 = np.c_[np.ones(len(X)), X]; b = np.zeros(X1.shape[1])
    for _ in range(100):
        p = expit(X1 @ b); W = p * (1 - p); g = X1.T @ (yb - p); Hm = X1.T @ (X1 * W[:, None]); step = np.linalg.solve(Hm, g); b += step
        if np.abs(step).max() < 1e-9: break
    return b, expit(X1 @ b), X1
X = X4.values; yy = w.y.values; K = 5; bs, ps = [], []
for j in range(K - 1):
    b, p, X1 = logit_irls(X, (yy > j).astype(float)); bs.append(b); ps.append(p)
q = X.shape[1]; V = np.zeros(((K - 1) * q, (K - 1) * q))
for j in range(K - 1):
    for l in range(j, K - 1):
        Wjj = ps[j] * (1 - ps[j]); Wll = ps[l] * (1 - ps[l]); Wjl = ps[l] - ps[j] * ps[l]
        Ijj = np.linalg.inv(X1.T @ (X1 * Wjj[:, None])); Ill = np.linalg.inv(X1.T @ (X1 * Wll[:, None]))
        Cjl = Ijj @ (X1.T @ (X1 * Wjl[:, None])) @ Ill
        V[j*q:(j+1)*q, l*q:(l+1)*q] = Cjl[1:, 1:]; V[l*q:(l+1)*q, j*q:(j+1)*q] = Cjl[1:, 1:].T
beta = np.concatenate([b[1:] for b in bs])
def brant(cols_idx):
    D = []
    for j in range(1, K - 1):
        for c in cols_idx:
            row = np.zeros((K - 1) * q); row[c] = 1; row[j * q + c] = -1; D.append(row)
    D = np.array(D); Db = D @ beta; W = Db @ np.linalg.pinv(D @ V @ D.T) @ Db; return W, D.shape[0], stats.chi2.sf(W, D.shape[0])
cols = list(X4.columns); br = [('Omnibus',) + brant(range(q))]
for c in foc + ['Text sentiment (z)']: br.append((c,) + brant([cols.index(c)]))
BR = pd.DataFrame(br, columns=['Term', 'χ²', 'df', 'p']); print(BR.round(4).to_string())
cut = pd.DataFrame({f'Rating > {j+1}': [bs[j][1 + cols.index(c)] for c in foc] for j in range(K - 1)}, index=foc); cut['Proportional-odds estimate'] = [r4['b'][cols.index(c)] for c in foc]; print(cut.round(2))

# ---- Human-coded subsample check (validation reviews with star rating): H1 and H2 using each coder's codes
d = pd.read_pickle('val_new.pkl'); HH = d['H']; kk = d['k']
vs = kk[['index']].copy(); vs = vs.merge(w[['index', 'y', 'sent']], on='index', how='left')
hs = []
for coder in ['A', 'B']:
    s = vs.copy(); s['J'] = HH[coder]['just'].values; s['K'] = HH[coder]['kano'].values; s = s.dropna(subset=['y', 'J', 'K'])
    s = s[s.J.isin(['Distributive', 'Procedural', 'Interactional'])]; s = s[s.K.isin(['Must-be', 'Performance'])]
    Jd = pd.get_dummies(s.J).astype(float)[['Distributive', 'Procedural']]; Kd = (s.K == 'Must-be').astype(float).rename('Must-be')
    X = pd.concat([Jd, Kd], axis=1); r = fit(X.values, s.y.astype(int).values, 5); t = H.tidy(r, list(X.columns))
    for f in X.columns: hs.append([f'Coder {coder}', f, len(s), f"{t.loc[f,'OR']:.2f} [{t.loc[f,'lo']:.2f}, {t.loc[f,'hi']:.2f}]", f"{t.loc[f,'p']:.3f}"])
HS = pd.DataFrame(hs, columns=['Codes', 'Term (ref. interactional; performance)', 'n', 'OR [95% CI]', 'p']); print(HS.to_string())

# ---- descriptives by carrier, KW tests, cross-tab
desc = []
for col, labl in [('S', 'Service dimension'), ('J', 'Organizational justice'), ('K', 'Kano-informed category')]:
    for c in sorted(df[col].unique()):
        g = df[df[col] == c]; row = [labl, c]
        for car in ['Turkish', 'Pegasus']: row.append(int((g.airline == car).sum()))
        row += [len(g), f"{len(g)/len(df)*100:.1f}", f"{g.sent.mean():.2f}", f"{g.star.mean():.2f}"]
        desc.append(row)
DESC = pd.DataFrame(desc, columns=['Classification', 'Category', 'Turkish Airlines', 'Pegasus Airlines', 'Total', '%', 'Mean sentiment', 'Mean star rating'])
kw = []
for col, labl in [('S', 'Service dimension'), ('J', 'Organizational justice'), ('K', 'Kano-informed category')]:
    for var, vl, dd in [('sent', 'Text sentiment', df), ('star', 'Star rating', w)]:
        groups = [g[var].dropna().values for _, g in dd.groupby(col)]; Hs, p = stats.kruskal(*groups); n = sum(len(g) for g in groups)
        kw.append([labl, vl, n, f"{Hs:.1f}", len(groups) - 1, f"{p:.3g}", f"{Hs/(n-1):.3f}"])
KW = pd.DataFrame(kw, columns=['Classification', 'Outcome', 'N', 'H', 'df', 'p', 'ε²'])
ct = pd.crosstab(df.J, df.K); chi2, p, dof, exp = stats.chi2_contingency(ct); n = ct.values.sum(); cv = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
rs = ct.sum(1).values[:, None] / n; cs = ct.sum(0).values[None, :] / n; adj = (ct.values - exp) / np.sqrt(exp * (1 - rs) * (1 - cs))
car = pd.crosstab(df.airline, df.J); chi2c, pc, dofc, _ = stats.chi2_contingency(car)
cark = pd.crosstab(df.airline, df.K); chi2k, pk, dofk, _ = stats.chi2_contingency(cark)
cars = pd.crosstab(df.airline, df.S); chi2s, ps_, dofs, _ = stats.chi2_contingency(cars)
cross = dict(chi2=chi2, p=p, dof=dof, V=cv, adj=pd.DataFrame(adj, index=ct.index, columns=ct.columns), ct=ct, minexp=exp.min(),
             carJ=(chi2c, dofc, pc, np.sqrt(chi2c / car.values.sum())), carK=(chi2k, dofk, pk, np.sqrt(chi2k / cark.values.sum())), carS=(chi2s, dofs, ps_, np.sqrt(chi2s / cars.values.sum())))
heat = w.groupby(['J', 'K']).star.agg(['mean', 'size']).reset_index()
print(DESC.to_string()); print(KW.to_string()); print(ct); print(cross['adj'].round(1)); print('chi2', chi2, dof, p, cv, exp.min())
print('carrier tests', cross['carJ'], cross['carK'], cross['carS'])
print(pd.crosstab(df.airline, df.J, normalize='index').round(3)); print(pd.crosstab(df.airline, df.K, normalize='index').round(3))
print(heat)
pd.to_pickle(dict(RB=RB, BR=BR, cut=cut, HS=HS, DESC=DESC, KW=KW, cross=cross, heat=heat), 'robust.pkl')
