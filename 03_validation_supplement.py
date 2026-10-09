import pandas as pd, numpy as np, json
from sklearn.metrics import cohen_kappa_score, f1_score
d = pd.read_pickle('val_new.pkl'); H, M = d['H'], d['M']
df = pd.read_pickle('df.pkl'); k = d['k']
rng = np.random.default_rng(20261008)

def ac1(x, y):
    cats = sorted(set(x) | set(y)); q = len(cats)
    if q < 2: return np.nan
    pa = (x == y).mean(); pi = np.array([((x == c).mean() + (y == c).mean()) / 2 for c in cats]); pe = (pi * (1 - pi)).sum() / (q - 1)
    return (pa - pe) / (1 - pe)
kap = lambda x, y: cohen_kappa_score(x, y)
qwk = lambda x, y: cohen_kappa_score(x, y, weights='quadratic')
acc = lambda x, y: (x == y).mean()
mf1 = lambda x, y: f1_score(x, y, average='macro')
pol = lambda v: np.where(v <= 2, 0, np.where(v == 3, 1, 2))
polacc = lambda x, y: (pol(x) == pol(y)).mean()
mae = lambda x, y: np.abs(x - y).mean()

def boot(fn, x, y, B=2000):
    m = x.notna() & y.notna(); x, y = x[m].values, y[m].values; n = len(x)
    if n == 0: return (np.nan, np.nan, np.nan, 0)
    e = fn(x, y); bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        try: bs.append(fn(x[i], y[i]))
        except Exception: pass
    return e, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), n
def fmt(e, l, u, pct=False):
    if np.isnan(e): return '–'
    return (f"{e*100:.1f}% [{l*100:.1f}, {u*100:.1f}]" if pct else f"{e:.2f} [{l:.2f}, {u:.2f}]")

out = {}
# ---- S7: inter-coder reliability with CIs
rows = []
spec = [('Sentiment (1–5)', 'sent', [('Exact agreement', acc, True), ('Cohen’s κ (unweighted)', kap, False), ('Quadratic-weighted κ', qwk, False), ('Gwet’s AC1', ac1, False), ('Polarity agreement', polacc, True), ('Mean absolute error', mae, False)]),
        ('Service dimension (9 categories)', 'serv', [('Agreement', acc, True), ('Cohen’s κ', kap, False), ('Gwet’s AC1', ac1, False)]),
        ('Organizational justice', 'just', [('Agreement', acc, True), ('Cohen’s κ', kap, False), ('Gwet’s AC1', ac1, False)]),
        ('Kano-informed category', 'kano', [('Agreement', acc, True), ('Cohen’s κ', kap, False), ('Gwet’s AC1', ac1, False)])]
for task, t, mets in spec:
    for name, fn, pct in mets:
        e, l, u, n = boot(fn, H['A'][t], H['B'][t])
        rows.append([task, name, n, fmt(e, l, u, pct)])
out['S7'] = pd.DataFrame(rows, columns=['Task', 'Metric', 'n', 'Estimate [95% CI]'])

# per-category kappa
rows = []
for t, lab in [('just', 'Organizational justice'), ('kano', 'Kano-informed category'), ('serv', 'Service dimension')]:
    m = H['A'][t].notna() & H['B'][t].notna(); a, b = H['A'][t][m], H['B'][t][m]
    for c in sorted(set(a) | set(b)):
        rows.append([lab, c, int((a == c).sum()), int((b == c).sum()), f"{cohen_kappa_score(a == c, b == c):.2f}"])
out['S7b'] = pd.DataFrame(rows, columns=['Classification', 'Category', 'Coder A (n)', 'Coder B (n)', 'Category-specific κ'])

# ---- S8: cross-tabulations coder A x coder B
for t, key in [('just', 'S8a'), ('kano', 'S8b')]:
    ct = pd.crosstab(H['A'][t], H['B'][t]); ct.index.name = 'Coder A \\ Coder B'
    out[key] = ct.reset_index()

# ---- S9: sentiment model and zero-shot vs each coder and agreed subset
rows = []
def agreed(t): return H['A'][t].where(H['A'][t] == H['B'][t])
sent_m = [('Exact agreement', acc, True), ('Quadratic-weighted κ', qwk, False), ('Polarity agreement', polacc, True), ('Mean absolute error', mae, False), ('Macro-F1', mf1, False)]
cat_m = [('Agreement', acc, True), ('Cohen’s κ', kap, False), ('Gwet’s AC1', ac1, False), ('Macro-F1', mf1, False)]
for task, t, mt, model, mets in [('Sentiment', 'sent', 'sent', 'Transformer sentiment model', sent_m),
                                  ('Service dimension (8 categories)', 'serv8', 'serv8', 'Zero-shot NLI', cat_m),
                                  ('Organizational justice', 'just', 'just', 'Zero-shot NLI', cat_m),
                                  ('Kano-informed category', 'kano', 'kano', 'Zero-shot NLI', cat_m)]:
    for name, fn, pct in mets:
        cells = []
        for ref in [H['A'][t], H['B'][t], agreed(t)]:
            e, l, u, n = boot(fn, M[mt], ref); cells.append(fmt(e, l, u, pct))
        n_ag = int(agreed(t).notna().sum())
        rows.append([task, model, name] + cells + [n_ag])
out['S9'] = pd.DataFrame(rows, columns=['Task', 'Model', 'Metric', 'vs Coder A', 'vs Coder B', 'vs agreed subset', 'n agreed'])

# ---- S10: label distribution
rows = []
for t, mt, lab in [('just', 'just', 'Organizational justice'), ('kano', 'kano', 'Kano-informed category'), ('serv', 'serv8', 'Service dimension')]:
    cats = sorted(set(H['A'][t].dropna()) | set(H['B'][t].dropna()) | set(M[mt].dropna()))
    for c in cats:
        rows.append([lab, c, int((H['A'][t] == c).sum()), int((H['B'][t] == c).sum()), int((M[mt] == c).sum())])
out['S10'] = pd.DataFrame(rows, columns=['Classification', 'Category', 'Coder A', 'Coder B', 'Zero-shot NLI'])

# ---- S12 panel C: validation sample vs star ratings
rows = []
star = M['star']
for lab, x in [('Coder A', H['A'].sent), ('Coder B', H['B'].sent), ('Transformer sentiment model', M.sent)]:
    m = x.notna() & star.notna(); xx, ss = x[m].astype(int), star[m].astype(int)
    rows.append([lab, int(m.sum()), f"{(xx == ss).mean()*100:.1f}%", f"{qwk(xx, ss):.2f}", f"{polacc(xx.values, ss.values)*100:.1f}%", f"{mae(xx.values, ss.values):.2f}"])
out['S12c'] = pd.DataFrame(rows, columns=['Sentiment source', 'n', 'Exact agreement', 'Quadratic-weighted κ', 'Polarity agreement', 'MAE'])

# ---- S6: validation sample construction
vs = k.merge(df[['index', 'source', 'airline', 'words']], on='index')
rows = [['Reviews in analytical corpus', '4,225'], ['Eligible complete reviews (sentence-final punctuation; 20–400 words; English; not duplicated; not in earlier sample)', '1,394'],
        ['Validation sample (stratified random; seed 20261002)', '250']]
for s in ['TripAdvisor', 'Trustpilot', 'Facebook Group']:
    rows.append([f'  {s.replace(" Group", "")}', str(int((vs.source == s).sum()))])
for a in sorted(vs.airline.unique()):
    rows.append([f'  {a} Airlines', str(int((vs.airline == a).sum()))])
rows.append(['Median review length in sample (words)', f"{vs.words.median():.1f}"])
rows.append(['Reviews with a star rating', str(int(M.star.notna().sum()))])
out['S6'] = pd.DataFrame(rows, columns=['Step / characteristic', 'Value'])

# ---- S1 panel B: source x carrier, ratings, excerpts
d2 = df.copy(); L = d2.text_length_chars
d2['excerpt'] = (d2.source == 'TripAdvisor') & L.between(270, 280)
rows = []
for s in ['TripAdvisor', 'Trustpilot', 'Facebook Group']:
    g = d2[d2.source == s]
    rows.append([s.replace(' Group', ''), *[int((g.airline == a).sum()) for a in ['Turkish', 'Pegasus']], len(g), int(g.star.notna().sum()), int(g.excerpt.sum())])
rows.append(['Total', int((d2.airline == 'Turkish').sum()), int((d2.airline == 'Pegasus').sum()), len(d2), int(d2.star.notna().sum()), int(d2.excerpt.sum())])
out['S1b'] = pd.DataFrame(rows, columns=['Source', 'Turkish Airlines', 'Pegasus Airlines', 'Total', 'With star rating', 'Excerpts (270–280 characters)'])
pd.to_pickle(out, 'val_supp.pkl')
for kk, v in out.items(): print(kk); print(v.to_string()); print()
