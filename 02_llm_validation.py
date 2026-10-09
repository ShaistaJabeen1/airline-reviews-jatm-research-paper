import pandas as pd, numpy as np, json
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support
d = pd.read_pickle('val_new.pkl'); H, M, k = d['H'], d['M'], d['k']
v = pd.read_csv('llm_labels_validation.csv').set_index('Code').loc[k.Code].reset_index()
rng = np.random.default_rng(20261009)
L = pd.DataFrame({'just': v.llm_justice.str.replace(' Justice', ''), 'kano': v.llm_kano, 'serv': v.llm_service,
                  'serv8': v.llm_service.replace('Booking and Pricing Transparency', 'General/Other'), 'sent': v.llm_sentiment.astype(float)})
def ac1(x, y):
    cats = sorted(set(x) | set(y)); q = len(cats)
    pa = (x == y).mean(); pi = np.array([((x == c).mean() + (y == c).mean()) / 2 for c in cats]); pe = (pi * (1 - pi)).sum() / (q - 1)
    return (pa - pe) / (1 - pe)
kap = lambda x, y: cohen_kappa_score(x, y); qwk = lambda x, y: cohen_kappa_score(x, y, weights='quadratic')
acc = lambda x, y: (x == y).mean(); mf1 = lambda x, y: f1_score(x, y, average='macro')
pol = lambda v: np.where(v <= 2, 0, np.where(v == 3, 1, 2)); polacc = lambda x, y: (pol(x) == pol(y)).mean()
mae = lambda x, y: np.abs(x - y).mean()
def boot(fn, x, y, B=2000):
    m = x.notna() & y.notna(); x, y = x[m].values, y[m].values; n = len(x); e = fn(x, y); bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        try: bs.append(fn(x[i], y[i]))
        except Exception: pass
    return e, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), n
def fmt(t, pct=False):
    e, l, u, n = t
    return f"{e*100:.1f}% [{l*100:.1f}, {u*100:.1f}]" if pct else f"{e:.2f} [{l:.2f}, {u:.2f}]"
agreed = lambda t: H['A'][t].where(H['A'][t] == H['B'][t])
R = {}; rows = []
spec = {'serv': [('Agreement', acc, True), ('Cohen’s κ', kap, False), ('Gwet’s AC1', ac1, False), ('Macro-F1', mf1, False)],
        'just': [('Agreement', acc, True), ('Cohen’s κ', kap, False), ('Gwet’s AC1', ac1, False), ('Macro-F1', mf1, False)],
        'kano': [('Agreement', acc, True), ('Cohen’s κ', kap, False), ('Gwet’s AC1', ac1, False), ('Macro-F1', mf1, False)],
        'sent': [('Exact agreement', acc, True), ('Quadratic-weighted κ', qwk, False), ('Polarity agreement', polacc, True), ('Mean absolute error', mae, False), ('Macro-F1', mf1, False)]}
names = {'serv': 'Service dimension (9 categories)', 'just': 'Organizational justice', 'kano': 'Kano-informed category', 'sent': 'Sentiment (LLM)'}
crit = {}
for t, mets in spec.items():
    for name, fn, pct in mets:
        rA = boot(fn, L[t], H['A'][t]); rB = boot(fn, L[t], H['B'][t]); rG = boot(fn, L[t], agreed(t)); rH = boot(fn, H['A'][t], H['B'][t])
        rows.append([names[t], name, fmt(rA, pct), fmt(rB, pct), f"{(rA[0]+rB[0])/2*(100 if pct else 1):.{1 if pct else 2}f}{'%' if pct else ''}", fmt(rG, pct), int(agreed(t).notna().sum()), fmt(rH, pct)])
        R[(t, name)] = dict(A=rA[0], B=rB[0], mean=(rA[0] + rB[0]) / 2, agreed=rG[0], agreed_ci=(rG[1], rG[2]), human=rH[0])
tab = pd.DataFrame(rows, columns=['Task', 'Metric', 'LLM vs Coder A', 'LLM vs Coder B', 'Mean', 'LLM vs agreed subset', 'n agreed', 'Coder A vs Coder B'])
# criterion
for t in ['serv', 'just', 'kano']:
    kmean, kh = R[(t, 'Cohen’s κ')]['mean'], R[(t, 'Cohen’s κ')]['human']
    amean, ah = R[(t, 'Gwet’s AC1')]['mean'], R[(t, 'Gwet’s AC1')]['human']
    crit[t] = dict(kappa_mean=round(kmean, 3), kappa_human=round(kh, 3), ac1_mean=round(amean, 3), ac1_human=round(ah, 3), passed=bool(kmean >= kh and amean >= ah))
print(tab.to_string()); print(json.dumps(crit, indent=1))
# class-level vs agreed subset
cl = []
for t, lab in [('just', 'Organizational justice'), ('kano', 'Kano-informed category'), ('serv', 'Service dimension')]:
    g = agreed(t); m = g.notna(); y, p = g[m], L[t][m]
    cats = sorted(set(y) | set(p))
    P, Rr, F, S = precision_recall_fscore_support(y, p, labels=cats, zero_division=0)
    for c, a, b, f, s in zip(cats, P, Rr, F, S): cl.append([lab, c, int(s), int((p == c).sum()), f"{a:.2f}", f"{b:.2f}", f"{f:.2f}"])
cl = pd.DataFrame(cl, columns=['Classification', 'Category', 'n (agreed)', 'n predicted', 'Precision', 'Recall', 'F1'])
print(cl.to_string())
dist = []
for t, lab in [('just', 'Organizational justice'), ('kano', 'Kano-informed category'), ('serv', 'Service dimension')]:
    for c in sorted(set(H['A'][t].dropna()) | set(H['B'][t].dropna()) | set(L[t])):
        dist.append([lab, c, int((H['A'][t] == c).sum()), int((H['B'][t] == c).sum()), int((L[t] == c).sum()), int((M[t if t != 'serv' else 'serv8'] == c).sum())])
dist = pd.DataFrame(dist, columns=['Classification', 'Category', 'Coder A', 'Coder B', 'LLM', 'Zero-shot NLI'])
print(dist.to_string())
pd.to_pickle(dict(tab=tab, crit=crit, R=R, cl=cl, dist=dist, L=L), 'val_llm.pkl')
