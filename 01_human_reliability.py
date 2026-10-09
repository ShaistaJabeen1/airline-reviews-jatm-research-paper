import pandas as pd, numpy as np, sys, json
from sklearn.metrics import cohen_kappa_score, f1_score
rng = np.random.default_rng(20261008)
hc = pd.read_csv('human_coding_validation.csv')
a = hc[hc.coder == 'A'].reset_index(drop=True); b = hc[hc.coder == 'B'].reset_index(drop=True)
k = pd.read_csv('validation_sample.csv').rename(columns={'zs_justice_dimension': 'justice_dimension', 'zs_kano_category': 'kano_category', 'zs_service_dimension': 'service_dimension', 'zs_sentiment_score': 'sentiment_score'})
df = pd.read_pickle('df.pkl')
assert (a.Code == b.Code).all() and (a.Code == k.Code).all()
# (text-integrity check against review texts omitted: texts are not redistributed)

def sent(s): return s.map(lambda v: int(str(v)[0]) if pd.notna(v) else np.nan)
def just(s): return s.map(lambda v: v if pd.isna(v) else v.split(' - ')[0].replace(' Justice', '').replace('General/Other', 'General/Other'))
def kano(s): return s.map(lambda v: v if pd.isna(v) else v.replace(' Quality', ''))
H = {}
for nm, d in [('A', a), ('B', b)]:
    H[nm] = pd.DataFrame({'sent': sent(d['Sentiment (1-5)']), 'just': just(d['Justice category']),
                          'kano': kano(d['Kano-informed category']), 'serv': d['Service dimension'],
                          'serv8': d['Service dimension'].replace('Booking and Pricing Transparency', 'General/Other')})
M = pd.DataFrame({'sent': k.sentiment_score.astype(float), 'just': just(k.justice_dimension),
                  'kano': kano(k.kano_category), 'serv': k.service_dimension, 'serv8': k.service_dimension})
M['star'] = df.set_index('index').loc[k['index'], 'star'].values

def boot(fn, x, y, B=2000):
    m = x.notna() & y.notna(); x, y = x[m].values, y[m].values; n = len(x)
    est = fn(x, y); bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        try: bs.append(fn(x[i], y[i]))
        except Exception: pass
    return est, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), n
kap = lambda x, y: cohen_kappa_score(x, y)
qwk = lambda x, y: cohen_kappa_score(x, y, weights='quadratic')
acc = lambda x, y: (x == y).mean()
def pol(v): return np.where(v <= 2, 'neg', np.where(v == 3, 'neu', 'pos'))
R = []
def row(task, comp, metric, fn, x, y):
    e, l, u, n = boot(fn, x, y); R.append(dict(task=task, comparison=comp, metric=metric, n=n, est=e, lo=l, hi=u))

# ---- inter-coder
row('Sentiment', 'Coder A vs B', '% exact', acc, H['A'].sent, H['B'].sent)
row('Sentiment', 'Coder A vs B', 'Cohen κ', kap, H['A'].sent, H['B'].sent)
row('Sentiment', 'Coder A vs B', 'Quadratic-weighted κ', qwk, H['A'].sent, H['B'].sent)
row('Sentiment', 'Coder A vs B', 'Polarity κ', lambda x, y: cohen_kappa_score(pol(x), pol(y)), H['A'].sent, H['B'].sent)
for t, lab in [('just', 'Justice'), ('kano', 'Kano'), ('serv', 'Service (9 cat.)')]:
    row(lab, 'Coder A vs B', '% agreement', acc, H['A'][t], H['B'][t])
    row(lab, 'Coder A vs B', 'Cohen κ', kap, H['A'][t], H['B'][t])
# ---- model vs each coder and vs agreed subset
for t, lab in [('sent', 'Sentiment'), ('just', 'Justice'), ('kano', 'Kano'), ('serv8', 'Service (8 cat.)')]:
    agree = H['A'][t].where(H['A'][t] == H['B'][t])
    for comp, ref in [('Model vs Coder A', H['A'][t]), ('Model vs Coder B', H['B'][t]), ('Model vs agreed items', agree)]:
        row(lab, comp, '% agreement', acc, M[t], ref)
        row(lab, comp, 'Cohen κ', kap, M[t], ref)
        if t == 'sent':
            row(lab, comp, 'Quadratic-weighted κ', qwk, M[t], ref)
            row(lab, comp, 'Polarity % agreement', lambda x, y: (pol(x) == pol(y)).mean(), M[t], ref)
    if t != 'sent':
        # selective: only items where the model assigned a substantive label
        sel = M[t] != 'General/Other'
        row(lab, 'Model (assigned only) vs agreed items', '% agreement', acc, M[t][sel], agree[sel])
        for c in ['A', 'B']:
            row(lab, f'Model (assigned only) vs Coder {c}', '% agreement', acc, M[t][sel], H[c][t][sel])
# humans vs star rating (sanity)
row('Sentiment', 'Coder A vs star', 'Quadratic-weighted κ', qwk, H['A'].sent, M.star)
row('Sentiment', 'Coder B vs star', 'Quadratic-weighted κ', qwk, H['B'].sent, M.star)
row('Sentiment', 'Model vs star', 'Quadratic-weighted κ', qwk, M.sent, M.star)
R = pd.DataFrame(R)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
print(R.round(3).to_string())
# coverage and distributions
cov = {}
for t in ['just', 'kano', 'serv8']:
    cov[t] = dict(model_GO=(M[t] == 'General/Other').mean(), A_GO=(H['A'][t] == 'General/Other').mean(),
                  B_GO=(H['B'][t] == 'General/Other').mean())
print(json.dumps(cov, indent=1))
for t in ['just', 'kano', 'serv']:
    print(pd.concat([H['A'][t].value_counts(), H['B'][t].value_counts(), M[t if t != 'serv' else 'serv8'].value_counts()], axis=1, keys=['A', 'B', 'Model']).fillna(0).astype(int))
print('booking chosen A,B,both:', (H['A'].serv == 'Booking and Pricing Transparency').sum(), (H['B'].serv == 'Booking and Pricing Transparency').sum(),
      ((H['A'].serv == 'Booking and Pricing Transparency') & (H['B'].serv == 'Booking and Pricing Transparency')).sum())
# per-category confusion model vs agreed for kano/justice
for t in ['just', 'kano']:
    agree = H['A'][t].where(H['A'][t] == H['B'][t])
    print(pd.crosstab(agree, M[t], margins=True))
# per-category kappa A vs B (one-vs-rest)
for t in ['just', 'kano', 'serv']:
    m = H['A'][t].notna() & H['B'][t].notna()
    for c in sorted(set(H['A'][t].dropna()) | set(H['B'][t].dropna())):
        print(t, c, round(cohen_kappa_score(H['A'][t][m] == c, H['B'][t][m] == c), 3))
pd.to_pickle(dict(R=R, H=H, M=M, cov=cov, k=k), 'val_new.pkl')
