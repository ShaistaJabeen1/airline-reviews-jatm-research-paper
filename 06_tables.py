import pandas as pd, numpy as np
h = pd.read_pickle('hyp.pkl'); r = pd.read_pickle('robust.pkl'); v = pd.read_pickle('val_llm.pkl')
F, T, LR = h['F'], h['T'], h['LR']
def orci(t, n):
    if n not in t.index: return ''
    x = t.loc[n]; star = '***' if x.p < .001 else '**' if x.p < .01 else '*' if x.p < .05 else ''
    return f"{x.OR:.2f} [{x.lo:.2f}, {x.hi:.2f}]{star}"
nice = lambda n: n.replace('Carrier: ', '').replace('Platform: ', '').replace('Period: ', 'Period ').replace(' (ref. Pegasus)', '').replace(' (ref. TripAdvisor)', '').replace('2020-2022', '2020–2022')
terms = [c for c in T['M4'].index]
rows = []
groups = [('Controls', [c for c in terms if c.startswith(('Carrier', 'Platform', 'Period', 'Review'))]),
          ('Service dimension (ref. General/Other)', [c for c in terms if c.startswith('Service')]),
          ('Organizational justice (ref. interactional)', ['Justice: Distributive', 'Justice: Procedural', 'Justice: General/Other']),
          ('Kano-informed category (ref. Performance)', ['Kano: Must-be', 'Kano: Attractive', 'Kano: General/Other']),
          ('Text sentiment', ['Text sentiment (z)'])]
for g, cs in groups:
    rows.append([g, '', ''])
    for c in cs: rows.append(['  ' + nice(c).split(': ')[-1] if ': ' in nice(c) else '  ' + nice(c), orci(T['M3'], c), orci(T['M4'], c)])
rows += [['Model fit', '', ''], ['  Log-likelihood', f"{F.loc['M3','ll']:.1f}", f"{F.loc['M4','ll']:.1f}"], ['  McFadden pseudo-R²', f"{F.loc['M3','r2']:.3f}", f"{F.loc['M4','r2']:.3f}"],
         ['  AIC', f"{F.loc['M3','aic']:.1f}", f"{F.loc['M4','aic']:.1f}"],
         ['  LR test vs. nested model', f"χ²(6) = {LR['M3 vs M2'][0]:.1f}*** (vs. Model 2)", f"χ²(6) = {LR['M4 vs M2b'][0]:.1f}*** (vs. Model 2b)"]]
T6 = pd.DataFrame(rows, columns=['Predictor', 'Model 3: OR [95% CI]', 'Model 4: OR [95% CI]'])
T6.loc[T6.Predictor == '  Turkish Airlines', 'Predictor'] = '  Turkish Airlines (ref. Pegasus Airlines)'
T6.loc[T6.Predictor == '  Trustpilot', 'Predictor'] = '  Trustpilot (ref. TripAdvisor)'
T6.loc[T6.Predictor.str.startswith('  Period'), 'Predictor'] = T6.loc[T6.Predictor.str.startswith('  Period'), 'Predictor'] + ' (ref. 2024)'
R = h['R']
def g(m, k, f='OR'): return R.loc[(m, k)][f]
T7 = pd.DataFrame([
 ['H1', 'Exploratory', 'Procedural vs. interactional; distributive vs. interactional (Model 3)', f"OR = {g('M3','H1a procedural vs interactional'):.2f} [{g('M3','H1a procedural vs interactional','lo'):.2f}, {g('M3','H1a procedural vs interactional','hi'):.2f}]; OR = {g('M3','H1b distributive vs interactional'):.2f} [{g('M3','H1b distributive vs interactional','lo'):.2f}, {g('M3','H1b distributive vs interactional','hi'):.2f}]", 'Not supported'],
 ['H2', 'Confirmatory', 'Must-be vs. Performance (Models 3 and 4)', f"OR = {g('M3','H2 must-be vs performance'):.2f} [{g('M3','H2 must-be vs performance','lo'):.2f}, {g('M3','H2 must-be vs performance','hi'):.2f}]; OR = {g('M4','H2 must-be vs performance'):.2f} [{g('M4','H2 must-be vs performance','lo'):.2f}, {g('M4','H2 must-be vs performance','hi'):.2f}]; Holm p < .001", 'Supported'],
 ['H3', 'Confirmatory (Kano); exploratory (justice)', 'Categories added beyond service dimension, controls and text sentiment', f"Kano beyond justice: LR χ²(3) = {2*(F.loc['M4','ll']-F.loc['M2b+J','ll']):.1f}, p < .001; justice beyond Kano: LR χ²(3) = {2*(F.loc['M4','ll']-F.loc['M2b+K','ll']):.1f}, p < .001", 'Supported'],
 ['H4', 'Exploratory', 'Must-be × distributive and Must-be × procedural (ref. Must-be × interactional; Model 5)', f"Wald χ²(2) = {g('M5','H4 joint (MB×Dist, MB×Proc)','W'):.2f}, p = {g('M5','H4 joint (MB×Dist, MB×Proc)','p'):.2f}", 'Not supported'],
 ['H5', 'Exploratory', 'Carrier × justice interactions (Model 6)', f"LR χ²(3) = {LR['M6 vs M4g'][0]:.1f}, Holm p = {h['holm']['H5']:.3f}; Turkish Airlines × distributive OR = {g('M6','H5 Turkish Airlines × Distributive'):.2f} [{g('M6','H5 Turkish Airlines × Distributive','lo'):.2f}, {g('M6','H5 Turkish Airlines × Distributive','hi'):.2f}]; × interactional OR = {g('M6','H5 Turkish Airlines × Interactional'):.2f} [{g('M6','H5 Turkish Airlines × Interactional','lo'):.2f}, {g('M6','H5 Turkish Airlines × Interactional','hi'):.2f}]", 'Partly supported (distributive penalty larger for the low-cost carrier; interactional effect opposite to prediction)'],
], columns=['Hypothesis', 'Status', 'Test', 'Result', 'Conclusion'])
D = r['DESC'].copy(); D['Category'] = D['Category'].str.replace(' and ', ' & ') if False else D['Category']
T5 = D.copy()
# supplement: model fit all models
S18a = F.loc[['M1', 'M2', 'M2b', 'M3', 'M4', 'M5', 'M6']].reset_index()
S18a['Model'] = ['Model 1: controls', 'Model 2: + service dimension', 'Model 2b: Model 2 + text sentiment', 'Model 3: Model 2 + justice + Kano', 'Model 4: Model 3 + text sentiment', 'Model 5: Model 4 + Must-be × justice', 'Model 6: Model 4 + carrier × justice']
S18a = pd.DataFrame({'Model': S18a.Model, 'Log-likelihood': S18a.ll.round(1), 'Parameters': S18a.k.astype(int), 'AIC': S18a.aic.round(1), 'McFadden R²': S18a.r2.round(3)})
LR['M4 vs M2b+J'] = (2*(F.loc['M4','ll']-F.loc['M2b+J','ll']), 3, 0.0); LR['M4 vs M2b+K'] = (2*(F.loc['M4','ll']-F.loc['M2b+K','ll']), 3, 0.0)
from scipy import stats as _st
LR['M4 vs M2b+J'] = (LR['M4 vs M2b+J'][0], 3, _st.chi2.sf(LR['M4 vs M2b+J'][0], 3)); LR['M4 vs M2b+K'] = (LR['M4 vs M2b+K'][0], 3, _st.chi2.sf(LR['M4 vs M2b+K'][0], 3))
lrrows = [[k, f"{v_[0]:.1f}", int(v_[1]), ('< .001' if v_[2] < .001 else f"{v_[2]:.3f}")] for k, v_ in LR.items()]
S18b = pd.DataFrame(lrrows, columns=['Comparison', 'LR χ²', 'df', 'p'])
def coef_tab(m):
    t = T[m]; return pd.DataFrame({'Term': [nice(c) for c in t.index], 'B': t.B.round(3), 'SE': t.SE.round(3), 'OR [95% CI]': [f"{a:.2f} [{b:.2f}, {c:.2f}]" for a, b, c in zip(t.OR, t.lo, t.hi)], 'p': [('< .001' if p < .001 else f"{p:.3f}") for p in t.p]})
S18c = coef_tab('M5'); S18d = coef_tab('M6')
simple = R[[k for k in R.index if k[1].startswith('simple')]].copy() if False else None
srows = []
for j in ['Distributive', 'Interactional', 'Procedural']:
    for car in ['Pegasus', 'Turkish Airlines']:
        x = R.loc[('M6', f'simple {j} | {car}')]; srows.append([j, car if car != 'Pegasus' else 'Pegasus Airlines', f"{x.OR:.2f} [{x.lo:.2f}, {x.hi:.2f}]", ('< .001' if x.p_two < .001 else f"{x.p_two:.3f}")])
S18e = pd.DataFrame(srows, columns=['Justice category (ref. General/Other)', 'Carrier', 'OR [95% CI]', 'p'])
pd.to_pickle(dict(T5=T5, T6=T6, T7=T7, S18a=S18a, S18b=S18b, S18c=S18c, S18d=S18d, S18e=S18e), 'final_tables.pkl')
print(T6.to_string()); print(T7.to_string()); print(S18b.to_string()); print(S18e)
