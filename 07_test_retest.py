"""Test-retest check: labels for the 250 validation reviews in the separate validation run vs. the full-corpus run."""
import pandas as pd
v = pd.read_csv('llm_labels_validation.csv').set_index('index')
a = pd.read_csv('llm_labels_all.csv').set_index('index').loc[v.index]
for c in ['llm_sentiment', 'llm_justice', 'llm_kano', 'llm_service']:
    print(f"{c}: identical in {(v[c].astype(str) == a[c].astype(str)).mean():.1%} of {len(v)} reviews")
