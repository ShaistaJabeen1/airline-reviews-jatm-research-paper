"""Builds df.pkl (the analysis dataset used by scripts 01-06) from analysis_dataset.csv."""
import pandas as pd
df = pd.read_csv('analysis_dataset.csv')
df = df.rename(columns={'zs_sentiment_score': 'sentiment_score', 'zs_justice_dimension': 'justice_dimension',
                        'zs_kano_category': 'kano_category', 'zs_service_dimension': 'service_dimension'})
df.to_pickle('df.pkl'); print(df.shape)
