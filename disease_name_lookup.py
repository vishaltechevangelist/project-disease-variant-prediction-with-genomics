import pandas as pd
import numpy as np

def get_top_diseases(df, gene, is_indel=1, top_n=3, synonym_map=None):
    # 1) filter by gene and indel flag
    sub = df[(df['Gene_Symbol'] == gene) & (df['IS_INDEL'] == is_indel)].copy()
    # print(sub.columns)
    if sub.empty:
        return []

    # 2) explode disease list (assumes 'disease_name' pipe-separated)
    sub['disease_list'] = sub['Clinical_Disease_Name'].fillna('not_provided').str.split('|')
    sub = sub.explode('disease_list')
    sub['disease_list'] = sub['disease_list'].str.replace('_', ' ').str.strip().str.lower()
    
    # optional synonyms map
    if synonym_map:
        sub['disease_list'] = sub['disease_list'].replace(synonym_map)

    # 3) compute frequency and pathogenic counts
    # assume there's a 'clinical_significance' column with strings like 'Pathogenic' etc.
    grp = sub.groupby('disease_list').agg(
        freq=('disease_list','size'),
        path_count=('Clinical_Significance', lambda s: (s.str.lower()=='pathogenic').sum()),
        accessions=('lookup_id', lambda s: ','.join(map(str, s.unique()))),
        review_status_list=('Clinical_Review_Status', lambda s: list(s.dropna().unique()))
    ).reset_index()

    # 4) normalize scores
    grp['freq_n'] = grp['freq'] / grp['freq'].max()
    grp['path_n'] = grp['path_count'] / (grp['path_count'].max() if grp['path_count'].max()>0 else 1)
    # simple review score: more distinct review statuses -> better (customize as needed)
    grp['review_n'] = grp['review_status_list'].apply(lambda lst: min(len(lst)/3,1.0))

    w_freq, w_path, w_rev = 0.5, 0.35, 0.15
    grp['score'] = w_freq*grp['freq_n'] + w_path*grp['path_n'] + w_rev*grp['review_n']

    # 5) rank and return top_n with provenance
    out = grp.sort_values('score', ascending=False).head(top_n)
    results = []
    for _, r in out.iterrows():
        results.append({
            'disease': r['disease_list'],
            'score': float(round(r['score'], 3)),
            'freq': int(r['freq']),
            'pathogenic_submissions': int(r['path_count']),
            'evidence_id': list(sub[sub['disease_list']==r['disease_list']]['lookup_id'].unique()),
            'review_statuses': r['review_status_list']
        })
    return results
