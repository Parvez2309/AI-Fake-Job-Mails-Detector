import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from ml_pipeline import text_agent, tfidf_agent


def load_dataset():
    candidates = [
        'final_combined_dataset.csv',
        'final_fake_review_dataset.csv',
        'fake_job_postings.csv',
        'fake_job_postings_oversampled.csv',
    ]

    for name in candidates:
        if os.path.exists(name):
            df = pd.read_csv(name)
            if 'text' in df.columns and 'label' in df.columns:
                return df[['text', 'label']].dropna().drop_duplicates()
            if 'fraudulent' in df.columns:
                cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
                for c in cols:
                    if c not in df.columns:
                        df[c] = ''
                df = df.fillna('')
                df['text'] = (
                    'TITLE ' + df['title']
                    + ' COMPANY ' + df['company_profile']
                    + ' DESCRIPTION ' + df['description']
                    + ' REQUIREMENTS ' + df['requirements']
                    + ' BENEFITS ' + df['benefits']
                )
                df['label'] = df['fraudulent']
                return df[['text', 'label']].dropna().drop_duplicates()

    raise FileNotFoundError(
        "No dataset file was found. Place one of these in the project folder: "
        "final_combined_dataset.csv, final_fake_review_dataset.csv, fake_job_postings.csv"
    )


def main():
    df = load_dataset()
    clean_text = text_agent(df['text'])
    tfidf_features, vectorizer = tfidf_agent(clean_text)
    chi2_scores, _ = chi2(tfidf_features, df['label'])
    top_idx = np.argsort(chi2_scores)[::-1][:100]
    auto_keywords = [vectorizer.get_feature_names_out()[i] for i in top_idx]

    print('TOP 100 AUTO KEYWORDS:')
    for i, kw in enumerate(auto_keywords, start=1):
        print(f'{i:03d}. {kw}')

    with open('auto_keywords_top_100.txt', 'w', encoding='utf-8') as f:
        for kw in auto_keywords:
            f.write(f'{kw}\n')

    print('\nSaved to auto_keywords_top_100.txt')


if __name__ == '__main__':
    main()
