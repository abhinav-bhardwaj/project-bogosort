"""
compute_eda_cache.py

Pre-computes all EDA statistics from training data with engineered features.
Exports as a single JSON file for Flask to load at startup.

This script should be run once (or on a schedule) when new data arrives.
It separates computation from serving, keeping Flask lightweight.

Usage:
    python -m app.services.compute_eda_cache \
        --train-path data/train_set_with_features.csv \
        --output data/eda_cache.json
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FEATURE_CATEGORIES = {
    'sentiment': [
        'vader_compound',
        'vader_neg',
        'vader_pos',
        'vader_is_negative',
        'vader_intensity',
        'vader_pos_minus_neg',
    ],
    'second_person': [
        'has_second_person',
        'second_person_count',
        'second_person_density',
    ],
    'profanity': ['profanity_count', 'obfuscated_profanity_count'],
    'slang': ['slang_count'],
    'text_shape': [
        'char_count',
        'word_count',
        'exclamation_count',
        'uppercase_ratio',
    ],
    'unique_words': ['unique_word_ratio'],
    'elongation': ['elongated_token_count', 'consecutive_punct_count'],
    'urls_ips': ['url_count', 'ip_count', 'has_url_or_ip'],
    'syntactic': ['negation_count', 'sentence_count', 'avg_sentence_length'],
    'identity': [
        'identity_mention_count',
        'identity_race',
        'identity_gender',
        'identity_sexuality',
        'identity_religion',
        'identity_disability',
        'identity_nationality',
    ],
}


def compute_effect_size(toxic_vals, nontoxic_vals):
    """Compute Cohen's d effect size between two groups."""
    toxic_mean = toxic_vals.mean()
    nontoxic_mean = nontoxic_vals.mean()

    n1, n2 = len(toxic_vals), len(nontoxic_vals)
    var1, var2 = toxic_vals.var(), nontoxic_vals.var()

    if var1 + var2 == 0:
        return 0.0

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0

    return float((toxic_mean - nontoxic_mean) / pooled_std)


def compute_histogram(series, bins=40):
    """Compute histogram counts and bin edges for a numeric series."""
    series = series.dropna()
    if len(series) == 0:
        return {
            'counts': [],
            'bin_centers': [],
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'std': None,
        }

    counts, bin_edges = np.histogram(series, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {
        'counts': counts.astype(int).tolist(),
        'bin_centers': bin_centers.tolist(),
        'min': float(series.min()),
        'max': float(series.max()),
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std()),
    }


def compute_overview(df, target_col):
    """Compute basic dataset overview stats."""
    toxic_count = int((df[target_col] == 1).sum())
    nontoxic_count = int((df[target_col] == 0).sum())
    toxic_rate = toxic_count / len(df)

    if nontoxic_count > 0:
        imbalance_ratio = nontoxic_count / toxic_count
    else:
        imbalance_ratio = 0

    return {
        'total_rows': len(df),
        'toxic_count': toxic_count,
        'nontoxic_count': nontoxic_count,
        'toxic_rate': round(toxic_rate, 4),
        'nontoxic_rate': round(1 - toxic_rate, 4),
        'imbalance_ratio': round(imbalance_ratio, 2),
        'missing_values': int(df.isna().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'total_features': len(df.columns),
    }


def compute_feature_categories_summary(df, target_col):
    """Compute summary stats for each feature category."""
    toxic_mask = df[target_col] == 1
    nontoxic_mask = df[target_col] == 0

    result = {}

    for cat_name, features in FEATURE_CATEGORIES.items():
        valid_features = [f for f in features if f in df.columns]
        if not valid_features:
            continue

        toxic_means = df.loc[toxic_mask, valid_features].mean().to_dict()
        nontoxic_means = df.loc[nontoxic_mask, valid_features].mean().to_dict()

        effect_sizes = {}
        for feat in valid_features:
            es = compute_effect_size(
                df.loc[toxic_mask, feat].dropna(),
                df.loc[nontoxic_mask, feat].dropna(),
            )
            effect_sizes[feat] = round(es, 4)

        result[cat_name] = {
            'features': valid_features,
            'toxic_means': {f: round(v, 6) for f, v in toxic_means.items()},
            'nontoxic_means': {f: round(v, 6) for f, v in nontoxic_means.items()},
            'effect_sizes': effect_sizes,
        }

    return result


def compute_top_features(df, target_col, top_n=15):
    """Rank all numeric features by effect size vs. target."""
    toxic_mask = df[target_col] == 1
    nontoxic_mask = df[target_col] == 0

    exclude_cols = {
        'id',
        'comment_text',
        target_col,
        'severe_toxic',
        'obscene',
        'threat',
        'insult',
        'identity_hate',
    }

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    features_ranked = []

    for col in numeric_cols:
        toxic_vals = df.loc[toxic_mask, col].dropna()
        nontoxic_vals = df.loc[nontoxic_mask, col].dropna()

        if len(toxic_vals) == 0 or len(nontoxic_vals) == 0:
            continue

        es = compute_effect_size(toxic_vals, nontoxic_vals)

        corr = df[[col, target_col]].corr(numeric_only=True).iloc[0, 1]

        category = None
        for cat_name, features in FEATURE_CATEGORIES.items():
            if col in features:
                category = cat_name
                break

        features_ranked.append(
            {
                'rank': 0,
                'name': col,
                'category': category,
                'effect_size': round(es, 4),
                'correlation': round(float(corr), 4),
                'toxic_mean': round(float(toxic_vals.mean()), 6),
                'nontoxic_mean': round(float(nontoxic_vals.mean()), 6),
                'toxic_median': round(float(toxic_vals.median()), 6),
                'nontoxic_median': round(float(nontoxic_vals.median()), 6),
            }
        )

    features_ranked.sort(key=lambda x: abs(x['effect_size']), reverse=True)

    for i, feat in enumerate(features_ranked[:top_n]):
        feat['rank'] = i + 1

    return features_ranked[:top_n]


def compute_feature_distributions(df, target_col):
    """Compute histograms for all numeric features."""
    toxic_mask = df[target_col] == 1
    nontoxic_mask = df[target_col] == 0

    exclude_cols = {
        'id',
        'comment_text',
        target_col,
        'severe_toxic',
        'obscene',
        'threat',
        'insult',
        'identity_hate',
    }

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    result = {}

    for col in numeric_cols:
        toxic_vals = df.loc[toxic_mask, col]
        nontoxic_vals = df.loc[nontoxic_mask, col]

        combined_vals = pd.concat([toxic_vals, nontoxic_vals]).dropna()
        if len(combined_vals) == 0:
            continue

        result[col] = {
            'toxic': compute_histogram(toxic_vals),
            'nontoxic': compute_histogram(nontoxic_vals),
        }

    return result


def compute_profanity_slang_analysis(df, target_col='toxic'):
    """Analyze profanity and slang patterns in comments."""
    if 'profanity_count' not in df.columns or 'slang_count' not in df.columns:
        return {
            'note': 'Profanity/slang features not available',
            'total_profanity': 0,
            'total_slang': 0,
        }

    toxic_mask = df[target_col] == 1
    nontoxic_mask = df[target_col] == 0

    profanity_stats = {
        'total_comments_with_profanity': int((df['profanity_count'] > 0).sum()),
        'avg_profanity_count': round(float(df['profanity_count'].mean()), 4),
        'max_profanity_count': int(df['profanity_count'].max()),
        'profanity_distribution': {
            'toxic': compute_histogram(df.loc[toxic_mask, 'profanity_count']),
            'nontoxic': compute_histogram(df.loc[nontoxic_mask, 'profanity_count']),
        },
    }

    slang_stats = {
        'total_comments_with_slang': int((df['slang_count'] > 0).sum()),
        'avg_slang_count': round(float(df['slang_count'].mean()), 4),
        'max_slang_count': int(df['slang_count'].max()),
        'slang_distribution': {
            'toxic': compute_histogram(df.loc[toxic_mask, 'slang_count']),
            'nontoxic': compute_histogram(df.loc[nontoxic_mask, 'slang_count']),
        },
    }

    obfuscation_stats = (
        {
            'avg_obfuscated_profanity': round(
                float(df['obfuscated_profanity_count'].mean()), 4
            ),
            'comments_with_obfuscation': int(
                (df['obfuscated_profanity_count'] > 0).sum()
            ),
        }
        if 'obfuscated_profanity_count' in df.columns
        else {}
    )

    return {
        'profanity': profanity_stats,
        'slang': slang_stats,
        'obfuscation': obfuscation_stats,
    }


def compute_identity_analysis(df, target_col):
    """Analyze identity mention patterns and toxicity rates."""
    identity_categories = [
        'race',
        'gender',
        'sexuality',
        'religion',
        'disability',
        'nationality',
    ]
    identity_cols = [f'identity_{cat}' for cat in identity_categories]
    identity_cols = [c for c in identity_cols if c in df.columns]

    if not identity_cols:
        return {'note': 'Identity features not available', 'categories': []}

    toxic_mask = df[target_col] == 1

    result = {'categories': []}

    for col in identity_cols:
        cat_name = col.replace('identity_', '')
        mentioned = (df[col] == 1).sum()

        if mentioned == 0:
            continue

        toxic_with_mention = ((df[col] == 1) & toxic_mask).sum()
        toxicity_rate = toxic_with_mention / mentioned if mentioned > 0 else 0
        baseline_rate = toxic_mask.sum() / len(df)
        relative_risk = toxicity_rate / baseline_rate if baseline_rate > 0 else 0

        result['categories'].append(
            {
                'category': cat_name,
                'mention_count': int(mentioned),
                'mention_pct': round(mentioned / len(df) * 100, 2),
                'toxic_with_mention': int(toxic_with_mention),
                'toxicity_rate': round(toxicity_rate, 4),
                'baseline_toxicity_rate': round(baseline_rate, 4),
                'relative_risk': round(relative_risk, 2),
            }
        )

    return result


def compute_text_shape_analysis(df, target_col='toxic'):
    """Analyze text structure features with toxic/nontoxic splits."""
    toxic_mask = df[target_col] == 1
    nontoxic_mask = df[target_col] == 0

    text_features = {
        'char_count': 'Character count',
        'word_count': 'Word count',
        'exclamation_count': 'Exclamation marks',
        'uppercase_ratio': 'Uppercase ratio',
        'unique_word_ratio': 'Unique word ratio',
    }

    result = {}

    for feat, label in text_features.items():
        if feat in df.columns:
            result[feat] = {
                'label': label,
                'toxic': compute_histogram(df.loc[toxic_mask, feat]),
                'nontoxic': compute_histogram(df.loc[nontoxic_mask, feat]),
            }

    return result


def compute_correlations(df, target_col):
    """Compute correlation matrix for top features."""
    exclude_cols = {
        'id',
        'comment_text',
        target_col,
        'severe_toxic',
        'obscene',
        'threat',
        'insult',
        'identity_hate',
    }

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    top_features = sorted(
        numeric_cols,
        key=lambda c: abs(df[[c, target_col]].corr(numeric_only=True).iloc[0, 1]),
        reverse=True,
    )[:12]

    top_features.append(target_col)

    corr_matrix = df[top_features].corr(numeric_only=True)

    return {
        'features': top_features,
        'correlation_matrix': corr_matrix.values.tolist(),
        'correlation_values': corr_matrix.to_dict(),
    }


def generate_eda_cache(df_path, target_col='toxic', output_path='eda_cache.json'):
    """Main function: load data, compute all stats, export JSON."""
    try:
        logger.info(f'Loading data from {df_path}...')
        df = pd.read_csv(df_path)
        logger.info(f'Loaded {len(df)} rows, {len(df.columns)} columns')

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        cache = {
            'metadata': {
                'computed_at': datetime.now().isoformat(),
                'data_path': str(df_path),
                'target_column': target_col,
                'data_shape': list(df.shape),
            },
            'overview': compute_overview(df, target_col),
            'feature_categories': compute_feature_categories_summary(df, target_col),
            'top_features': compute_top_features(df, target_col, top_n=15),
            'feature_distributions': compute_feature_distributions(df, target_col),
            'profanity_slang': compute_profanity_slang_analysis(df, target_col),
            'identity_analysis': compute_identity_analysis(df, target_col),
            'text_shape': compute_text_shape_analysis(df, target_col),
            'correlations': compute_correlations(df, target_col),
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)

        cache_size_kb = output_path.stat().st_size / 1024
        logger.info(f'EDA cache saved to {output_path}')
        logger.info(f'Cache size: {cache_size_kb:.1f} KB')

        return cache

    except FileNotFoundError:
        logger.error(f'Data file not found: {df_path}')
        raise
    except pd.errors.ParserError as e:
        logger.error(f'Error parsing CSV file: {e}')
        raise
    except Exception as e:
        logger.error(f'Error generating EDA cache: {e}', exc_info=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-compute EDA statistics')
    parser.add_argument(
        '--train-path',
        type=str,
        default='data/train_set_with_features.csv',
        help='Path to training data CSV',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/eda_cache.json',
        help='Output path for EDA cache JSON',
    )
    parser.add_argument(
        '--target',
        type=str,
        default='toxic',
        help='Target column name',
    )

    args = parser.parse_args()

    try:
        generate_eda_cache(args.train_path, target_col=args.target, output_path=args.output)
        logger.info('EDA cache generation completed successfully')
    except Exception as e:
        logger.error(f'EDA cache generation failed: {e}')
        exit(1)
