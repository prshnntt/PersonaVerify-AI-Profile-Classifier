"""
predictor/explainer.py

Explainable AI (XAI) module for PersonaVerify.

Uses the Random Forest model's internal structure to compute
per-prediction feature contributions — showing exactly WHY
the model classified a profile as Fake or Real.

Approach:
  1. Each tree in the Random Forest traverses from root → leaf.
  2. At each split node, the prediction changes based on a specific feature.
  3. We trace each tree's decision path and accumulate how much each
     feature shifted the prediction away from the base rate.
  4. Averaging across all trees gives us the per-instance contribution
     of every feature.

This is equivalent to the TreeInterpreter / TreeSHAP method
and provides faithful, model-specific explanations.
"""

import numpy as np
from .model_loader import ModelLoader


# ─── Human-readable feature descriptions ─────────────────────────────────────

FEATURE_LABELS = {
    'profile pic': 'Profile Picture',
    'nums/length username': 'Numeric Characters in Username',
    'fullname words': 'Full Name Word Count',
    'nums/length fullname': 'Numeric Characters in Full Name',
    'name==username': 'Name Matches Username',
    'description length': 'Bio Length',
    'external URL': 'External URL Present',
    'private': 'Private Account',
    '#posts': 'Post Count',
    '#followers': 'Follower Count',
    '#follows': 'Following Count',
    'followers_following_ratio': 'Followers-to-Following Ratio',
    'post_per_follower': 'Posts per Follower',
    'has_bio': 'Has Bio',
    'is_active': 'Is Active (Has Posts)',
    'log_followers': 'Follower Count (log scale)',
    'log_follows': 'Following Count (log scale)',
    'log_posts': 'Post Count (log scale)',
    'spammy_username': 'Spammy Username Pattern',
    'no_pic_no_bio': 'No Profile Pic and No Bio',
}


def _compute_tree_contributions(tree, X_scaled):
    """
    Trace the decision path of a single decision tree and compute
    how much each feature contributed to the final prediction.

    For each split node on the path:
      contribution[feature] += (child_value - parent_value)

    Returns:
      contributions: array of shape (n_features,) — per-feature contribution
      bias: float — the base prediction (root node value)
    """
    tree_model = tree.tree_
    node_indicator = tree.decision_path(X_scaled)
    node_indices = node_indicator.indices

    # Base value: the mean prediction at the root node
    # tree_model.value[node_id] has shape [n_outputs, n_classes]
    # For binary classification: value[node_id][0] = [count_class_0, count_class_1]
    def node_probability(node_id):
        """Get P(Fake) at a given node."""
        values = tree_model.value[node_id][0]
        total = values.sum()
        if total == 0:
            return 0.5
        return values[1] / total  # P(class=1) = P(Fake)

    n_features = X_scaled.shape[1]
    contributions = np.zeros(n_features)
    bias = node_probability(node_indices[0])

    for i in range(len(node_indices) - 1):
        parent_id = node_indices[i]
        child_id = node_indices[i + 1]

        feature_used = tree_model.feature[parent_id]
        if feature_used >= 0:  # Internal node (not leaf)
            delta = node_probability(child_id) - node_probability(parent_id)
            contributions[feature_used] += delta

    return contributions, bias


def explain_prediction(features_scaled, raw_features_df):
    """
    Generate a full Explainable AI report for a single prediction.

    Args:
        features_scaled: numpy array of scaled features (1, n_features)
        raw_features_df: pandas DataFrame of unscaled features (1 row)

    Returns:
        dict with:
          - feature_contributions: list of {feature, contribution, direction, value, description}
          - bias (base rate)
          - risk_factors: human-readable list of why the profile looks fake/real
    """
    model = ModelLoader.get_model()
    feature_names = ModelLoader.get_feature_names()
    n_features = len(feature_names)

    # ── Compute per-tree contributions, then average ──────────────────────

    all_contributions = []
    all_biases = []

    for tree in model.estimators_:
        contribs, bias = _compute_tree_contributions(tree, features_scaled)
        all_contributions.append(contribs)
        all_biases.append(bias)

    # Average across all trees in the forest
    mean_contributions = np.mean(all_contributions, axis=0)
    mean_bias = np.mean(all_biases)

    # ── Build per-feature explanation ─────────────────────────────────────

    feature_explanations = []
    for i, fname in enumerate(feature_names):
        contrib = float(mean_contributions[i])
        raw_value = float(raw_features_df.iloc[0][fname])

        # Direction: positive contribution → pushes toward Fake
        if abs(contrib) < 0.001:
            direction = 'neutral'
        elif contrib > 0:
            direction = 'toward_fake'
        else:
            direction = 'toward_real'

        feature_explanations.append({
            'feature': fname,
            'label': FEATURE_LABELS.get(fname, fname),
            'contribution': round(contrib, 4),
            'direction': direction,
            'value': raw_value,
        })

    # Sort by absolute contribution (most impactful first)
    feature_explanations.sort(key=lambda x: abs(x['contribution']), reverse=True)

    # ── Generate human-readable risk factors ──────────────────────────────

    risk_factors = _generate_risk_factors(raw_features_df, feature_explanations)

    # ── Global feature importance (from RF training) ──────────────────────

    importances_df = ModelLoader.get_feature_importances()
    global_importance = [
        {
            'feature': row['Feature'],
            'label': FEATURE_LABELS.get(row['Feature'], row['Feature']),
            'importance': round(float(row['Importance']), 4),
        }
        for _, row in importances_df.head(10).iterrows()
    ]

    return {
        'bias': round(mean_bias, 4),
        'feature_contributions': feature_explanations[:10],  # Top 10
        'risk_factors': risk_factors,
        'global_feature_importance': global_importance,
    }


def _generate_risk_factors(raw_df, contributions):
    """
    Generate human-readable risk factor descriptions
    based on the actual feature values and their contributions.
    """
    row = raw_df.iloc[0]
    factors = []

    # Check each feature and generate plain-English explanations
    checks = [
        (row.get('profile pic', 1) == 0,
         'toward_fake',
         'No profile picture — common in fake accounts'),
        (row.get('#followers', 0) < 50,
         'toward_fake',
         f'Very low follower count ({int(row.get("#followers", 0))}) — suspicious for a real account'),
        (row.get('#followers', 0) >= 1000,
         'toward_real',
         f'High follower count ({int(row.get("#followers", 0))}) — indicates established presence'),
        (row.get('#follows', 0) > 1000 and row.get('#followers', 0) < 100,
         'toward_fake',
         f'Follows {int(row.get("#follows", 0))} but only {int(row.get("#followers", 0))} followers — follow-spam pattern'),
        (row.get('#posts', 0) == 0,
         'toward_fake',
         'Zero posts — account has no content'),
        (row.get('#posts', 0) > 20,
         'toward_real',
         f'{int(row.get("#posts", 0))} posts — shows regular activity'),
        (row.get('no_pic_no_bio', 0) == 1,
         'toward_fake',
         'No profile picture AND no bio — strong fake signal'),
        (row.get('spammy_username', 0) == 1,
         'toward_fake',
         'Username contains mostly numbers — spammy pattern'),
        (row.get('has_bio', 1) == 0,
         'toward_fake',
         'No bio/description — minimal account effort'),
        (row.get('has_bio', 0) == 1 and row.get('description length', 0) > 30,
         'toward_real',
         f'Has a detailed bio ({int(row.get("description length", 0))} chars)'),
        (row.get('followers_following_ratio', 0) < 0.1 and row.get('#follows', 0) > 100,
         'toward_fake',
         f'Very low followers-to-following ratio ({row.get("followers_following_ratio", 0):.2f}) — one-sided engagement'),
        (row.get('private', 0) == 1,
         'toward_real',
         'Private account — fake accounts are rarely private'),
        (row.get('external URL', 0) == 1,
         'toward_real',
         'Has external URL — shows more account investment'),
    ]

    for condition, direction, message in checks:
        if condition:
            factors.append({
                'direction': direction,
                'message': message,
            })

    return factors
