"""Conservative rule-based safety checks."""


def rules_engine(row):
    """Return True only for accounts with extreme, multi-signal automation patterns."""
    duplicate_burst_rule = (
        row["near_duplicate_ratio"] >= 0.50
        and row["tweet_count"] >= 20
        and row["burst_ratio_1h"] >= 0.35
    )
    coordinated_repost_rule = (
        row["cross_user_repost_ratio"] >= 0.60
        and row["template_duplicate_ratio"] >= 0.45
        and row["hour_uniform_chi2"] >= 18.0
    )
    regularity_rule = (
        row["duplicate_tweet_ratio"] >= 0.45
        and row["cv_time_delta"] <= 0.12
        and row["hour_entropy"] <= 1.2
        and row["tweet_count"] >= 15
    )
    zscore_rule = (
        row["z_score"] >= 2.0
        and row["near_duplicate_ratio"] >= 0.35
        and row["tweets_per_hour"] >= 1.5
    )

    return bool(duplicate_burst_rule or coordinated_repost_rule or regularity_rule or zscore_rule)
