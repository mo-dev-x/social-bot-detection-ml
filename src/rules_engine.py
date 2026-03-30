"""Conservative rule-based safety checks."""


def rules_engine(row):
    """Return True only for accounts with extreme, multi-signal automation patterns."""
    language = row.get("_language", row.get("language", "en"))

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

    if language == "fr":
        fr_periodic_campaign_rule = (
            row["periodic_interval_ratio"] >= 0.99
            and row["avg_similarity_between_tweets"] <= 0.07
        )
        fr_low_periodicity_volume_rule = (
            row["periodic_interval_ratio"] <= 0.02
            and row["tweet_count"] >= 18
        )
        return bool(
            duplicate_burst_rule
            or coordinated_repost_rule
            or regularity_rule
            or zscore_rule
            or fr_periodic_campaign_rule
            or fr_low_periodicity_volume_rule
        )

    return bool(duplicate_burst_rule or coordinated_repost_rule or regularity_rule or zscore_rule)
