"""Rule-based detection helpers."""


def rules_engine(row):
    if row["z_score"] > 2.5 and row["cv_time_delta"] < 0.4:
        return True
    if row["duplicate_tweet_ratio"] > 0.30:
        return True
    if row["avg_cosine_similarity"] > 0.90:
        return True
    if row["hour_entropy"] < 0.25 and row["tweets_per_day_avg"] > 20:
        return True
    if row["max_tweets_in_10min"] > 25:
        return True
    return False
