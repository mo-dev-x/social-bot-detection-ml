"""Feature extraction for user-level bot detection."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from .utils import (
    estimate_language,
    extract_urls,
    jaccard_similarity,
    load_json_dataset,
    normalize_for_similarity,
    parse_timestamp,
    safe_divide,
    shannon_entropy,
    tokenize_words,
)


class FeatureExtractor:
    """Convert raw user/posts JSON into user-level feature rows."""

    def __init__(self, language: str = "en"):
        self.language = language
        stop_words = "english" if language == "en" else None
        self.tfidf = TfidfVectorizer(max_features=150, lowercase=True, stop_words=stop_words)

    def extract_all_features(self, users_dict, posts_list):
        """Build a dataframe with one row per user."""
        posts_by_author = defaultdict(list)
        for post in posts_list:
            author_id = post.get("author_id")
            if author_id is not None:
                posts_by_author[author_id].append(post)

        feature_rows = []
        for user in tqdm(users_dict, desc=f"Extracting features ({self.language.upper()})"):
            user_id = user.get("id")
            tweets = posts_by_author.get(user_id, [])
            if not tweets:
                continue

            row = {"user_id": user_id}
            row.update(self._extract_temporal_features(tweets))
            row.update(self._extract_text_features(tweets))
            row.update(self._extract_profile_features(user))
            row.update(self._extract_activity_features(user, tweets))
            feature_rows.append(row)

        features_df = pd.DataFrame(feature_rows)
        if features_df.empty:
            return features_df
        return features_df.fillna(0)

    def _extract_temporal_features(self, tweets):
        timestamps = [parse_timestamp(tweet.get("created_at")) for tweet in tweets]
        timestamps = [ts for ts in timestamps if ts is not None]
        timestamps.sort()

        base = {
            "mean_time_delta": 0.0,
            "std_time_delta": 0.0,
            "cv_time_delta": 0.0,
            "median_time_delta": 0.0,
            "min_time_delta": 0.0,
            "max_time_delta": 0.0,
            "hour_entropy": 0.0,
            "day_entropy": 0.0,
            "night_activity_ratio": 0.0,
            "peak_hour_ratio": 0.0,
            "hours_active": 0.0,
            "tweets_per_day_avg": 0.0,
            "std_tweets_per_day": 0.0,
            "activity_span_days": 0.0,
            "max_tweets_in_10min": 0.0,
            "max_tweets_in_1hour": 0.0,
            "max_tweets_in_1day": 0.0,
            "tweets_during_sleep": 0.0,
            "tweets_during_work": 0.0,
            "ratio_sleep_to_active": 0.0,
            "weekend_ratio": 0.0,
        }

        if len(timestamps) < 2:
            return base

        time_deltas = [
            (timestamps[index + 1] - timestamps[index]).total_seconds()
            for index in range(len(timestamps) - 1)
        ]

        base["mean_time_delta"] = float(np.mean(time_deltas))
        base["std_time_delta"] = float(np.std(time_deltas))
        base["median_time_delta"] = float(np.median(time_deltas))
        base["min_time_delta"] = float(np.min(time_deltas))
        base["max_time_delta"] = float(np.max(time_deltas))
        base["cv_time_delta"] = safe_divide(base["std_time_delta"], base["mean_time_delta"])

        hours = [ts.hour for ts in timestamps]
        dates = [ts.date() for ts in timestamps]
        weekdays = [ts.weekday() for ts in timestamps]
        daily_counts = Counter(dates)

        base["hour_entropy"] = shannon_entropy(hours)
        base["day_entropy"] = shannon_entropy(dates)
        base["hours_active"] = float(len(set(hours)))
        base["night_activity_ratio"] = safe_divide(sum(1 for hour in hours if 1 <= hour <= 5), len(hours))
        base["peak_hour_ratio"] = safe_divide(max(Counter(hours).values()), len(hours))
        base["activity_span_days"] = float(len(daily_counts))
        base["tweets_per_day_avg"] = safe_divide(len(timestamps), len(daily_counts))
        base["std_tweets_per_day"] = float(np.std(list(daily_counts.values()))) if daily_counts else 0.0
        base["max_tweets_in_10min"] = float(self._max_tweets_in_window(timestamps, 600))
        base["max_tweets_in_1hour"] = float(self._max_tweets_in_window(timestamps, 3600))
        base["max_tweets_in_1day"] = float(max(daily_counts.values())) if daily_counts else 0.0
        base["tweets_during_sleep"] = float(sum(1 for hour in hours if 2 <= hour <= 6))
        base["tweets_during_work"] = float(sum(1 for hour in hours if 9 <= hour <= 17))
        base["ratio_sleep_to_active"] = safe_divide(base["tweets_during_sleep"], len(hours))
        base["weekend_ratio"] = safe_divide(sum(1 for day in weekdays if day >= 5), len(weekdays))
        return base

    @staticmethod
    def _max_tweets_in_window(timestamps, window_seconds: int) -> int:
        if len(timestamps) < 2:
            return len(timestamps)

        max_count = 1
        left = 0
        for right in range(len(timestamps)):
            while (timestamps[right] - timestamps[left]).total_seconds() > window_seconds:
                left += 1
            max_count = max(max_count, right - left + 1)
        return max_count

    def _extract_text_features(self, tweets):
        texts = [tweet.get("text", "") for tweet in tweets]
        texts = [text for text in texts if text]

        base = {
            "duplicate_tweet_ratio": 0.0,
            "avg_cosine_similarity": 0.0,
            "near_duplicate_ratio": 0.0,
            "unique_words_count": 0.0,
            "unique_words_ratio": 0.0,
            "vocabulary_repetition": 0.0,
            "word_entropy": 0.0,
            "avg_words_per_tweet": 0.0,
            "std_words_per_tweet": 0.0,
            "avg_word_length": 0.0,
            "punctuation_ratio": 0.0,
            "emoji_ratio": 0.0,
            "uppercase_ratio": 0.0,
            "digit_ratio": 0.0,
            "avg_hashtags_per_tweet": 0.0,
            "hashtag_reuse_ratio": 0.0,
            "url_ratio": 0.0,
            "url_diversity": 0.0,
            "primary_language_consistency": 0.0,
            "language_switch_count": 0.0,
            "mention_ratio": 0.0,
            "reply_ratio": 0.0,
        }

        if not texts:
            return base

        normalized_texts = [normalize_for_similarity(text) for text in texts]
        unique_texts = set(normalized_texts)
        base["duplicate_tweet_ratio"] = 1 - safe_divide(len(unique_texts), len(normalized_texts))

        if len(normalized_texts) > 1:
            try:
                matrix = self.tfidf.fit_transform(normalized_texts)
                similarities = (matrix @ matrix.T).toarray()
                similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)
                mask = np.triu(np.ones_like(similarities), k=1).astype(bool)
                similarity_values = similarities[mask]
                if similarity_values.size:
                    base["avg_cosine_similarity"] = float(np.mean(similarity_values))
                    base["near_duplicate_ratio"] = safe_divide((similarity_values > 0.8).sum(), similarity_values.size)
            except Exception:
                pass

        per_tweet_tokens = [tokenize_words(text) for text in texts]
        all_words = [word for words in per_tweet_tokens for word in words]
        unique_words = set(all_words)
        word_lengths = [len(word) for word in all_words]

        base["unique_words_count"] = float(len(unique_words))
        base["unique_words_ratio"] = safe_divide(len(unique_words), len(all_words))
        base["vocabulary_repetition"] = 1 - safe_divide(len(unique_words), len(all_words))
        base["word_entropy"] = shannon_entropy(all_words)
        base["avg_words_per_tweet"] = safe_divide(len(all_words), len(texts))
        base["std_words_per_tweet"] = float(np.std([len(words) for words in per_tweet_tokens]))
        base["avg_word_length"] = float(np.mean(word_lengths)) if word_lengths else 0.0

        full_text = " ".join(texts)
        base["punctuation_ratio"] = safe_divide(sum(full_text.count(ch) for ch in "!?.,-;:"), len(full_text))
        base["emoji_ratio"] = safe_divide(sum(1 for char in full_text if ord(char) > 127), len(full_text))
        base["uppercase_ratio"] = safe_divide(sum(1 for char in full_text if char.isupper()), len(full_text))
        base["digit_ratio"] = safe_divide(sum(1 for char in full_text if char.isdigit()), len(full_text))
        base["avg_hashtags_per_tweet"] = safe_divide(sum(text.count("#") for text in texts), len(texts))
        base["mention_ratio"] = safe_divide(sum(text.count("@") for text in texts), len(texts))
        base["reply_ratio"] = safe_divide(sum(1 for text in texts if text.lstrip().startswith("@")), len(texts))

        hashtags = []
        urls = []
        guessed_languages = []
        for text in texts:
            words = text.split()
            hashtags.extend(word for word in words if word.startswith("#"))
            urls.extend(extract_urls(text))
            guessed_languages.append(estimate_language([text]))

        base["hashtag_reuse_ratio"] = 1 - safe_divide(len(set(hashtags)), len(hashtags)) if hashtags else 0.0
        base["url_ratio"] = safe_divide(sum(1 for text in texts if extract_urls(text)), len(texts))
        base["url_diversity"] = safe_divide(len(set(urls)), len(urls)) if urls else 0.0
        base["primary_language_consistency"] = safe_divide(
            sum(1 for lang in guessed_languages if lang == self.language),
            len(guessed_languages),
        )

        switch_count = 0
        for previous, current in zip(guessed_languages, guessed_languages[1:]):
            if previous != current and "unknown" not in {previous, current}:
                switch_count += 1
        base["language_switch_count"] = float(switch_count)
        return base

    def _extract_profile_features(self, user):
        username = user.get("username", "")
        description = user.get("description", "")
        location = user.get("location", "")
        name = user.get("name", "")

        vowels = sum(1 for char in username.lower() if char in "aeiou")
        return {
            "username_length": float(len(username)),
            "username_digit_ratio": safe_divide(sum(1 for char in username if char.isdigit()), len(username)),
            "username_entropy": shannon_entropy(list(username.lower())) if username else 0.0,
            "username_vowel_ratio": safe_divide(vowels, len(username)),
            "has_description": 1.0 if description else 0.0,
            "description_length": float(len(description)),
            "description_contains_url": 1.0 if "http" in description.lower() else 0.0,
            "description_wordcount": float(len(description.split())),
            "has_location": 1.0 if location else 0.0,
            "location_length": float(len(location)),
            "name_length": float(len(name)),
            "name_matches_username": jaccard_similarity(name, username),
        }

    def _extract_activity_features(self, user, tweets):
        timestamps = [parse_timestamp(tweet.get("created_at")) for tweet in tweets]
        timestamps = [ts for ts in timestamps if ts is not None]

        if not timestamps:
            return {
                "tweet_count": float(user.get("tweet_count", 0)),
                "z_score": float(user.get("z_score", 0)),
                "tweets_per_hour_avg": 0.0,
                "tweets_per_day_avg": 0.0,
                "activity_days_ratio": 0.0,
            }

        unique_hours = {(timestamp.date(), timestamp.hour) for timestamp in timestamps}
        unique_days = {timestamp.date() for timestamp in timestamps}
        span_days = max((max(unique_days) - min(unique_days)).days + 1, 1)

        return {
            "tweet_count": float(user.get("tweet_count", len(tweets))),
            "z_score": float(user.get("z_score", 0)),
            "tweets_per_hour_avg": safe_divide(len(tweets), len(unique_hours)),
            "tweets_per_day_avg": safe_divide(len(tweets), len(unique_days)),
            "activity_days_ratio": safe_divide(len(unique_days), span_days),
        }


def create_feature_dataframe(dataset_filepath, language: str = "en"):
    """Load a JSON dataset and convert it into model features."""
    data = load_json_dataset(dataset_filepath)
    extractor = FeatureExtractor(language=language)
    return extractor.extract_all_features(data["users"], data["posts"])


__all__ = ["FeatureExtractor", "create_feature_dataframe", "load_json_dataset"]


if __name__ == "__main__":
    df_en = create_feature_dataframe("data/train_en/dataset.posts&users.json", language="en")
    print(f"English features shape: {df_en.shape}")
    print(f"Columns: {df_en.columns.tolist()}")
