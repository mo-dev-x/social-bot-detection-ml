"""Feature extraction for user-level bot detection."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from .utils import load_json_dataset, normalize_for_similarity, parse_timestamp, safe_divide, shannon_entropy, tokenize_words

URL_PATTERN = re.compile(r"https?://\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
NUMBER_PATTERN = re.compile(r"\b\d+\b")
WORD_PATTERN = re.compile(r"\b[a-zA-Z\u00C0-\u00FF_]+\b", re.UNICODE)
ACCENTED_VOWELS = set("\u00E0\u00E2\u00E4\u00E6\u00E8\u00E9\u00EA\u00EB\u00EE\u00EF\u00F4\u0153\u00F9\u00FB\u00FC\u00FF")
VOWELS = set("aeiouy") | ACCENTED_VOWELS


class FeatureExtractor:
    """Convert raw dataset JSON into one feature row per author."""

    def __init__(self, language: str = "en"):
        self.language = language
        stop_words = "english" if language == "en" else None
        self.tfidf = TfidfVectorizer(max_features=200, lowercase=True, stop_words=stop_words)

    def extract_all_features(self, users, posts):
        posts_by_author = defaultdict(list)
        text_authors = defaultdict(set)
        text_counts = Counter()
        template_counts = Counter()

        for post in posts:
            author_id = post.get("author_id")
            if author_id:
                posts_by_author[author_id].append(post)
                normalized_text = normalize_for_similarity(post.get("text", ""))
                if normalized_text:
                    text_authors[normalized_text].add(author_id)
                    text_counts[normalized_text] += 1
                    template_counts[self._normalize_template(post.get("text", ""))] += 1

        rows = []
        for user in tqdm(users, desc=f"Extracting user features ({self.language.upper()})"):
            user_id = user.get("id")
            author_posts = posts_by_author.get(user_id, [])
            if not author_posts:
                continue

            row = {"user_id": user_id}
            row.update(self._extract_temporal_features(author_posts))
            row.update(self._extract_text_features(author_posts, text_authors, text_counts, template_counts))
            row.update(self._extract_profile_features(user))
            row.update(self._extract_activity_features(user, author_posts))
            rows.append(row)

        features_df = pd.DataFrame(rows)
        if features_df.empty:
            return features_df
        return features_df.replace([np.inf, -np.inf], 0).fillna(0)

    def _extract_temporal_features(self, posts):
        timestamps = [parse_timestamp(post.get("created_at")) for post in posts]
        timestamps = sorted(timestamp for timestamp in timestamps if timestamp is not None)

        base = {
            "mean_time_delta": 0.0,
            "std_time_delta": 0.0,
            "interval_variance": 0.0,
            "min_time_delta": 0.0,
            "max_time_delta": 0.0,
            "cv_time_delta": 0.0,
            "periodic_interval_ratio": 0.0,
            "successive_delay_ratio": 0.0,
            "min_rolling_cv": 0.0,
            "hour_entropy": 0.0,
            "hour_uniform_chi2": 0.0,
            "night_activity_ratio": 0.0,
            "tweets_per_hour": 0.0,
            "burst_ratio_1h": 0.0,
            "burst_ratio_extreme": 0.0,
        }

        if not timestamps:
            return base

        hours = [timestamp.hour for timestamp in timestamps]
        base["hour_entropy"] = shannon_entropy(hours)
        base["night_activity_ratio"] = safe_divide(sum(1 for hour in hours if 0 <= hour <= 5), len(hours))
        base["hour_uniform_chi2"] = self._hour_chi_square(hours)

        if len(timestamps) == 1:
            base["tweets_per_hour"] = 1.0
            base["burst_ratio_1h"] = 1.0
            return base

        deltas = np.array(
            [(right - left).total_seconds() for left, right in zip(timestamps[:-1], timestamps[1:])],
            dtype=float,
        )
        span_seconds = max((timestamps[-1] - timestamps[0]).total_seconds(), 1.0)

        base["mean_time_delta"] = float(np.mean(deltas))
        base["std_time_delta"] = float(np.std(deltas))
        base["interval_variance"] = float(np.var(deltas))
        base["min_time_delta"] = float(np.min(deltas))
        base["max_time_delta"] = float(np.max(deltas))
        base["cv_time_delta"] = safe_divide(base["std_time_delta"], base["mean_time_delta"])
        base["periodic_interval_ratio"] = self._periodic_interval_ratio(deltas)
        base["successive_delay_ratio"] = self._successive_delay_ratio(deltas)
        base["min_rolling_cv"] = self._rolling_burst_index(timestamps)
        base["tweets_per_hour"] = safe_divide(len(timestamps), span_seconds / 3600.0)
        base["burst_ratio_1h"] = safe_divide(self._max_tweets_in_window(timestamps, 3600), len(timestamps))
        base["burst_ratio_extreme"] = self._short_delay_ratio(deltas, threshold_seconds=5.0)
        return base

    def _extract_text_features(self, posts, text_authors, text_counts, template_counts):
        texts = [post.get("text", "") for post in posts]
        texts = [text for text in texts if text]

        base = {
            "avg_tweet_length": 0.0,
            "std_tweet_length": 0.0,
            "type_token_ratio": 0.0,
            "unique_words_ratio": 0.0,
            "duplicate_tweet_ratio": 0.0,
            "avg_similarity_between_tweets": 0.0,
            "max_similarity": 0.0,
            "near_duplicate_ratio": 0.0,
            "template_duplicate_ratio": 0.0,
            "template_top_ratio": 0.0,
            "cross_user_repost_ratio": 0.0,
            "top10_word_concentration": 0.0,
            "accent_density": 0.0,
        }

        if not texts:
            return base

        tweet_lengths = [len(text) for text in texts]
        token_lists = [tokenize_words(text) for text in texts]
        all_tokens = [token for tokens in token_lists for token in tokens]
        unique_tokens = set(all_tokens)
        normalized_texts = [normalize_for_similarity(text) for text in texts]
        unique_texts = set(normalized_texts)
        templates = [self._normalize_template(text) for text in texts]
        template_counter = Counter(templates)

        base["avg_tweet_length"] = float(np.mean(tweet_lengths))
        base["std_tweet_length"] = float(np.std(tweet_lengths))
        base["type_token_ratio"] = safe_divide(len(unique_tokens), len(all_tokens))
        base["unique_words_ratio"] = safe_divide(len(unique_tokens), len(texts))
        if len(all_tokens) >= 20:
            top10_count = sum(count for _, count in Counter(all_tokens).most_common(10))
            base["top10_word_concentration"] = safe_divide(top10_count, len(all_tokens))
        base["duplicate_tweet_ratio"] = 1.0 - safe_divide(len(unique_texts), len(normalized_texts))
        base["template_duplicate_ratio"] = 1.0 - safe_divide(len(template_counter), len(templates))
        base["template_top_ratio"] = safe_divide(max(template_counter.values()), len(templates)) if template_counter else 0.0
        base["accent_density"] = self._accent_density(texts)
        shared_reposts = 0
        for normalized_text in normalized_texts:
            if not normalized_text:
                continue
            if len(text_authors[normalized_text]) > 1 or text_counts[normalized_text] > 1:
                shared_reposts += 1
        base["cross_user_repost_ratio"] = safe_divide(shared_reposts, len(normalized_texts))

        if len(normalized_texts) > 1:
            try:
                matrix = self.tfidf.fit_transform(normalized_texts)
                similarity_matrix = (matrix @ matrix.T).toarray()
                mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
                similarity_values = similarity_matrix[mask]
                if similarity_values.size:
                    base["avg_similarity_between_tweets"] = float(np.mean(similarity_values))
                    base["max_similarity"] = float(np.max(similarity_values))
                    base["near_duplicate_ratio"] = safe_divide((similarity_values >= 0.85).sum(), similarity_values.size)
            except ValueError:
                pass

        return base

    def _extract_profile_features(self, user):
        username = user.get("username", "") or ""
        description = user.get("description", "") or ""
        return {
            "username_length": float(len(username)),
            "digit_ratio": safe_divide(sum(1 for char in username if char.isdigit()), len(username)),
            "username_entropy": shannon_entropy(list(username.lower())) if username else 0.0,
            "bio_length": float(len(description)),
            "has_description": 1.0 if description.strip() else 0.0,
        }

    def _extract_activity_features(self, user, posts):
        timestamps = [parse_timestamp(post.get("created_at")) for post in posts]
        timestamps = sorted(timestamp for timestamp in timestamps if timestamp is not None)

        if not timestamps:
            return {
                "tweet_count": float(user.get("tweet_count", len(posts))),
                "z_score": float(user.get("z_score", 0.0)),
                "tweet_time_span": 0.0,
            }

        time_span_seconds = max((timestamps[-1] - timestamps[0]).total_seconds(), 0.0)
        return {
            "tweet_count": float(user.get("tweet_count", len(posts))),
            "observed_post_count": float(len(posts)),
            "tweet_count_gap_ratio": safe_divide(abs(float(user.get("tweet_count", len(posts))) - len(posts)), max(float(user.get("tweet_count", len(posts))), len(posts), 1.0)),
            "z_score": float(user.get("z_score", 0.0)),
            "tweet_time_span": float(time_span_seconds),
        }

    @staticmethod
    def _normalize_template(text: str) -> str:
        template = text.lower()
        template = URL_PATTERN.sub(" URL ", template)
        template = MENTION_PATTERN.sub(" MENTION ", template)
        template = HASHTAG_PATTERN.sub(" HASHTAG ", template)
        template = NUMBER_PATTERN.sub(" NUM ", template)
        template = WORD_PATTERN.sub(" WORD ", template)
        return " ".join(template.split())

    @staticmethod
    def _max_tweets_in_window(timestamps, window_seconds: int) -> int:
        if not timestamps:
            return 0

        left = 0
        best = 1
        for right in range(len(timestamps)):
            while (timestamps[right] - timestamps[left]).total_seconds() > window_seconds:
                left += 1
            best = max(best, right - left + 1)
        return best

    @staticmethod
    def _hour_chi_square(hours) -> float:
        if not hours:
            return 0.0
        expected = len(hours) / 24.0
        counts = Counter(hours)
        return float(
            sum(((counts.get(hour, 0) - expected) ** 2) / expected for hour in range(24) if expected > 0)
        )

    @staticmethod
    def _periodic_interval_ratio(deltas: np.ndarray) -> float:
        if len(deltas) < 4:
            return 0.0

        median_delta = float(np.median(deltas))
        if median_delta <= 0:
            return 0.0

        residuals = np.mod(deltas, median_delta)
        mirrored = np.minimum(residuals, np.abs(median_delta - residuals))
        return float(np.mean(mirrored <= (0.15 * median_delta)))

    @staticmethod
    def _short_delay_ratio(deltas: np.ndarray, threshold_seconds: float = 5.0) -> float:
        if len(deltas) == 0:
            return 0.0
        return safe_divide(np.sum(deltas < threshold_seconds), len(deltas))

    @staticmethod
    def _successive_delay_ratio(deltas: np.ndarray) -> float:
        if len(deltas) < 5:
            return 0.0
        perc_10 = float(np.percentile(deltas, 10))
        median_delta = float(np.median(deltas))
        return safe_divide(perc_10, median_delta)

    @staticmethod
    def _rolling_burst_index(timestamps) -> float:
        if len(timestamps) < 10:
            return 0.0

        ts_seconds = np.array([timestamp.timestamp() for timestamp in timestamps], dtype=float)
        deltas = np.diff(ts_seconds)
        window_size = 5
        if len(deltas) < window_size:
            return 0.0

        rolling_cvs = []
        for start in range(len(deltas) - window_size + 1):
            window = deltas[start : start + window_size]
            window_mean = float(np.mean(window))
            if window_mean <= 0:
                continue
            rolling_cvs.append(float(np.std(window)) / window_mean)
        return min(rolling_cvs) if rolling_cvs else 1.0

    @staticmethod
    def _accent_density(texts) -> float:
        joined_text = " ".join(texts).lower()
        vowel_count = sum(char in VOWELS for char in joined_text)
        accented_count = sum(char in ACCENTED_VOWELS for char in joined_text)
        return safe_divide(accented_count, vowel_count)


def create_feature_dataframe(dataset_filepath, language: str = "en"):
    """Load a dataset JSON file and convert it into user-level features."""
    data = load_json_dataset(dataset_filepath)
    extractor = FeatureExtractor(language=language or data.get("lang", "en"))
    return extractor.extract_all_features(data["users"], data["posts"])


__all__ = ["FeatureExtractor", "create_feature_dataframe", "load_json_dataset"]
