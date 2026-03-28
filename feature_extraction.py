"""Compatibility entry point for feature extraction helpers."""

from src.feature_extraction import FeatureExtractor, create_feature_dataframe
from src.utils import load_json_dataset

__all__ = ["FeatureExtractor", "create_feature_dataframe", "load_json_dataset"]
