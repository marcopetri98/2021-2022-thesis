import os.path
import pathlib
import unittest

import numpy as np
import pandas as pd

from models.time_series.anomaly.machine_learning import TSAIsolationForest


class TestTSAIsolationForest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sinusoidal_path = "test_data/sinusoidal_wave.csv"
        dir_path = pathlib.Path(__file__).parent.resolve()

        cls.sinusoidal_path = os.path.join(dir_path, sinusoidal_path)
        cls.sinusoidal_df = pd.read_csv(cls.sinusoidal_path)

    def setUp(self) -> None:
        self.sinusoidal_series = self.sinusoidal_df["value"].copy()
        self.sinusoidal_series_len = self.sinusoidal_series.values.shape[0]
        self.sinusoidal_targets = self.sinusoidal_df["target"].copy()

    def test_reproducibility(self):
        configurations = [
            {"window": 10, "n_estimators": 50},
            {"window": 5, "n_estimators": 50},
            {"window": 15, "n_estimators": 50},
            {"window": 20, "n_estimators": 50},
            {"window": 10, "n_estimators": 20},
            {"window": 10, "n_estimators": 100},
            {"window": 10, "n_estimators": 200}
        ]
        reproductions = 10

        training_subseries = self.sinusoidal_series[:4000].values
        testing_subseries = self.sinusoidal_series[4000:].values
        for config in configurations:
            scores = None
            model = TSAIsolationForest(classification="points_score",
                                       scaling="none",
                                       random_state=22,
                                       **config)

            for _ in range(reproductions):
                model.fit(training_subseries.reshape(-1, 1))
                model_scores = model.anomaly_score(testing_subseries.reshape(-1, 1))

                if scores is None:
                    scores = model_scores
                else:
                    comparison = (scores == model_scores) | (np.isnan(scores) & np.isnan(model_scores))
                    self.assertTrue(comparison.all())
