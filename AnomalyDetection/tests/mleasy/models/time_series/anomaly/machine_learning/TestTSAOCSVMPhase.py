import os.path
import pathlib
import unittest

import numpy as np
import pandas as pd

from models.time_series.anomaly.machine_learning import TSAOCSVMPhase


class TestTSAOCSVMPhase(unittest.TestCase):
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
            {"windows": [5, 10, 15, 20], "gamma": 0.0001, "nu": 0.5, "tol": 1e-10},
            {"windows": [5, 10, 15, 20], "gamma": 0.1, "nu": 0.5, "tol": 1e-10},
            {"windows": [5, 10, 15, 20], "gamma": 0.0001, "nu": 0.2, "tol": 1e-10},
            {"windows": [5, 10, 15, 20], "gamma": 0.0001, "nu": 0.5, "tol": 1e-2},
            {"windows": [5, 10, 15, 20], "gamma": 0.1, "nu": 0.5, "tol": 1e-2},
            {"windows": [5, 10, 15, 20], "gamma": 0.0001, "nu": 0.2, "tol": 1e-2}
        ]
        reproductions = 10

        training_subseries = self.sinusoidal_series[:4000].values
        testing_subseries = self.sinusoidal_series[4000:].values
        for config in configurations:
            scores = None
            model = TSAOCSVMPhase(classification="points_score",
                                  scaling="none",
                                  **config)

            for _ in range(reproductions):
                model.fit(training_subseries.reshape(-1, 1))
                model_scores = model.anomaly_score(testing_subseries.reshape(-1, 1))

                if scores is None:
                    scores = model_scores
                else:
                    comparison = (scores == model_scores) | (np.isnan(scores) & np.isnan(model_scores))
                    self.assertTrue(comparison.all())