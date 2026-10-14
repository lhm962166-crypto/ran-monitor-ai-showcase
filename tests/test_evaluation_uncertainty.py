import unittest

import numpy as np

from ran_monitor_ai.evaluation import cluster_bootstrap_binary_metrics


class ClusterBootstrapTests(unittest.TestCase):
    def test_resamples_overlap_groups_reproducibly_and_returns_intervals(self):
        labels = np.array([0, 1] * 12)
        scores = np.array([0.1, 0.8, 0.2, 0.7, 0.1, 0.9] * 4)
        groups = np.repeat(np.arange(12), 2)
        first = cluster_bootstrap_binary_metrics(labels, scores, 0.5, groups,
                                                 replicates=100, seed=4)
        second = cluster_bootstrap_binary_metrics(labels, scores, 0.5, groups,
                                                  replicates=100, seed=4)
        self.assertEqual(first, second)
        self.assertEqual(first["independent_groups"], 12)
        self.assertEqual(first["metrics"]["f1"]["estimate"], 1.0)
        self.assertEqual(first["metrics"]["average_precision"]["estimate"], 1.0)
        for metric in first["metrics"].values():
            self.assertEqual(len(metric["ci95"]), 2)
            self.assertEqual(metric["valid_replicates"], 100)

    def test_rejects_bad_or_insufficient_grouped_inputs(self):
        with self.assertRaisesRegex(ValueError, "至少需要 2 个不重叠时间组"):
            cluster_bootstrap_binary_metrics([0, 1], [0.1, 0.9], 0.5, [1, 1])
        with self.assertRaisesRegex(ValueError, "重复次数"):
            cluster_bootstrap_binary_metrics([0, 1], [0.1, 0.9], 0.5, [1, 2], replicates=10)
        with self.assertRaisesRegex(ValueError, "同时包含正常和异常"):
            cluster_bootstrap_binary_metrics([0, 0], [0.1, 0.2], 0.5, [1, 2])

    def test_average_precision_skips_bootstrap_draws_without_positive_groups(self):
        report = cluster_bootstrap_binary_metrics(
            [0, 0, 0, 0, 1, 1], [.1, .2, .1, .2, .8, .9], .5,
            [0, 0, 1, 1, 2, 2], replicates=100, seed=7,
        )
        ap = report["metrics"]["average_precision"]
        self.assertLess(ap["valid_replicates"], 100)
        self.assertGreater(ap["valid_replicates"], 0)


if __name__ == "__main__":
    unittest.main()
