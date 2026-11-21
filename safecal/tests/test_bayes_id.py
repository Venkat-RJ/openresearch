import unittest

try:
    import numpy  # noqa: F401
    from bayes_id import BayesianID, policy_bayes, prior_bank
except ImportError:
    BayesianID = None

from safecal import CALIBRATIONS, SafeCalEnv, TEST_BUDGET


@unittest.skipIf(BayesianID is None, "NumPy is an optional classical baseline dependency")
class BayesianTests(unittest.TestCase):
    def test_prior_reproducible_and_independent_of_test_seeds(self):
        bank, probes = prior_bank()
        self.assertEqual(len(bank), 384)
        self.assertEqual(probes.shape, (384, 7, 2))

    def test_posterior_normalized_and_changes_after_measurement(self):
        model = BayesianID(samples=8)
        start = model.weights.copy()
        record = SafeCalEnv(900012).step("pulse_pos_high")
        model.update(record)
        self.assertAlmostEqual(float(sum(model.weights)), 1.0, places=12)
        self.assertGreater(float(max(abs(start-model.weights))), 0.001)

    def test_bayes_policy_valid_and_deterministic(self):
        a, b = SafeCalEnv(1200054, combined=True), SafeCalEnv(1200054, combined=True)
        ca, cb = policy_bayes(a, samples=8), policy_bayes(b, samples=8)
        self.assertEqual(ca, cb)
        self.assertIn(ca, CALIBRATIONS)
        self.assertEqual(a.records, b.records)
        self.assertLessEqual(a.finish(ca)["test_cost"], TEST_BUDGET)

if __name__ == "__main__":
    unittest.main()
