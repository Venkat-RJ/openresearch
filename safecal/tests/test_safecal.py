import unittest
from safecal import (SafeCalEnv, Probe, Plant, Calibration, PROBES, CALIBRATIONS,
                     TEST_BUDGET, integrate_pulse, measurement_prediction,
                     DEFAULT_CALIBRATION, policy_calibration, calibration_quality)


class SafeCalTests(unittest.TestCase):
    def test_deterministic_probe_sequence(self):
        a, b = SafeCalEnv(123), SafeCalEnv(123)
        for name in ("reference_check", "pulse_pos_low", "pulse_neg_high"):
            self.assertEqual(a.step(name), b.step(name))
        self.assertEqual(a.finish(DEFAULT_CALIBRATION), b.finish(DEFAULT_CALIBRATION))

    def test_budget_and_duplicate_enforced(self):
        env = SafeCalEnv(44)
        env.step("reference_check")
        with self.assertRaises(ValueError):
            env.step("reference_check")
        for name in ("pulse_pos_low", "pulse_pos_high", "pulse_pos_offset"):
            env.step(name)
        self.assertEqual(env.remaining_budget, 0)
        with self.assertRaises(ValueError):
            env.step("pulse_neg_low")

    def test_shield_rejects_unsafe_controls(self):
        with self.assertRaises(ValueError):
            integrate_pulse(Plant(), 1.4, 0, 0.5)
        with self.assertRaises(ValueError):
            integrate_pulse(Plant(), 0.5, 0.4, 0.5)
        with self.assertRaises(ValueError):
            integrate_pulse(Plant(), 0.5, 0, 2.0)

    def test_hidden_truth_not_in_observation(self):
        env = SafeCalEnv(99)
        v = env.step("pulse_pos_high")
        self.assertEqual(set(v), {"probe", "observed", "cost"})
        self.assertNotIn("motor", str(v))
        self.assertNotIn("fault", str(v))

    def test_terminal_action_guard(self):
        env = SafeCalEnv(43)
        with self.assertRaises(ValueError):
            env.finish(Calibration(5, 1, 1))
        env.finish(DEFAULT_CALIBRATION)
        with self.assertRaises(ValueError):
            env.step("pulse_pos_low")
        with self.assertRaises(ValueError):
            env.finish(DEFAULT_CALIBRATION)

    def test_all_registered_probes_safe_on_nominal(self):
        plant = Plant()
        for probe in PROBES:
            measurement_prediction(plant, probe)

    def test_shifted_physics_differs_and_has_valid_oracle(self):
        from safecal import plant_for_seed, oracle_quality
        a, fam = plant_for_seed(1500000, novel=True)
        b, _ = plant_for_seed(1500000)
        self.assertEqual(fam, "deadzone_shift")
        self.assertGreater(a.torque_deadzone, 0.1)
        self.assertEqual(b.torque_deadzone, 0.0)
        self.assertGreater(oracle_quality(a), 0.0)

    def test_baselines_finish(self):
        for policy in ("default", "random", "fixed", "greedy"):
            env = SafeCalEnv(54)
            cal = policy_calibration(env, policy)
            self.assertIn(cal, CALIBRATIONS)
            r = env.finish(cal)
            self.assertLessEqual(r["test_cost"], TEST_BUDGET)
            self.assertTrue(0 <= r["quality"] <= 1)

    def test_calibration_uses_true_plant_for_scoring(self):
        plant = Plant(sensor_bias=0.08)
        a = calibration_quality(plant, Calibration(1.0, 0.08, 0.08))
        b = calibration_quality(plant, Calibration(1.0, 0.08, -0.08))
        self.assertGreater(a, b)


if __name__ == "__main__":
    unittest.main()
