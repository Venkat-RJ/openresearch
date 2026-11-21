import json
import unittest
from safecal_v1.protocol import SafeCalDialogue, SYSTEM_PROMPT


class ProtocolTests(unittest.TestCase):
    def test_interactive_sequence_deterministic_and_safe(self):
        a, b = SafeCalDialogue(123), SafeCalDialogue(123)
        for d in (a, b):
            initial = json.loads(d.observation())
            self.assertNotIn("motor", d.observation())
            self.assertEqual(initial["remaining_budget"], 4.5)
            msg, done = d.act('{"probe":"pulse_pos_high"}')
            self.assertFalse(done)
            parsed = json.loads(msg)
            self.assertIn("probe_result", parsed)
            self.assertNotIn("motor", msg)
            msg, done = d.act('{"calibration":{"gain":1.0,"friction_ff":0.08,"sensor_offset":0.0}}')
            self.assertTrue(done)
        self.assertEqual(a.reward(), b.reward())
        self.assertGreater(a.reward(), 0.0)

    def test_rejects_infinite_and_invalid_calibration(self):
        for bad in ('{"calibration":{"gain":NaN,"friction_ff":0.08,"sensor_offset":0}}',
                    '{"calibration":{"gain":99,"friction_ff":0.08,"sensor_offset":0}}'):
            d = SafeCalDialogue(15)
            msg, ended = d.act(bad)
            self.assertFalse(ended)
            self.assertEqual(json.loads(msg)["error"], "invalid_action")

    def test_invalid_probe_consumes_turn_and_bounded(self):
        d = SafeCalDialogue(25)
        _, ended = d.act('{"probe":"unsafe_torque"}')
        self.assertFalse(ended)
        _, ended = d.act('{"probe":"unsafe_torque"}')
        self.assertTrue(ended)
        self.assertEqual(d.reward(), 0.0)

if __name__ == "__main__":
    unittest.main()
