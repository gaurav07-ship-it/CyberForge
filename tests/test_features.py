"""
Feature Engineering Unit Tests
Tests graph metrics: in-degree, out-degree, cycle detection, fan-in/fan-out ratios.
"""

import unittest
from src.features_graph import StreamingGraphState


class TestGraphFeatures(unittest.TestCase):

    def setUp(self):
        self.state = StreamingGraphState()
        self.base_time = 1700000000.0

    def test_cycle_detection(self):
        """
        Tests that a circular sequence A -> B -> C -> A correctly flags
        a 2-hop or 3-hop cycle.
        """
        # Step 1: A -> B
        t1 = self.base_time
        f1 = self.state.compute_features("ACC_A", "ACC_B", t1, 5000.0)
        self.assertEqual(f1["is_reciprocal_return"], 0.0)
        self.assertEqual(f1["cycle_2hop_detected"], 0.0)
        self.state.update_state("ACC_A", "ACC_B", t1, 5000.0)

        # Step 2: B -> C
        t2 = t1 + 3600
        f2 = self.state.compute_features("ACC_B", "ACC_C", t2, 4900.0)
        self.assertEqual(f2["is_reciprocal_return"], 0.0)
        self.state.update_state("ACC_B", "ACC_C", t2, 4900.0)

        # Step 3: Direct return B -> A (2-cycle)
        t3_recip = t1 + 7200
        f3_recip = self.state.compute_features("ACC_B", "ACC_A", t3_recip, 4800.0)
        self.assertEqual(f3_recip["is_reciprocal_return"], 1.0)

        # Step 4: C -> A (completing 3-cycle A -> B -> C -> A)
        t3 = t2 + 3600
        f3 = self.state.compute_features("ACC_C", "ACC_A", t3, 4800.0)
        self.state.update_state("ACC_C", "ACC_A", t3, 4800.0)
        # B sent to C, and A was sent from C
        self.assertEqual(f3["cycle_2hop_detected"], 1.0)

    def test_fan_out_structuring(self):
        """
        Tests that an account dispersing to 5 distinct recipients within hours
        triggers a high fan-out score.
        """
        src = "ACC_SMURF_BOSS"
        for i in range(5):
            t = self.base_time + i * 600
            mule = f"ACC_MULE_{i}"
            self.state.update_state(src, mule, t, 9500.0)

        # Check features for 6th transfer
        t_next = self.base_time + 3600
        feats = self.state.compute_features(src, "ACC_MULE_NEW", t_next, 9500.0)
        self.assertGreaterEqual(feats["src_unique_recipients_24h"], 5.0)
        self.assertGreater(feats["src_fan_out_score"], 2.0)


if __name__ == "__main__":
    unittest.main()
