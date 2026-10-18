"""
FastAPI Service Integration Tests
Tests scoring endpoint, case grouping, SAR decisions, and metric reports.
"""

import unittest
from fastapi.testclient import TestClient
from src.api import app, load_champion_model


class TestAMLServiceAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_champion_model()
        cls.client = TestClient(app)

    def test_score_endpoint(self):
        payload = {
            "Timestamp": "2026-03-01 12:00:00",
            "From_Bank": "BANK_001",
            "Account": "ACC_TEST_01",
            "To_Bank": "BANK_002",
            "Account.1": "ACC_TEST_02",
            "Amount_Paid": 9800.0,
            "Payment_Currency": "USD",
            "Payment_Format": "Wire"
        }
        resp = self.client.post("/score", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("risk_score", data)
        self.assertIn("risk_tier", data)
        self.assertIn("narrative", data)
        self.assertIn("top_drivers", data)
        self.assertIn("subgraph", data)

    def test_cases_and_actions(self):
        # 1. Fetch cases
        resp = self.client.get("/cases")
        self.assertEqual(resp.status_code, 200)
        cases_data = resp.json()
        self.assertIn("cases", cases_data)

        # 2. Score a multi-step structuring flow to guarantee a case is created
        for i in range(4):
            tx = {
                "Timestamp": f"2026-03-01 13:{i*10:02d}:00",
                "From_Bank": "BANK_005",
                "Account": "ACC_RING_BOSS",
                "To_Bank": "BANK_008",
                "Account.1": f"ACC_MULE_{i}",
                "Amount_Paid": 9750.0,
                "Payment_Currency": "USD",
                "Payment_Format": "Wire"
            }
            self.client.post("/score", json=tx)

        # Re-fetch cases
        resp2 = self.client.get("/cases")
        cases = resp2.json()["cases"]
        self.assertGreater(len(cases), 0)

        first_case_id = cases[0]["case_id"]

        # Action: Confirm SAR
        action_payload = {"decision": "CONFIRMED_SAR", "notes": "Verified fan-out smurfing ring."}
        act_resp = self.client.post(f"/cases/{first_case_id}/action", json=action_payload)
        self.assertEqual(act_resp.status_code, 200)

        # Check detail
        detail_resp = self.client.get(f"/cases/{first_case_id}")
        self.assertEqual(detail_resp.status_code, 200)
        detail = detail_resp.json()
        self.assertEqual(detail["status"], "CONFIRMED_SAR")

    def test_metrics_endpoint(self):
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("models", data)


if __name__ == "__main__":
    unittest.main()
