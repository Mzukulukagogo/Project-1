import json
import tempfile
import unittest
from pathlib import Path

import categoriser


class CategoriserTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_paths = (
            categoriser.MODEL_DIR,
            categoriser.MODEL_PATH,
            categoriser.FEEDBACK_PATH,
            categoriser.ENTITY_PATH,
        )
        model_dir = self.temp_dir.name
        categoriser.MODEL_DIR = model_dir
        categoriser.MODEL_PATH = str(Path(model_dir) / "transaction_classifier.pkl")
        categoriser.FEEDBACK_PATH = str(Path(model_dir) / "feedback.json")
        categoriser.ENTITY_PATH = str(Path(model_dir) / "entity_profiles.json")

    def tearDown(self):
        (
            categoriser.MODEL_DIR,
            categoriser.MODEL_PATH,
            categoriser.FEEDBACK_PATH,
            categoriser.ENTITY_PATH,
        ) = self.original_paths
        self.temp_dir.cleanup()

    def test_entity_frequency_is_current_statement_only(self):
        transactions = [
            {
                "date": "01-08-2026",
                "merchant": "RAJU",
                "deposit": 0,
                "withdrawal": 100,
                "balance": 900,
                "is_upi": True,
            },
            {
                "date": "02-08-2026",
                "merchant": "RAJU",
                "deposit": 0,
                "withdrawal": 150,
                "balance": 750,
                "is_upi": True,
            },
            {
                "date": "03-08-2026",
                "merchant": "RAJU",
                "deposit": 0,
                "withdrawal": 200,
                "balance": 550,
                "is_upi": True,
            },
        ]

        first = categoriser.summarise(transactions)
        second = categoriser.summarise(transactions[:1])

        self.assertFalse(Path(categoriser.ENTITY_PATH).exists())
        self.assertTrue(
            all(t["category"] == "Peer Transfers" for t in first["transactions"])
        )
        self.assertTrue(all(t["entity_frequency"] == 3 for t in first["transactions"]))
        self.assertEqual(second["transactions"][0]["entity_frequency"], 1)
        self.assertEqual(second["transactions"][0]["category"], "Travel & Transport")

    def test_known_merchants_are_protected_from_frequency_heuristic(self):
        transactions = [
            {
                "date": f"0{day}-08-2026",
                "merchant": "SWIGGY",
                "deposit": 0,
                "withdrawal": 300,
                "balance": 1000 - day,
                "is_upi": True,
            }
            for day in range(1, 5)
        ]

        result = categoriser.summarise(transactions)

        self.assertTrue(
            all(t["category"] == "Food & Dining" for t in result["transactions"])
        )
        self.assertTrue(
            all(t["classification_source"] == "rule" for t in result["transactions"])
        )

    def test_feedback_keeps_upi_entity_learning(self):
        self.assertTrue(
            categoriser.record_feedback(
                "UPI/DR/12345/APPLE/ORDER",
                "Entertainment",
            )
        )

        feedback = json.loads(Path(categoriser.FEEDBACK_PATH).read_text())
        profiles = json.loads(Path(categoriser.ENTITY_PATH).read_text())

        self.assertEqual(feedback[0]["text"], "UPI/DR/12345/APPLE/ORDER")
        self.assertEqual(profiles["APPLE"]["known_categories"]["Entertainment"], 1)
        self.assertEqual(
            categoriser.categorise("UPI/DR/99999/APPLE/REPEAT", is_upi=True),
            "Entertainment",
        )


if __name__ == "__main__":
    unittest.main()
