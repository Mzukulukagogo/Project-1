import io
import unittest

from parser import parse_csv


class CsvParsingTests(unittest.TestCase):
    def test_signed_amount_column_without_type(self):
        data = (
            "Date,Description,Amount,Balance\n"
            "2026-08-01,Salary,5000,15000\n"
            "2026-08-02,Coffee,-125.50,14874.50\n"
        )

        transactions = parse_csv(io.BytesIO(data.encode("utf-8")))

        self.assertEqual(transactions[0]["date"], "01-08-2026")
        self.assertEqual(transactions[0]["deposit"], 5000.0)
        self.assertEqual(transactions[0]["withdrawal"], 0.0)
        self.assertEqual(transactions[1]["deposit"], 0.0)
        self.assertEqual(transactions[1]["withdrawal"], 125.50)

    def test_amount_column_with_dr_cr_type(self):
        data = (
            "Txn Date,Narration,Transaction Amount,DR/CR,Closing Balance\n"
            "02/08/2026,UPI rider,220,D,9780\n"
            "03/08/2026,Refund,100,C,9880\n"
        )

        transactions = parse_csv(io.StringIO(data))

        self.assertEqual(transactions[0]["withdrawal"], 220.0)
        self.assertEqual(transactions[0]["deposit"], 0.0)
        self.assertEqual(transactions[1]["withdrawal"], 0.0)
        self.assertEqual(transactions[1]["deposit"], 100.0)


if __name__ == "__main__":
    unittest.main()
