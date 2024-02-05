import unittest
import sys
import os
from dotenv import load_dotenv
load_dotenv()
root_dir = os.getenv("ROOT")
sys.path.insert(0, root_dir)
from plugins.AISearch.vsearch import VSearch

class TestVSearch(unittest.TestCase):
    def setUp(self):
        self.vsearch = VSearch()

    def test_normal_case(self):
        json_object = {
            "ticker": ["AAPL", "MSFT"],
            "location": ["USA"],
            "dates": ["2021-01-01", "2022-01-01"]
        }
        expected = [
            "referenced_entity eq 'AAPL' and referenced_location eq 'USA' and referenced_year eq '2021'",
            "referenced_entity eq 'MSFT' and referenced_location eq 'USA' and referenced_year eq '2021'"
        ]
        self.assertEqual(self.vsearch.build_query_filter(json_object), expected)

    def test_no_tickers(self):
        json_object = {"ticker": None, "location": ["USA"], "dates": ["2021-01-01"]}
        expected = ["referenced_location eq 'USA' and referenced_year eq '2021'"]
        self.assertEqual(self.vsearch.build_query_filter(json_object), expected)

    def test_no_location(self):
        json_object = {"ticker": ["AAPL"], "dates": ["2021-01-01"]}
        expected = ["referenced_entity eq 'AAPL' and referenced_year eq '2021'"]
        self.assertEqual(self.vsearch.build_query_filter(json_object), expected)

    def test_no_dates(self):
        json_object = {"ticker": ["AAPL"], "location": ["USA"]}
        expected = ["referenced_entity eq 'AAPL' and referenced_location eq 'USA'"]
        self.assertEqual(self.vsearch.build_query_filter(json_object), expected)

    def test_empty_json_object(self):
        json_object = {}
        self.assertIsNone(self.vsearch.build_query_filter(json_object))

    def test_invalid_json_structure(self):
        json_object = {"unexpected_key": ["value"]}
        self.assertIsNone(self.vsearch.build_query_filter(json_object))

    def test_build_query_filter_returns_none(self):
        # Case 1: json_object is empty
        json_object = {}
        self.assertIsNone(self.vsearch.build_query_filter(json_object))

        # Case 2: json_object contains keys but they have no meaningful data
        json_object = {"ticker": [], "location": None, "dates": None}
        self.assertIsNone(self.vsearch.build_query_filter(json_object))

        # Case 3: json_object contains keys but they are empty
        json_object = {"ticker": [], "location": [], "dates": []}
        self.assertIsNone(self.vsearch.build_query_filter(json_object))

        # Case 4: json_object contains None values for the keys
        json_object = {"ticker": None, "location": None, "dates": None}
        self.assertIsNone(self.vsearch.build_query_filter(json_object))    

    def test_exception_handling(self):
        json_object = "Not a JSON object"
        with self.assertRaises(TypeError):
            self.vsearch.build_query_filter(json_object)


if __name__ == '__main__':
    unittest.main()