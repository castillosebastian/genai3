import os
import sys
import unittest
from unittest.mock import patch, AsyncMock
import asyncio
import semantic_kernel as sk
from dotenv import load_dotenv
load_dotenv()
root_dir = os.getenv("ROOT")
sys.path.insert(0, root_dir)
from plugins.AISearch.query import query


class TestQueryFunction(unittest.TestCase):

    @patch('semantic_kernel.Kernel')  # Replace with the actual import path
    def test_query_successful(self, mock_kernel):
        # Mocking kernel and its methods
        mock_kernel.return_value.run = AsyncMock(return_value={"input": "some response"})

        # Test cases where the function is expected to succeed
        test_cases = [
            "Give me a summary of MD&A of Pfizer and Microsoft?",
            # ... Add other successful test cases here
        ]

        for ask in test_cases:
            with self.subTest(ask=ask):
                result = query(ask)
                self.assertNotEqual(result, [])
                self.assertIsInstance(result, list)

    @patch('semantic_kernel.Kernel')  # Replace with the actual import path
    def test_query_failed(self, mock_kernel):
        # Mocking kernel and its methods to simulate failure
        mock_kernel.return_value.run = AsyncMock(side_effect=Exception("Some error"))

        # Test cases where the function is expected to fail
        test_cases = [
            "Waht elements are mentioned in the MD&A of Pfizer?",
            "What elements are mentioned in the MD&A of US BestBuy?",
            # ... Add other failing test cases here
        ]

        for ask in test_cases:
            with self.subTest(ask=ask):
                result = query(ask)
                self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()
