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

    @patch('semantic_kernel.Kernel')
    def test_query_function(self, mock_kernel):
        # Mock the kernel methods used in the query function
        mock_kernel.return_value.run = AsyncMock(return_value={"input": "some response"})

        # Define test cases
        test_cases = [
            "Give me a summary of MD&A of Pfizer and Microsoft",
            "Waht elements are mentioned in the Microsoft and BestBuy Income Statement?",
            "What is the Debt status of BestBuy and Pfizer?",
            "What is the total Revenue of Microsoft for the years 2023,2022,2021?",
            "Is Best Buy trying to enrich the lives of consumers through technology?",
            "Give a summary overview of Best Buy challenges",
            "Compare the market capitalization of Microsoft and Apple in the last quarter",
            "Analyze the impact of COVID-19 on the revenue streams of Pfizer",
            "What are the projections for Tesla’s electric vehicle sales in Europe?",
            "How does Amazon's cloud business profitability compare to its e-commerce?",
            "Evaluate the effectiveness of Google's current digital advertising strategy",
            "What are the ethical implications of Facebook’s data privacy policies?",
            "Assess the long-term sustainability of Netflix's content creation model",
            "Discuss the potential risks and rewards of Tesla’s investment in Bitcoin",
            "What are the emerging trends in renewable energy and how might they affect ExxonMobil?"
        ]
        
        async def run_test(ask):
            # Call the query function with the test case
            result = await query(ask)

            # Assertions
            self.assertIsNotNone(result)
            self.assertIsInstance(result, list)

        # Run the asynchronous tests for each case
        for ask in test_cases:
            with self.subTest(ask=ask):
                asyncio.run(run_test(ask))

if __name__ == '__main__':
    unittest.main()

