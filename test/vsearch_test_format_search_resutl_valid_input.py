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

    def test_format_search_results_normal(self):
        documents = [
            {
                "filter": "some_filter",
                "retrieved_info": [
                    {"id": "1", "document": "Document 1", "filename": "File1.pdf"},
                    {"id": "2", "document": "Document 2", "filename": "File2.pdf"}
                ]
            }
        ]
        expected_output = "\n\n```\n" + "id: 1\ndocument: Document 1\nfilename: File1.pdf\n\nid: 2\ndocument: Document 2\nfilename: File2.pdf" + "\n```"
        self.assertEqual(self.vsearch.format_search_results(documents), expected_output)

    def test_format_search_results_empty_list(self):
        documents = []
        entities = None
        expected_message = "No documents found for this question's related search: None"
        self.assertEqual(self.vsearch.format_search_results(documents, entities), expected_message)

    def test_format_search_results_handling_exceptions(self):
        # Example of a structure of documents with incorrect data type to simulate the error
        documents = "some incorrect string data"
        with self.assertRaises(AttributeError):
            self.vsearch.format_search_results(documents, {})


if __name__ == '__main__':
    unittest.main()