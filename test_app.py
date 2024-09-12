import unittest
import json
from app import app  # Import the Flask app from your app module

class FlaskTestCase(unittest.TestCase):
    def setUp(self):
        # Set up the test client
        self.app = app.test_client()
        self.app.testing = True

    def test_start_route(self):
        # Define the payload for the POST request
        payload = {"deviceSn": "RX2505ACA10J0A160016"}

        # Send a POST request to the /start route
        response = self.app.post('/start', data=json.dumps(payload), content_type='application/json')

        # Check the status code of the response
        self.assertEqual(response.status_code, 200)

        # Check the response data
        response_data = json.loads(response.data)
        self.assertEqual(response_data['status'], 'success')
        self.assertEqual(response_data['message'], 'Scheduler started')

if __name__ == '__main__':
    unittest.main()