from api.app import app
import unittest
from fastapi.testclient import TestClient

class FastAPITest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "API is healthy"})

    def test_predict_endpoint(self):
        payload = {
            'Time_spent_Alone': 4.0,
            'Stage_fear': 'No',
            'Social_event_attendance': 4.0,
            'Going_outside': 6.0,
            'Drained_after_socializing': 'No',
            'Friends_circle_size': 13.0,
            'Post_frequency': 5.0
        }

        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())

if __name__ == "__main__":
    unittest.main()