from locust import HttpUser, between, task, tag, events
from loguru import logger


class ModelUser(HttpUser):
    # Wait between 1 and 3 seconds between requrests
    # wait_time = between(1, 3)

    # def on_start(self):
    #     logger.info("Load your model here")
    
    @events.test_start.add_listener
    def on_test_start(environment, **kwargs):
        print("A new test is starting")
        
    @events.test_stop.add_listener
    def on_test_stop(environment, **kwargs):
        print("A new test is ending")

    @tag("task1")   
    @task(1)
    def predict_skinlesion(self):
        logger.info("Sending POST requests!")
        image = open('./image.jpeg', 'rb')
        files = [("image", image)]
        self.client.post(
            "/worker_generate",
            params={"id_thread": 1},
            files=files,
        )
    @tag("task2")   
    @task(1)
    def predict_facereg(self):
        logger.info("Sending POST requests!")
        image = open('./hieu.jpg', 'rb')
        self.client.post(
            url="/api/searchUserv2",
            files=[("image", image)],
        )
        
# locust -f test_performance.py --tags task2