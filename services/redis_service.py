import redis
from config import REDIS_HOST, REDIS_PORT
import json

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def store_metadata(image_id, data):
    r.set(image_id, json.dumps(data))

def get_all_images():
    keys = r.keys("*")
    return [json.loads(r.get(k)) for k in keys]
