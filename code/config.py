import os

REDIS_CONFIG = {"uri": os.getenv("REDIS_URL", "redis://localhost")}

RABBITMQ_CONFIG = {
    "uri": os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
}

DOMAIN_CONFIG = {"domain": os.getenv("DOMAIN", "api.ooblex.com")}


