import json
from kafka import KafkaConsumer

print("Listening for messages on 'intel-lab-sensors' topic (Press Ctrl+C to stop)...")
consumer = KafkaConsumer(
    "intel-lab-sensors",
    bootstrap_servers=["localhost:29092"],
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)

for msg in consumer:
    print(f"Partition {msg.partition} | Key={msg.key.decode('utf-8') if msg.key else None} | Data: {msg.value}")
