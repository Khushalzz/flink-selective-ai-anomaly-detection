import argparse
import os
import sys
import time
import orjson
import pandas as pd
from kafka import KafkaProducer


def run_producer(
    file_path: str,
    bootstrap_servers: str,
    topic: str,
    delay: float = 0.0,
    max_records: int | None = None,
    batch_size: int = 2000,
):
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        sys.exit(1)

    print(f"Connecting to Kafka broker at {bootstrap_servers}...")
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers.split(","),
        value_serializer=lambda v: orjson.dumps(v),
        key_serializer=lambda k: str(k).encode("utf-8") if k is not None else None,
        acks=1,
        linger_ms=5,             # Micro-batching in Kafka client for ultra-high throughput
        batch_size=65536,        # 64 KB buffer per partition
        compression_type="lz4",  # High-speed low-CPU compression
    )
    print(f"Connected! High-speed streaming to topic: '{topic}'")
    print(f"Source: {file_path}")
    print(f"Delay: {delay}s | Max records: {max_records or 'Unlimited'}")
    print("-" * 60)

    start_time = time.time()
    sent_count = 0

    try:
        if file_path.endswith(".parquet"):
            # Stream from pre-sorted chronological Parquet file
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(file_path)
            stop_early = False

            for batch in parquet_file.iter_batches(batch_size=batch_size):
                pydict = batch.to_pydict()
                n = len(pydict["date"])
                for i in range(n):
                    mote = int(pydict["moteid"][i])
                    d_str = pydict["date"][i]
                    t_str = pydict["time"][i]
                    record = {
                        "date": d_str,
                        "time": t_str,
                        "timestamp": f"{d_str}T{t_str}",
                        "epoch": int(pydict["epoch"][i]),
                        "moteid": mote,
                        "temperature": float(pydict["temperature"][i]),
                        "humidity": float(pydict["humidity"][i]),
                        "light": float(pydict["light"][i]),
                        "voltage": float(pydict["voltage"][i]),
                    }

                    producer.send(topic, key=mote, value=record)
                    sent_count += 1

                    if delay > 0:
                        time.sleep(delay)

                    if max_records and sent_count >= max_records:
                        stop_early = True
                        break

                elapsed = time.time() - start_time
                rate = sent_count / elapsed if elapsed > 0 else 0
                print(f"[Streaming] Sent {sent_count:,} events | Speed: {rate:,.0f} msgs/sec")

                if stop_early:
                    break
        else:
            # Fallback to plain text reading
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 8:
                        continue
                    mote = int(parts[3])
                    d_str = parts[0]
                    t_str = parts[1]
                    record = {
                        "date": d_str,
                        "time": t_str,
                        "timestamp": f"{d_str}T{t_str}",
                        "epoch": int(parts[2]),
                        "moteid": mote,
                        "temperature": float(parts[4]),
                        "humidity": float(parts[5]),
                        "light": float(parts[6]),
                        "voltage": float(parts[7]),
                    }
                    producer.send(topic, key=mote, value=record)
                    sent_count += 1

                    if delay > 0:
                        time.sleep(delay)
                    if max_records and sent_count >= max_records:
                        break
                    if sent_count % 5000 == 0:
                        elapsed = time.time() - start_time
                        rate = sent_count / elapsed if elapsed > 0 else 0
                        print(f"[Streaming] Sent {sent_count:,} events | Speed: {rate:,.0f} msgs/sec")

    except KeyboardInterrupt:
        print("\nStreaming interrupted by user.")
    finally:
        print("\nFlushing remaining messages to Kafka...")
        producer.flush()
        producer.close()
        total_time = time.time() - start_time
        final_rate = sent_count / total_time if total_time > 0 else 0
        print(f"Finished. Total sent: {sent_count:,} in {total_time:.2f}s ({final_rate:,.0f} msgs/sec)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Speed Intel Lab Kafka Producer")
    parser.add_argument(
        "--file",
        default="data_chronological.parquet" if os.path.exists("data_chronological.parquet") else os.path.join("archive (5)", "data.txt"),
        help="Path to dataset file (.parquet or data.txt)",
    )
    parser.add_argument(
        "--broker",
        default="localhost:29092",
        help="Kafka bootstrap server",
    )
    parser.add_argument(
        "--topic",
        default="intel-lab-sensors",
        help="Kafka topic name",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between messages (default: 0 for max speed)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of messages to send (optional)",
    )

    args = parser.parse_args()
    run_producer(
        file_path=args.file,
        bootstrap_servers=args.broker,
        topic=args.topic,
        delay=args.delay,
        max_records=args.limit,
    )
