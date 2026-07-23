"""
Kafka producer that reads claims from a CSV file and streams them to a Kafka topic.
Usage:
    python kafka/producer.py --csv data/raw/claims.csv --delay 0.5
"""
import argparse
import json
import time

import pandas as pd
from kafka import KafkaProducer

TOPIC = "claims-events"


def make_producer(bootstrap_servers: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )


def replay(csv_path: str, bootstrap_servers: str, delay: float) -> None:
    df = pd.read_csv(csv_path)
    producer = make_producer(bootstrap_servers)

    print(f"Replaying {len(df)} claims from {csv_path} to topic '{TOPIC}'")
    for i, row in df.iterrows():
        event = row.to_dict()
        producer.send(TOPIC, value=event)
        if i % 50 == 0:
            print(f"  sent {i} events...")
        time.sleep(delay)

    producer.flush()
    print("Done. All claims streamed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/claims.csv")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--delay", type=float, default=0.3, help="seconds between events")
    args = parser.parse_args()

    replay(args.csv, args.bootstrap_servers, args.delay)
