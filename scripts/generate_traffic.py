"""Send random patient requests to the API so the monitoring dashboard has data.

Usage: python scripts/generate_traffic.py --n 200
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request


def random_patient():
    return {
        "age": random.randint(30, 77), "sex": random.randint(0, 1), "cp": random.randint(1, 4),
        "trestbps": random.randint(95, 190), "chol": random.randint(130, 400),
        "fbs": random.randint(0, 1), "restecg": random.randint(0, 2),
        "thalach": random.randint(90, 200), "exang": random.randint(0, 1),
        "oldpeak": round(random.uniform(0, 4.5), 1), "slope": random.randint(1, 3),
        "ca": random.randint(0, 3), "thal": random.choice([3, 6, 7]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200, help="number of requests")
    args = parser.parse_args()
    counts = {}
    for i in range(args.n):
        patient = random_patient()
        if random.random() < 0.05:
            patient["age"] = 999            # about 5% invalid on purpose, to show errors
        req = urllib.request.Request(f"{args.url}/predict", data=json.dumps(patient).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            status = urllib.request.urlopen(req, timeout=5).status
        except urllib.error.HTTPError as err:
            status = err.code
        counts[status] = counts.get(status, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{args.n} requests sent, status counts: {counts}")
        time.sleep(0.2)
    print("Done:", counts)


if __name__ == "__main__":
    main()