# scripts/test_endpoint.py
import json
import requests
import argparse

def main(endpoint_url, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    example_features = {
        "mean": 0.4,
        "std": 0.2,
        "sobel": 0.12,
        "prewitt": 0.15,
        # add any features your model expects
    }

    response = requests.post(
        endpoint_url,
        headers=headers,
        data=json.dumps(example_features)
    )

    print("Status:", response.status_code)
    print("Response:", response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    main(args.url, args.key)
