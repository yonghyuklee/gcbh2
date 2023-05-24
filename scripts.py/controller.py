import argparse, json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--options", type=str, default="./options.json")
    args = parser.parse_args()
    with open(args.options) as f:
        options = json.load(f)
