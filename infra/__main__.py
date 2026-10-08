"""Locate cloud starter modules without provisioning or requiring cloud credentials."""
import argparse
from importlib import import_module


def main():
    parser = argparse.ArgumentParser(description="Cloud deployment starters; no deployment is implemented.")
    parser.add_argument("--cloud", required=True, choices=["azure", "aws", "gcp"])
    args = parser.parse_args()
    module = import_module(f"infra.{args.cloud}")
    print(f"Starter module: {module.__file__}")
    print("Provisioning is intentionally unimplemented. See infra/README.md for next steps.")

if __name__ == "__main__":
    main()
