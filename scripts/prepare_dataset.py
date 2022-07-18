import argparse
import logging
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DPATH = os.path.join(PROJECT_ROOT, "data")
CONFIG_DPATH = os.path.join(PROJECT_ROOT, "configs")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("data_preparation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", help="data version", required=True)

    parser.parse_args().version


if __name__ == "__main__":
    main()
