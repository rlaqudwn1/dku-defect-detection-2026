"""Prepare an offline text packet; never open a browser or upload evidence."""
import argparse
import json
from pathlib import Path

from defectlab.packets import prepare_packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('specification')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    result = prepare_packet(Path(__file__).resolve().parents[1], args.specification, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
