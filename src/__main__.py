"""``python -m src --check`` — validate the data contract without side effects."""

from __future__ import annotations

import argparse
import json
import sys

from .data import data_contract, load_raw, missingness_table, validate_schema


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src", description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate data/raw/framingham.csv")
    args = parser.parse_args(argv)

    if not args.check:
        parser.print_help()
        return 0

    try:
        df = load_raw()
        validate_schema(df)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("data contract OK")
    print(json.dumps(data_contract(df), indent=2))
    print()
    print(missingness_table(df).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
