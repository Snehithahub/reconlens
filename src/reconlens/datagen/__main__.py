"""CLI: python -m reconlens.datagen --payments 10000 --out data/demo"""

from __future__ import annotations

import argparse
from collections import Counter

from reconlens.config import load_recon_config
from reconlens.datagen.generator import GenConfig, generate


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic PayFlow/bank/ledger data with gold labels")
    ap.add_argument("--payments", type=int, default=10_000)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/demo")
    args = ap.parse_args()
    data = generate(GenConfig(n_payments=args.payments, days=args.days, seed=args.seed), load_recon_config())
    data.write(args.out)
    print(f"wrote {args.out}: {data.gold['counts']}")
    for kind, c in sorted(Counter(lab["type"] for lab in data.gold["labels"]).items()):
        print(f"  {kind:24s} {c}")


if __name__ == "__main__":
    main()
