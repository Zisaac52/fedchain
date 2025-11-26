#!/usr/bin/env python3
"""
Utility script to sweep state-vector weights (epsilon coefficients) for DDMLTS.

Usage:
    python scripts/state_weight_sweep.py \
        --command "python fl/main.py" \
        --weights 0.6,0.2,0.2 0.2,0.6,0.2 0.2,0.2,0.6 0.3333,0.3333,0.3333

By default it runs the built-in weight grid on `python fl/main.py` and writes logs
under logs/sweep/. Summary statistics (final accuracy/F1 and average C_gmax) are
printed after the run.
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

DEFAULT_WEIGHTS: Tuple[Tuple[float, float, float], ...] = (
    (0.6, 0.2, 0.2),
    (0.2, 0.6, 0.2),
    (0.2, 0.2, 0.6),
    (1 / 3, 1 / 3, 1 / 3),
)

LOG_DIR = Path("logs") / "sweep"


def parse_weight_arg(arg: str) -> Tuple[float, ...]:
    parts = [p.strip() for p in arg.split(',') if p.strip() != ""]
    if not parts:
        raise ValueError("Empty weight specification.")
    return tuple(float(p) for p in parts)


def run_command(command: str, weights: Sequence[float], log_path: Path):
    env = os.environ.copy()
    env['STATE_VECTOR_WEIGHTS'] = ','.join(f"{w:.6f}" for w in weights)
    cmd_parts = command.split()
    with log_path.open('w') as log_file:
        subprocess.run(
            cmd_parts,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
            env=env
        )


def summarize_log(log_path: Path):
    last_metrics = None
    cmax_values: List[float] = []
    with log_path.open() as log_file:
        for raw_line in log_file:
            line = raw_line.strip()
            if not line:
                continue
            if 'C_gmax(sec)' in line:
                try:
                    cmax_values.append(float(line.rsplit(' ', 1)[-1]))
                except ValueError:
                    continue
            if 'INFO -' in line and ',' in line:
                payload = line.split('INFO -', 1)[1].strip()
                if not payload or not (payload[0].isdigit() or payload[0] == '-'):
                    continue
                tokens = payload.split(',')
                if len(tokens) >= 6 and tokens[0].lstrip('-').isdigit():
                    try:
                        last_metrics = {
                            'epoch': int(tokens[0]),
                            'accuracy': float(tokens[1]),
                            'loss': float(tokens[2]),
                            'precision': float(tokens[3]),
                            'recall': float(tokens[4]),
                            'f1': float(tokens[5]),
                        }
                    except ValueError:
                        continue
    avg_cmax = sum(cmax_values) / len(cmax_values) if cmax_values else None
    return last_metrics, avg_cmax


def format_weights(weights: Sequence[float]) -> str:
    return '(' + ','.join(f"{w:.2f}" for w in weights) + ')'


def main():
    parser = argparse.ArgumentParser(description="Sweep DDMLTS state-vector weights.")
    parser.add_argument(
        '--command',
        default="python fl/main.py",
        help="Command to execute for each run (default: python fl/main.py)."
    )
    parser.add_argument(
        '--weights',
        nargs='*',
        help="List of weight tuples (comma-separated). Example: 0.6,0.2,0.2 0.2,0.6,0.2"
    )
    parser.add_argument(
        '--prefix',
        default="ddmlts_weights",
        help="Filename prefix for generated logs."
    )
    args = parser.parse_args()

    if args.weights:
        weight_sets = tuple(parse_weight_arg(arg) for arg in args.weights)
    else:
        weight_sets = DEFAULT_WEIGHTS

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, weights in enumerate(weight_sets):
        sanitized = '_'.join(str(w).replace('.', 'p') for w in weights)
        log_path = LOG_DIR / f"{args.prefix}_{idx}_{sanitized}.log"
        print(f"[+] Running {args.command} with weights={weights} -> {log_path}")
        run_command(args.command, weights, log_path)
        metrics, avg_cmax = summarize_log(log_path)
        results.append((weights, log_path, metrics, avg_cmax))

    print("\nSummary:")
    header = ["Weights", "Final Acc", "Final F1", "Avg C_gmax", "Log"]
    print("{:<20} {:>10} {:>10} {:>12}  {}".format(*header))
    for weights, log_path, metrics, avg_cmax in results:
        acc = f"{metrics['accuracy']:.2f}" if metrics else "n/a"
        f1 = f"{metrics['f1']:.2f}" if metrics else "n/a"
        cgmax = f"{avg_cmax:.3f}" if avg_cmax is not None else "n/a"
        print("{:<20} {:>10} {:>10} {:>12}  {}".format(
            format_weights(weights), acc, f1, cgmax, log_path
        ))


if __name__ == '__main__':
    main()
