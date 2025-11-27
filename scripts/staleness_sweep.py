#!/usr/bin/env python3
"""
Sweep different staleness-weighting functions (reciprocal / polynomial / exponential / constant)
and summarize their impact on FL training logs.

Example:
    python scripts/staleness_sweep.py \
        --command "python fl/main.py" \
        --cases reciprocal polynomial:power=0.5 exponential:lambda=0.5 constant
"""

import argparse
import concurrent.futures
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_CASES = (
    "reciprocal",
    "polynomial:power=0.5",
    "exponential:lambda=0.5",
    "constant",
)

LOG_DIR = Path("logs") / "staleness"


def parse_case(case: str) -> Tuple[str, Dict[str, float]]:
    """
    Convert a case string like 'polynomial:power=0.5' into mode + params.
    """
    if ':' not in case:
        return case.lower(), {}
    mode, rest = case.split(':', 1)
    params: Dict[str, float] = {}
    for token in rest.split(','):
        token = token.strip()
        if not token:
            continue
        if '=' not in token:
            continue
        key, value = token.split('=', 1)
        params[key.strip()] = float(value.strip())
    return mode.lower(), params


def run_case(command: str, mode: str, params: Dict[str, float], log_path: Path):
    env = os.environ.copy()
    env['STALENESS_MODE'] = mode
    if 'power' in params:
        env['STALENESS_POWER'] = str(params['power'])
    if 'lambda' in params:
        env['STALENESS_LAMBDA'] = str(params['lambda'])
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


def main():
    parser = argparse.ArgumentParser(description="Staleness weighting sweep utility.")
    parser.add_argument(
        '--command',
        default="python fl/main.py",
        help="Command to execute for each case (default: python fl/main.py)."
    )
    parser.add_argument(
        '--cases',
        nargs='*',
        help="List of cases, e.g., reciprocal polynomial:power=0.5 exponential:lambda=1.0 constant."
    )
    parser.add_argument(
        '--prefix',
        default="staleness",
        help="Filename prefix for generated logs."
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=1,
        help="Maximum number of concurrent runs (default: 1, sequential)."
    )
    args = parser.parse_args()

    case_specs = args.cases if args.cases else DEFAULT_CASES
    parsed_cases = [parse_case(case) for case in case_specs]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    def run_case_and_summary(idx, mode, params):
        suffix = mode
        if params:
            suffix += '_' + '_'.join(f"{k}{v}" for k, v in params.items())
        log_path = LOG_DIR / f"{args.prefix}_{idx}_{suffix}.log"
        print(f"[+] Running {args.command} with mode={mode}, params={params} -> {log_path}")
        run_case(args.command, mode, params, log_path)
        metrics, avg_cmax = summarize_log(log_path)
        return mode, params, log_path, metrics, avg_cmax

    results = []
    max_workers = max(1, args.max_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {}
        for idx, (mode, params) in enumerate(parsed_cases):
            future = executor.submit(run_case_and_summary, idx, mode, params)
            future_to_case[future] = (mode, params)
        for future in concurrent.futures.as_completed(future_to_case):
            results.append(future.result())

    print("\nSummary:")
    header = ["Mode", "Params", "Final Acc", "Final F1", "Avg C_gmax", "Log"]
    print("{:<15} {:<20} {:>10} {:>10} {:>12}  {}".format(*header))
    for mode, params, log_path, metrics, avg_cmax in results:
        param_str = ','.join(f"{k}={v}" for k, v in params.items()) if params else "-"
        acc = f"{metrics['accuracy']:.2f}" if metrics else "n/a"
        f1 = f"{metrics['f1']:.2f}" if metrics else "n/a"
        cgmax = f"{avg_cmax:.3f}" if avg_cmax is not None else "n/a"
        print("{:<15} {:<20} {:>10} {:>10} {:>12}  {}".format(
            mode, param_str, acc, f1, cgmax, log_path
        ))


if __name__ == '__main__':
    main()
