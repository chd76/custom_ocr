"""Run the stages switched on in settings.yaml (stages: ...), each in its own Python environment.

    python run.py                                   the stages that are true in settings.yaml
    python run.py --stages paddle,csv               only these stages (ignores the true/false in settings.yaml)
    python run.py --set run_name=test --set select.limit=5 --stages paddle,csv,check
    python run.py --redo                            redo pages that already have results

Any Python with PyYAML can start run.py, e.g. .venv-paddle/Scripts/python.exe run.py; every stage is started with the environment from settings.yaml
(python.filter: .venv-main; python.default: .venv-paddle). Each stage can also be run on its own:
    python s1_pages.py / s2_filter.py / s3_paddle.py / s4_csv.py / s5_check.py   (same --set / --redo options)
"""

import subprocess
import sys

from common import HERE, load_settings, python_for, stage_args

STAGES = {"pages": "s1_pages.py", "filter": "s2_filter.py", "paddle": "s3_paddle.py", "csv": "s4_csv.py",
          "check": "s5_check.py"}


def main():
    ap = stage_args(__doc__)
    ap.add_argument("--stages", help=f"comma-separated, from: {', '.join(STAGES)}")
    args = ap.parse_args()
    s = load_settings(args)
    if args.stages:
        chosen = [x.strip() for x in args.stages.split(",") if x.strip()]
        bad = [x for x in chosen if x not in STAGES]
        if bad:
            raise SystemExit(f"unknown stage(s): {bad}; choose from {list(STAGES)}")
    else:
        chosen = [k for k in STAGES if s["stages"].get(k)]
    chosen = [k for k in STAGES if k in chosen]  # always in pipeline order

    passthrough = ["--settings", args.settings] + [x for item in args.set for x in ("--set", item)]
    if args.redo:
        passthrough.append("--redo")
    for name in chosen:
        python = python_for(s, name)
        if not python.exists():
            raise SystemExit(f"Python for stage '{name}' not found: {python} (settings.yaml -> python)")
        print(f"\n===== {name} ({STAGES[name]}) =====", flush=True)
        code = subprocess.call([str(python), str(HERE / STAGES[name])] + passthrough, cwd=HERE)
        if code:
            raise SystemExit(f"stage '{name}' failed (exit code {code}); later stages not run")
    print("\nall done")


if __name__ == "__main__":
    sys.exit(main())
