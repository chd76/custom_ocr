"""Create the Python environments of the project (needs uv: https://docs.astral.sh/uv/). Works on Windows and Linux.

    .venv-main     page filter tool and page filter model             (ultralytics + PyTorch)
    .venv-paddle   PaddleOCR and all other stages                          (paddlepaddle-gpu + paddleocr)

    python setup/setup_envs.py                       all environments
    python setup/setup_envs.py --only paddle         only one
    python setup/setup_envs.py --locked              exact versions from lock-*.txt (as on the laptop where it was made)
    python setup/setup_envs.py --torch-cuda cu126 --paddle-cuda cu126    other CUDA builds (check the GPU driver first)
    python setup/setup_envs.py --freeze              write lock-*.txt from the current environments

They are separate because PyTorch and PaddlePaddle bring their own CUDA libraries.
RTX 50xx (Blackwell) needs paddle cu129 and torch cu128 or newer, and a driver for CUDA 12.9+.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP = ROOT / "setup"
PYTHON_VERSION = "3.12"


def venv_python(venv):
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


ENVS = {
    "main": {"venv": ROOT / ".venv-main", "index": "https://download.pytorch.org/whl/{cuda}", "cuda_arg": "torch_cuda",
             "check": "import torch, ultralytics; print('torch', torch.__version__, '| ultralytics', ultralytics.__version__,"
                      " '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"},
    "paddle": {"venv": ROOT / ".venv-paddle", "index": "https://www.paddlepaddle.org.cn/packages/stable/{cuda}/",
               "cuda_arg": "paddle_cuda",
               "check": "import paddle, paddleocr; print('paddle', paddle.__version__, '| paddleocr', paddleocr.__version__,"
                        " '| GPU:', paddle.device.cuda.device_count() > 0 and paddle.device.cuda.get_device_name(0)"
                        " or 'NOT FOUND')"},
}


def run(cmd):
    print(">", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(ENVS))
    ap.add_argument("--locked", action="store_true", help="install the exact versions from lock-<env>.txt")
    ap.add_argument("--torch-cuda", default="cu128", help="PyTorch build: cu128, cu126, cpu ...")
    ap.add_argument("--paddle-cuda", default="cu129", help="PaddlePaddle build: cu129, cu126, cu118 ...")
    ap.add_argument("--freeze", action="store_true", help="only write lock-<env>.txt from the existing environments")
    args = ap.parse_args()
    for name, e in ENVS.items():
        if args.only and name != args.only:
            continue
        py = venv_python(e["venv"])
        if args.freeze:
            out = subprocess.run(["uv", "pip", "freeze", "--python", str(py)], check=True, capture_output=True, text=True)
            (SETUP / f"lock-{name}.txt").write_text(out.stdout, encoding="utf-8")
            print(f"lock-{name}.txt written ({len(out.stdout.splitlines())} packages)")
            continue
        print(f"\n===== {name}: {e['venv']}")
        if not py.exists():
            run(["uv", "venv", e["venv"], "--python", PYTHON_VERSION])
        reqs = SETUP / (f"lock-{name}.txt" if args.locked else f"requirements-{name}.txt")
        index = e["index"].format(cuda=getattr(args, e["cuda_arg"]))
        run(["uv", "pip", "install", "--python", py, "-r", reqs, "--index-url", index,
             "--extra-index-url", "https://pypi.org/simple", "--index-strategy", "unsafe-best-match"])
        run([py, "-c", e["check"]])
    print("\ndone")


if __name__ == "__main__":
    main()
