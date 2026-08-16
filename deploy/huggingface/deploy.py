"""
Automated Hugging Face Spaces deployment for the Brain Tumor Detection System.

What it does:
  1. Verifies you are logged in to Hugging Face.
  2. Creates (or reuses) a Docker Space on your account.
  3. Stages app code, runtime requirements, and the trained model.
  4. Initializes git-lfs and pushes everything to the Space.

Usage:
    python deploy/huggingface/deploy.py --space-name my-brain-tumor-app
    # or with a token directly:
    python deploy/huggingface/deploy.py --space-name my-brain-tumor-app --token hf_xxxx

Environment:
    HF_TOKEN=<token>  (alternative to --token or an interactive login)
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

HF_SRC = REPO_ROOT / "deploy" / "huggingface"
APP_SRC = REPO_ROOT / "app"
MODEL_SRC = REPO_ROOT / "models" / "best_model.h5"


def run(cmd, cwd=None, check=True):
    print(f"$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")
    return result


def get_token(args):
    token = args.token or os.environ.get("HF_TOKEN", "")
    if token:
        return token
    # Fall back to an existing local login
    return None


def main():
    parser = argparse.ArgumentParser(description="Deploy app to a Hugging Face Docker Space")
    parser.add_argument("--space-name", required=True, help="Name of the Space (e.g. brain-tumor-detection)")
    parser.add_argument("--token", help="Hugging Face write token (or set HF_TOKEN env var)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    token = get_token(args)
    api = HfApi(token=token) if token else HfApi()

    if token:
        try:
            who = api.whoami(token=token)
        except Exception as e:
            print(f"Token rejected: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            who = api.whoami()
        except Exception as e:
            print(f"Not logged in: {e}", file=sys.stderr)
            print("Log in with: huggingface-cli login   (or pass --token / set HF_TOKEN)", file=sys.stderr)
            sys.exit(1)

    username = who["name"]
    print(f"Logged in as: {username}")

    if not MODEL_SRC.exists():
        print(f"Model not found: {MODEL_SRC}", file=sys.stderr)
        print("Train the model first so models/best_model.h5 exists.", file=sys.stderr)
        sys.exit(1)
    print(f"Model: {MODEL_SRC} ({MODEL_SRC.stat().st_size / 1e6:.1f} MB)")

    if not args.yes:
        answer = input(f"Deploy as Space '{username}/{args.space_name}'? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    # 1. Create the Space if it does not exist
    print("\n[1/4] Ensuring Space exists...")
    try:
        api.create_repo(
            repo_id=f"{username}/{args.space_name}",
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            token=token,
        )
        print(f"Space ready: https://huggingface.co/spaces/{username}/{args.space_name}")
    except Exception as e:
        print(f"Error creating Space: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Clone the Space repo into a temp staging dir
    print("\n[2/4] Cloning Space repo...")
    staging = Path(tempfile.mkdtemp(prefix="hf_space_"))
    space_url = f"https://huggingface.co/spaces/{username}/{args.space_name}"
    run(["git", "clone", space_url, str(staging)], check=True)

    # 3. Copy files
    print("\n[3/4] Staging files...")
    shutil.copytree(APP_SRC, staging / "app", dirs_exist_ok=True)
    shutil.copy2(HF_SRC / "Dockerfile", staging / "Dockerfile")
    shutil.copy2(HF_SRC / "requirements.txt", staging / "requirements.txt")
    (staging / "models").mkdir(exist_ok=True)
    shutil.copy2(MODEL_SRC, staging / "models" / "best_model.h5")
    shutil.copy2(HF_SRC / "README.md", staging / "README.md")
    # Keep the Space repo clean of local runtime artifacts
    for name in ("results", "logs", "uploads"):
        d = staging / name
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    print(f"Staged at {staging}")

    # 4. Push with git-lfs for the 128 MB model
    print("\n[4/4] Committing and pushing...")
    run(["git", "-c", "user.email=deploy@example.com", "-c", "user.name=deploy",
         "config", "user.email", "deploy@example.com"], cwd=str(staging), check=False)
    run(["git", "-c", "user.email=deploy@example.com", "-c", "user.name=deploy",
         "config", "user.name", "deploy"], cwd=str(staging), check=False)
    run(["git", "lfs", "install"], cwd=str(staging), check=False)
    run(["git", "lfs", "track", "models/best_model.h5"], cwd=str(staging), check=False)
    run(["git", "add", "-A"], cwd=str(staging), check=True)
    run(["git", "-c", "user.email=deploy@example.com", "-c", "user.name=deploy",
         "commit", "-m", "Deploy Brain Tumor Detection System (Grad-CAM + localization)"], cwd=str(staging), check=True)

    if token:
        run(["git", "push", f"https://{username}:{token}@huggingface.co/spaces/{username}/{args.space_name}.git",
             "main"], cwd=str(staging), check=True)
    else:
        run(["git", "push", "origin", "main"], cwd=str(staging), check=True)

    print("\nDeployed!")
    print(f"  Space:    https://huggingface.co/spaces/{username}/{args.space_name}")
    print(f"  Building: watch the 'Factory' tab until the build succeeds.")
    print("Note: free-tier Spaces sleep after ~48h of inactivity and wake on request.")


if __name__ == "__main__":
    main()
