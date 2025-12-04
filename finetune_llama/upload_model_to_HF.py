#!/usr/bin/env python
import argparse
import os
import sys
from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser(
        description="Upload a local merged LoRA model folder to Hugging Face Hub."
    )
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Path to the local model directory (e.g., 'model_lora').",
    )
    parser.add_argument(
        "--repo_id",
        default="excilalala/llama-3.1-8b-causal-CFT",
        help="Hugging Face repo ID to upload to (default: excilalala/llama-3.1-8b-causal-CFT).",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        help=(
            "Hugging Face access token. "
            "If not provided, will try HF_TOKEN or HUGGINGFACE_HUB_TOKEN env vars."
        ),
    )
    parser.add_argument(
        "--commit_message",
        default="Upload LoRA contextual finetuned model",
        help="Commit message for the upload.",
    )

    args = parser.parse_args()

    # --- Basic checks ---
    if not os.path.isdir(args.local_dir):
        print(f"[ERROR] local_dir '{args.local_dir}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    if not args.token:
        print(
            "[ERROR] No HF token provided. Pass --token or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] Using repo_id: {args.repo_id}")
    print(f"[INFO] Uploading from local_dir: {args.local_dir}")

    # --- Ensure repo exists ---
    print("[INFO] Ensuring repo exists on Hugging Face...")
    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        token=args.token,
        exist_ok=True,
    )

    # --- Upload folder ---
    api = HfApi(token=args.token)
    print("[INFO] Starting upload to Hugging Face Hub...")
    api.upload_folder(
        folder_path=args.local_dir,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
        # create_pr=True,  # uncomment if you only have PR permissions on the repo
    )
    print("[INFO] Upload complete ✅")


if __name__ == "__main__":
    main()