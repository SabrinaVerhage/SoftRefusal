from __future__ import annotations

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Residual activation mapping; UMAP axes are not refusal scores")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("doctor", help="Check hardware and access without downloading weights")
    p.add_argument("--check-access", action="store_true")
    p = commands.add_parser("chat", help="Open a local chat and recording server")
    p.add_argument("--config", default="configs/gemma3-4b-chat.json")
    p.add_argument("--out", default="runs/chat")
    p.add_argument("--port", type=int, default=8765)
    p = commands.add_parser("collect")
    p.add_argument("--config", default="configs/gemma3-4b-local.json")
    p.add_argument("--input", default="data/neutral.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--synthetic", action="store_true", help="Clearly marked random-vector plumbing test; not Gemma")
    p = commands.add_parser("map", help="Donut and per-layer UMAP of every recorded conversation, for ../map")
    p.add_argument("--runs", default="runs/gpu-chat")
    p.add_argument("--out", default="../map/data/recordings.js")
    p.add_argument("--layer", type=int, help="Layer shown first (default: second recorded layer)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neighbors", type=int, default=10)
    p.add_argument("--min-dist", type=float, default=.1)
    for name in ["reduce", "features", "probe", "score"]:
        p = commands.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--layer", type=int, default=17)
        if name == "reduce":
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--neighbors", type=int, default=10)
            p.add_argument("--min-dist", type=float, default=.1)
        if name == "probe":
            p.add_argument("--annotations", required=True)
        if name == "score":
            p.add_argument("--probe", required=True)
    args = parser.parse_args()
    try:
        if args.command == "chat":
            from .chat import serve
            serve(args.config, args.out, args.port)
        elif args.command == "doctor":
            import torch
            import psutil
            from .io import environment
            info = {"environment": environment(), "ram_available_gib": round(psutil.virtual_memory().available / 2**30, 2),
                    "cuda_available": torch.cuda.is_available()}
            if torch.cuda.is_available():
                info.update(gpu=torch.cuda.get_device_name(), gpu_memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)
            if args.check_access:
                from huggingface_hub import hf_hub_download
                try:
                    hf_hub_download("google/gemma-3-4b-it", "config.json", revision="093f9f388b31de276ce2de164bdc2081324b9767")
                    info["gemma_access"] = "confirmed_config_download"
                except Exception as e:
                    info["gemma_access"] = f"not_confirmed: {type(e).__name__}; accept model terms and run hf auth login"
            print(json.dumps(info, indent=2))
        elif args.command == "collect":
            from .collect import collect
            print(collect(args.config, args.input, args.out, args.synthetic))
        elif args.command == "map":
            from .mapping import export_map
            print(export_map(args.runs, args.out, args.layer, args.seed, args.neighbors, args.min_dist))
        elif args.command == "reduce":
            from .reduce import reduce
            print(reduce(args.run, args.out, args.layer, args.seed, args.neighbors, args.min_dist))
        elif args.command == "features":
            from .scope import features
            print(features(args.run, args.out, args.layer))
        elif args.command == "probe":
            from .probe import train_probe
            print(train_probe(args.run, args.annotations, args.out, args.layer))
        elif args.command == "score":
            from .probe import score
            print(score(args.run, args.probe, args.out, args.layer))
    except (ValueError, FileExistsError, KeyError) as e:
        parser.exit(2, f"Cannot continue: {e}\n")


if __name__ == "__main__":
    main()
