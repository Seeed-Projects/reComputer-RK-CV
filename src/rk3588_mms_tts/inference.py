import argparse
import json
import time
from pathlib import Path

from task_runtime import create_runtime


def parse_prompts(value):
    if not value:
        return None
    return [item.strip() for item in value.replace("\n", "|").split("|") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run one RKNN model inference")
    parser.add_argument("--platform", choices=["rk3576", "rk3588"], required=True)
    parser.add_argument("--model_dir", default="model")
    parser.add_argument("--text")
    parser.add_argument("--file")
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--query")
    parser.add_argument("--prompts", help="Candidate prompts separated with |")
    parser.add_argument("--topk", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--hop_seconds", type=float)
    parser.add_argument("--speaking_rate", type=float, default=1.0, help="Speech speed from 0.6 to 1.4")
    parser.add_argument("--output", default="output.wav")
    args = parser.parse_args()

    runtime = create_runtime(args.platform, Path(args.model_dir))
    params = runtime.get_config()
    if args.prompts:
        params["prompts"] = parse_prompts(args.prompts)
    for key in ("topk", "threshold", "hop_seconds"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    params["speaking_rate"] = args.speaking_rate

    started = time.perf_counter()
    if args.files:
        if runtime.name != "clip" or not args.query:
            parser.error("--files and --query are only valid for CLIP retrieval")
        images = [(Path(path).name, Path(path).read_bytes()) for path in args.files]
        result = runtime.retrieve(images, args.query, args.topk)
    elif runtime.input_kind == "text":
        if not args.text:
            parser.error("--text is required for this model")
        result = runtime.predict(args.text, params)
    else:
        if not args.file:
            parser.error("--file is required for this model")
        result = runtime.predict(Path(args.file).read_bytes(), params)

    audio = result.pop("_audio_bytes", None)
    if audio is not None:
        Path(args.output).write_bytes(audio)
        result["output_file"] = args.output
    print(json.dumps({
        "success": True,
        "model": runtime.name,
        "platform": runtime.platform,
        "inference_time_ms": round((time.perf_counter() - started) * 1000, 2),
        "result": result,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
