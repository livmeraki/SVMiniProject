#!/usr/bin/env python3
"""Three-task WoodScape fisheye ODD benchmark using SmolVLM on Mac/MPS."""

# Let unsupported MPS operations fall back to CPU. This must precede torch import.
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
BUCKET_VALUES = {"0", "1", "2", "3_to_5", "6_to_10", "11_plus", "unknown"}

PROMPT_ENVIRONMENT = """You are analyzing one WoodScape RGB fisheye autonomous-driving image.
This task is ODD classification, NOT image captioning.

Return ONLY one valid JSON object using exactly these fields in this order:
{
  "scene_id": "",
  "weather": "",
  "lighting": "",
  "road_type": "",
  "road_geometry": "",
  "intersection_present": "",
  "traffic_density": ""
}

Allowed values:
weather: clear, rain, snow, fog, wet_road_visible, unknown
lighting: day, night, dusk_dawn, tunnel_or_underpass, backlit_or_glare, unknown
road_type: urban, residential, highway, parking_lot, service_road, unknown
road_geometry: straight, curved, intersection, roundabout_like, parking_area_layout, unknown
intersection_present: yes, no, unknown
traffic_density: low, medium, high, unknown

Select exactly one allowed value per field. Do not copy an allowed-value list. Use
"unknown" when visual evidence is insufficient. Do not add Markdown, extra fields,
commentary, or text outside the JSON."""

PROMPT_ACTOR_BUCKETS = """You are analyzing one WoodScape RGB fisheye autonomous-driving image.
This task is coarse visible-actor counting, NOT image captioning.

Return ONLY one valid JSON object using exactly these fields in this order:
{
  "scene_id": "",
  "vehicle_count_bucket": "",
  "pedestrian_count_bucket": "",
  "cyclist_count_bucket": "",
  "large_vehicle_count_bucket": ""
}

For every bucket field choose exactly one of:
0, 1, 2, 3_to_5, 6_to_10, 11_plus, unknown

Definitions:
- vehicle: visible car, truck, bus, or van
- pedestrian: visible person
- cyclist: visible cyclist or bicycle being used as a road user
- large_vehicle: visible bus, truck, or clearly large van

Count only visible actors in this frame. Do not infer hidden actors. Edge objects may be
distorted by the fisheye lens; use "unknown" when too uncertain. Do not copy the option
list.

Mandatory visual scan before choosing buckets:
1. inspect the entire left sidewalk and left image edge
2. inspect the center roadway and distant traffic
3. inspect the entire right sidewalk and right image edge

Standing or walking human figures on either sidewalk count as pedestrians. Use "0" only
after confirming no actor of that type is visible anywhere. Do not use all-zero buckets
as a default response. Do not add Markdown, extra fields, commentary, or text outside
the JSON."""

PROMPT_OBSERVABILITY = """You are analyzing one WoodScape RGB fisheye autonomous-driving image.
This task is observability and fisheye-difficulty tagging, NOT image captioning.

Return ONLY one valid JSON object using exactly these fields in this order:
{
  "scene_id": "",
  "drivable_area_visibility": "",
  "lane_marking_visibility": "",
  "occlusion_level": "",
  "fisheye_distortion_impact": ""
}

Allowed values:
drivable_area_visibility: clear, partially_visible, poor, unknown
lane_marking_visibility: clear, faint_or_partial, not_visible, unknown
occlusion_level: low, medium, high, unknown
fisheye_distortion_impact: low, medium, high, unknown

Select exactly one allowed value per field. Do not copy an allowed-value list. Use
"unknown" when visual evidence is insufficient. Do not add Markdown, extra fields,
commentary, or text outside the JSON."""

TASKS = {
    "environment": {
        "prompt": PROMPT_ENVIRONMENT,
        "fields": {
            "weather": {"clear", "rain", "snow", "fog", "wet_road_visible", "unknown"},
            "lighting": {"day", "night", "dusk_dawn", "tunnel_or_underpass", "backlit_or_glare", "unknown"},
            "road_type": {"urban", "residential", "highway", "parking_lot", "service_road", "unknown"},
            "road_geometry": {"straight", "curved", "intersection", "roundabout_like", "parking_area_layout", "unknown"},
            "intersection_present": {"yes", "no", "unknown"},
            "traffic_density": {"low", "medium", "high", "unknown"},
        },
    },
    "actor_buckets": {
        "prompt": PROMPT_ACTOR_BUCKETS,
        "fields": {
            "vehicle_count_bucket": BUCKET_VALUES,
            "pedestrian_count_bucket": BUCKET_VALUES,
            "cyclist_count_bucket": BUCKET_VALUES,
            "large_vehicle_count_bucket": BUCKET_VALUES,
        },
    },
    "observability": {
        "prompt": PROMPT_OBSERVABILITY,
        "fields": {
            "drivable_area_visibility": {"clear", "partially_visible", "poor", "unknown"},
            "lane_marking_visibility": {"clear", "faint_or_partial", "not_visible", "unknown"},
            "occlusion_level": {"low", "medium", "high", "unknown"},
            "fisheye_distortion_impact": {"low", "medium", "high", "unknown"},
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("woodscape_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--actor-crop-ratio",
        type=float,
        default=0.72,
        help="Top fraction of frame used by actor task (default: 0.72)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_device(requested: str) -> tuple[torch.device, torch.dtype]:
    mps_available = torch.backends.mps.is_available()
    if requested == "mps" and not mps_available:
        raise RuntimeError("--device mps requested, but MPS is unavailable")
    use_mps = requested == "mps" or (requested == "auto" and mps_available)
    device = torch.device("mps" if use_mps else "cpu")
    # torch 2.2 MPS is more reliable with float16 than bfloat16.
    dtype = torch.float16 if device.type == "mps" else torch.float32
    return device, dtype


def load_model_and_processor(
    model_id: str, device: torch.device, dtype: torch.dtype
) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        _attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, processor


def find_images(image_dir: Path, limit: Optional[int]) -> list[Path]:
    images = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        images = images[:limit]
    return images


def response_prefill(scene_id: str, first_field: str) -> str:
    # Supplying the JSON beginning greatly reduces generic prose from this 500M model.
    return (
        '{"scene_id":'
        + json.dumps(scene_id, ensure_ascii=False)
        + ',"'
        + first_field
        + '":"'
    )


def run_inference(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    scene_id: str,
    first_field: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    prefill = response_prefill(scene_id, first_field)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": prefill}],
        },
    ]
    chat_prompt = processor.apply_chat_template(
        messages, continue_final_message=True
    )
    inputs = processor(text=chat_prompt, images=[image], return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    continuation = processor.batch_decode(
        generated[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return prefill + continuation


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def extract_json_block(text: str) -> dict[str, Any]:
    """Return the first decodable JSON object, ignoring fences and surrounding text."""
    cleaned = strip_markdown_fences(text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No valid JSON object found in model response")


def normalize_label(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    value = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "3_5": "3_to_5",
        "3to5": "3_to_5",
        "6_10": "6_to_10",
        "6to10": "6_to_10",
        "11+": "11_plus",
    }
    return aliases.get(value, value)


def normalize_task_json(
    obj: dict[str, Any],
    scene_id: str,
    allowed_fields: dict[str, set[str]],
) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    expected = {"scene_id", *allowed_fields.keys()}
    extra = sorted(set(obj) - expected)
    if extra:
        errors.append(f"Ignored unexpected fields: {extra}")

    returned_scene_id = obj.get("scene_id")
    if returned_scene_id != scene_id:
        errors.append("scene_id was missing or incorrect; injected from filename")
    normalized = {"scene_id": scene_id}

    for field, allowed in allowed_fields.items():
        if field not in obj:
            errors.append(f"Missing {field}; set to unknown")
            normalized[field] = "unknown"
            continue

        original = obj[field]
        if not isinstance(original, str):
            errors.append(f"{field} was not a string; set to unknown")
            normalized[field] = "unknown"
            continue
        if not original.strip():
            errors.append(f"{field} was empty; set to unknown")
            normalized[field] = "unknown"
            continue
        if "|" in original:
            errors.append(f"{field} copied multiple options; set to unknown")
            normalized[field] = "unknown"
            continue

        value = normalize_label(original)
        if value not in allowed:
            errors.append(f"Invalid {field}={original!r}; set to unknown")
            value = "unknown"
        normalized[field] = value

    return normalized, errors


def empty_task_result(
    scene_id: str, allowed_fields: dict[str, set[str]], error: str
) -> dict[str, Any]:
    normalized = {"scene_id": scene_id}
    normalized.update({field: "unknown" for field in allowed_fields})
    return {
        "raw_response": None,
        "parsed_json": None,
        "normalized_json": normalized,
        "validation_errors": [error],
        "format_success": False,
        "success": False,
        "elapsed_seconds": 0.0,
    }


def process_task(
    model: Any,
    processor: Any,
    image: Image.Image,
    scene_id: str,
    task: dict[str, Any],
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    fields = task["fields"]
    raw_response: Optional[str] = None
    parsed: Optional[dict[str, Any]] = None
    try:
        raw_response = run_inference(
            model,
            processor,
            image,
            task["prompt"],
            scene_id,
            next(iter(fields)),
            device,
            max_new_tokens,
        )
        parsed = extract_json_block(raw_response)
        normalized, errors = normalize_task_json(parsed, scene_id, fields)
        return {
            "raw_response": raw_response,
            "parsed_json": parsed,
            "normalized_json": normalized,
            # Format validity is not the same as semantic/visual correctness.
            "validation_errors": errors,
            "format_success": not errors,
            "success": not errors,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:
        fallback = empty_task_result(
            scene_id, fields, f"{type(exc).__name__}: {exc}"
        )
        fallback["raw_response"] = raw_response
        fallback["parsed_json"] = parsed
        fallback["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        return fallback


def process_single_image(
    model: Any,
    processor: Any,
    image_path: Path,
    model_id: str,
    device: torch.device,
    max_new_tokens: int,
    actor_crop_ratio: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    scene_id = image_path.stem
    result: dict[str, Any] = {
        "image": str(image_path),
        "model": model_id,
        "device": device.type,
        "scene_id": scene_id,
    }
    try:
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        crop_height = max(1, round(image.height * actor_crop_ratio))
        actor_image = image.crop((0, 0, image.width, crop_height))
        result["actor_preprocessing"] = {
            "strategy": "top_scene_crop",
            "crop_ratio": actor_crop_ratio,
            "original_size": [image.width, image.height],
            "actor_input_size": [actor_image.width, actor_image.height],
        }
        for task_name, task in TASKS.items():
            print(f"    {task_name}")
            task_image = actor_image if task_name == "actor_buckets" else image
            result[task_name] = process_task(
                model, processor, task_image, scene_id, task, device, max_new_tokens
            )
            if device.type == "mps":
                torch.mps.empty_cache()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        error = f"Image error: {type(exc).__name__}: {exc}"
        for task_name, task in TASKS.items():
            result[task_name] = empty_task_result(scene_id, task["fields"], error)

    result["elapsed_seconds_total"] = round(time.perf_counter() - started, 3)
    return result


def output_path_for(image_path: Path, image_dir: Path, output_dir: Path) -> Path:
    relative = image_path.relative_to(image_dir)
    return (output_dir / relative).with_suffix(relative.suffix + ".json")


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if not args.image_dir.is_dir():
        print(f"Error: image directory not found: {args.image_dir}", file=sys.stderr)
        return 2
    if not 0.25 <= args.actor_crop_ratio <= 1.0:
        print("Error: --actor-crop-ratio must be between 0.25 and 1.0", file=sys.stderr)
        return 2
    if torch.__version__.startswith("2.2.") and int(np.__version__.split(".")[0]) >= 2:
        print(
            "Error: PyTorch 2.2 requires NumPy 1.x here. Run: "
            "python -m pip install 'numpy==1.26.4'",
            file=sys.stderr,
        )
        return 2

    try:
        images = find_images(args.image_dir, args.limit)
        device, dtype = get_device(args.device)
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if not images:
        print(f"Error: no PNG/JPG images found under {args.image_dir}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.model} on {device} with {dtype} ...")
    try:
        model, processor = load_model_and_processor(args.model, device, dtype)
    except Exception as exc:
        print(f"Model load failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if device.type == "mps":
            print("Retry with --device cpu for an MPS-specific error.", file=sys.stderr)
        return 1

    processed = skipped = task_failures = 0
    for index, image_path in enumerate(images, start=1):
        output_path = output_path_for(image_path, args.image_dir, args.output_dir)
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(images)}] skip {image_path.name}")
            skipped += 1
            continue
        print(f"[{index}/{len(images)}] {image_path}")
        result = process_single_image(
            model,
            processor,
            image_path,
            args.model,
            device,
            args.max_new_tokens,
            args.actor_crop_ratio,
        )
        save_json(output_path, result)
        task_failures += sum(
            not result[name]["success"] for name in TASKS
        )
        processed += 1

    print(
        f"Done: processed={processed}, skipped={skipped}, "
        f"task_failures_or_repairs={task_failures}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
