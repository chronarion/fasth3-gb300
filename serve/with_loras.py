"""Write the serve config with every LoRA in a folder on the per-request allow-list.

    python3 with_loras.py BASE_CONFIG LORA_DIR OUT_CONFIG

Each *.safetensors file in LORA_DIR (the compose services mount serve/loras at /loras) becomes an adapter whose id is
the file name without .safetensors. A request selects one with "lora": {"name": "<id>", "scale": 0.8}; scale is the
adapter's strength, 0.8 when left out, from 0 to 2. Without LoRA files, or when BASE_CONFIG already has a lora
section, the base config is written unchanged.
"""
import pathlib
import sys

import yaml

base, folder, out = sys.argv[1:4]
config = yaml.safe_load(pathlib.Path(base).read_text())
lora_dir = pathlib.Path(folder)
files = sorted(lora_dir.glob("*.safetensors")) if lora_dir.is_dir() else []
if files and "lora" not in config:
    config["lora"] = {"adapters": [{"id": f.stem, "path": str(f), "default_strength": 0.8, "min_strength": 0.0,
                                    "max_strength": 2.0} for f in files]}
    print("LoRA adapters on the allow-list: " + ", ".join(f.stem for f in files), flush=True)
pathlib.Path(out).write_text(yaml.safe_dump(config, sort_keys=False))
