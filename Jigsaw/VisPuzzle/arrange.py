import os
import json
import io
import pandas as pd
from PIL import Image
from tqdm import tqdm

PARQUET_PATH = "/mnt/shared-storage-user/hezefeng/data/DiffThinker/Jigsaw/test-00000-of-00001.parquet"

OUT_ROOT = "/mnt/shared-storage-user/hezefeng/data/DiffThinker/Jigsaw/VisPuzzle"
IMG_DIR = os.path.join(OUT_ROOT, "images")
JSONL_PATH = os.path.join(OUT_ROOT, "data.jsonl")

os.makedirs(IMG_DIR, exist_ok=True)

df = pd.read_parquet(
    PARQUET_PATH,
    columns=[
        "pid",
        "question",
        "options",
        "answer",
        "problem_image",
    ]
)

with open(JSONL_PATH, "w", encoding="utf-8") as f:
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pid = row["pid"]

        img_bytes = row["problem_image"]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img_name = f"{pid}_problem.jpg"
        img_path = os.path.join(IMG_DIR, img_name)
        img.save(img_path, quality=95)

        record = {
            "pid": pid,
            "question": row["question"],
            "options": list(row["options"]),
            "answer": row["answer"],
            "problem_image": f"images/{img_name}"
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")
