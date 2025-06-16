import os
import json
import sys
import requests
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path
from huggingface_hub import login
import ijson

# Authenticate Hugging Face API

try:
    from datasets import load_dataset
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    import kagglehub
except ImportError:
    kagglehub = None
# Add TokenSim to path for utility
sys.path.append(str(Path(__file__).resolve().parent.parent))
from TokenSim.utils import get_generation_lens

DATASETS = {
   # "longbench": ("longbench_input.json", "THUDM/LongBench-v2"),
   # "needle_in_a_haystack": ("haystack.json", None),
   # "sharegpt": ("sharegpt.json", None),
   "bookcorpus": ("bookcorpus.json", None),  # Use Kaggle local JSONL
   # "wikipedia_structured": ("wiki.json", None),
}

MODELS = {
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "internlm2-7b": "internlm/internlm2-chat-7b"
}

GEN_LEN_MEAN = 128
GEN_LEN_RANGE = 64
GEN_DIST = "uniform"

def extract_prompt(dataset_name, entry):
    if dataset_name == "longbench":
        context = entry.get("context", "")
        question = entry.get("question", "")
        choices = [entry.get(f"choice_{c}", "") for c in ["A", "B", "C", "D"]]
        return context + "\n" + question + "\n" + "\n".join([f"{l}. {c}" for l, c in zip("ABCD", choices)])
    elif dataset_name == "needle_in_a_haystack":
        return entry.get("context", "") + "\n" + entry.get("question", "")
    elif dataset_name == "sharegpt":
        try:
            convos = entry.get("conversations", [])
            if isinstance(convos, str):
                convos = json.loads(convos)
            if isinstance(convos, list):
                return "\n".join([
                    f"{c.get('role', c.get('from', ''))}: {c.get('content', c.get('value', ''))}"
                    for c in convos if isinstance(c, dict)
                ])
        except Exception:
            return ""
        return entry.get("prompt", "")
    elif dataset_name in ["bookcorpus", "wikipedia_structured"]:
        return entry.get("text", "")
    return ""

def fallback_sharegpt_stream():
    base_path = Path("dataset") / "sharegpt" / "raw"
    files = ["sg_90k_part1.json", "sg_90k_part2.json"]
    def generator():
        for file in files:
            file_path = base_path / file
            print(f"📄 Streaming ShareGPT from: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                for obj in ijson.items(f, "item"):
                    if isinstance(obj.get("conversations", None), list):
                        yield obj
    return generator()
def load_bookcorpus_local():
    data_dir = Path("dataset/bookcorpus/raw")
    text_files = list(data_dir.glob("*.txt"))
    
    for txt_file in text_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield {"text": line}

# MAIN LOOP
for dataset_name, (local_filename, hf_id) in DATASETS.items():
    local_path = Path("dataset") / dataset_name / local_filename
    dataset_data = None

    if dataset_name == "sharegpt":
        print("🔁 Always using fallback loader for ShareGPT...")
        dataset_data = fallback_sharegpt_stream()
    if dataset_name == "bookcorpus":
        print(f"📦 Loading local BookCorpus .txt files...")
        dataset_data = load_bookcorpus_local()
        
    elif dataset_name == "wikipedia_structured":
        try:
            print("📡 Downloading Wikipedia Structured from Kaggle via kagglehub...")
            kaggle_path = kagglehub.dataset_download("wikimedia-foundation/wikipedia-structured-contents")
            file_path = Path(kaggle_path) / "enwiki-20240520-cirrussearch-content.json"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    dataset_data = [json.loads(line) for line in f if line.strip()]
            else:
                print(f"❌ File not found at expected location: {file_path}")
                continue
        except Exception as e:
            print(f"❌ Failed to load Wikipedia Structured from Kaggle: {e}")
            continue
    elif hf_id and HUGGINGFACE_AVAILABLE:
        try:
            print(f"📡 Loading {dataset_name} from Hugging Face: {hf_id}")
            dataset_data = load_dataset(hf_id, split="train", streaming=True,download_mode="force_redownload")
        except Exception as e:
            print(f"⚠️ HF load failed for {hf_id}: {e}")

    if dataset_data is None:
        if local_path.exists():
            print(f"📂 Loading {dataset_name} from local: {local_path}")
            with open(local_path, "r", encoding="utf-8") as f:
                dataset_data = json.load(f)
        elif dataset_name == "needle_in_a_haystack":
            print("❌ Skipping needle_in_a_haystack: no data found")
            continue
        else:
            print(f"❌ Skipping {dataset_name}: no data found")
            continue

    print(f"✅ Loaded dataset: {dataset_name}")

    for model_name, tokenizer_id in MODELS.items():
        output_path = Path("dataset") / dataset_name / "converted" / f"{model_name}.json"
        if output_path.exists():
            print(f"⏩ Skipping {dataset_name}-{model_name}, output already exists.")
            continue

        print(f"\n🚀 Tokenizing with {model_name}: {tokenizer_id}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        
        prompt_lens = []
        entry_count = 0

        iterator = dataset_data if isinstance(dataset_data, list) else iter(dataset_data)

        for raw_entry in tqdm(iterator, desc=f"{dataset_name}-{model_name}"):
            try:
                entry = dict(raw_entry)
                prompt = extract_prompt(dataset_name, entry)
                tokens = tokenizer(prompt, truncation=True)["input_ids"]
                prompt_lens.append(len(tokens))
                entry_count += 1
            except Exception:
                continue

        print(f"✅ {dataset_name}: {entry_count} prompts processed for {model_name}")

        generation_lens = get_generation_lens(
            distribution=GEN_DIST,
            len_mean=GEN_LEN_MEAN,
            len_range=GEN_LEN_RANGE,
            num_prompt=len(prompt_lens)
        )

        output_data = list(zip(prompt_lens, generation_lens))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"💾 Saved {len(output_data)} to {output_path}")
