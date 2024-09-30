import itertools
import json
from pathlib import Path


output_path = Path.cwd() / "data" / "hyperparams.jsonl"


if __name__ == "__main__":

    def is_cached(config: dict):
        if not output_path.exists():
            return False
        content = output_path.read_text()
        lines = content.split("\n")
        for line in lines:
            if not line:
                continue
            result = json.loads(line)
            if result["config"] == config:
                return True

    searchspace = {
        "dataset": ["cifar10", "cifar100"],  # paper used pretrained imagenet weights with cifar10, cifar100
        "lr": [1e-1, 1e-4, 1e-5, 1.7e-5, 1e-6, 1e-7],  # paper found 1.7e-5 to be most robust
        "num_epochs": [4, 8, 16],  # paper only used 1 epoch
        "crossmax_k": [2, 3],  # 2 is the classic vickery consensus
    }
    combinations = [dict(zip(searchspace.keys(), values)) for values in itertools.product(*searchspace.values())]
    print(f"searching {len(combinations)} combinations")

    for combination in combinations:
        if is_cached(combination):
            continue

        print(f"missing: {combination}")
