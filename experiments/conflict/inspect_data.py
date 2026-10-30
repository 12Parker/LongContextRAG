"""
Print the schema of an unverified Hugging Face benchmark so a loader can be written.

    python -m experiments.conflict.inspect_data faitheval
"""
import json
import sys

HF = {
    "faitheval": ("Salesforce/FaithEval-inconsistent-v1.0", {"split": "test"}),
    "confrag": ("OracleY/ConfRAG", {}),
    "wikicontradict": ("ibm/Wikipedia_contradict_benchmark", {}),
}


def main():
    name = sys.argv[1]
    from datasets import load_dataset
    repo, kw = HF[name]
    ds = load_dataset(repo, **kw)
    print(ds)
    split = ds if "split" in kw else ds[list(ds.keys())[0]]
    row = split[0]
    print(json.dumps({k: (v if len(str(v)) < 300 else str(v)[:300] + "...") for k, v in row.items()},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
