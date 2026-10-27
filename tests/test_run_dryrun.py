"""End-to-end plumbing with a fake LLM: no network, no cost."""
import json

from experiments.conflict import analyze, run
from tests.conftest import needs_data


@needs_data
def test_dry_run_and_resume(tmp_path):
    args = ["--dataset", "ramdocs", "--n", "6", "--seeds", "0,1", "--out", str(tmp_path), "--dry-run",
            "--conditions", "plain,instruct,oracle+flag,oracle+remove,random+flag,madam", "--random-rate", "0.2"]
    run.main(args)
    rows = [json.loads(l) for l in open(tmp_path / "results.jsonl")]
    assert len(rows) == 6 * 6 * 2
    assert all("primary" in r and "detect_s" in r for r in rows)
    assert any(r["n_flags"] > 0 for r in rows if r["condition"] == "oracle+flag")
    run.main(args)  # resume: nothing new
    assert len(open(tmp_path / "results.jsonl").readlines()) == 72
    analyze.main([str(tmp_path)])
    assert (tmp_path / "analysis.json").exists()
