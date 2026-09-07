"""Redaction guards for the code-and-setup judge arm."""

from tools.outcome_prediction import rpm_judge_code as arm


def test_redactor_hides_other_candidate_and_accuracies_but_keeps_hyperparameters():
    walk = arm.redactor("exp-04", ["/home/ben/task/runs/rft2", "/home/ben/task/runs/rft2/final"])
    setup = {
        "data": [
            {
                "selection": "reuse exp-04 traces (exp-04 scored 0.483, 48.3% on n=300) from runs/rft2",
                "source": "derived:exp-01 + exp_04 outputs",
            }
        ],
        "method": {"hyperparams": {"lr": 1e-5, "warmup": 0.03, "other": "lr 1e-5; warmup 0.03; 2 epochs"}},
        "command": {"argv": ["python", "train.py", "--parent", "runs/rft2", "--epochs", "2"]},
    }
    out = walk(setup)
    text = str(out)
    assert "exp-04" not in text and "exp_04" not in text and "rft2" not in text
    assert "0.483" not in text and "48.3%" not in text
    assert "[sibling]" in text and "[n]" in text
    # Numeric JSON values and lr-style strings are not accuracies.
    assert out["method"]["hyperparams"]["lr"] == 1e-5
    assert "1e-5" in out["method"]["hyperparams"]["other"]
    # Unrelated experiment identifiers survive.
    assert "exp-01" in out["data"][0]["source"]


def test_redactor_ignores_short_directory_basenames():
    walk = arm.redactor("exp-02", ["/x/a"])
    assert walk("a sample from a run") == "a sample from a run"
