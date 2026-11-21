"""Doctor formatting and exit codes (it is allowed to say 'not ready' here:
the test machine has no GPU and no 50 GB checkpoint)."""

from __future__ import annotations

import json

from app.config import Settings
from app.doctor import format_report, run_checks


def test_report_has_all_sections(settings: Settings):
    report = run_checks(settings)
    sections = {check.section for check in report.checks}
    assert {"system", "gpu", "libraries", "wan", "weights", "media", "storage", "service"} <= sections
    assert report.exit_code() in {0, 2, 3}


def test_report_is_json_serialisable(settings: Settings):
    payload = json.loads(json.dumps(run_checks(settings).as_dict()))
    assert payload["checks"]
    assert isinstance(payload["ready"], bool)


def test_format_mentions_the_verdict(settings: Settings):
    text = format_report(run_checks(settings))
    assert "Wan2.2-S2V container preflight" in text
    assert ("READY" in text) or ("NOT READY" in text)


def test_missing_weights_is_a_failure(settings: Settings):
    report = run_checks(settings)
    assert report.failures, "an empty model dir must be a blocking problem"
    assert report.exit_code() == 2
