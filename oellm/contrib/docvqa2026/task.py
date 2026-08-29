"""Task definition for the DocVQA 2026 competition benchmark."""

from __future__ import annotations

from oellm.core.base_task import BaseTask
from oellm.task_groups import DatasetSpec

SUITE_NAME = "docvqa2026"


class DocVQA2026ValTask(BaseTask):
    """DocVQA 2026 validation split: 80 questions over 25 multi-page documents.

    Only ``val`` is scorable here. The test split ships with its answers
    withheld and is graded solely by the RRC platform.
    """

    @property
    def name(self) -> str:
        return "docvqa2026_val"

    @property
    def suite(self) -> str:
        return SUITE_NAME

    @property
    def task_group_name(self) -> str:
        """Mirrors the core ``image-docvqa`` / ``docvqa_val`` pairing: the group
        carries the modality prefix, the task inside it carries the split."""
        return "image-docvqa2026"

    @property
    def n_shots(self) -> list[int]:
        return [0]

    @property
    def primary_metric(self) -> str:
        return "accuracy"

    @property
    def description(self) -> str:
        return (
            "DocVQA 2026 (ICDAR) reasoning over multi-page documents in eight "
            "domains, scored with the competition's own strict number/unit/date "
            "matcher and ANLS fallback."
        )

    @property
    def dataset_specs(self) -> list[DatasetSpec]:
        return [DatasetSpec(repo_id="VLR-CVC/DocVQA-2026")]
