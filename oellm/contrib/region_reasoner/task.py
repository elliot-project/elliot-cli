"""RegionReasoner task definition — single source of truth for task metadata."""

from oellm.core.base_task import BaseTask


class RegionReasonerTask(BaseTask):
    """Multi-turn region grounding benchmark on RefCOCOg.

    All task metadata lives here.  ``suite.py`` generates its ``TASK_GROUPS``
    dict directly from :meth:`to_task_groups_dict` so nothing is duplicated.
    """

    @property
    def name(self) -> str:
        return "regionreasoner_refcocog"

    @property
    def suite(self) -> str:
        return "region_reasoner"

    @property
    def n_shots(self) -> list[int]:
        return [0]

    @property
    def task_group_name(self) -> str:
        return "region-reasoner"

    @property
    def description(self) -> str:
        return (
            "RegionReasoner multi-turn region grounding benchmark (RefCOCOg). "
            "Requires REGION_REASONER_DIR on cluster."
        )

    @property
    def primary_metric(self) -> str:
        return "gIoU"

    @property
    def hf_models(self) -> list[str]:
        return ["Ricky06662/TaskRouter-1.5B", "facebook/sam2-hiera-large"]

    @property
    def hf_dataset_files(self) -> list[dict]:
        return [
            {
                "repo_id": "lmsdss/regionreasoner_test_data",
                "patterns": [
                    "raw/refcocog_multi_turn.json",
                    "raw/refcocog_test_multi_bbox_images/*",
                ],
            }
        ]
