"""RegionReasoner task definition.

Uses the existing :class:`oellm.core.base_task.BaseTask` interface unchanged.
"""

from oellm.core.base_task import BaseTask
from oellm.task_groups import DatasetSpec


class RegionReasonerTask(BaseTask):
    """Multi-turn region grounding benchmark on RefCOCOg.

    The evaluation dataset is hosted at ``lmsdss/regionreasoner_data`` on the
    Hugging Face Hub.  Pre-download is handled automatically by the scheduling
    engine via :attr:`dataset_specs`.

    The suite identifier ``"region_reasoner"`` routes execution through the
    contrib dispatch system (``oellm/contrib/dispatch.py``) which calls
    :func:`oellm.contrib.region_reasoner.suite.run`.
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
    def dataset_specs(self) -> list[DatasetSpec]:
        return [DatasetSpec(repo_id="lmsdss/regionreasoner_data")]
