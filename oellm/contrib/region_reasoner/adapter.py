"""RegionReasoner model adapter."""

from pathlib import Path

from oellm.core.base_model_adapter import BaseModelAdapter


class RegionReasonerModelAdapter(BaseModelAdapter):
    """Translates a model path into eval-engine argument strings.

    Also implements :meth:`to_contrib_flags` so that ``detect_model_flags``
    in ``suite.py`` has a single place to maintain model-type routing logic.
    """

    def __init__(self, model_path: str) -> None:
        self._path = model_path

    @property
    def model_path(self) -> str:
        return self._path

    def to_lm_eval_args(self) -> str:
        return f"pretrained={self._path},trust_remote_code=True"

    def to_lmms_eval_args(self) -> str:
        return f"pretrained={self._path}"

    def to_contrib_flags(self) -> str | None:
        name = Path(self._path).name.lower()
        if "regionreasoner" in name or "region_reasoner" in name:
            return "vision_reasoner"
        if "qwen2" in name:
            return "qwen2"
        if "qwen" in name:
            return "qwen"
        return "vision_reasoner"
