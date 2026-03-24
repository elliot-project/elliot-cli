from abc import ABC, abstractmethod
from pathlib import Path


class BaseModelAdapter(ABC):
    """Abstract base class for model adapters.

    Translates a model path/config into engine-specific argument strings
    passed on the command line.

    Example::

        class HFModelAdapter(BaseModelAdapter):
            def __init__(self, path: str, trust_remote_code: bool = True):
                self._path = path
                self._trust = trust_remote_code

            @property
            def model_path(self) -> str:
                return self._path

            def to_lm_eval_args(self) -> str:
                return f"pretrained={self._path},trust_remote_code={self._trust}"

            def to_lmms_eval_args(self) -> str:
                return f"pretrained={self._path}"
    """

    @property
    @abstractmethod
    def model_path(self) -> str | Path:
        """Path to the model weights or HuggingFace repo ID."""

    @abstractmethod
    def to_lm_eval_args(self) -> str:
        """Return the ``--model_args`` string for lm-eval-harness.

        Example: ``"pretrained=/path/to/model,trust_remote_code=True"``
        """

    @abstractmethod
    def to_lmms_eval_args(self) -> str:
        """Return the ``--model_args`` string for lmms-eval.

        Example: ``"pretrained=/path/to/model"``
        """

    def to_contrib_flags(self) -> str | None:
        """Return the model-type flag for contrib suite routing.

        This is the value returned by ``detect_model_flags()`` in
        ``suite.py``.  Override to distinguish between inference backends
        for the same benchmark (e.g. ``"vision_reasoner"`` vs ``"qwen2"``).

        Returns ``None`` by default (no model-type distinction needed).
        """
        return None
