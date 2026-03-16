"""Tests for the contrib plugin registry (oellm/registry.py)."""

import pytest

from oellm import registry


class TestRegistryDiscovery:
    def test_region_reasoner_is_discovered(self):
        """The RegionReasoner suite must be auto-discovered from contrib/."""
        suites = registry.get_all_suites()
        suite_names = [getattr(mod, "SUITE_NAME", None) for mod in suites]
        assert "region_reasoner" in suite_names

    def test_get_suite_returns_module(self):
        mod = registry.get_suite("region_reasoner")
        assert mod is not None
        assert hasattr(mod, "SUITE_NAME")
        assert mod.SUITE_NAME == "region_reasoner"

    def test_get_suite_unknown_raises_keyerror(self):
        with pytest.raises(KeyError, match="nonexistent_suite_xyz"):
            registry.get_suite("nonexistent_suite_xyz")

    def test_keyerror_message_lists_known_suites(self):
        with pytest.raises(KeyError) as exc_info:
            registry.get_suite("nonexistent_suite_xyz")
        # The error message should mention known suites to help diagnose typos
        assert "region_reasoner" in str(exc_info.value)

    def test_get_all_suites_returns_list(self):
        suites = registry.get_all_suites()
        assert isinstance(suites, list)
        assert len(suites) >= 1

    def test_suite_has_required_protocol_attributes(self):
        mod = registry.get_suite("region_reasoner")
        assert hasattr(mod, "SUITE_NAME"), "suite.py must expose SUITE_NAME"
        assert hasattr(mod, "TASK_GROUPS"), "suite.py must expose TASK_GROUPS"
        assert callable(getattr(mod, "run", None)), "suite.py must expose run()"
        assert callable(getattr(mod, "parse_results", None)), "suite.py must expose parse_results()"


class TestRegistryTaskGroupMerge:
    def test_task_metrics_contains_region_reasoner(self):
        merged = registry.get_all_task_groups()
        assert "regionreasoner_refcocog" in merged.get("task_metrics", {})

    def test_task_groups_contains_region_reasoner(self):
        merged = registry.get_all_task_groups()
        assert "region-reasoner" in merged.get("task_groups", {})

    def test_merged_task_group_has_correct_suite(self):
        merged = registry.get_all_task_groups()
        tg = merged["task_groups"]["region-reasoner"]
        assert tg["suite"] == "region_reasoner"

    def test_merged_primary_metric(self):
        merged = registry.get_all_task_groups()
        assert merged["task_metrics"]["regionreasoner_refcocog"] == "gIoU"
