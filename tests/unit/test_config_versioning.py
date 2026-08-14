"""Tests for the current, v2-only configuration-version contract."""

from __future__ import annotations

import pytest

from dslighting.core.config.versioning import (
    ConfigVersionManager,
    ConfigVersionManagerFactory,
    InvalidVersionError,
    MigrationNotSupportedError,
    detect_config_version,
    get_version_manager,
    is_config_compatible,
    migrate_config,
)


@pytest.fixture
def manager() -> ConfigVersionManager:
    return ConfigVersionManager()


def test_version_contract_is_v2_only(manager: ConfigVersionManager) -> None:
    assert manager.VERSION == "2.0"
    assert manager.SUPPORTED_VERSIONS == ["2.0"]
    assert manager.MIN_COMPATIBLE_VERSION == "2.0"


def test_detects_explicit_or_implicit_current_version(
    manager: ConfigVersionManager,
) -> None:
    assert manager.detect_version({"_version": "2.0"}) == "2.0"
    assert manager.detect_version({"llm": {"model": "test"}}) == "2.0"
    assert manager.detect_version({}) == "2.0"


def test_unknown_explicit_version_is_not_treated_as_current(
    manager: ConfigVersionManager,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = {"_version": "1.1"}

    assert manager.detect_version(config) == "1.1"
    assert "Unknown config version '1.1'" in caplog.text
    assert manager.is_compatible(config) is False


def test_migrate_normalizes_an_unversioned_v2_config_without_mutating_it(
    manager: ConfigVersionManager,
) -> None:
    original = {"llm": {"model": "test"}}

    migrated = manager.migrate(original)

    assert migrated == {**original, "_version": "2.0"}
    assert migrated is not original
    assert "_version" not in original


def test_migrate_current_version_is_noop(manager: ConfigVersionManager) -> None:
    config = {"_version": "2.0", "custom": True}

    assert manager.migrate(config) is config


@pytest.mark.parametrize("legacy_version", ["0.9", "1.0", "1.1"])
def test_migrate_rejects_removed_legacy_versions(
    manager: ConfigVersionManager,
    legacy_version: str,
) -> None:
    with pytest.raises(MigrationNotSupportedError, match="no longer supported"):
        manager.migrate({"_version": legacy_version})


def test_migrate_rejects_an_explicit_legacy_source(
    manager: ConfigVersionManager,
) -> None:
    with pytest.raises(MigrationNotSupportedError, match="no longer supported"):
        manager.migrate({}, from_version="0.9")


def test_migrate_rejects_an_unknown_target(manager: ConfigVersionManager) -> None:
    with pytest.raises(InvalidVersionError):
        manager.migrate({}, to_version="3.0")


def test_migration_path_exists_only_for_the_current_version(
    manager: ConfigVersionManager,
) -> None:
    assert manager.get_migration_path("2.0", "2.0") == []

    for version in ["0.9", "1.0", "1.1", "3.0", "invalid"]:
        with pytest.raises(InvalidVersionError):
            manager.get_migration_path(version, "2.0")


def test_version_validation_and_support(manager: ConfigVersionManager) -> None:
    manager._validate_version("2.0")
    assert manager.is_version_supported("2.0") is True
    assert manager.is_version_supported("1.1") is False

    with pytest.raises(InvalidVersionError):
        manager._validate_version("1.1")


def test_custom_migration_registration(manager: ConfigVersionManager) -> None:
    def migration(config: dict) -> dict:
        return config

    manager.register_migration("custom", "2.0", migration)

    assert manager._migration_registry[("custom", "2.0")] is migration


def test_factory_reuses_named_managers() -> None:
    factory = ConfigVersionManagerFactory()

    assert factory.get_manager("v2-test") is factory.get_manager("v2-test")
    assert factory.get_manager("v2-test") is not factory.get_manager("other-v2-test")


def test_convenience_functions_follow_the_v2_contract() -> None:
    assert get_version_manager() is get_version_manager()
    assert get_version_manager().VERSION == "2.0"
    assert detect_config_version({}) == "2.0"
    assert migrate_config({"task": {}}) == {"task": {}, "_version": "2.0"}
    assert is_config_compatible({"_version": "2.0"}) is True
    assert is_config_compatible({"_version": "1.1"}) is False
