"""Tests for the multi-user registry and per-user state isolation."""

from src.portfolio.users import UserRegistry, slug, DEFAULT_USER


def test_default_user_created(tmp_path):
    reg = UserRegistry(str(tmp_path))
    assert reg.current() == DEFAULT_USER
    assert reg.users() == [DEFAULT_USER]


def test_add_switch_remove(tmp_path):
    reg = UserRegistry(str(tmp_path))
    assert reg.add("Maman") is True
    assert reg.add("Maman") is False  # duplicate
    assert reg.switch("Maman") is True
    assert reg.current() == "Maman"
    assert reg.switch("ghost") is False
    assert reg.remove("Maman") is True
    assert reg.current() == DEFAULT_USER


def test_rename_moves_state(tmp_path):
    reg = UserRegistry(str(tmp_path))
    (tmp_path / "users" / slug(DEFAULT_USER) / "watchlist.json").write_text('{"symbols": ["AAPL"]}')
    assert reg.rename(DEFAULT_USER, "MasterLaplace") is True
    assert reg.current() == "MasterLaplace"
    moved = tmp_path / "users" / slug("MasterLaplace") / "watchlist.json"
    assert moved.exists()
    assert reg.rename("MasterLaplace", "MasterLaplace") is False  # name taken


def test_cannot_remove_last_user(tmp_path):
    reg = UserRegistry(str(tmp_path))
    assert reg.remove(DEFAULT_USER) is False


def test_per_user_directories_differ(tmp_path):
    reg = UserRegistry(str(tmp_path))
    reg.add("papa")
    assert reg.dir_for(DEFAULT_USER) != reg.dir_for("papa")


def test_registry_persists(tmp_path):
    reg = UserRegistry(str(tmp_path))
    reg.add("papa")
    reg.switch("papa")
    reloaded = UserRegistry(str(tmp_path))
    assert reloaded.current() == "papa"
    assert set(reloaded.users()) == {DEFAULT_USER, "papa"}


def test_legacy_migration(tmp_path):
    # A pre-existing single-user file should move into the default user's dir.
    (tmp_path / "portfolio_real.json").write_text('{"cash": 0, "positions": {}}')
    reg = UserRegistry(str(tmp_path))
    migrated = tmp_path / "users" / slug(DEFAULT_USER) / "portfolio_real.json"
    assert migrated.exists()
    assert not (tmp_path / "portfolio_real.json").exists()
