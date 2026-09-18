"""Tests for the vault backup engine, installer, and restore tool.

All filesystem-only: no network, and every rclone interaction is mocked.
"""

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from core.backup import backup_vault, install_backup_job, restore_vault

# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    (root / "System" / "integrations").mkdir(parents=True)
    (root / "05-Areas" / "People").mkdir(parents=True)
    (root / "05-Areas" / "People" / "Ada_Lovelace.md").write_text("# Ada\n")
    (root / "docs").mkdir()
    (root / backup_vault.RUNBOOK_SOURCE).write_text("# Restoring a Dex vault\n")
    (root / ".env").write_text("ANTHROPIC_API_KEY=secret\n")
    (root / ".mcp.json").write_text("{}\n")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "bulk.js").write_text("x")
    return root


def write_config(vault, destination, backend="folder", extra=""):
    (vault / "System" / "integrations" / "config.yaml").write_text(
        "backup:\n"
        "  enabled: true\n"
        f"  backend: {backend}\n"
        f"  destination: {destination}\n"
        + extra
    )


def read_stamp(vault):
    return json.loads((vault / "System" / ".dex" / "backup-last-run.json").read_text())


def make_set(dest, stamp):
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{backup_vault.PREFIX}{stamp}.tar.gz").write_bytes(b"archive")
    (dest / f"{backup_vault.PREFIX}{stamp}.sha256").write_text("sums\n")


# --- retention maths --------------------------------------------------------

def test_prune_keeps_gfs_ladder_and_deletes_the_rest(tmp_path):
    dest = tmp_path / "backups"
    stamps = [
        "20260711-020000",  # newest: today
        "20260710-020000",  # yesterday
        "20260709-020000",  # older daily, outside daily=2
        "20260630-020000",  # previous ISO week and month
        "20260501-020000",  # older month, outside monthly=2
    ]
    for stamp in stamps:
        make_set(dest, stamp)
    backend = backup_vault.FolderBackend({"destination": str(dest)})
    pruned = backup_vault.prune(backend, {"daily": 2, "weekly": 1, "monthly": 2})

    assert sorted(pruned) == ["20260501-020000", "20260709-020000"]
    remaining = backend.list_sets()
    assert remaining == ["20260711-020000", "20260710-020000", "20260630-020000"]
    # a pruned set loses every file, not just the archive
    assert not list(dest.glob("*20260709*"))


def test_prune_never_deletes_the_newest_set_even_with_zero_retention(tmp_path):
    dest = tmp_path / "backups"
    make_set(dest, "20260711-020000")
    make_set(dest, "20260710-020000")
    backend = backup_vault.FolderBackend({"destination": str(dest)})
    pruned = backup_vault.prune(backend, {"daily": 0, "weekly": 0, "monthly": 0})
    assert pruned == ["20260710-020000"]
    assert backend.list_sets() == ["20260711-020000"]


def test_prune_never_deletes_an_unrecognised_stamp(tmp_path):
    dest = tmp_path / "backups"
    make_set(dest, "20260711-020000")
    make_set(dest, "not-a-timestamp")
    backend = backup_vault.FolderBackend({"destination": str(dest)})
    pruned = backup_vault.prune(backend, {"daily": 1, "weekly": 0, "monthly": 0})
    assert pruned == []
    assert (dest / f"{backup_vault.PREFIX}not-a-timestamp.tar.gz").exists()


# --- exclusion filter -------------------------------------------------------

@pytest.mark.parametrize("relative,expected", [
    (".env", True),
    (".env.local", True),
    (".mcp.json", True),
    ("node_modules/bulk.js", True),
    ("04-Projects/demo/node_modules/dep/index.js", True),
    (".venv/bin/python", True),
    ("System/backup/backup.log", True),
    (".claude/worktrees/scratch/file.md", True),
    ("05-Areas/People/Ada_Lovelace.md", False),
    ("System/user-profile.yaml", False),
    ("04-Projects/environments.md", False),
])
def test_excluded_paths(relative, expected):
    assert backup_vault.excluded(relative) is expected


def test_archive_excludes_secrets_and_bulk_but_keeps_notes(vault, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    artifacts = backup_vault.build_artifacts(vault, workdir, "20260711-020000")
    archive = artifacts[0]
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert f"{backup_vault.ARCNAME}/05-Areas/People/Ada_Lovelace.md" in names
    assert not any(".env" in name for name in names)
    assert not any(".mcp.json" in name for name in names)
    assert not any("node_modules" in name for name in names)


# --- the restore runbook travels inside the archive, not the release --------

def test_archive_carries_the_restore_runbook_for_a_bare_machine(vault, tmp_path):
    """The runbook must be readable from the archive alone, on a dead machine."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    artifacts = backup_vault.build_artifacts(vault, workdir, "20260711-020000")

    with tarfile.open(artifacts[0]) as tar:
        member = f"{backup_vault.ARCNAME}/{backup_vault.RUNBOOK_IN_ARCHIVE}"
        assert member in tar.getnames()
        assert tar.extractfile(member).read() == b"# Restoring a Dex vault\n"


def test_a_stale_vault_copy_does_not_duplicate_the_runbook(vault, tmp_path):
    """A vault restored from an older backup already has a copy at that path."""
    stale = vault / backup_vault.RUNBOOK_IN_ARCHIVE
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("# stale copy from an older restore\n")
    workdir = tmp_path / "work"
    workdir.mkdir()
    artifacts = backup_vault.build_artifacts(vault, workdir, "20260711-020000")

    member = f"{backup_vault.ARCNAME}/{backup_vault.RUNBOOK_IN_ARCHIVE}"
    with tarfile.open(artifacts[0]) as tar:
        assert tar.getnames().count(member) == 1
        assert tar.extractfile(member).read() == b"# Restoring a Dex vault\n"


def test_a_vault_without_the_runbook_still_backs_up_but_says_so(vault, tmp_path):
    (vault / backup_vault.RUNBOOK_SOURCE).unlink()
    workdir = tmp_path / "work"
    workdir.mkdir()
    warnings: list[str] = []
    artifacts = backup_vault.build_artifacts(vault, workdir, "20260711-020000",
                                             warnings=warnings)

    with tarfile.open(artifacts[0]) as tar:
        assert f"{backup_vault.ARCNAME}/05-Areas/People/Ada_Lovelace.md" in tar.getnames()
    assert any("restore runbook" in warning for warning in warnings)


def test_no_release_ever_ships_a_path_under_system_backup():
    """Releases are built from the tracked tree, and no released contract before
    v1.95 can classify System/backup/. Shipping anything there refuses the whole
    update for every existing install - it stranded v1.94.0 once already.
    """
    repo = Path(__file__).resolve().parents[2]
    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "System/backup"],
        capture_output=True, text=True, check=True).stdout.split()

    assert tracked == [], (
        "these paths would ship to users who cannot classify them: "
        f"{tracked}; keep the runbook in docs/ and let the backup engine write "
        "it into each archive instead")


# --- stamp on every outcome -------------------------------------------------

def test_failure_writes_stamp_with_the_error(vault):
    write_config(vault, destination="")  # deliberately unconfigured
    assert backup_vault.run_backup(vault) == 1
    stamp = read_stamp(vault)
    assert stamp["ok"] is False
    assert "backup.destination is not set" in stamp["error"]


def test_unknown_backend_fails_loudly_and_is_stamped(vault, tmp_path):
    write_config(vault, destination=str(tmp_path / "dest"), backend="carrier-pigeon")
    assert backup_vault.run_backup(vault) == 1
    stamp = read_stamp(vault)
    assert stamp["ok"] is False
    assert "unknown backup backend" in stamp["error"]


def test_successful_run_stores_set_and_writes_ok_stamp(vault, tmp_path):
    dest = tmp_path / "backups"
    write_config(vault, destination=str(dest))
    assert backup_vault.run_backup(vault) == 0
    stamp = read_stamp(vault)
    assert stamp["ok"] is True
    assert stamp["backend"] == "folder"
    archives = list(dest.glob(f"{backup_vault.PREFIX}*.tar.gz"))
    sidecars = list(dest.glob(f"{backup_vault.PREFIX}*.sha256"))
    assert len(archives) == 1 and len(sidecars) == 1
    assert not list(dest.glob("*.partial"))


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_git_vault_gets_a_verified_history_bundle(vault, tmp_path):
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
           "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    subprocess.run(["git", "init", "-q"], cwd=vault, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=vault, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=vault, check=True, env=env)
    dest = tmp_path / "backups"
    write_config(vault, destination=str(dest))
    assert backup_vault.run_backup(vault) == 0
    bundles = list(dest.glob(f"{backup_vault.PREFIX}*.bundle"))
    assert len(bundles) == 1
    sidecar = next(dest.glob(f"{backup_vault.PREFIX}*.sha256"))
    assert bundles[0].name in sidecar.read_text()


# --- backend selection and rclone degradation -------------------------------

def test_rclone_backend_fails_gracefully_when_binary_is_absent(monkeypatch):
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("rclone")
    monkeypatch.setattr(backup_vault.subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="not installed or not on PATH"):
        backup_vault.RcloneBackend({"remote": "b2:dex-backups"})


def test_rclone_backend_requires_a_remote(monkeypatch):
    monkeypatch.setattr(
        backup_vault.subprocess, "run",
        lambda *_a, **_k: pytest.fail("must not probe rclone without a remote"),
    )
    with pytest.raises(RuntimeError, match="backup.remote is not set"):
        backup_vault.RcloneBackend({"remote": ""})


def test_rclone_failure_during_a_run_is_stamped(vault, monkeypatch):
    write_config(vault, destination="", backend="rclone",
                 extra="  remote: b2:dex-backups\n")

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("rclone")
    monkeypatch.setattr(backup_vault.subprocess, "run", missing)
    assert backup_vault.run_backup(vault) == 1
    stamp = read_stamp(vault)
    assert stamp["ok"] is False
    assert "rclone" in stamp["error"]


# --- config parsing ---------------------------------------------------------

def test_config_without_yaml_module_still_parses(monkeypatch):
    text = (
        "enabled:\n"
        "  notion: false\n"
        "backup:\n"
        "  enabled: true\n"
        "  backend: folder  # a comment\n"
        "  destination: '/Volumes/Backup Drive/Dex Backups'\n"
        "  retention:\n"
        "    daily: 5\n"
        "    weekly: 2\n"
        "hooks:\n"
        "  meeting_prep:\n"
        "    use_notion: false\n"
    )
    block = backup_vault.parse_backup_block_without_yaml(text)
    assert block["enabled"] == "true"
    assert block["backend"] == "folder"
    assert block["destination"] == "/Volumes/Backup Drive/Dex Backups"
    assert block["retention"] == {"daily": "5", "weekly": "2"}


def test_load_config_honours_legacy_retention_days(vault):
    (vault / "System" / "integrations" / "config.yaml").write_text(
        "backup:\n"
        "  enabled: true\n"
        "  destination: /backups\n"
        "  retention_days: 9\n"
    )
    config = backup_vault.load_config(vault)
    assert config["retention"] == {"daily": 9, "weekly": 0, "monthly": 0}


def test_load_config_defaults_when_block_is_absent(vault):
    config = backup_vault.load_config(vault)
    assert config["enabled"] is False
    assert config["backend"] == "folder"
    assert config["retention"] == {"daily": 7, "weekly": 4, "monthly": 3}


# --- installer --------------------------------------------------------------

def test_build_plist_pins_interpreter_environment_and_schedule(tmp_path):
    payload = install_backup_job.build_plist(
        tmp_path, "/opt/example/bin/python3", hour=12, minute=30)
    assert payload["Label"] == "com.dex.vault-backup"
    assert payload["ProgramArguments"][0] == "/opt/example/bin/python3"
    assert payload["ProgramArguments"][1] == str(
        tmp_path / "core" / "backup" / "backup_vault.py")
    assert payload["EnvironmentVariables"]["VAULT_PATH"] == str(tmp_path)
    assert "/usr/bin" in payload["EnvironmentVariables"]["PATH"]
    assert payload["StartCalendarInterval"] == {"Hour": 12, "Minute": 30}
    assert payload["WorkingDirectory"] == str(tmp_path)


def test_installer_refuses_a_temporary_checkout(tmp_path):
    worktree = tmp_path / "worktrees" / "scratch"
    worktree.mkdir(parents=True)
    assert install_backup_job.looks_like_temporary_checkout(worktree)
    plain = tmp_path / "real-vault"
    plain.mkdir()
    assert not install_backup_job.looks_like_temporary_checkout(plain)
    (plain / ".git").write_text("gitdir: elsewhere\n")
    assert install_backup_job.looks_like_temporary_checkout(plain)


def test_installer_prints_cron_guidance_on_other_platforms(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(install_backup_job.sys, "platform", "linux")
    code = install_backup_job.main(["--vault", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 2
    assert "crontab -e" in out
    assert "Nothing was installed" in out


# --- restore tool -----------------------------------------------------------

@pytest.fixture
def backed_up(vault, tmp_path):
    dest = tmp_path / "backups"
    write_config(vault, destination=str(dest))
    assert backup_vault.run_backup(vault) == 0
    stamp = read_stamp(vault)["set"]
    return vault, dest, stamp


def test_verify_passes_on_an_intact_set(backed_up):
    _vault, dest, stamp = backed_up
    findings = restore_vault.verify_set(dest, stamp)
    assert any("checksum matches" in finding for finding in findings)


def test_verify_detects_a_damaged_archive(backed_up):
    _vault, dest, stamp = backed_up
    archive = dest / f"{backup_vault.PREFIX}{stamp}.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"corruption")
    with pytest.raises(restore_vault.RestoreError, match="does not match"):
        restore_vault.verify_set(dest, stamp)


def test_test_mode_extracts_to_a_temporary_folder_only(backed_up, capsys):
    vault_root, dest, _stamp = backed_up
    code = restore_vault.main(["test", "--source", str(dest),
                              "--vault", str(vault_root)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Test restore succeeded" in out
    assert "Nothing was changed" in out


def test_restore_refuses_the_live_vault_and_non_empty_targets(backed_up, tmp_path):
    vault_root, dest, stamp = backed_up
    with pytest.raises(restore_vault.RestoreError, match="live vault"):
        restore_vault.restore(dest, stamp, vault_root, vault_root)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "existing.txt").write_text("keep me")
    with pytest.raises(restore_vault.RestoreError, match="not empty"):
        restore_vault.restore(dest, stamp, occupied, vault_root)
    assert (occupied / "existing.txt").read_text() == "keep me"


def test_restore_extracts_into_a_fresh_folder(backed_up, tmp_path, capsys):
    vault_root, dest, stamp = backed_up
    target = tmp_path / "restored"
    restore_vault.restore(dest, stamp, target, vault_root)
    restored_note = (target / backup_vault.ARCNAME / "05-Areas" / "People"
                     / "Ada_Lovelace.md")
    assert restored_note.read_text() == "# Ada\n"
    assert not (target / backup_vault.ARCNAME / ".env").exists()


def test_pick_set_prefers_newest_and_validates_requests(tmp_path):
    dest = tmp_path / "backups"
    make_set(dest, "20260710-020000")
    make_set(dest, "20260711-020000")
    assert restore_vault.pick_set(dest, None) == "20260711-020000"
    assert restore_vault.pick_set(dest, "20260710-020000") == "20260710-020000"
    with pytest.raises(restore_vault.RestoreError, match="not found"):
        restore_vault.pick_set(dest, "20990101-000000")


# --- degradation, cleanup, and restore honesty (review follow-up) -----------

@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_unbundleable_history_still_backs_up_the_notes_and_warns(vault, tmp_path):
    """A git repo with no commits must not cost the user their notes backup."""
    subprocess.run(["git", "init", "-q"], cwd=vault, check=True)
    dest = tmp_path / "backups"
    write_config(vault, destination=str(dest))

    assert backup_vault.run_backup(vault) == 0
    stamp = read_stamp(vault)
    assert stamp["ok"] is True
    assert len(list(dest.glob(f"{backup_vault.PREFIX}*.tar.gz"))) == 1
    assert not list(dest.glob(f"{backup_vault.PREFIX}*.bundle"))
    assert any("version history could not be bundled" in warning
               for warning in stamp["warnings"])
    assert "WARNING" in (vault / "System" / "backup" / "backup.log").read_text()


def test_a_failed_store_leaves_no_half_written_set_behind(vault, tmp_path, monkeypatch):
    """A set missing its checksum file would be picked as newest and be unusable.

    The previous good set must also survive untouched: a run that cannot
    finish is exactly when the last successful copy matters most.
    """
    dest = tmp_path / "backups"
    write_config(vault, destination=str(dest))
    good = "20260101-000000"
    dest.mkdir(parents=True)
    for artifact in backup_vault.build_artifacts(vault, tmp_path, good):
        shutil.copy2(artifact, dest / artifact.name)

    real_copyfileobj = backup_vault.shutil.copyfileobj
    calls = {"n": 0}

    def fail_on_the_sidecar(src, dst, length=0):
        calls["n"] += 1
        if calls["n"] == 2:  # archive lands, checksum file does not
            raise OSError(28, "No space left on device")
        return real_copyfileobj(src, dst, length)

    monkeypatch.setattr(backup_vault.shutil, "copyfileobj", fail_on_the_sidecar)
    assert backup_vault.run_backup(vault) == 1

    assert read_stamp(vault)["ok"] is False
    assert not list(dest.glob("*.partial")), "orphaned partial files accumulate"
    assert restore_vault.pick_set(dest, None) == good, \
        "a half-stored set must not become the one a restore picks"
    restore_vault.verify_set(dest, good)


def test_links_pointing_outside_the_vault_are_reported_at_backup_time(vault, tmp_path):
    """The restore tool cannot unpack these; learning that in a crisis is too late."""
    (tmp_path / "outside.md").write_text("elsewhere")
    (vault / "linked.md").symlink_to(tmp_path / "outside.md")
    dest = tmp_path / "backups"
    write_config(vault, destination=str(dest))

    assert backup_vault.run_backup(vault) == 0
    warnings = read_stamp(vault)["warnings"]
    assert any("point outside it" in warning for warning in warnings)
    assert any("linked.md" in warning for warning in warnings)


def test_a_vault_without_stray_links_records_no_warnings(vault, tmp_path):
    write_config(vault, destination=str(tmp_path / "backups"))
    assert backup_vault.run_backup(vault) == 0
    assert read_stamp(vault)["warnings"] == []


def test_restore_explains_an_unusable_archive_and_removes_the_partial_folder(
        vault, tmp_path):
    """Never leave a half-unpacked folder that reads as a finished restore."""
    (tmp_path / "outside.md").write_text("elsewhere")
    (vault / "linked.md").symlink_to(tmp_path / "outside.md")
    dest = tmp_path / "backups"
    write_config(vault, destination=str(dest))
    assert backup_vault.run_backup(vault) == 0
    stamp = read_stamp(vault)["set"]

    target = tmp_path / "restored"
    with pytest.raises(restore_vault.RestoreError, match="could not be unpacked"):
        restore_vault.restore(dest, stamp, target, vault)
    assert not target.exists(), "a half-unpacked restore target was left behind"


def test_test_mode_reports_an_unusable_archive_as_a_plain_stop(vault, tmp_path, capsys):
    (tmp_path / "outside.md").write_text("elsewhere")
    (vault / "linked.md").symlink_to(tmp_path / "outside.md")
    dest = tmp_path / "backups"
    write_config(vault, destination=str(dest))
    assert backup_vault.run_backup(vault) == 0

    report_path = tmp_path / "test-report.json"
    code = restore_vault.main(["test", "--source", str(dest),
                               "--vault", str(vault),
                               "--report", str(report_path)])
    out = capsys.readouterr().out
    assert code == 1, "an unrestorable set must not report a successful test"
    assert "Stopped:" in out and "could not be unpacked" in out
    assert "Traceback" not in out
    report = json.loads(report_path.read_text())
    assert report["status"] == "stopped"
    assert report["error"]["code"] == "unsafe_link"


def test_a_damaged_newest_set_points_at_the_newest_intact_one(vault, tmp_path):
    dest = tmp_path / "backups"
    write_config(vault, destination=str(dest))
    assert backup_vault.run_backup(vault) == 0
    older = read_stamp(vault)["set"]
    for artifact in sorted(dest.glob(f"{backup_vault.PREFIX}{older}.*")):
        newer = artifact.with_name(artifact.name.replace(older, "29991231-235959"))
        shutil.copy2(artifact, newer)
    damaged = dest / f"{backup_vault.PREFIX}29991231-235959.tar.gz"
    damaged.write_bytes(damaged.read_bytes() + b"rot")

    code = restore_vault.main(["verify", "--source", str(dest), "--vault", str(vault)])
    assert code == 1
    error = _capture_restore_error(dest, "29991231-235959")
    assert f"--set {older}" in error


def _capture_restore_error(dest, stamp):
    try:
        restore_vault.verify_set(dest, stamp)
    except restore_vault.RestoreError as first:
        return str(restore_vault._with_fallback_hint(first, dest, stamp))
    raise AssertionError("expected the damaged set to fail verification")


def test_escaping_link_detection_accepts_links_that_stay_inside(vault, tmp_path):
    inside = vault / "05-Areas" / "People" / "Ada_Lovelace.md"
    (vault / "shortcut.md").symlink_to(Path("05-Areas/People/Ada_Lovelace.md"))
    assert inside.exists()
    dest = tmp_path / "backups"
    write_config(vault, destination=str(dest))
    assert backup_vault.run_backup(vault) == 0
    assert read_stamp(vault)["warnings"] == []


# --- hardened, member-by-member safe extraction ------------------------------
#
# A restore runs on the least-controlled machine in the system (a new one)
# against the least-controlled input (an archive that sat in a synced
# folder). These tests craft archives directly so the dangerous shapes never
# depend on what the honest backup engine would produce.

def _add_dir(tar, name, mode=0o755):
    info = tarfile.TarInfo(name if name.endswith("/") else name + "/")
    info.type = tarfile.DIRTYPE
    info.mode = mode
    info.mtime = 1700000000
    tar.addfile(info)


def _add_file(tar, name, data=b"x", mode=0o644, typ=tarfile.REGTYPE):
    info = tarfile.TarInfo(name)
    info.type = typ
    info.mode = mode
    info.mtime = 1700000000
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _add_sym(tar, name, target, mode=0o777):
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.mode = mode
    info.mtime = 1700000000
    info.linkname = target
    tar.addfile(info)


def _add_hard(tar, name, target):
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE
    info.mode = 0o644
    info.mtime = 1700000000
    info.linkname = target
    tar.addfile(info)


def _crafted_archive(tmp_path, build):
    archive = tmp_path / "crafted.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        build(tar)
    return archive


def test_safe_extract_stages_files_links_and_modes_then_verifies(tmp_path):
    def build(tar):
        _add_dir(tar, "dex-vault/")
        _add_dir(tar, "dex-vault/notes/")
        _add_file(tar, "dex-vault/notes/a.md", b"# Ada\n", mode=0o600)
        _add_file(tar, "dex-vault/run", b"#!/bin/sh\n", mode=0o755)
        _add_sym(tar, "dex-vault/alias", "notes/a.md")
        _add_file(tar, "dex-vault/orig", b"same bytes\n")
        _add_hard(tar, "dex-vault/twin", "dex-vault/orig")

    archive = _crafted_archive(tmp_path, build)
    staging = tmp_path / "stage"
    staging.mkdir()
    result = restore_vault.safe_extract(archive, staging)

    assert result["counts"] == {"members": 7, "files": 3, "directories": 2,
                                "symlinks": 1, "hardlinks": 1}
    note = staging / "dex-vault" / "notes" / "a.md"
    assert note.read_bytes() == b"# Ada\n"
    assert note.stat().st_mode & 0o777 == 0o600
    run = staging / "dex-vault" / "run"
    assert run.stat().st_mode & 0o100 == 0o100
    alias = staging / "dex-vault" / "alias"
    assert alias.is_symlink() and os.readlink(alias) == "notes/a.md"
    orig = staging / "dex-vault" / "orig"
    twin = staging / "dex-vault" / "twin"
    assert (orig.stat().st_dev, orig.stat().st_ino) \
        == (twin.stat().st_dev, twin.stat().st_ino)
    assert result["files_verified"] == 3
    note_record = next(e for e in result["entries"]
                       if e["path"] == "dex-vault/notes/a.md")
    assert note_record["sha256"] == backup_vault.file_digest(note)
    assert result["bytes"] == len(b"# Ada\n") + len(b"#!/bin/sh\n") \
        + len(b"same bytes\n")


def _build_parent_traversal(tar):
    _add_dir(tar, "dex-vault/")
    _add_file(tar, "dex-vault/../evil", b"x")


def _build_absolute_name(tar):
    _add_file(tar, "/dex-vault/evil", b"x")


def _build_backslash_name(tar):
    _add_file(tar, "dex-vault/back\\slash", b"x")


def _build_outside_prefix(tar):
    _add_dir(tar, "dex-vault/")
    _add_file(tar, "other/x", b"x")


def _build_char_device(tar):
    _add_dir(tar, "dex-vault/")
    info = tarfile.TarInfo("dex-vault/null")
    info.type = tarfile.CHRTYPE
    info.mode = 0o20666
    info.devmajor, info.devminor = 0o1, 0o3
    info.mtime = 1700000000
    tar.addfile(info)


def _build_fifo(tar):
    _add_dir(tar, "dex-vault/")
    info = tarfile.TarInfo("dex-vault/pipe")
    info.type = tarfile.FIFOTYPE
    info.mode = 0o644
    info.mtime = 1700000000
    tar.addfile(info)


def _build_absolute_symlink(tar):
    _add_dir(tar, "dex-vault/")
    _add_sym(tar, "dex-vault/link", "/etc/passwd")


def _build_escaping_symlink(tar):
    _add_dir(tar, "dex-vault/")
    _add_dir(tar, "dex-vault/sub/")
    _add_sym(tar, "dex-vault/sub/link", "../../../outside")


def _build_backslash_symlink(tar):
    _add_dir(tar, "dex-vault/")
    _add_sym(tar, "dex-vault/link", "..\\evil")


def _build_forward_hardlink(tar):
    _add_dir(tar, "dex-vault/")
    _add_hard(tar, "dex-vault/h", "dex-vault/missing")


def _build_duplicate(tar):
    _add_dir(tar, "dex-vault/")
    _add_file(tar, "dex-vault/same", b"one")
    _add_file(tar, "dex-vault/same", b"two")


def _build_file_under_file(tar):
    _add_dir(tar, "dex-vault/")
    _add_file(tar, "dex-vault/a", b"x")
    _add_file(tar, "dex-vault/a/b", b"x")


def _build_through_symlinked_dir(tar):
    _add_dir(tar, "dex-vault/")
    _add_sym(tar, "dex-vault/portal", ".")
    _add_file(tar, "dex-vault/portal/evil", b"x")


def _build_top_level_file(tar):
    _add_file(tar, "dex-vault", b"x")


@pytest.mark.parametrize("build,code", [
    (_build_parent_traversal, "unsafe_member_name"),
    (_build_absolute_name, "unsafe_member_name"),
    (_build_backslash_name, "unsafe_member_name"),
    (_build_outside_prefix, "unsafe_member_name"),
    (_build_top_level_file, "unsafe_member_name"),
    (_build_char_device, "unsafe_member_type"),
    (_build_fifo, "unsafe_member_type"),
    (_build_absolute_symlink, "unsafe_link"),
    (_build_escaping_symlink, "unsafe_link"),
    (_build_backslash_symlink, "unsafe_link"),
    (_build_forward_hardlink, "unsafe_link"),
    (_build_duplicate, "duplicate_member"),
    (_build_file_under_file, "conflicting_member"),
    (_build_through_symlinked_dir, "conflicting_member"),
])
def test_unsafe_archives_are_rejected_and_stage_is_emptied(tmp_path, build, code):
    archive = _crafted_archive(tmp_path, build)
    staging = tmp_path / "stage"
    staging.mkdir()
    with pytest.raises(restore_vault.RestoreError) as raised:
        restore_vault.safe_extract(archive, staging)
    assert raised.value.code == code
    assert "could not be unpacked" in str(raised.value)
    # Nothing landed in staging, and nothing escaped next to it.
    assert list(staging.iterdir()) == []
    assert not (tmp_path / "evil").exists()
    assert not (tmp_path / "outside").exists()


def test_an_inside_symlink_and_parent_dot_target_are_accepted(tmp_path):
    def build(tar):
        _add_dir(tar, "dex-vault/")
        _add_dir(tar, "dex-vault/notes/")
        _add_file(tar, "dex-vault/notes/a.md", b"# Ada\n")
        # lexical . and ../notes stay inside the staging root
        _add_sym(tar, "dex-vault/notes/here", ".")
        _add_sym(tar, "dex-vault/again", "notes/../notes/a.md")

    archive = _crafted_archive(tmp_path, build)
    staging = tmp_path / "stage"
    staging.mkdir()
    result = restore_vault.safe_extract(archive, staging)
    assert result["counts"]["symlinks"] == 2


def test_staged_tampering_fails_verification(tmp_path):
    def build(tar):
        _add_dir(tar, "dex-vault/")
        _add_file(tar, "dex-vault/note.md", b"original\n")

    archive = _crafted_archive(tmp_path, build)
    staging = tmp_path / "stage"
    staging.mkdir()
    planned = restore_vault.plan_extraction(archive)
    manifest, _counts, _bytes, _warnings = restore_vault._materialize(
        archive, staging, planned)
    (staging / "dex-vault" / "note.md").write_text("tampered\n")
    with pytest.raises(restore_vault.RestoreError) as raised:
        restore_vault._verify_staged_tree(staging, manifest)
    assert raised.value.code == "verify_mismatch"


def test_an_extra_file_in_staging_fails_verification(tmp_path):
    def build(tar):
        _add_dir(tar, "dex-vault/")
        _add_file(tar, "dex-vault/note.md", b"original\n")

    archive = _crafted_archive(tmp_path, build)
    staging = tmp_path / "stage"
    staging.mkdir()
    planned = restore_vault.plan_extraction(archive)
    manifest, _counts, _bytes, _warnings = restore_vault._materialize(
        archive, staging, planned)
    (staging / "intruder").write_text("not from the archive")
    with pytest.raises(restore_vault.RestoreError, match="not accounted for"):
        restore_vault._verify_staged_tree(staging, manifest)


def test_restore_keeps_links_that_stay_inside_and_writes_an_ok_report(
        vault, tmp_path):
    (vault / "shortcut.md").symlink_to("05-Areas/People/Ada_Lovelace.md")
    dest = tmp_path / "backups"
    write_config(vault, destination=str(dest))
    assert backup_vault.run_backup(vault) == 0
    stamp = read_stamp(vault)["set"]

    target = tmp_path / "restored"
    restore_vault.restore(dest, stamp, target, vault)
    link = target / backup_vault.ARCNAME / "shortcut.md"
    assert link.is_symlink()
    assert os.readlink(link) == "05-Areas/People/Ada_Lovelace.md"
    assert link.read_text() == "# Ada\n"

    report_path = target.parent / f".{target.name}.restore-report-{stamp}.json"
    report = json.loads(report_path.read_text())
    assert report["schema"] == restore_vault.REPORT_SCHEMA
    assert report["status"] == "ok"
    assert report["error"] is None
    assert report["files_verified"] >= 1
    shortcut = next(e for e in report["entries"]
                    if e["path"].endswith("shortcut.md"))
    assert shortcut["type"] == "symlink"
    assert shortcut["target"] == "05-Areas/People/Ada_Lovelace.md"


def test_an_unsafe_set_leaves_no_target_and_writes_a_stopped_report(
        vault, tmp_path):
    (tmp_path / "outside.md").write_text("elsewhere")
    (vault / "linked.md").symlink_to(tmp_path / "outside.md")
    dest = tmp_path / "backups"
    write_config(vault, destination=str(dest))
    assert backup_vault.run_backup(vault) == 0
    stamp = read_stamp(vault)["set"]

    target = tmp_path / "restored"
    with pytest.raises(restore_vault.RestoreError, match="could not be unpacked"):
        restore_vault.restore(dest, stamp, target, vault)
    assert not target.exists()
    leftover = [p for p in target.parent.glob(f".{target.name}.restore-*")
                if not p.name.endswith(".json")]
    assert leftover == [], "a staging folder was left behind"
    report_path = target.parent / f".{target.name}.restore-report-{stamp}.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "stopped"
    assert report["error"]["code"] == "unsafe_link"


def test_restore_publishes_into_an_empty_existing_folder(backed_up, tmp_path):
    vault_root, dest, stamp = backed_up
    target = tmp_path / "restored"
    target.mkdir()
    restore_vault.restore(dest, stamp, target, vault_root)
    assert (target / backup_vault.ARCNAME / "05-Areas" / "People"
            / "Ada_Lovelace.md").read_text() == "# Ada\n"


def test_test_mode_writes_a_machine_report_only_when_asked(
        backed_up, tmp_path, capsys):
    vault_root, dest, _stamp = backed_up
    report_path = tmp_path / "report.json"
    code = restore_vault.main(["test", "--source", str(dest),
                               "--vault", str(vault_root),
                               "--report", str(report_path)])
    assert code == 0
    report = json.loads(report_path.read_text())
    assert report["mode"] == "test"
    assert report["status"] == "ok"
    assert report["target"] is None
    assert report["files_verified"] >= 1
    assert "Test restore succeeded" in capsys.readouterr().out


def test_a_truncated_archive_is_rejected_and_stage_is_emptied(tmp_path):
    def build(tar):
        _add_dir(tar, "dex-vault/")
        _add_file(tar, "dex-vault/note.md", b"original bytes\n" * 100)

    archive = _crafted_archive(tmp_path, build)
    archive.write_bytes(archive.read_bytes()[:-120])  # cut the gzip stream
    staging = tmp_path / "stage"
    staging.mkdir()
    with pytest.raises(restore_vault.RestoreError) as raised:
        restore_vault.safe_extract(archive, staging)
    assert raised.value.code == "archive_error"
    assert list(staging.iterdir()) == []


def test_bundle_is_staged_alongside_and_checksum_verified(tmp_path):
    def build(tar):
        _add_dir(tar, "dex-vault/")
        _add_file(tar, "dex-vault/note.md", b"x")

    archive = _crafted_archive(tmp_path, build)
    bundle = tmp_path / "dex-vault-stamp.bundle"
    bundle.write_bytes(b"git bundle bytes\n")
    staging = tmp_path / "stage"
    staging.mkdir()
    result = restore_vault.safe_extract(archive, staging, bundle)
    assert (staging / bundle.name).read_bytes() == b"git bundle bytes\n"
    assert result["bundle"]["name"] == bundle.name
    record = next(e for e in result["entries"] if e["path"] == bundle.name)
    assert record["role"] == "history-bundle"
    assert record["sha256"] == backup_vault.file_digest(bundle)


def _build_absolute_hardlink(tar):
    _add_dir(tar, "dex-vault/")
    _add_hard(tar, "dex-vault/h", "/etc/passwd")


def test_an_absolute_hardlink_target_is_rejected(tmp_path):
    archive = _crafted_archive(tmp_path, _build_absolute_hardlink)
    staging = tmp_path / "stage"
    staging.mkdir()
    with pytest.raises(restore_vault.RestoreError) as raised:
        restore_vault.safe_extract(archive, staging)
    assert raised.value.code == "unsafe_link"


def test_a_publish_failure_writes_a_stopped_report(backed_up, tmp_path,
                                                   monkeypatch):
    vault_root, dest, stamp = backed_up
    target = tmp_path / "restored"

    def refuse_replace(src, dst):
        # Only block the staging -> target rename; let the report's atomic
        # write (.partial swap) through.
        if ".restore-report" not in str(src):
            raise OSError("simulated rename failure")
        return real_replace(src, dst)

    real_replace = restore_vault.os.replace
    monkeypatch.setattr(restore_vault.os, "replace", refuse_replace)
    with pytest.raises(restore_vault.RestoreError) as raised:
        restore_vault.restore(dest, stamp, target, vault_root)
    assert raised.value.code == "publish_failed"
    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.restore-*")) or \
        all(p.suffix == ".json" for p in target.parent.glob(
            f".{target.name}.restore-*"))
    report_path = target.parent / f".{target.name}.restore-report-{stamp}.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "stopped"
    assert report["error"]["code"] == "publish_failed"


def test_restore_without_to_prints_a_plain_stop(backed_up, capsys):
    vault_root, dest, _stamp = backed_up
    code = restore_vault.main(["restore", "--source", str(dest),
                               "--vault", str(vault_root)])
    assert code == 1
    assert "needs --to" in capsys.readouterr().out


# --- secrets must never reach a synced folder -------------------------------

def test_backup_excludes_everything_the_contract_hard_denies():
    """The engine's exclusion list must not drift behind the repo's authority.

    core.portable_contract.HARD_DENY_PATTERNS is what this repo means by "a
    credential". backup_vault duplicates the secret half of it (deliberately:
    the engine stays stdlib-only so a scheduled run cannot die on an import),
    so this test is what stops the copy going stale. Adding a deny pattern
    without teaching the backup about it fails here rather than in someone's
    cloud folder.
    """
    from core import portable_contract

    # .git is denied for write-safety, not secrecy, and is intentionally kept:
    # restoring a working repository is the point of the archive.
    representative = {
        ".env": ".env",
        ".env.*": ".env.production",
        "System/credentials": "System/credentials",
        "System/credentials/*": "System/credentials/todoist.json",
        "*token.json": "System/.gmail-oauth-token.json",
        "*.key": "04-Projects/deploy/id_rsa.key",
        "*.pem": "05-Areas/certs/client.pem",
    }
    covered = {".git", ".git/*"}
    unhandled = [pattern for pattern in portable_contract.HARD_DENY_PATTERNS
                 if pattern not in representative and pattern not in covered]
    assert unhandled == [], (
        "new hard-deny patterns exist that this test does not pin; add them "
        "to backup_vault.GLOB_EXCLUDES/ANCHORED_EXCLUDES and to this mapping")

    for pattern, sample in representative.items():
        assert backup_vault.excluded(sample) is True, \
            f"{sample} (hard-denied by {pattern}) would be uploaded in a backup"


def test_a_real_oauth_token_never_reaches_the_archive(vault, tmp_path):
    """The Google Workspace setup writes this exact file into the vault."""
    (vault / "System" / ".gmail-oauth-token.json").write_text('{"refresh_token": "x"}')
    (vault / "System" / "credentials").mkdir()
    (vault / "System" / "credentials" / "todoist.json").write_text('{"api_key": "x"}')
    (vault / "id_rsa.key").write_text("PRIVATE KEY")

    artifacts = backup_vault.build_artifacts(vault, tmp_path, "20260711-020000")
    with tarfile.open(artifacts[0]) as tar:
        names = tar.getnames()

    assert f"{backup_vault.ARCNAME}/05-Areas/People/Ada_Lovelace.md" in names
    for secret in ("gmail-oauth-token.json", "credentials", "id_rsa.key"):
        assert not any(secret in name for name in names), \
            f"{secret} was archived and would sync to the user's cloud folder"


def test_ordinary_notes_that_merely_look_secret_are_still_backed_up(vault, tmp_path):
    """The globs match file shapes, not topics: notes about keys are notes."""
    (vault / "04-Projects" ).mkdir(exist_ok=True)
    (vault / "04-Projects" / "API_keys_rotation_plan.md").write_text("# plan\n")
    artifacts = backup_vault.build_artifacts(vault, tmp_path, "20260711-020000")
    with tarfile.open(artifacts[0]) as tar:
        names = tar.getnames()
    assert f"{backup_vault.ARCNAME}/04-Projects/API_keys_rotation_plan.md" in names
