#!/usr/bin/env python3
"""Verify, test-restore, or restore a vault backup set.

Usage:
    python3 core/backup/restore_vault.py verify  [--set STAMP] [--source DIR]
    python3 core/backup/restore_vault.py test    [--set STAMP] [--source DIR]
                                                 [--report FILE]
    python3 core/backup/restore_vault.py restore --to DIR [--set STAMP]
                                                 [--source DIR] [--report FILE]

verify   recompute the checksums against the .sha256 sidecar and, when a
         history bundle is present, run `git bundle verify` on it.
test     verify, then fully extract the archive into a throwaway temporary
         folder, count what came out, and delete the extraction. Proves a
         restore works without touching anything.
restore  verify, then extract into a folder you choose. The target must be
         empty or absent; this tool never overwrites the live vault or any
         existing files. Reconnecting keys, schedules, and permissions is a
         separate, deliberate step: see docs/backup-restore.md.

Extraction never delegates to TarFile.extractall and never relies on a
runtime-provided tar filter: an explicit, member-by-member extractor runs
identically on every supported Python version. It refuses absolute or
path-traversing names, device/FIFO entries, links that leave the restore
folder, duplicate or type-conflicting entries, and any unexpected member
type. Everything lands in a private staging folder first; each file is
fsynced, then the whole tree is re-read from disk and checksum-verified
against the per-entry manifest derived from the (already sidecar-verified)
archive, and only then is the staging folder atomically renamed to the
target. restore mode always writes a machine-readable JSON report next to
the target (or at --report); test mode writes one only with --report.

--source defaults to the folder backend's configured destination. For the
rclone backend, first copy one set down to a local folder
(`rclone copy remote:path/dex-vault-<stamp>.* /some/folder/`) and point
--source at it; this tool stays honest by not reaching into the network.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zlib
from datetime import datetime
from pathlib import Path

try:
    from core.backup.backup_vault import (PREFIX, ARCNAME, file_digest,
                                          load_config, resolve_vault_root)
except ImportError:  # invoked by file path: put the vault root on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from core.backup.backup_vault import (PREFIX, ARCNAME, file_digest,
                                          load_config, resolve_vault_root)

REPORT_SCHEMA = "dex-restore-report/1"
# Old sets carry no embedded manifest: the manifest is derived from the
# archive members themselves. The archive's own bytes were already proven
# against the .sha256 sidecar, so the trust chain is sidecar -> archive ->
# derived manifest -> bytes re-read from disk. This keeps every existing
# backup restorable on a bare machine.
MANIFEST_SOURCE = "derived-from-archive"
# Truncated/corrupt payloads surface as several distinct exception types.
# gzip.BadGzipFile is itself an OSError, so it must be named explicitly
# *before* the disk-IO handler or a corrupt set would be reported as a full
# disk.
_ARCHIVE_ERRORS = (tarfile.TarError, EOFError, gzip.BadGzipFile, zlib.error)
_CHUNK = 1024 * 1024
_DRIVE_RE = re.compile(r"^[A-Za-z]:")

_FILE_TYPES = frozenset({tarfile.REGTYPE, tarfile.AREGTYPE})
_ALLOWED_TYPES = _FILE_TYPES | {tarfile.DIRTYPE, tarfile.SYMTYPE,
                                tarfile.LNKTYPE}
_SPECIAL_TYPE_NAMES = {
    tarfile.CHRTYPE: "character device",
    tarfile.BLKTYPE: "block device",
    tarfile.FIFOTYPE: "named pipe (FIFO)",
    tarfile.CONTTYPE: "contiguous file",
}
_KIND_TO_COUNT = {"f": "files", "d": "directories",
                  "s": "symlinks", "h": "hardlinks"}


class RestoreError(RuntimeError):
    """A condition that must stop the run with a plain explanation.

    ``code`` is a stable, machine-readable slug (it lands in the JSON
    report's error block); the prose message stays human-first.
    """

    def __init__(self, message: str, code: str = "restore_failed"):
        super().__init__(message)
        self.code = code


def resolve_source(vault: Path, override: str | None) -> Path:
    if override:
        source = Path(override).expanduser()
        if not source.is_dir():
            raise RestoreError(f"--source {source} is not a folder")
        return source
    config = load_config(vault)
    if config["backend"] != "folder" or not config["destination"]:
        raise RestoreError(
            "No local backup folder is configured. For the rclone backend, "
            "copy one backup set to a local folder first and pass --source.")
    source = Path(config["destination"]).expanduser()
    if not source.is_dir():
        raise RestoreError(f"The configured backup folder {source} does not exist")
    return source


def pick_set(source: Path, requested: str | None) -> str:
    stamps = sorted({p.name[len(PREFIX):-len(".tar.gz")]
                     for p in source.glob(f"{PREFIX}*.tar.gz")}, reverse=True)
    if not stamps:
        raise RestoreError(f"No backup sets found in {source}")
    if requested is None:
        return stamps[0]
    if requested not in stamps:
        raise RestoreError(f"Set {requested} not found in {source}. "
                           f"Available: {', '.join(stamps)}")
    return requested


def verify_set(source: Path, stamp: str) -> list[str]:
    """Return plain-language findings; raise RestoreError on any mismatch."""
    findings: list[str] = []
    sidecar = source / f"{PREFIX}{stamp}.sha256"
    if not sidecar.exists():
        raise RestoreError(f"The checksum file {sidecar.name} is missing; "
                           "this set cannot be proven intact")
    checked: set[str] = set()
    for line in sidecar.read_text().splitlines():
        expected, _, name = line.strip().partition("  ")
        if not name:
            continue
        checked.add(name)
        artifact = source / name
        if not artifact.exists():
            raise RestoreError(f"{name} is listed in the checksum file but missing")
        actual = file_digest(artifact)
        if actual != expected:
            raise RestoreError(f"{name} does not match its recorded checksum; "
                               "the copy in storage is damaged")
        findings.append(f"{name}: checksum matches")
    # The sidecar names the files it covers, so a sidecar carried over from a
    # different set (a copy, a rename, a sync conflict) would otherwise verify
    # that other set's files and report this one intact without ever reading
    # the archive about to be unpacked.
    archive_name = f"{PREFIX}{stamp}.tar.gz"
    if archive_name not in checked:
        raise RestoreError(
            f"{sidecar.name} does not cover {archive_name}, so the archive "
            "this set would restore has not been checked at all. The checksum "
            "file belongs to a different set; do not rely on this copy.")
    bundle = source / f"{PREFIX}{stamp}.bundle"
    if bundle.exists():
        if bundle.name not in checked:
            raise RestoreError(
                f"{bundle.name} is present but {sidecar.name} does not cover "
                "it, so the version history in this set is unchecked.")
        git = shutil.which("git")
        if git is None:
            findings.append(f"{bundle.name}: present, but git is not installed "
                            "so the history could not be verified here")
        else:
            result = subprocess.run([git, "bundle", "verify", str(bundle)],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                raise RestoreError(f"git bundle verify failed for {bundle.name}: "
                                   f"{(result.stderr or result.stdout).strip()[:300]}")
            findings.append(f"{bundle.name}: history verified as complete")
    else:
        findings.append("No history bundle in this set (the vault had no "
                        "version history when it was taken)")
    return findings


def _with_fallback_hint(error: RestoreError, source: Path,
                        failed: str) -> RestoreError:
    """Add the newest set that does verify, when the requested one does not."""
    others = [s for s in sorted(
        {p.name[len(PREFIX):-len(".tar.gz")]
         for p in source.glob(f"{PREFIX}*.tar.gz")}, reverse=True)
        if s != failed]
    for candidate in others:
        try:
            verify_set(source, candidate)
        except RestoreError:
            continue
        return RestoreError(
            f"{error} The most recent set that does check out intact is "
            f"{candidate}; add --set {candidate} to use it.")
    if others:
        return RestoreError(f"{error} No older set in {source} checks out "
                            "intact either.")
    return error


def _one_line(error: BaseException) -> str:
    text = str(error).strip() or error.__class__.__name__
    return " ".join(text.split())[:300]


# --- safe extraction: plan, stage, verify, publish --------------------------
#
# Why this exists: TarFile.extractall only grew a safe filter in Python 3.12.
# A restore is the one moment when the Python runtime and the archive's
# provenance are least under anyone's control (a brand-new machine, a synced
# folder), so "use the filter when present, fall back to raw extractall" was
# the wrong shape: the fallback writes absolute paths, traversal entries,
# links, and device nodes wherever the archive says. The extractor below
# makes every decision itself and behaves identically on every runtime.

def _validated_parts(raw_name: str) -> list[str]:
    """Split one archive member name into safe path parts or refuse.

    Rules, applied lexically so behaviour does not depend on the platform
    extracting the archive: non-empty, no NUL bytes or backslashes (a path
    separator on Windows and an ambiguity everywhere else), not absolute
    (leading slash or a drive letter), no "."/".." components, and everything
    must sit under the single top-level folder the backup engine writes.
    """
    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise RestoreError(
            f"archive entry {raw_name!r} has an unsafe name (empty, or it "
            "contains a backslash or control byte)", "unsafe_member_name")
    if raw_name.startswith("/") or _DRIVE_RE.match(raw_name):
        raise RestoreError(
            f"archive entry {raw_name!r} is an absolute path; entries must "
            f"live inside the {ARCNAME}/ folder", "unsafe_member_name")
    parts = raw_name.rstrip("/").split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise RestoreError(
            f"archive entry {raw_name!r} contains an empty, current-dir, or "
            "parent-dir component, which could write outside the restore "
            "folder", "unsafe_member_name")
    if parts[0] != ARCNAME:
        raise RestoreError(
            f"archive entry {raw_name!r} sits outside the archive's single "
            f"top-level folder ({ARCNAME}/)", "unsafe_member_name")
    return parts


def _link_is_absolute(linkname: str) -> bool:
    return (not linkname or "\x00" in linkname or "\\" in linkname
            or linkname.startswith(("/", "\\"))
            or bool(_DRIVE_RE.match(linkname)))


def _symlink_escapes(parent_parts: list[str], linkname: str) -> bool:
    """Would a symlink at parent_parts resolve outside the staging root?"""
    depth = len(parent_parts)  # components below the staging root
    for part in linkname.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            depth -= 1
            if depth < 0:
                return True
        else:
            depth += 1
    return False


def _kind(member: tarfile.TarInfo) -> str:
    if member.type in _FILE_TYPES:
        return "f"
    if member.type == tarfile.DIRTYPE:
        return "d"
    if member.type == tarfile.SYMTYPE:
        return "s"
    return "h"


def plan_extraction(archive: Path) -> list[dict]:
    """Read every member and decide, before writing anything, what may land.

    Returns one record per member in archive order. Any refusal aborts the
    whole set: a partially "accepted" malicious archive is no answer on the
    day someone needs a restore.
    """
    try:
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
    except tarfile.TarError as error:
        raise RestoreError(
            f"the archive could not be read: {_one_line(error)}",
            "archive_error") from error
    entries: list[dict] = []
    seen: dict[str, str] = {}
    for member in members:
        if member.type not in _ALLOWED_TYPES:
            what = _SPECIAL_TYPE_NAMES.get(
                member.type, f"unknown tar member type {member.type!r}")
            raise RestoreError(
                f"archive entry {member.name!r} is a {what}; only regular "
                "files, folders, and inside-the-folder links are allowed",
                "unsafe_member_type")
        parts = _validated_parts(member.name)
        key = "/".join(parts)
        kind = _kind(member)
        if len(parts) == 1 and kind != "d":
            raise RestoreError(
                f"the archive top level {ARCNAME}/ must be one folder, not a "
                "file or link", "unsafe_member_name")
        if key in seen:
            raise RestoreError(
                f"archive entry {key!r} appears more than once; a later "
                "duplicate could overwrite what the first one wrote",
                "duplicate_member")
        for depth in range(1, len(parts)):
            ancestor = "/".join(parts[:depth])
            if ancestor in seen and seen[ancestor] != "d":
                raise RestoreError(
                    f"archive entry {key!r} would sit inside {ancestor!r}, "
                    "which the archive records as a file or link rather than "
                    "a folder", "conflicting_member")
        target = None
        if kind == "s":
            linkname = member.linkname
            if _link_is_absolute(linkname) or \
                    _symlink_escapes(parts[:-1], linkname):
                raise RestoreError(
                    f"symlink {key!r} -> {member.linkname!r} points outside "
                    "the folder being restored", "unsafe_link")
        elif kind == "h":
            linkname = member.linkname
            if _link_is_absolute(linkname):
                raise RestoreError(
                    f"hard link {key!r} -> {linkname!r} is an absolute or "
                    "backslash-bearing target", "unsafe_link")
            target = "/".join(_validated_parts(linkname))
            if seen.get(target) != "f":
                raise RestoreError(
                    f"hard link {key!r} targets {target!r}, which is not an "
                    "earlier regular file in this archive", "unsafe_link")
        seen[key] = kind
        entries.append({
            "key": key, "kind": kind,
            "size": member.size if kind == "f" else 0,
            "mode": member.mode & 0o777,  # never carry setuid/setgid/sticky
            "mtime": int(member.mtime),
            "link": member.linkname if kind == "s" else None,
            "target": target,
        })
    return entries


def _descend(root: Path, parts: list[str], create: bool) -> Path:
    """Walk root/parts, refusing to pass through any symlinked component."""
    current = root
    for part in parts:
        nxt = current / part
        try:
            st = os.lstat(nxt)
        except FileNotFoundError:
            if not create:
                raise RestoreError(
                    f"expected staging folder {nxt} is missing", "unsafe_link")
            os.mkdir(nxt, 0o700)
        else:
            if stat.S_ISLNK(st.st_mode):
                raise RestoreError(
                    f"the staging path to {nxt} goes through a symlink; a "
                    "link must never redirect where an entry lands",
                    "unsafe_link")
            if not stat.S_ISDIR(st.st_mode):
                raise RestoreError(
                    f"{nxt} exists in staging but is not a folder",
                    "conflicting_member")
        current = nxt
    return current


def _write_regular_file(staging: Path, parts: list[str],
                        tar: tarfile.TarFile, member: tarfile.TarInfo
                        ) -> tuple[str, int]:
    """Stream one regular member to disk with O_EXCL and a running hash."""
    dest = staging.joinpath(*parts)
    source = tar.extractfile(member)
    if source is None:
        raise RestoreError(f"{member.name!r} could not be read as a file",
                           "archive_error")
    hasher = hashlib.sha256()
    written = 0
    fd = os.open(str(dest), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        while True:
            chunk = source.read(_CHUNK)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written_here = os.write(fd, view)
                view = view[written_here:]
            hasher.update(chunk)
            written += len(chunk)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    if written != member.size:
        raise RestoreError(
            f"{member.name!r} delivered {written} bytes but its archive "
            f"header declares {member.size}", "archive_truncated")
    return hasher.hexdigest(), written


def _make_symlink(staging: Path, parts: list[str], target: str) -> None:
    dest = staging.joinpath(*parts)
    try:
        os.symlink(target, str(dest))
    except FileExistsError as error:
        raise RestoreError(
            f"{'/'.join(parts)} would overwrite an entry already staged",
            "duplicate_member") from error
    except OSError as error:
        raise RestoreError(
            f"symlink {'/'.join(parts)} could not be created: {_one_line(error)}",
            "io_error") from error


def _physical_regular(staging: Path, parts: list[str]) -> Path:
    """Resolve a hard-link target on disk as a real regular file, no links."""
    _descend(staging, parts[:-1], create=False)
    path = staging.joinpath(*parts)
    try:
        st = os.lstat(path)
    except OSError as error:
        raise RestoreError(
            f"hard-link target {'/'.join(parts)} is not present in staging",
            "unsafe_link") from error
    if not stat.S_ISREG(st.st_mode):
        raise RestoreError(
            f"hard-link target {'/'.join(parts)} is not a regular file on disk",
            "unsafe_link")
    return path


def _make_hardlink(staging: Path, parts: list[str],
                   target_parts: list[str]) -> None:
    target = _physical_regular(staging, target_parts)
    dest = staging.joinpath(*parts)
    try:
        os.link(str(target), str(dest))
    except FileExistsError as error:
        raise RestoreError(
            f"{'/'.join(parts)} would overwrite an entry already staged",
            "duplicate_member") from error
    except OSError as error:
        raise RestoreError(
            f"hard link {'/'.join(parts)} could not be created: "
            + _one_line(error), "io_error") from error


def _implicit_directories(planned: list[dict]) -> list[dict]:
    """Parent folders an entry needs that the archive does not list itself.

    The runbook is written into System/backup/ while that folder may not
    exist in the vault when the archive is taken, so its parent directories
    exist only as extraction scaffolding. They still land on disk, so they
    must enter the manifest and the verification pass as real directories.
    Planning already proved no such path is a file or link in the archive.
    """
    listed_dirs = {e["key"] for e in planned if e["kind"] == "d"}
    implicit: dict[str, dict] = {}
    for entry in planned:
        parts = entry["key"].split("/")
        for depth in range(1, len(parts)):
            ancestor = "/".join(parts[:depth])
            if ancestor not in listed_dirs and ancestor not in implicit:
                implicit[ancestor] = {
                    "key": ancestor, "kind": "d", "size": 0,
                    "mode": 0o755, "mtime": None, "link": None, "target": None}
    return [implicit[key] for key in
            sorted(implicit, key=lambda k: k.count("/"))]


def _apply_metadata(staging: Path, planned: list[dict],
                    warnings: list[str]) -> None:
    """Restore permissions and mtimes only after every write is finished.

    Applying modes during extraction can lock a read-only folder before its
    contents exist. Files get their modes first, then folders deepest-last;
    timestamps follow in the same order. Special mode bits were stripped at
    planning time, and symlinks are never chmod'ed (that would touch the
    link's target).
    """
    files = [e for e in planned if e["kind"] == "f"]
    dirs = [e for e in planned if e["kind"] == "d"]
    links = [e for e in planned if e["kind"] == "s"]
    for entry in files:
        os.chmod(staging.joinpath(*entry["key"].split("/")), entry["mode"])
    for entry in sorted(dirs, key=lambda e: e["key"].count("/"), reverse=True):
        os.chmod(staging.joinpath(*entry["key"].split("/")), entry["mode"])
    for entry in files:
        os.utime(staging.joinpath(*entry["key"].split("/")),
                 (entry["mtime"], entry["mtime"]))
    symlink_time_failures = 0
    for entry in links:
        try:
            os.utime(staging.joinpath(*entry["key"].split("/")),
                     (entry["mtime"], entry["mtime"]),
                     follow_symlinks=False)
        except (OSError, NotImplementedError, ValueError):
            symlink_time_failures += 1
    if symlink_time_failures:
        warnings.append(
            f"{symlink_time_failures} symlink timestamp(s) could not be "
            "restored on this platform; the links and their targets are "
            "otherwise intact")
    for entry in sorted(dirs, key=lambda e: e["key"].count("/"), reverse=True):
        if entry["mtime"] is not None:  # implicit scaffold folders
            os.utime(staging.joinpath(*entry["key"].split("/")),
                     (entry["mtime"], entry["mtime"]))


def _materialize(archive: Path, staging: Path, planned: list[dict]
                 ) -> tuple[list[dict], dict, int, list[str]]:
    """Extract the planned entries into staging, streaming each once."""
    counts = {"members": len(planned), "files": 0, "directories": 0,
              "symlinks": 0, "hardlinks": 0}
    manifest: list[dict] = []
    total_bytes = 0
    warnings: list[str] = []
    implicit = _implicit_directories(planned)
    for record in implicit:  # shallowest first, before any member needs them
        _descend(staging, record["key"].split("/"), create=True)
        manifest.append({"path": record["key"], "type": "directory",
                         "implicit": True})
    with tarfile.open(archive) as tar:
        for order, member in enumerate(tar):
            if order >= len(planned):
                raise RestoreError(
                    "the archive yielded more entries while unpacking than "
                    "it listed when scanned", "archive_changed")
            record = planned[order]
            if member.name.rstrip("/") != record["key"]:
                raise RestoreError(
                    "the archive's entries changed between the safety scan "
                    "and unpacking", "archive_changed")
            kind = record["kind"]
            counts[_KIND_TO_COUNT[kind]] += 1
            parts = record["key"].split("/")
            if kind == "d":
                _descend(staging, parts, create=True)
                manifest.append({"path": record["key"], "type": "directory"})
            elif kind == "f":
                _descend(staging, parts[:-1], create=True)
                digest, size = _write_regular_file(staging, parts, tar, member)
                total_bytes += size
                manifest.append({"path": record["key"], "type": "file",
                                 "size": size, "sha256": digest})
            elif kind == "s":
                _descend(staging, parts[:-1], create=True)
                _make_symlink(staging, parts, record["link"])
                manifest.append({"path": record["key"], "type": "symlink",
                                 "target": record["link"]})
            else:
                _descend(staging, parts[:-1], create=True)
                _make_hardlink(staging, parts, record["target"].split("/"))
                manifest.append({"path": record["key"], "type": "hardlink",
                                 "target": record["target"]})
    member_records = len(manifest) - len(implicit)
    if member_records != len(planned):
        raise RestoreError(
            "the archive yielded fewer entries while unpacking than it "
            "listed when scanned", "archive_changed")
    counts["directories"] += len(implicit)
    _apply_metadata(staging, planned + implicit, warnings)
    return manifest, counts, total_bytes, warnings


def _stage_bundle(bundle: Path, staging: Path) -> dict:
    """Copy the verified history bundle into staging with the archive tree."""
    dest = staging / bundle.name
    hasher = hashlib.sha256()
    written = 0
    fd = os.open(str(dest), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with bundle.open("rb") as source:
            while True:
                chunk = source.read(_CHUNK)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    here = os.write(fd, view)
                    view = view[here:]
                hasher.update(chunk)
                written += len(chunk)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    return {"path": bundle.name, "type": "file", "size": written,
            "sha256": hasher.hexdigest(), "role": "history-bundle"}


def _scan_tree(root: Path) -> dict[str, tuple[str, str | None, os.stat_result]]:
    """List staging without following links: path -> (kind, linkname, stat)."""
    found: dict[str, tuple[str, str | None, os.stat_result]] = {}

    def walk(folder: Path, prefix: str) -> None:
        with os.scandir(folder) as entries:
            for entry in entries:
                rel = prefix + entry.name
                st = entry.stat(follow_symlinks=False)
                if entry.is_symlink():
                    found[rel] = ("s", os.readlink(entry.path), st)
                elif entry.is_dir(follow_symlinks=False):
                    found[rel] = ("d", None, st)
                    walk(Path(entry.path), rel + "/")
                elif entry.is_file(follow_symlinks=False):
                    found[rel] = ("f", None, st)
                else:
                    raise RestoreError(
                        f"an unexpected special file ({stat.S_IFMT(st.st_mode):#o}) "
                        f"appeared in staging at {rel}", "unsafe_member_type")

    walk(root, "")
    return found


def _verify_staged_tree(staging: Path, manifest: list[dict]) -> int:
    """Re-read everything staged and prove it matches the manifest exactly.

    No missing paths, no extra paths, matching types and link targets, and
    every regular file's size and SHA-256 recomputed from the bytes now on
    disk. Returns the number of files verified.
    """
    on_disk = _scan_tree(staging)
    expected = {m["path"] for m in manifest}
    actual = set(on_disk)
    if actual != expected:
        detail: list[str] = []
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            detail.append("missing after staging: " + ", ".join(missing[:5]))
        if extra:
            detail.append("not accounted for by the archive: "
                          + ", ".join(extra[:5]))
        raise RestoreError(
            "the staged copy did not match the archive manifest ("
            + "; ".join(detail) + ")", "verify_mismatch")
    verified = 0
    for record in manifest:
        kind, linkname, st = on_disk[record["path"]]
        if record["type"] == "file":
            if kind != "f":
                raise RestoreError(
                    f"{record['path']} is not a regular file on disk",
                    "verify_mismatch")
            if st.st_size != record["size"]:
                raise RestoreError(
                    f"{record['path']} has {st.st_size} bytes on disk but "
                    f"{record['size']} in the manifest", "verify_mismatch")
            digest = file_digest(staging / record["path"])
            if digest != record["sha256"]:
                raise RestoreError(
                    f"the checksum of {record['path']} re-read from disk "
                    "does not match what was unpacked", "verify_mismatch")
            verified += 1
        elif record["type"] == "directory":
            if kind != "d":
                raise RestoreError(
                    f"{record['path']} is not a directory on disk",
                    "verify_mismatch")
        elif record["type"] == "symlink":
            if kind != "s" or linkname != record["target"]:
                raise RestoreError(
                    f"symlink {record['path']} does not point where the "
                    "archive declares", "verify_mismatch")
        else:  # hardlink: must share an inode with its archived target
            if kind != "f":
                raise RestoreError(
                    f"hard link {record['path']} is not a regular file on disk",
                    "verify_mismatch")
            target = on_disk.get(record["target"])
            if target is None or target[0] != "f" \
                    or (st.st_dev, st.st_ino) != (target[2].st_dev,
                                                  target[2].st_ino):
                raise RestoreError(
                    f"hard link {record['path']} is not linked to its "
                    f"archived target {record['target']}", "verify_mismatch")
    return verified


def _fsync_dir(folder: Path) -> None:
    fd = os.open(str(folder), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(staging: Path) -> None:
    """Flush file data (done per file) and every directory entry update."""
    on_disk = _scan_tree(staging)
    dirs = [staging / path for path, (kind, _l, _st) in on_disk.items()
            if kind == "d"]
    for folder in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        _fsync_dir(folder)
    _fsync_dir(staging)


def _purge_staging(staging: Path) -> None:
    """A failed extraction must leave no folder that looks finished."""
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(mode=0o700)


def safe_extract(archive: Path, staging: Path,
                 bundle: Path | None = None) -> dict:
    """Plan, extract into staging, verify, and fsync. Never publishes.

    On any failure the staging folder is emptied before the error escapes,
    so callers never face a half-restored tree.
    """
    try:
        planned = plan_extraction(archive)
        manifest, counts, total_bytes, warnings = _materialize(
            archive, staging, planned)
        bundle_record = None
        if bundle is not None:
            bundle_record = _stage_bundle(bundle, staging)
            manifest.append(bundle_record)
        files_verified = _verify_staged_tree(staging, manifest)
        _fsync_tree(staging)
    except RestoreError as error:
        _purge_staging(staging)
        raise RestoreError(
            f"{archive.name} could not be unpacked: {error} Nothing usable "
            "was written. Try `verify` on this set, or pick an older one "
            "with --set.", error.code) from error
    except _ARCHIVE_ERRORS as error:
        _purge_staging(staging)
        raise RestoreError(
            f"{archive.name} could not be unpacked: {_one_line(error)}. "
            "Nothing usable was written. Try `verify` on this set, or pick "
            "an older one with --set.", "archive_error") from error
    except OSError as error:
        _purge_staging(staging)
        raise RestoreError(
            f"{archive.name} could not be unpacked: {_one_line(error)}. "
            "This usually means the disk is full or the folder is not "
            "writable. Nothing usable was written.", "io_error") from error
    return {"counts": counts, "entries": manifest, "bytes": total_bytes,
            "files_verified": files_verified,
            "bundle": ({"name": bundle_record["path"],
                        "size": bundle_record["size"],
                        "sha256": bundle_record["sha256"]}
                       if bundle_record is not None else None),
            "warnings": warnings}


def _publish(staging: Path, target: Path) -> None:
    """Move the verified tree to its name atomically (same filesystem)."""
    try:
        os.chmod(staging, 0o755)
        if target.exists():
            os.rmdir(target)  # the caller proved it empty; never recurse
        os.replace(str(staging), str(target))
    except OSError as error:
        raise RestoreError(
            f"the verified restore could not be moved into {target}: "
            f"{_one_line(error)}", "publish_failed") from error
    _fsync_dir(target.parent)


# --- machine-readable report ------------------------------------------------

def _new_report(source: Path, stamp: str, archive_name: str, mode: str,
                target: Path | None) -> dict:
    return {
        "schema": REPORT_SCHEMA,
        "status": "running",
        "mode": mode,
        "set": stamp,
        "source": str(source),
        "archive": archive_name,
        "target": str(target) if target is not None else None,
        "started_at": _now(),
        "finished_at": None,
        "python": sys.version.split()[0],
        "manifest_source": MANIFEST_SOURCE,
        "checksum_algorithm": "sha256",
        "counts": {"members": 0, "files": 0, "directories": 0,
                   "symlinks": 0, "hardlinks": 0},
        "bytes_written": 0,
        "files_verified": 0,
        "bundle": None,
        "entries": [],
        "warnings": [],
        "error": None,
    }


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _finalize_report(report: dict, result: dict | None,
                     error: RestoreError | None) -> None:
    report["finished_at"] = _now()
    if error is not None:
        report["status"] = "stopped"
        report["error"] = {"code": error.code, "message": str(error)[:500]}
        return
    report["status"] = "ok"
    report["counts"] = result["counts"]
    report["bytes_written"] = result["bytes"]
    report["files_verified"] = result["files_verified"]
    report["bundle"] = result["bundle"]
    report["entries"] = result["entries"]
    report["warnings"] = result["warnings"]


def _save_report(path: Path, report: dict) -> bool:
    """Write the report atomically; a failed report write never hides a
    successful restore, but it must be visible."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        fd = os.open(str(partial),
                     os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            view = memoryview(payload)
            while view:
                here = os.write(fd, view)
                view = view[here:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(partial, path)
        return True
    except OSError as error:
        print(f"Warning: the machine-readable restore report could not be "
              f"written to {path}: {_one_line(error)}")
        return False


# --- modes ------------------------------------------------------------------

def test_restore(source: Path, stamp: str,
                 report_path: str | None = None) -> None:
    archive = source / f"{PREFIX}{stamp}.tar.gz"
    bundle = source / f"{PREFIX}{stamp}.bundle"
    report = _new_report(source, stamp, archive.name, "test", None)
    with tempfile.TemporaryDirectory() as parent:
        staging = Path(tempfile.mkdtemp(prefix=".test-restore-", dir=parent))
        try:
            result = safe_extract(
                archive, staging, bundle if bundle.exists() else None)
        except RestoreError as error:
            _finalize_report(report, None, error)
            if report_path:
                _save_report(Path(report_path).expanduser(), report)
            raise
        _finalize_report(report, result, None)
        if report_path:
            _save_report(Path(report_path).expanduser(), report)
    counts = result["counts"]
    print(
        f"Test restore succeeded: {counts['members']} entries "
        f"({counts['files']} files, {counts['directories']} folders, "
        f"{counts['symlinks'] + counts['hardlinks']} links), each file "
        "re-read from disk and checksum-verified, extracted to a temporary "
        "folder and cleaned up. Nothing was changed.")


def restore(source: Path, stamp: str, target: Path, vault: Path,
            report_path: str | None = None) -> None:
    target = target.expanduser()
    resolved_target = target.resolve()
    resolved_vault = vault.resolve()
    if resolved_target == resolved_vault \
            or resolved_vault in resolved_target.parents \
            or resolved_target in resolved_vault.parents:
        raise RestoreError(
            "Refusing to restore into (or over) the live vault. Restore to a "
            "fresh folder, inspect it, then move it into place yourself.")
    if target.exists() and any(target.iterdir()):
        raise RestoreError(f"Refusing to restore into {target}: the folder is "
                           "not empty. This tool never overwrites existing files.")
    archive = source / f"{PREFIX}{stamp}.tar.gz"
    bundle_path = source / f"{PREFIX}{stamp}.bundle"
    bundle = bundle_path if bundle_path.exists() else None
    # Staging is created as a sibling of the target so the final move is an
    # atomic same-filesystem rename; the target itself is never written to
    # until the verified tree is published.
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.restore-",
                                    dir=target.parent))
    report_file = (Path(report_path).expanduser() if report_path
                   else target.parent / f".{target.name}.restore-report-{stamp}.json")
    report = _new_report(source, stamp, archive.name, "restore", target)
    try:
        result = safe_extract(archive, staging, bundle)
    except RestoreError as error:
        shutil.rmtree(staging, ignore_errors=True)
        _finalize_report(report, None, error)
        _save_report(report_file, report)
        raise
    try:
        _publish(staging, target)
    except RestoreError as error:
        shutil.rmtree(staging, ignore_errors=True)
        _finalize_report(report, result, error)
        _save_report(report_file, report)
        raise
    _finalize_report(report, result, None)
    _save_report(report_file, report)
    counts = result["counts"]
    print(
        f"Restored {counts['members']} entries to {target} "
        f"({counts['files']} files, {counts['directories']} folders, "
        f"{counts['symlinks'] + counts['hardlinks']} links). Every file was "
        "fsynced, re-read from disk, and checksum-verified in a staging "
        "folder before anything was moved into place.")
    if bundle is not None:
        print(f"Copied {bundle.name} alongside; recover the version history "
              f"with: git clone {bundle.name} restored-history")
    for warning in result["warnings"]:
        print(f"Warning: {warning}")
    print("Secrets, schedules, and permissions are deliberately not in a "
          "backup; see docs/backup-restore.md for what to re-establish.")
    print(f"Machine-readable restore report: {report_file}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify or restore vault backups")
    parser.add_argument("mode", choices=["verify", "test", "restore"])
    parser.add_argument("--set", dest="stamp", help="backup set stamp "
                        "(default: the newest set)")
    parser.add_argument("--source", help="folder holding the backup sets")
    parser.add_argument("--to", help="restore target folder (restore mode)")
    parser.add_argument("--vault", help="vault root (default: VAULT_PATH or "
                                        "this file's own vault)")
    parser.add_argument("--report", help="write the machine-readable JSON "
                        "restore report to this path (restore mode also "
                        "writes one next to --to by default)")
    args = parser.parse_args(argv)
    vault = resolve_vault_root(args.vault)
    try:
        source = resolve_source(vault, args.source)
        stamp = pick_set(source, args.stamp)
        print(f"Backup set {stamp} in {source}:")
        try:
            findings = verify_set(source, stamp)
        except RestoreError as error:
            # Being told the newest copy is damaged, with no hint that an
            # intact older one is sitting right there, is the worst possible
            # answer on the day someone actually needs this.
            raise _with_fallback_hint(error, source, stamp) from None
        for finding in findings:
            print(f"  {finding}")
        if args.mode == "test":
            test_restore(source, stamp, report_path=args.report)
        elif args.mode == "restore":
            if not args.to:
                raise RestoreError("restore mode needs --to <empty folder>")
            restore(source, stamp, Path(args.to), vault,
                    report_path=args.report)
        return 0
    except RestoreError as error:
        print(f"Stopped: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
