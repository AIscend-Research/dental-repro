#!/usr/bin/env python3
"""
Check whether any revision of the DENTEX test-data file other than the one
this project pins has ever existed on the HuggingFace Hub.

The Deep-Caries finding (``DENTEX/test_data.zip`` contains no code for Deep
Caries -- see ``data_convert.TEST_LABEL_FINDING``) is only as trustworthy as
the claim that we looked at the release, not at one stale copy of it. This
script answers that with the repo's actual commit history rather than an
assumption:

1. Pull every commit on ``main`` for ``ibrahimhamamci/DENTEX`` from the Hub
   API (no ``huggingface_hub`` dependency -- plain HTTPS + stdlib json).
2. For each commit, check whether it touched ``DENTEX/test_data.zip`` (the
   Hub API exposes this per-commit via the file diff endpoint).
3. Report every revision that ever changed that file, and its blob oid, so a
   change in file *content* under an unchanged path would also be visible.

Run it with:

    python tools/verify_test_release_history.py

Network access to huggingface.co is required; no dataset download happens.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

REPO_ID = "ibrahimhamamci/DENTEX"
TARGET_PATH = "DENTEX/test_data.zip"
TARGET_DIR = "DENTEX"
API_ROOT = "https://huggingface.co/api/datasets/{}".format(REPO_ID)


class NotFound(Exception):
    """The path did not exist at this revision -- not a transient error."""


def _get_json(url: str, retries: int = 8) -> object:
    """
    The Hub API's tree endpoint returns HTTP 200 with body
    ``{"error": "maximum queue size reached"}`` under light load, not a
    retriable status code -- so retries are driven by the parsed body, not
    just the HTTP status. A 404 means the path genuinely does not exist at
    that revision (e.g. before the directory was first uploaded) and is
    raised immediately as ``NotFound``, not retried as if it were transient.
    """
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise NotFound(url) from None
            last_error = error
            time.sleep(5 * (attempt + 1))
            continue
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            time.sleep(5 * (attempt + 1))
            continue
        if isinstance(data, dict) and data.get("error") == "maximum queue size reached":
            time.sleep(5 * (attempt + 1))
            continue
        return data
    raise RuntimeError("giving up on {} after {} attempts: {!r}".format(
        url, retries, last_error))


def list_commits() -> list:
    return _get_json(API_ROOT + "/commits/main?limit=200")


def file_content_oid(revision: str, dirpath: str, filename: str) -> str:
    """
    The content-addressed LFS oid of ``dirpath/filename`` at ``revision``, or
    ``None`` if the path did not exist at that revision. This changes if and
    only if the file's *bytes* changed -- unlike a commit title, it cannot be
    fooled by an undescriptive "Upload N files" message.
    """
    try:
        tree = _get_json("{}/tree/{}/{}".format(API_ROOT, revision, dirpath))
    except NotFound:
        return None
    for entry in tree:
        if entry.get("path") == "{}/{}".format(dirpath, filename):
            lfs = entry.get("lfs")
            return lfs["oid"] if lfs else entry.get("oid")
    return None


def main() -> int:
    current_head = _get_json(API_ROOT + "/refs")["branches"][0]["targetCommit"]
    commits = list(reversed(list_commits()))  # oldest first
    print("HEAD of {}: {}".format(REPO_ID, current_head))
    print("Walking {} commits on main, oldest first, tracking the content "
          "oid of {!r}...".format(len(commits), TARGET_PATH))

    history = []
    last_oid = "<absent>"
    for commit in commits:
        commit_id = commit["id"]
        oid = file_content_oid(commit_id, TARGET_DIR, "test_data.zip")
        changed = oid != last_oid
        if changed:
            history.append({
                "commit": commit_id,
                "title": commit.get("title"),
                "date": commit.get("date"),
                "content_oid": oid,
            })
        last_oid = oid
        time.sleep(2.0)

    report = {
        "repo_id": REPO_ID,
        "target_path": TARGET_PATH,
        "current_head": current_head,
        "commits_scanned": len(commits),
        "distinct_content_revisions": history,
        "single_revision": len([h for h in history if h["content_oid"] is not None]) == 1,
    }
    print(json.dumps(report, indent=2))

    real_revisions = [h for h in history if h["content_oid"] is not None]
    if len(real_revisions) == 0:
        print("\nFAILED: {!r} was never found in the tree at any commit -- "
              "the path assumed by this script is wrong, this is not "
              "evidence the file never changed.".format(TARGET_PATH))
        return 1
    if len(real_revisions) == 1:
        rev = real_revisions[0]
        print("\n{!r} has exactly one content revision, introduced at {} "
              "({}, {}) and unchanged (same LFS content oid {}) through the "
              "current HEAD across all {} commits on main. There is no other "
              "revision to check for a Deep Caries label -- the file this "
              "project's Deep-Caries finding is based on is the only version "
              "of this file that has ever existed on the Hub.".format(
                  TARGET_PATH, rev["commit"][:12], rev["title"], rev["date"],
                  rev["content_oid"][:16], len(commits)))
        return 0
    print("\n{!r} has changed content {} times -- the Deep-Caries finding "
          "needs to be checked against EVERY revision listed above, not "
          "just current HEAD.".format(TARGET_PATH, len(real_revisions)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
