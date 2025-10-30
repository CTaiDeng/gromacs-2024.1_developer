#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPDX-License-Identifier: GPL-3.0-only
Copyright (C) 2010- The GROMACS Authors
Copyright (C) 2025 GaoZheng

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 only as
published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


# ------------------------------
# Data structures
# ------------------------------


@dataclass
class BlobMatch:
    commit: str
    path: str
    line: int
    text: str
    date: str = ""  # 提交入库时间（committer date，ISO）


@dataclass
class MessageMatch:
    commit: str
    author: str
    date: str
    subject: str


@dataclass
class SearchResult:
    query: str
    regex: bool
    ignore_case: bool
    repo_root: str
    head: str
    since: Optional[str]
    until: Optional[str]
    author: Optional[str]
    path_globs: List[str]
    timestamp: str
    blob_matches: List[BlobMatch]
    message_matches: List[MessageMatch]


# ------------------------------
# Utilities
# ------------------------------


def _run_git(args: Sequence[str], cwd: Optional[Path] = None, check_ok_codes: Tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
    """Run a git command and return CompletedProcess.

    Args:
        args: Full git command including 'git' as first element.
        cwd: Working directory.
        check_ok_codes: Acceptable return codes. Default (0,).

    Raises:
        RuntimeError on unacceptable exit code.
    """
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode not in check_ok_codes:
        cmd = " ".join(shlex.quote(a) for a in args)
        raise RuntimeError(f"git command failed ({proc.returncode}): {cmd}\n{proc.stderr}")
    return proc


def _ensure_git_root(cwd: Optional[Path] = None) -> Path:
    proc = _run_git(["git", "rev-parse", "--show-toplevel"], cwd=cwd)
    root = Path(proc.stdout.strip())
    if not root.exists():
        raise RuntimeError("未检测到有效的 Git 仓库（rev-parse 无法定位到根目录）")
    return root


def _get_head_sha(cwd: Path) -> str:
    proc = _run_git(["git", "rev-parse", "HEAD"], cwd=cwd)
    return proc.stdout.strip()


def _rev_list_all(cwd: Path, since: Optional[str], until: Optional[str]) -> List[str]:
    args = ["git", "rev-list", "--all"]
    if since:
        args += [f"--since={since}"]
    if until:
        args += [f"--until={until}"]
    proc = _run_git(args, cwd=cwd)
    revs = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return revs


def _chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def _grep_blobs(
    cwd: Path,
    revs: Sequence[str],
    query: str,
    regex: bool,
    ignore_case: bool,
    path_globs: Sequence[str],
    limit: int,
) -> List[BlobMatch]:
    """Search across file snapshots in all commits using git grep.

    Notes:
        - Uses chunked batches of revisions to avoid command-line length limits.
        - Accepts exit code 1 from git grep as "no matches" for a batch.
    """
    results: List[BlobMatch] = []
    if not revs:
        return results

    base = ["git", "grep", "-n", "-I", "--color=never"]
    if regex:
        base += ["-E"]
    else:
        base += ["-F"]
    if ignore_case:
        base += ["-i"]
    base += ["-e", query]

    for group in _chunked(revs, 64):
        args = list(base) + list(group)
        if path_globs:
            args += ["--"] + list(path_globs)

        try:
            proc = _run_git(args, cwd=cwd, check_ok_codes=(0, 1))
        except RuntimeError as e:
            # Continue after reporting; one bad batch should not stop everything.
            sys.stderr.write(str(e) + "\n")
            continue

        for line in proc.stdout.splitlines():
            # Expected format: <rev>:<path>:<lineno>:<content>
            # Beware that <content> can contain colons.
            if not line:
                continue
            # split first two ':' positions safely
            # 1) rev
            p1 = line.find(":")
            if p1 <= 0:
                continue
            rev = line[:p1]
            rest = line[p1 + 1 :]
            # 2) path
            p2 = rest.find(":")
            if p2 <= 0:
                continue
            path = rest[:p2]
            rest2 = rest[p2 + 1 :]
            # 3) line number
            p3 = rest2.find(":")
            if p3 <= 0:
                continue
            lineno_str = rest2[:p3]
            try:
                lineno = int(lineno_str)
            except ValueError:
                continue
            content = rest2[p3 + 1 :]
            results.append(BlobMatch(commit=rev, path=path, line=lineno, text=content))
            if 0 <= limit <= len(results):
                return results

    return results


def _search_commit_messages(
    cwd: Path,
    query: str,
    regex: bool,
    ignore_case: bool,
    since: Optional[str],
    until: Optional[str],
    author: Optional[str],
    limit: int,
) -> List[MessageMatch]:
    args = [
        "git",
        "log",
        "--all",
        "--date=iso",
        # 统一使用提交者时间（committer date）体现“入库时间”
        "--pretty=format:%H\t%an\t%cd\t%s",
    ]

    # --grep uses POSIX regex by default; use -E for extended if we escaped pattern
    pattern = query if regex else re.escape(query)
    args += [f"--grep={pattern}"]
    if not regex:
        args += ["-E"]
    if ignore_case:
        args += ["-i"]
    if since:
        args += [f"--since={since}"]
    if until:
        args += [f"--until={until}"]
    if author:
        args += [f"--author={author}"]

    proc = _run_git(args, cwd=cwd, check_ok_codes=(0, 128))
    results: List[MessageMatch] = []
    for line in proc.stdout.splitlines():
        if not line:
            continue
        parts = line.split("\t", 3)
        if len(parts) < 4:
            continue
        commit, author_name, date_str, subject = parts
        results.append(MessageMatch(commit=commit, author=author_name, date=date_str, subject=subject))
        if 0 <= limit <= len(results):
            break
    return results


def _fill_blob_commit_dates(cwd: Path, matches: List[BlobMatch]) -> None:
    if not matches:
        return
    uniq = sorted({m.commit for m in matches})
    if not uniq:
        return
    mapping = {}
    for group in _chunked(uniq, 256):
        args = [
            "git",
            "show",
            "-s",
            "--no-patch",
            "--date=iso",
            "--pretty=format:%H\t%cd",
        ] + list(group)
        proc = _run_git(args, cwd=cwd)
        for line in proc.stdout.splitlines():
            if not line:
                continue
            sha, _, date = line.partition("\t")
            if sha and date:
                mapping[sha] = date
    for m in matches:
        m.date = mapping.get(m.commit, "")


def search_git_history(
    query: str,
    *,
    regex: bool = False,
    ignore_case: bool = False,
    include_blobs: bool = True,
    include_messages: bool = True,
    since: Optional[str] = None,
    until: Optional[str] = None,
    author: Optional[str] = None,
    path_globs: Optional[Sequence[str]] = None,
    limit_blobs: int = -1,
    limit_messages: int = -1,
    output_dir: Optional[Path] = None,
    output_prefix: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Search across entire Git history and write reports.

    Returns:
        Tuple of (markdown_report_path, json_report_path)
    """
    root = _ensure_git_root()
    head = _get_head_sha(root)
    revs: List[str] = []
    if include_blobs:
        revs = _rev_list_all(root, since=since, until=until)

    blob_matches: List[BlobMatch] = []
    if include_blobs:
        blob_matches = _grep_blobs(
            root,
            revs,
            query,
            regex,
            ignore_case,
            list(path_globs or ()),
            limit_blobs,
        )

    message_matches: List[MessageMatch] = []
    if include_messages:
        message_matches = _search_commit_messages(
            root,
            query,
            regex,
            ignore_case,
            since,
            until,
            author,
            limit_messages,
        )

    # 为文件内容匹配补充入库时间（提交者时间）
    if include_blobs and blob_matches:
        _fill_blob_commit_dates(root, blob_matches)

    sr = SearchResult(
        query=query,
        regex=regex,
        ignore_case=ignore_case,
        repo_root=str(root),
        head=head,
        since=since,
        until=until,
        author=author,
        path_globs=list(path_globs or ()),
        timestamp=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        blob_matches=blob_matches,
        message_matches=message_matches,
    )

    base_out_dir = output_dir or (root / "out" / "out_git_search")
    base_out_dir.mkdir(parents=True, exist_ok=True)
    # 目录与文件名统一使用“秒级时间戳”
    ts_sec = str(int(time.time()))
    prefix = output_prefix or "report"
    report_dir = base_out_dir / f"{prefix}_{ts_sec}"
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path = report_dir / f"{prefix}_{ts_sec}.md"
    json_path = report_dir / f"{prefix}_{ts_sec}.json"

    _write_json(json_path, sr)
    _write_markdown(md_path, sr, json_path)

    return md_path, json_path


def _write_json(path: Path, sr: SearchResult) -> None:
    def to_dict(obj):
        if isinstance(obj, BlobMatch):
            return {
                "commit": obj.commit,
                "date": obj.date,
                "path": obj.path,
                "line": obj.line,
                "text": obj.text,
            }
        if isinstance(obj, MessageMatch):
            return {"commit": obj.commit, "author": obj.author, "date": obj.date, "subject": obj.subject}
        if isinstance(obj, SearchResult):
            return {
                "query": obj.query,
                "regex": obj.regex,
                "ignore_case": obj.ignore_case,
                "repo_root": obj.repo_root,
                "head": obj.head,
                "since": obj.since,
                "until": obj.until,
                "author": obj.author,
                "path_globs": obj.path_globs,
                "timestamp": obj.timestamp,
                "results": {
                    "blob_content": {
                        "match_count": len(obj.blob_matches),
                        "unique_commits": len({m.commit for m in obj.blob_matches}),
                        "unique_files": len({m.path for m in obj.blob_matches}),
                        "matches": [to_dict(m) for m in obj.blob_matches],
                    },
                    "commit_messages": {
                        "match_count": len(obj.message_matches),
                        "commits": [to_dict(m) for m in obj.message_matches],
                    },
                },
            }
        raise TypeError(f"Unsupported type: {type(obj)!r}")

    text = json.dumps(to_dict(sr), ensure_ascii=False, indent=2)
    data = text.replace("\r\n", "\n")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)


def _truncate(s: str, limit: int = 200) -> str:
    s = s.replace("\r", "").replace("\t", "    ")
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _write_markdown(path: Path, sr: SearchResult, json_path: Path) -> None:
    lines: List[str] = []

    lines.append(f"# Git 历史全文搜索报告")
    lines.append("")
    lines.append(f"- 查询：{sr.query}")
    lines.append(f"- 模式：{'正则' if sr.regex else '字面量'}{'（不区分大小写）' if sr.ignore_case else ''}")
    lines.append(f"- 仓库：{sr.repo_root}")
    lines.append(f"- HEAD：{sr.head}")
    if sr.since:
        lines.append(f"- 起始时间：{sr.since}")
    if sr.until:
        lines.append(f"- 截止时间：{sr.until}")
    if sr.author:
        lines.append(f"- 作者过滤：{sr.author}")
    if sr.path_globs:
        lines.append(f"- 路径过滤：{' '.join(sr.path_globs)}")
    lines.append(f"- 生成时间：{sr.timestamp}")
    lines.append("")

    # Summary
    blob_unique_commits = len({m.commit for m in sr.blob_matches})
    blob_unique_files = len({m.path for m in sr.blob_matches})
    lines.append("## 概览")
    lines.append(f"- 文件内容匹配：{len(sr.blob_matches)} 条（{blob_unique_commits} 个提交，{blob_unique_files} 个文件）")
    lines.append(f"- 提交信息匹配：{len(sr.message_matches)} 个提交")
    lines.append(f"- 完整 JSON：`{json_path.as_posix()}`")
    lines.append("")

    # Blob content samples
    lines.append("## 文件内容匹配（示例最多 100 条）")
    for i, m in enumerate(sr.blob_matches[:100], 1):
        snippet = _truncate(m.text, 200)
        when = f" {m.date}" if m.date else ""
        lines.append(f"- {i}. `{m.commit}`{when} `{m.path}:{m.line}` — {snippet}")
    if len(sr.blob_matches) == 0:
        lines.append("- （无）")
    lines.append("")

    # Commit message matches
    lines.append("## 提交信息匹配（示例最多 100 条）")
    for i, mm in enumerate(sr.message_matches[:100], 1):
        lines.append(f"- {i}. `{mm.commit}` {mm.date} {mm.author} — {mm.subject}")
    if len(sr.message_matches) == 0:
        lines.append("- （无）")
    lines.append("")

    text = "\n".join(lines)
    data = text.replace("\r\n", "\n")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="git_search",
        description=(
            "在整个 Git 历史上进行全文搜索（文件内容与提交信息），并输出报告到 out/git_search。"
        ),
    )
    p.add_argument("query", help="要搜索的字符串；默认按字面量匹配")
    p.add_argument("--regex", action="store_true", help="将查询解释为正则表达式（默认字面量）")
    p.add_argument("-i", "--ignore-case", action="store_true", help="大小写不敏感匹配")
    p.add_argument("--no-blobs", dest="include_blobs", action="store_false", help="不搜索文件内容，仅搜索提交信息")
    p.add_argument("--no-messages", dest="include_messages", action="store_false", help="不搜索提交信息，仅搜索文件内容")
    p.add_argument("--since", help="仅搜索此时间之后的提交（git 支持的日期格式）", default=None)
    p.add_argument("--until", help="仅搜索此时间之前的提交（git 支持的日期格式）", default=None)
    p.add_argument("--author", help="按作者过滤提交信息搜索", default=None)
    p.add_argument(
        "--path-glob",
        action="append",
        default=None,
        help="限制文件内容搜索的路径通配（可多次指定，如 --path-glob '*.c' --path-glob 'src/**'）",
    )
    p.add_argument("--limit-blobs", type=int, default=-1, help="文件内容匹配最大条数（-1 为不限）")
    p.add_argument("--limit-messages", type=int, default=-1, help="提交信息匹配最大条数（-1 为不限）")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认 out/git_search）",
    )
    p.add_argument("--output-prefix", type=str, default="report", help="输出文件名前缀（默认 report）")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir) if args.output_dir else None

    try:
        md, js = search_git_history(
            args.query,
            regex=args.regex,
            ignore_case=args.ignore_case,
            include_blobs=args.include_blobs,
            include_messages=args.include_messages,
            since=args.since,
            until=args.until,
            author=args.author,
            path_globs=args.path_glob,
            limit_blobs=args.limit_blobs,
            limit_messages=args.limit_messages,
            output_dir=out_dir,
            output_prefix=args.output_prefix,
        )
    except Exception as e:
        sys.stderr.write(f"[git_search] 错误：{e}\n")
        return 2

    sys.stdout.write(f"Markdown 报告：{md.as_posix()}\n")
    sys.stdout.write(f"JSON 报告：{js.as_posix()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
