 10-digit prefix under project_docs.
    for ts, paths in list(used_proj_prefixes.items()):
        if len(paths) > 1:
            try:
                def _title_key(p: Path) -> str:
                    name = p.name
                    rest = name.split("_", 1)[1] if "_" in name else name
                    return rest
                first = sorted(paths, key=_title_key)[0]
                used_proj_prefixes[ts] = {first}
            except Exception:
                # If sorting fails for any reason, keep original set; unique enforcement will still work
                pass
    if not targets:
        print("No target files found under my_docs/ or my_project/**/docs")
        return 0
    for p in targets:
        name = p.name
        ts_use: int | None = None
        if "_" in name:
            prefix, rest = name.split("_", 1)
            if prefix.isdigit():
                ts_filename = int(prefix)
                exempt = (
                    p.as_posix().startswith("my_docs/project_docs/")
                    and ts_filename in EXEMPT_PREFIXES
                )
                if exempt:
                    # Reserve current prefix to prevent others from taking it; do not change it
                    used_proj_prefixes.setdefault(ts_filename, set()).add(p)
                    ts_use = ts_filename
                else:
                    ts_git = first_add_timestamp(p)
                    if ts_git is not None:
                        # Strip unwanted '标题：'/'标题:' prefix from the title part of filename
                        rest_stripped = rest
                        for lab in ("标题：", "标题:", "標題：", "標題:"):
                            if rest_stripped.startswith(lab):
                                rest_stripped = rest_stripped[len(lab):].lstrip()
                                break
                        # Enforce unique prefix under project_docs by stepping back seconds if needed
                        ts_final = _ensure_unique_projdocs_ts(ts_git, p, used_proj_prefixes)
                        # 1) rename if needed
                        new_name = f"{ts_final}_{rest_stripped}"
                        if new_name != name:
                            new_path = p.with_name(new_name)
                            subprocess.check_call(["git", "mv", "-f", "--", str(p), str(new_path)])
                            p = new_path
                            name = p.name
                            renamed.append(str(new_path))
                        ts_use = ts_final
        # 2) ensure date line in markdown
        if p.suffix.lower() == ".md" and ts_use is not None:
            if ensure_date_in_markdown(p, ts_use):
                dated.append(str(p))
            if normalize_h1_prefix(p):
                # treat as date-updated category for reporting simplicity
                if str(p) not in dated:
                    dated.append(str(p))
        # 3) ensure O3 note when keywords present (markdown only)
        if p.suffix.lower() == ".md":
            if ensure_o3_note(p):
                noted.append(str(p))
            # 4) Always normalize H1 to drop timestamp prefix if present
            normalize_h1_prefix(p)
            # 4.1) Remove leading '标题：'/'标题:' label from H1
            normalize_h1_remove_title_label(p)
            # 5) Cleanup redundant sections and enforce header spacing even if date step skipped
            cleanup_redundant_sections(p)
            # 6) Ensure author bullet exists (default GaoZheng)
            ensure_author_bullet(p)
    print(f"Renamed {len(renamed)} file(s)")
    for f in renamed:
        print(" -", f)
    print(f"Updated date in {len(dated)} markdown file(s)")
    for f in dated:
        print(" -", f)
    print(f"Inserted O3 note in {len(noted)} markdown file(s)")
    for f in noted:
        print(" -", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


