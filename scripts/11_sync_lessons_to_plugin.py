#!/usr/bin/env python3
"""
scripts/11_sync_lessons_to_plugin.py
Automatic Cross-Repository Experience Inheritance Pipeline:
  aster-d51-sim/docs/LESSONS_LEARNED.md  <-->  tools/live-steam-twin-plugin/  <-->  ~/.gemini/config/plugins/live-steam-twin/ (origin/main)

Triggered automatically via:
  1. Antigravity IDE/Agent PostToolUse hook (.agents/hooks.json) whenever LESSONS_LEARNED.md or tools/live-steam-twin-plugin/ is edited;
  2. Git pre-commit hook (.git/hooks/pre-commit --stage-workspace) so tools/live-steam-twin-plugin/ stays synced in the same commit;
  3. Git post-commit hook (.git/hooks/post-commit --commit-plugin) so ~/.gemini/config/plugins/live-steam-twin auto-commits in lockstep;
  4. Git pre-push hook (.git/hooks/pre-push --push-plugin) so `git push` on aster-d51-sim automatically pushes live-steam-twin to GitHub!
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LESSONS_MD = PROJECT_ROOT / "docs" / "LESSONS_LEARNED.md"
ARCHIVE_PLUGIN_DIR = PROJECT_ROOT / "tools" / "live-steam-twin-plugin"
GLOBAL_PLUGIN_DIR = Path.home() / ".gemini" / "config" / "plugins" / "live-steam-twin"

AUTO_MARKER = "\n<!-- AUTO-SYNCED-INCREMENTAL-LESSONS-START -->\n"


def parse_lessons_markdown(md_text: str) -> List[Dict[str, object]]:
    """Extract all `### 失误 N：Title` blocks with root-cause and mandatory-rule summaries."""
    pattern = re.compile(r"^###\s*失误\s*(\d+)[：:]\s*(.+?)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(md_text))
    lessons: List[Dict[str, object]] = []
    for idx, m in enumerate(matches):
        num = int(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(md_text)
        body = md_text[start:end].strip()
        # Trim trailing section headers (e.g., `## 二、...`)
        sec_cut = re.search(r"^##\s+", body, re.MULTILINE)
        if sec_cut:
            body = body[: sec_cut.start()].strip()
        body = re.sub(r"\n---\s*$", "", body).strip()

        # Extract concise rule / root-cause summary lines for Encyclopedia & Bonsai Auditor
        rule_lines = []
        for line in body.splitlines():
            s = line.strip()
            if any(k in s for k in ("强制门禁", "强制规范", "修复方案", "准则：", "铁律")):
                rule_lines.append(re.sub(r"^[-*\d.\s]+", "", s))
            elif rule_lines and len(rule_lines) < 4 and s.startswith(("-", "1.", "2.", "3.")):
                rule_lines.append(re.sub(r"^[-*\d.\s]+", "", s))
        summary = " ".join(rule_lines)[:360] if rule_lines else " ".join(body.splitlines()[:3])[:360]
        lessons.append(
            {
                "id": num,
                "title": title,
                "rule_summary": summary,
            }
        )
    return lessons


def update_plugin_tree(plugin_root: Path, md_text: str, lessons: List[Dict[str, object]]) -> None:
    ref_dir = plugin_root / "skills" / "live-steam-01-pdf-calibration" / "references"
    if not ref_dir.exists():
        return

    # 1. Write full canonical D51 lessons reference
    (ref_dir / "lessons_learned_d51_full.md").write_text(md_text, encoding="utf-8")

    # 2. Write structured index consumed dynamically by bonsai_semantic_auditor.py
    index_payload = {
        "source_repo": "libinghui96/aster-d51-sim",
        "source_file": "docs/LESSONS_LEARNED.md",
        "total_lessons": len(lessons),
        "max_lesson_id": max((int(x["id"]) for x in lessons), default=0),
        "lessons": lessons,
    }
    (ref_dir / "auto_synced_lessons_index.json").write_text(
        json.dumps(index_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 3. Append any incremental lessons (ID > 27) into mistakes_27_encyclopedia.md idempotently
    enc_path = ref_dir / "mistakes_27_encyclopedia.md"
    if enc_path.exists():
        enc_txt = enc_path.read_text(encoding="utf-8")
        base_txt = enc_txt.split(AUTO_MARKER)[0].rstrip() + "\n"
        extra = [it for it in lessons if int(it["id"]) > 27]
        if extra:
            lines = [AUTO_MARKER.lstrip("\n"), "## E. Incremental Auto-Synced Lessons (Mistakes 28+)\n"]
            for it in extra:
                lines.append(f"- **M{it['id']} ({it['title']})**: {it['rule_summary']}\n")
            enc_path.write_text(base_txt + "\n" + "".join(lines), encoding="utf-8")
        else:
            enc_path.write_text(base_txt, encoding="utf-8")


def mirror_plugin_dirs(src: Path, dst: Path) -> None:
    """Sync plugin files from src to dst, preserving dst/.git and Archive's install_to_global.sh."""
    if not src.exists() or not dst.exists():
        return
    subprocess.run(
        [
            "rsync",
            "-a",
            "--delete",
            "--exclude=.git",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=install_to_global.sh",
            f"{src}/",
            f"{dst}/",
        ],
        check=True,
    )


def sync_all(prefer_global: bool = False) -> int:
    if not LESSONS_MD.exists():
        return 0
    md_text = LESSONS_MD.read_text(encoding="utf-8")
    lessons = parse_lessons_markdown(md_text)

    if GLOBAL_PLUGIN_DIR.exists() and ARCHIVE_PLUGIN_DIR.exists():
        if prefer_global:
            mirror_plugin_dirs(GLOBAL_PLUGIN_DIR, ARCHIVE_PLUGIN_DIR)
        else:
            mirror_plugin_dirs(ARCHIVE_PLUGIN_DIR, GLOBAL_PLUGIN_DIR)

    for target_dir in (ARCHIVE_PLUGIN_DIR, GLOBAL_PLUGIN_DIR):
        if target_dir.exists():
            update_plugin_tree(target_dir, md_text, lessons)
    return len(lessons)


def commit_global_plugin() -> bool:
    if not (GLOBAL_PLUGIN_DIR / ".git").exists():
        return False
    st = subprocess.run(
        ["git", "-C", str(GLOBAL_PLUGIN_DIR), "status", "--porcelain"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not st:
        return False

    d51_sha = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip() or "worktree"
    d51_subj = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "log", "-1", "--pretty=%s"],
        capture_output=True,
        text=True,
    ).stdout.strip()[:72]

    subprocess.run(["git", "-C", str(GLOBAL_PLUGIN_DIR), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(GLOBAL_PLUGIN_DIR),
            "commit",
            "-m",
            f"sync(d51): auto-inherit lessons from aster-d51-sim@{d51_sha} ({d51_subj})",
        ],
        check=True,
    )
    return True


def push_global_plugin() -> None:
    if not (GLOBAL_PLUGIN_DIR / ".git").exists():
        return
    commit_global_plugin()
    ahead = subprocess.run(
        ["git", "-C", str(GLOBAL_PLUGIN_DIR), "rev-list", "--count", "@{u}..HEAD"],
        capture_output=True,
        text=True,
    )
    if ahead.returncode == 0 and int(ahead.stdout.strip() or "0") > 0:
        print("[live-steam-twin] Auto-pushing synced lessons to git@github.com:libinghui96/live-steam-twin.git ...", file=sys.stderr)
        res = subprocess.run(["git", "-C", str(GLOBAL_PLUGIN_DIR), "push", "origin", "main"])
        if res.returncode != 0:
            print("[live-steam-twin WARN] Failed to auto-push global plugin (will retry on next push).", file=sys.stderr)


def handle_post_tool_hook() -> None:
    try:
        raw = sys.stdin.read()
        if raw.strip():
            payload = json.loads(raw)
            args = (payload.get("toolCall") or {}).get("args") or {}
            target = str(args.get("Path") or args.get("TargetFile") or args.get("AbsolutePath") or "")
            if "LESSONS_LEARNED.md" in target or "live-steam-twin" in target:
                prefer_g = ".gemini/config/plugins/live-steam-twin" in target
                sync_all(prefer_global=prefer_g)
    except Exception:
        pass
    sys.stdout.write("{}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync aster-d51-sim lessons learned to global & remote live-steam-twin plugin")
    ap.add_argument("--post-tool-hook", action="store_true", help="Run as Antigravity PostToolUse hook")
    ap.add_argument("--stage-workspace", action="store_true", help="Sync and git-add tools/live-steam-twin-plugin in aster-d51-sim")
    ap.add_argument("--commit-plugin", action="store_true", help="Auto-commit changes inside ~/.gemini/config/plugins/live-steam-twin")
    ap.add_argument("--push-plugin", action="store_true", help="Auto-commit and push ~/.gemini/config/plugins/live-steam-twin to origin/main")
    ap.add_argument("--prefer-global", action="store_true", help="Mirror ~/.gemini/config/plugins/live-steam-twin -> tools/ first")
    args = ap.parse_args()

    if args.post_tool_hook:
        handle_post_tool_hook()
        return

    count = sync_all(prefer_global=args.prefer_global)
    if args.stage_workspace and ARCHIVE_PLUGIN_DIR.exists():
        subprocess.run(["git", "-C", str(PROJECT_ROOT), "add", "tools/live-steam-twin-plugin"], check=True)
    if args.commit_plugin:
        committed = commit_global_plugin()
        if committed:
            print(f"[live-steam-twin] Auto-committed {count} synced lessons into ~/.gemini/config/plugins/live-steam-twin")
    if args.push_plugin:
        push_global_plugin()
    if not (args.stage_workspace or args.commit_plugin or args.push_plugin):
        print(f"[SYNC-OK] Synced {count} lessons from docs/LESSONS_LEARNED.md -> live-steam-twin")


if __name__ == "__main__":
    main()
