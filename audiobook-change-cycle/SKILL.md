---
name: audiobook-change-cycle
description: Use for changes in J:\audiobook when the user expects each change to be tested, committed, and pushed automatically.
metadata:
  short-description: Test, commit, and push audiobook changes
---

# Audiobook change cycle

Apply this workflow to every code, dictionary, GUI, documentation, or skill change in `J:\audiobook`.

1. Read `AGENTS.md` and the relevant `CLAUDE.md` section before editing. Preserve the project rules: no Docker, dependencies only through `install.bat`, do not change ESpeech RL-V2, and do not commit `fb2_reader_settings.json` or `dialogue_characters.json`.
2. Make the smallest complete change. Keep unrelated working-tree changes out of the commit.
3. Run the narrowest meaningful checks after the change. For Python, run `python -m py_compile` on changed modules. For stress or pronunciation changes, run `.venv\Scripts\python.exe tools\test_accentuation.py`. For GUI changes, also check the affected imports and helpers. Do not substitute system Python when the project environment is available.
4. If a check fails because of an environment issue, diagnose it before committing. Record the exact command and failure in the final report. Do not claim the tests passed.
5. Inspect `git diff --check` and `git status`. Commit the verified change with a concise message.
6. Push the commit to the current upstream branch. Confirm the commit hash and that the working tree is clean or clearly report remaining unrelated changes.

The user’s instruction to test, commit, and push authorizes this routine repository workflow. Do not ask again for permission after each ordinary change. Ask only if the requested operation would be destructive, targets a different repository, or requires an unapproved external action.

When a later user request adds a new constraint, apply it to the current change before testing and committing. Report three things briefly: what changed, what checks ran and their result, and the pushed commit.
