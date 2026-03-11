# Git upload checklist

This folder was cleaned so only source and data needed to **run** the project are tracked.

## Already removed (not in repo)
- `__pycache__/` and `*.pyc` — Python cache (regenerated on run)
- `.DS_Store` — macOS metadata
- `.miniconda.exe` — installer binary (~94MB); install Miniconda separately if needed
- Nested `.git` inside `20kLogRegChallenge/` — avoids nested repo; clone again if you need that subfolder as its own repo

## Ignored by `.gitignore` (won’t be committed)
- Virtual environments (`.venv/`, `venv/`)
- `*.exe`, `*.pdf`
- IDE folders, `.env`

## First commit
```bash
cd "/path/to/Ivan_Code/Code"
git init
git add .
git status   # review
git commit -m "Initial commit"
```

## If `git add` still shows unwanted files
Add a pattern to `.gitignore` or remove the file from disk if it is safe to delete.
