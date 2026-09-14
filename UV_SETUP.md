# Working with this repo using `uv`

This project used to be set up with `pyenv` + `python -m venv` + `pip install -r requirements.txt`.
**Those files have been removed.** Everything is now managed by
[uv](https://docs.astral.sh/uv/), which handles the Python interpreter, the virtual environment,
dependency resolution and locking in one tool. You do **not** need pyenv, conda, or pip anymore.

---

## 1. Install uv (once per machine)

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatives: `brew install uv`, `winget install --id=astral-sh.uv`, or `pipx install uv`.

Restart your shell, then verify:

```bash
uv --version
```

To upgrade uv later: `uv self update`.

---

## 2. Clone the repo

```bash
git clone <repo-url>
cd capstone-Matchas
```

---

## 3. Create the environment — one command

```bash
uv sync --all-groups
```

That single command does all of this for you:

1. Reads `pyproject.toml` to find the required Python version (`>=3.11`).
2. **Downloads and installs a matching Python interpreter automatically** if you don't have one —
   no pyenv, no system Python needed.
3. Creates a virtual environment in `.venv/` inside the project.
4. Installs every runtime dependency **and** the `dev` group (JupyterLab, pytest, black, …).
5. Writes/uses `uv.lock` so everyone on the team gets byte-identical versions.

Variants:

```bash
uv sync                  # runtime dependencies only (deployment / CI inference)
uv sync --all-groups     # runtime + dev tools (what you want for development)
uv sync --frozen         # install exactly what uv.lock says, never re-resolve (use in CI)
```

> There is no `pyenv local`, no `python -m venv`, no `pip install`. The old
> `requirements.txt`, `requirements_dev.txt` and `environment.yml` have been deleted —
> `pyproject.toml` plus `uv.lock` replace all three. (`make setup` still works; it now
> just calls `uv sync --all-groups`.)

---

## 4. Running things

The idiomatic way is `uv run`. It makes sure the environment is in sync **before** running,
so you never work against a stale venv, and you never have to activate anything:

```bash
uv run python -m modeling.train      # train the model
uv run python -m modeling.predict models/linear data/X_test.csv data/y_test.csv
uv run pytest                        # run tests
uv run black .                       # format
uv run jupyter lab                   # open the notebooks
uv run mlflow ui                     # start the MLflow UI at http://127.0.0.1:5000
```

If you prefer a classic activated shell, that still works:

```bash
source .venv/bin/activate     # macOS / Linux
.venv\Scripts\activate        # Windows (PowerShell / cmd)
python -m modeling.train
```

but `uv run` is recommended — it's the only form that guarantees the env matches `uv.lock`.

The `Makefile` wraps the common ones: `make setup`, `make train`, `make predict`, `make lab`,
`make mlflow`, `make test`, `make format`.

---

## 5. Managing packages

Never call `pip install` again. Use uv, which updates `pyproject.toml` and `uv.lock` for you:

```bash
uv add seaborn                   # add a runtime dependency
uv add --group dev ruff          # add a dev-only dependency
uv add "pandas>=2.2"             # add with a version constraint
uv remove statsmodels            # remove a dependency
```

Upgrading to the newest releases:

```bash
uv lock --upgrade                # re-resolve everything to the newest compatible versions
uv lock --upgrade-package mlflow # bump a single package
uv sync --all-groups             # apply the new lock to your .venv
```

Inspect what you have:

```bash
uv pip list                      # installed packages
uv tree                          # dependency tree
```

---

## 6. Python version

The required version lives in `pyproject.toml`:

```toml
requires-python = ">=3.11"
```

uv picks (and downloads, if needed) a suitable interpreter on its own. To pin a specific one
for this project:

```bash
uv python pin 3.12          # writes .python-version, uv will honour it
uv python list              # see available / installable interpreters
uv python install 3.12      # install an interpreter explicitly
```

Note: `.python-version` is currently listed in `.gitignore`. If the team wants everyone on the
exact same interpreter, remove that line and commit the file.

---

## 7. What to commit

| File | Commit? | Why |
|---|---|---|
| `pyproject.toml` | Yes | declares the dependencies |
| `uv.lock` | Yes | guarantees reproducible installs for everyone |
| `.venv/` | No | already in `.gitignore`, rebuilt by `uv sync` |
| `requirements*.txt`, `environment.yml` | Deleted | replaced by `pyproject.toml` + `uv.lock`; recover from git history if ever needed |

If some external system still needs a `requirements.txt`, generate it instead of hand-editing:

```bash
uv export --no-dev --format requirements-txt -o requirements.txt
```

---

## 8. Current dependency set

Declared in `pyproject.toml` (no pinned versions — uv resolves the newest compatible releases
and records the exact ones in `uv.lock`):

**Runtime:** `matplotlib`, `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `mlflow`, `parsenvy`

**Dev group:** `jupyterlab`, `seaborn`, `pytest`, `testbook`, `black`, `nbdime`

Verified working resolution (Sept 2026): Python 3.14.6, pandas 3.0.5, numpy 2.5.3,
scikit-learn 1.9.1, mlflow 3.16.0, statsmodels 0.15.0, matplotlib 3.11.2.

---

## 9. MLflow URI (unchanged)

Still not stored in git. Either write it to a local file:

```bash
echo http://127.0.0.1:5000/ > .mlflow_uri
```

or export it as an environment variable:

```bash
export MLFLOW_URI=http://127.0.0.1:5000/      # macOS / Linux
$env:MLFLOW_URI = "http://127.0.0.1:5000/"    # Windows PowerShell
```

`modeling/config.py` reads the file first, then the env var.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `uv: command not found` | Restart the shell; make sure `~/.local/bin` (or `%USERPROFILE%\.local\bin`) is on `PATH`. |
| Environment feels stale / wrong versions | `uv sync --all-groups` — it reconciles `.venv` with the lock, including removals. |
| Want a clean rebuild | Delete `.venv/` and run `uv sync --all-groups`. |
| Resolution conflict after `uv add` | Loosen the constraint, or run `uv lock --upgrade` to re-resolve the whole graph. |
| Notebook uses the wrong kernel | Launch with `uv run jupyter lab`, or register a kernel: `uv run python -m ipykernel install --user --name capstone-matchas`. |
| CI must not change the lock | Use `uv sync --frozen` (fails if `uv.lock` is out of date). |
