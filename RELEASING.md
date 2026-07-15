# Releasing Tile

Tile publishes the `tile-runtime` distribution to PyPI from a published GitHub
release. GitHub Actions authenticates through PyPI Trusted Publishing; no PyPI
API token is stored in GitHub.

## One-time setup

1. Create a PyPI account, enable two-factor authentication, and save the
   recovery codes.
2. Create a GitHub environment named `pypi`. Add protection rules appropriate
   for the repository; requiring maintainer approval is recommended.
3. In the PyPI account's **Publishing** settings, add a pending GitHub
   publisher with these exact values:

   | Field | Value |
   |---|---|
   | PyPI project name | `tile-runtime` |
   | GitHub owner | `muneebshahid` |
   | GitHub repository | `tile` |
   | Workflow filename | `release.yml` |
   | Environment | `pypi` |

The pending publisher does not reserve the project name. Publish promptly after
configuring it.

## Release procedure

1. Update the version in `pyproject.toml` and add release notes under
   `docs/releases/`.
2. From a clean checkout of `main`, run:

   ```bash
   make format
   make type_check
   make test
   rm -rf dist
   uv build
   uvx twine check --strict dist/*
   ```

3. Confirm the wheel installs in an isolated environment and imports the public
   package:

   ```bash
   temporary_environment="$(mktemp -d)"
   uv venv "${temporary_environment}/venv" --python 3.13
   uv pip install --python "${temporary_environment}/venv/bin/python" dist/*.whl
   "${temporary_environment}/venv/bin/python" -c "import tile"
   rm -rf "${temporary_environment}"
   ```

4. Create a draft GitHub release whose tag is `v` followed by the exact package
   version. For `0.1.0`:

   ```bash
   gh release create v0.1.0 \
     --draft \
     --title "Tile 0.1.0" \
     --notes-file docs/releases/v0.1.0.md
   ```

5. Review the draft and publish it. Publishing the GitHub release triggers
   `.github/workflows/release.yml`, which verifies the tag, builds and validates
   the distributions, and publishes them through the `pypi` environment.
6. Verify the published package from a new isolated environment:

   ```bash
   temporary_environment="$(mktemp -d)"
   uv venv "${temporary_environment}/venv" --python 3.13
   uv pip install \
     --python "${temporary_environment}/venv/bin/python" \
     "tile-runtime==0.1.0"
   "${temporary_environment}/venv/bin/python" -c "import tile"
   rm -rf "${temporary_environment}"
   ```

PyPI release files are immutable. If a published release is defective, yank it
and publish a new patch version rather than attempting to replace its files.
