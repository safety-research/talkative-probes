## Server quickstart (lightweight)

Minimal setup to run only `talkative_autoencoder` and the website. Skips safety-tooling and other submodules.

### One-command setup and run

```bash
make lightweight-run
```

- Starts the backend on `http://localhost:8000`.

### Two-step setup

```bash
make lightweight
cd talkative_autoencoder/website
make run
```

### Run on RunPod

```bash
cd talkative_autoencoder/website
make run RUNPOD_PORT=8000
```

### Notes

- **Targets**: `lightweight` and `lightweight-run` are defined in the root `Makefile`.
- **What it installs**: Only `talkative_autoencoder` and the website backend/frontend deps (via `uv`).
- **Website directory**: `talkative_autoencoder/website`.
- **Frontend**: optional local server at `http://localhost:3000`:

```bash
cd talkative_autoencoder/website
make run-frontend
```


