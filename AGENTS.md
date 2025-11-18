# Repository Guidelines

## Project Structure & Module Organization
This repository couples blockchain networking with personalized federated learning. `main.py` and `Console.py` orchestrate runtime flows, while `blockchain/node/` contains SN/EN services, gRPC stubs in `base_package/`, database adapters, and split-FL logic. The stand-alone `fl/` package (Configurator, Server, Client, loaders) runs FedPer simulations. Shared configs live in `config.py` and `nodeconfig.json`, and artifacts such as datasets, checkpoints, and attacks are stored in `data/`, `models.py`, `DLGAttack/`, and evaluation harnesses under `multi_model_test/`.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: create a local Python 3.7 sandbox.
- `pip install -r requirements.txt`: install PyTorch/grpc dependencies (set README mirrors when needed).
- `python -m grpc_tools.protoc -I blockchain/node/proto --python_out=blockchain/node/base_package --grpc_python_out=blockchain/node/base_package blockchain/node/proto/data.proto`: regenerate RPC bindings after proto edits.
- `python main.py --port 8080 --nt SN --entry 127.0.0.1:8080 -z`: start the first super node; reuse with new ports for more SN/EN nodes.
- `python Console.py --ip 127.0.0.1 --port 8080` then `train()` / `trainFL()`: issue runtime commands.
- `python fl/main.py`: run the serialized FedPer loop for offline experiments.

## Coding Style & Naming Conventions
Follow PEP8 with 4-space indentation, `snake_case` modules, and PascalCase classes (`Server`, `Client`, `Configurator`). Fetch settings through `config.my_conf` or `Configurator()` instead of literals, and surface behavior toggles as CLI flags when possible. Reuse the global `logging` formatter from `fl/main.py`, and never hand-edit generated gRPC files in `blockchain/node/base_package`—rerun the proto command instead.

## Testing Guidelines
`python model_eval_test.py` validates saved checkpoints, and `python multi_model_test/main.py` compares architectures. Stage exploratory scripts next to `my_test.py` or inside `multi_model_test/`, naming helpers `*_test.py`. Seed `torch`, `random`, and data loaders before collecting accuracy/loss artifacts under `data/flm` or `data/model` so regressions stay reproducible.

## Commit & Pull Request Guidelines
Recent commits (`新增初始化模型client节点`) show terse imperative subjects; follow that style and mention the touched subsystem (`blockchain/node`, `fl`, `data`). In PRs, cite the motivating issue, list reproduction commands, call out config or dataset changes, and attach logs/screens whenever accuracy shifts.

## Security & Configuration Tips
Mask real IPs and credentials inside `nodeconfig.json`, overriding them via environment variables in deployment scripts. Keep large datasets or private weights outside Git, referencing paths through `config.py`, and scrub grpc certificates or checkpoints under `data/` before sharing.
