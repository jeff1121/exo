# exo Evolution Plan

> Generated: 2026-06-29 | Branch: `copilot/update-build-evolution-plan`

---

## Executive Summary

**exo** is a distributed AI inference system that connects multiple devices into a cluster, enabling large language models to run across multiple machines using MLX as the primary inference backend and zenoh for peer-to-peer networking. The system uses an event-sourcing architecture with Rust bindings (exo_rs) for high-performance networking.

This plan outlines a phased evolution across six pillars: reliability, performance, developer experience, platform expansion, security hardening, and observability.

---

## Current State Assessment

### Architecture Strengths
- Clean event-sourcing pattern (immutable `State`, pure `apply()` function)
- Typed pub/sub messaging via `TypedTopic[T]`
- Pydantic models with `frozen=True` and `strict=True` throughout
- Rust-backed networking (zenoh + RDMA) with PyO3 bindings
- OpenAI / Claude / Ollama API compatibility

### Known Technical Debt (from TODO.md + code inline TODOs)
| Priority | Item | Location |
|---|---|---|
| P1 | `EXO_BOOTSTRAP_PEERS` is broken | TODO.md#1 |
| P1 | Tensor-parallel shard rank assignments are ugly | `placement_utils.py` |
| P2 | New prompt during decode blocks existing batch (no continuous batching interleaving) | TODO.md#7 |
| P2 | Offline model detection uses internet for partial-file check | TODO.md#8 |
| P2 | Memory pressure not tracked (only memory used) | TODO.md#13 |
| P3 | Network connection type not shown in UI (TB5 vs Ethernet) | TODO.md#14 |
| P3 | No dynamic switch to higher-priority connection | TODO.md#16 |
| P3 | No model streaming between cluster nodes | TODO.md#17 |
| P3 | Log cleanup / per-module filters | TODO.md#27 |
| P3 | Validate RDMA connections with `ibv_devinfo` | TODO.md#28 |
| P3 | Profiled network latency/bandwidth display | TODO.md#4, #5 |

---

## Phase 1 — Foundation Hardening (0–2 months)

### 1.1 Fix Critical Bugs
- **EXO_BOOTSTRAP_PEERS** — Repair peer bootstrap logic so manual cluster formation is reliable without relying solely on mDNS.
- **Offline model detection** — Replace internet-dependent partial-file check with a local heuristic: non-empty model folder with zero `.partial` files = fully downloaded.

### 1.2 Type Safety & Linting
- Resolve all remaining `basedpyright` strict-mode violations; enforce zero-error CI gate.
- Add `ruff` auto-fix rules for common patterns (`UP`, `B`, `SIM`).
- Annotate all `# TODO(ciaran): type this` and `# TODO(evan): ...` callsites with proper types.

### 1.3 Test Coverage Expansion
- Add unit tests for `apply.py` covering all event types (currently partial coverage).
- Add integration tests for peer discovery / topology formation (uses mocked zenoh).
- Add regression test for EXO_BOOTSTRAP_PEERS fix.

### 1.4 Security Baseline
- Audit all HTTP endpoints in `src/exo/api/main.py` for authentication bypass risks.
- Validate Pydantic model input lengths / ranges to prevent resource exhaustion.
- Dependency audit: pin all transitive Python deps; run `pip-audit` in CI.

---

## Phase 2 — Performance & Scalability (2–4 months)

### 2.1 Continuous Batching
- Implement interleaved prefill/decode so a new prompt does not block an in-progress decode batch (TODO.md#7).
- Design `BatchScheduler` abstraction in `worker/engines/mlx/generator/` to decouple scheduling from generation.

### 2.2 Memory Pressure Tracking
- Replace static "memory used" metrics with real-time memory pressure signals (macOS `vm_stat`, Linux `/proc/meminfo`).
- Surface pressure levels to the master's placement algorithm to avoid OOM-induced crashes.

### 2.3 Topology-Aware Placement Improvements
- Fix the tensor-parallel shard condition (`placement_utils.py` TODO).
- Encode rank assignments explicitly in `ShardAssignment` type rather than inferring from ordering.
- Profile and instrument actual connection speeds (TODO.md#4, #5) to feed placement decisions.

### 2.4 Model Streaming Between Nodes
- Implement chunk-based model transfer from a node that already has the model to a new node (TODO.md#17), eliminating redundant HuggingFace downloads within a LAN cluster.

### 2.5 RDMA Validation
- Integrate `ibv_devinfo` output into the info-gatherer (TODO.md#28) to surface RDMA adapter state in the dashboard.

---

## Phase 3 — Developer Experience (3–5 months)

### 3.1 Structured Logging
- Replace ad-hoc `logger.debug` calls with structured log records (key=value pairs).
- Add per-module log level filters; default to `WARNING` in production, `DEBUG` in `EXO_TESTS=1` environments.
- Consolidate log output into a single file rotated daily (currently scattered to stdout).

### 3.2 Observability Dashboard
- Expose per-link bandwidth utilization and latency in the Svelte dashboard (TODO.md#5).
- Show connection type (Thunderbolt 5, Ethernet, WiFi) for each topology edge (TODO.md#14).
- Add a live "cluster health" view showing memory pressure, download progress, and runner states.

### 3.3 CLI Improvements
- Add `exo status` subcommand: print current cluster state, running models, and node health.
- Add `exo models list --local` to enumerate locally downloaded models without starting the full daemon.
- Add `exo models copy <model> <peer>` for manual model-to-peer transfer.

### 3.4 API Surface Hardening
- Add OpenAPI schema validation tests for all API endpoints.
- Add rate-limiting middleware to the FastAPI server.
- Document all public API endpoints with accurate response schemas.

---

## Phase 4 — Platform Expansion (4–8 months)

### 4.1 Linux / CUDA Backend
- Extend the inference backend beyond MLX to support NVIDIA CUDA (via PyTorch / vLLM shim).
- Abstract `Runner` interface so `mlx`, `cuda`, and future backends are interchangeable.
- Add CI matrix for Linux + CUDA in GitHub Actions (using self-hosted GPU runners).

### 4.2 Windows Preview
- Evaluate DirectML / WSL2 as inference paths on Windows.
- Ship a Windows installer via `packaging/` scripts.

### 4.3 Mobile / Edge Nodes
- Investigate iOS / Android device participation as read-only cluster nodes (receive inference results, contribute memory shards).

### 4.4 Dynamic Connection Upgrade
- Implement priority-based connection switching: automatically promote a node to a higher-bandwidth link when it becomes available (TODO.md#16), integrating with `InstanceReplacedAtomically`.

---

## Phase 5 — Security Hardening (ongoing, accelerated in months 2–6)

### 5.1 Authentication & Authorization
- Add an optional token-based auth layer to the API server (configurable via `~/.exo/config.toml`).
- Implement node identity verification using ed25519 keys during zenoh peer handshake.

### 5.2 Secret & Credential Management
- Document that API keys (HuggingFace, OpenAI-compat) must be passed via environment variables, not config files.
- Add `secret_scanning` to CI to prevent accidental credential commits.

### 5.3 Dependency Supply-Chain
- Pin all Rust crates in `Cargo.lock` and audit with `cargo audit` in CI.
- Pin all Python deps via `uv.lock` and audit with `pip-audit` in CI.
- Enable GitHub Dependabot for both `Cargo.toml` and `pyproject.toml`.

### 5.4 Network Security
- Evaluate zenoh TLS transport for inter-node communication on untrusted networks.
- Add optional mTLS for RDMA control-plane connections.

---

## Phase 6 — Ecosystem & Community (6–12 months)

### 6.1 Plugin / Backend SDK
- Define a stable `Runner` protocol that third-party backends can implement.
- Publish SDK documentation and a minimal "echo runner" reference implementation.

### 6.2 Model Registry
- Build a lightweight registry service that clusters can query for available model cards.
- Support custom / private model registries via config.

### 6.3 Benchmarking Suite
- Formalize the existing benchmark scripts into a reproducible `exo bench` CLI command.
- Publish automated benchmark results to a public dashboard on each main-branch commit.

### 6.4 Community Infrastructure
- Add a `CHANGELOG.md` with semantic versioning.
- Set up GitHub Release automation from the existing `packaging/` scripts.
- Create contributor-facing architecture diagram (auto-generated from module dependency graph).

---

## Success Metrics

| Metric | Current | Phase 1 Target | Phase 3 Target |
|---|---|---|---|
| `basedpyright` errors | Run `uv run basedpyright` to establish baseline¹ | 0 | 0 (enforced in CI) |
| Test coverage | Partial (no coverage gate) | +20 pp | +50 pp |
| Time-to-first-token (2-node, 70B) | Measure with `uv run exo bench` | Baseline | −15% (batch sched.) |
| CI pipeline time | Measure from Actions run history² | < 10 min | < 8 min |
| Dashboard load time | Measure with Playwright `page.metrics()`² | < 2 s | < 1 s |
| Open TODO/FIXME items in code | ~30 (grep count as of 2026-06-29) | < 20 | < 10 |

> ¹ `uv run basedpyright` requires the full uv/Python dev environment on macOS — run locally before starting Phase 1.
> ² Baseline measurements should be captured in the first sprint of Phase 1 and committed to `docs/baselines.md`.

---

## Quick Wins (this sprint)

1. Fix offline model detection (no internet required for local model check).
2. Resolve `basedpyright` violations in `src/exo/shared/types/`.
3. Add `cargo audit` and `pip-audit` steps to CI.
4. Expose connection type (TB5/Ethernet) in topology edges in the dashboard.
5. Add `exo status` CLI subcommand.
