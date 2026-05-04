# content-manager

A Rust CLI tool that transforms raw social-platform exports into structured, AI-enriched datasets. Import your saved posts, run them through an LLM for summaries, tags, and categories, then export to JSON or SQLite — and browse the results in a self-contained HTML viewer.

**Current supported import source:** Facebook saved posts via the [J2Team](https://j2team.dev/) browser extension.

---

## Features

- **Import** — Parse J2Team Facebook export files with streaming reads (handles 10k+ records)
- **Process** — Analyze posts with Ollama (local) or OpenAI, producing summaries, tags, categories, and relevance scores
- **Export** — Write results to JSON or SQLite for downstream use
- **Viewer** — Browse, filter, and search your data in a zero-dependency static HTML page
- **Datasets** — Manage multiple independent named datasets
- **Stats** — Inspect dataset composition at a glance

---

## Prerequisites

- [Rust](https://rustup.rs/) 1.75 or later
- One of:
  - [Ollama](https://ollama.com/) running locally (default, no API key needed)
  - An OpenAI API key (set `OPENAI_API_KEY` in your environment)

---

## Build

```bash
git clone https://github.com/your-username/content-manager.git
cd content-manager

cargo build --release
```

The binary is placed at `target/release/content-manager`.

To run without a separate build step:

```bash
cargo run --release -- <subcommand> [options]
```

---

## Configuration

The app reads `config.toml` from the **current working directory** at startup. All fields are optional — defaults are shown below.

```toml
[llm]
provider = "ollama"          # "ollama" | "openai"
endpoint = "http://localhost:11434/api/generate"  # Ollama only
model    = "llama3"

[llm.batch]
size        = 10     # posts per LLM request
max_tokens  = 2048
temperature = 0.3

[processing]
skip_existing     = true   # skip posts already analyzed
min_content_length = 20    # ignore posts shorter than this (title + link chars)
max_items         = 0      # 0 = unlimited; set > 0 to cap a single run

[output]
dir = "output"

[storage]
path = "data/content-manager.db"

[logging]
level = "info"   # error | warn | info | debug | trace
```

When using OpenAI, set your API key before running:

```bash
export OPENAI_API_KEY=sk-...
```

---

## Usage

All commands are run from the project root (where `config.toml` lives).

### Import

Parse a J2Team Facebook export file and load it into a named dataset.

```bash
content-manager import \
  --source facebook \
  --dataset my_saved_posts \
  --file /path/to/j2team_export.json
```

Output:
```
Imported: parsed=1234, inserted=1230, skipped=4
```

### Process

Run LLM analysis on all unprocessed posts in a dataset.

```bash
content-manager process --dataset my_saved_posts
```

Output:
```
Processed: processed=1230, skipped=0, ignored=12
```

### Export

Export a processed dataset to a file.

```bash
# JSON (for the viewer)
content-manager export \
  --dataset my_saved_posts \
  --format json \
  --output output/my_saved_posts.json

# SQLite
content-manager export \
  --dataset my_saved_posts \
  --format sqlite \
  --output output/my_saved_posts.db
```

Output:
```
Exported 1218 items to output/my_saved_posts.json
```

### List datasets

```bash
content-manager datasets
```

Output:
```
my_saved_posts (facebook) created 2026-05-03T10:00:00Z
```

### Dataset stats

```bash
content-manager stats --dataset my_saved_posts
```

Output:
```
Dataset: my_saved_posts
  Total:       1234
  Valid:       1218
  Ignored:     12
  Unprocessed: 0
  Category distribution:
    Technology: 420
    Business: 310
    Education: 200
    Entertainment: 150
    Other: 138
    Travel: 24
    Personal: 16
```

---

## Viewer

Open `viewer/index.html` in any browser. Click **Load JSON** and select the file produced by `export --format json`. No server or build step required.

Features:
- Filter by category
- Filter by one or more tags (AND logic)
- Search by title or tag
- Click any card title to open the original link

---

## Full pipeline example

```bash
# 1. Import
content-manager import --source facebook --dataset demo --file ~/Downloads/saved.json

# 2. Process (uses Ollama by default)
content-manager process --dataset demo

# 3. Export
content-manager export --dataset demo --format json --output output/demo.json

# 4. Open the viewer
open viewer/index.html   # macOS
# or just double-click viewer/index.html in your file manager
```

---

## Running tests

```bash
cargo test
```

---

## Project structure

```
content-manager/
├── src/
│   ├── cli/          # Argument parsing and command dispatch (clap)
│   ├── config/       # TOML config loading and validation
│   ├── importer/     # Source-specific parsers (Facebook/J2Team)
│   ├── processor/    # LLM batching, filtering, retry logic
│   ├── storage/      # SQLite persistence (rusqlite + tokio-rusqlite)
│   ├── exporter/     # JSON and SQLite output writers
│   ├── models/       # Shared data types
│   └── main.rs
├── tests/
│   └── integration_test.rs
├── viewer/
│   └── index.html    # Self-contained static HTML viewer
├── data/             # SQLite database (created at runtime)
├── output/           # Export output (created at runtime)
└── config.toml       # Your local config (not committed)
```

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0`  | Success |
| `1`  | User error (bad config, dataset not found, unknown source/format) |
| `2`  | Internal error (unexpected storage or LLM failure) |

---

## License

MIT
