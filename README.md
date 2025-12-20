# llm-kit

**llm-kit** is a small, opinionated collection of reusable utilities for building **production-grade LLM and RAG systems**.

It is **not** a framework.
It does **not** try to replace LangChain or LlamaIndex.

The goal is simple:

> **Extract the boring, repeated infrastructure that shows up across real LLM-backed services — and keep it minimal, explicit, and boring.**

---

## Why this exists

After building multiple LLM-powered services (RAG systems, agentic workflows, tool-driven backends), the same patterns keep repeating:

* embedding generation
* chunking with metadata
* vector database access
* LLM tool calling
* prompt versioning

Copy-pasting these across services leads to:

* subtle inconsistencies
* poor observability
* unclear contracts
* harder maintenance

**llm-kit exists to solve only that problem — nothing more.**

---

## Design principles

llm-kit follows a few strict rules:

* **Thin abstractions only**
  If an abstraction doesn’t remove duplication across multiple apps, it doesn’t belong here.

* **Constraints over flexibility**
  Fewer supported options, chosen deliberately.

* **Explicit > clever**
  No magic, no hidden behavior, no DSLs.

* **Infrastructure, not orchestration**
  Business logic and agent flows stay in the application layer.

* **Easy to delete**
  If llm-kit ever becomes more expensive than useful, it should be easy to remove.

---

## What’s included

### 1. Embeddings client

A small wrapper around embedding providers to standardize:

* model selection
* batching
* retries / timeouts
* metadata passthrough

*Not a vector database.*

---

### 2. Chunking utilities

A single, predictable chunking strategy:

* sliding window
* metadata preservation

No recursive chunking zoo.
No document-type heuristics.

---

### 3. Vector DB interface

A narrow interface for vector stores such as:

* pgvector
* Qdrant

Exposes only:

* `upsert`
* `search`
* `delete`

No query DSLs. No smart ranking logic.

---

### 4. LLM tool-calling helpers

Utilities to:

* define tool schemas
* invoke tools safely
* capture tool usage for logging / observability
* handle retries and failures cleanly

This is infrastructure glue, not agent logic.

---

### 5. Prompt registry

A deliberately simple approach to prompt management:

* prompts stored as files
* explicit versioning
* clear change history

No runtime mutation.
No prompt DSLs.
No “prompt management platform”.

---

## What this is **not**

llm-kit explicitly does **not** aim to be:

* a full RAG framework
* an agent orchestration engine
* a prompt engineering playground
* a research toolkit
* a replacement for LangChain / LlamaIndex

Those tools are excellent at what they do.
llm-kit focuses on **shared infrastructure**, not orchestration or experimentation.

---

## Installation

Using Poetry (recommended):

```bash
poetry add git+https://github.com/<your-username>/llm-kit.git@v0.1.0
```

You can also use it as a local path dependency during development.

---

## Versioning

* `v0.x` — APIs may change
* `v1.0.0` — public API considered stable

Breaking changes are introduced only with version bumps.

---

## License

This project is **unlicensed**.

You are free to:

* use it
* copy it
* modify it
* learn from it

No guarantees. No warranties.
If it breaks, you get to keep both pieces.

---

## Who this is for

llm-kit is useful if you:

* build multiple LLM-backed services
* care about engineering discipline
* want consistency without over-engineering
* prefer small, readable code over large frameworks

If you’re looking for a feature-rich AI framework — this is not it.

---

## Philosophy

> **llm-kit exists to make LLM systems boring — so the interesting problems stay in your application.**
