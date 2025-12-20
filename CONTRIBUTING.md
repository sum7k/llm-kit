# Contributing to llm-kit

Thanks for your interest in contributing.

llm-kit is intentionally small and opinionated.
Contributions are welcome **only if they align with its goals**.

---

## What belongs here

Contributions should satisfy **at least one** of the following:

* Remove duplication across multiple LLM/RAG services
* Improve clarity, correctness, or reliability of existing utilities
* Tighten interfaces or reduce unnecessary complexity
* Fix bugs or edge cases in current functionality
* Improve documentation where behavior is unclear

If a change adds flexibility without clear need, it probably doesn’t belong.

---

## What does *not* belong here

Please avoid proposals that:

* Turn llm-kit into a framework
* Add speculative abstractions
* Introduce orchestration, agents, or workflows
* Re-implement features from LangChain / LlamaIndex
* Add “just in case” configuration options

Business logic and experimentation belong in application code, not here.

---

## Design expectations

* Keep abstractions **thin**
* Prefer **explicit code** over cleverness
* Add **constraints**, not options
* Optimize for **readability and deletion**
* If something can’t be explained simply, it’s probably too complex

---

## Pull requests

Before opening a PR:

1. Make sure the change has a clear, narrow purpose
2. Keep the diff small and focused
3. Add or update tests where appropriate
4. Ensure existing tests pass

PRs may be declined if they increase complexity without clear benefit.

---

## Issues

Issues are welcome for:

* bugs
* incorrect behavior
* unclear documentation
* missing edge cases

Feature requests should include a concrete, real-world use case.

---

## Code of conduct

Be respectful, concise, and constructive.
