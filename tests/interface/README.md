# Interface Tests

This directory contains interface and architecture-boundary tests.

These tests sit between `tests/unit` and `tests/smoke`:

- broader than a single function or class
- lighter than end-to-end smoke tests
- focused on contracts, abstract base classes, registries, and dependency boundaries

Typical scope:

- abstract base class method contracts
- required registry surface contracts
- clean-architecture dependency rules
- package-layer boundaries such as “core must not import CLI”

Run only the interface tests:

```bash
uv run pytest tests/interface -q
```
