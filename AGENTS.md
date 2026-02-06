- Code must adhere to the design standards set in design.md
- File formatting, typing and shapes should adhere to the standards set in design.md
- Docstrings should be implemented as follows:
    - Clarify the expected inputs and outputs with typing and shapes
    - Clarify the semantic goal of the functions / classes
    - Show example usage of the function / class
    - Use inline comments for clarity in the implementations.

- Code formatting should adhere to ruff standards. Use ruff to ensure code style and for deduplication.
- Before implementing any code, first search for already implemented modules that can be used to solve the problem (modular design).

For each function in the codebase we require:
- A unit test testing regular (expected use)
- Unit tests for edge cases

For pipelines (policies) inside the framework we will need a smoke test:
- Load sample data (utils/test_samples.py)
- Put these through the entire pipeline
- For the pipelines, measure time metrics / memory metrics.

- Use pytest as the testing framework. Ensure calculations in tests inside the testing suite are not conducted double and are thus re-used to ensure efficiency.

- Programmatically assess whether the dependency structure (Interfaces/Drivers -> Adapters -> Applications/Use Cases -> Domain) holds and where it fails.


