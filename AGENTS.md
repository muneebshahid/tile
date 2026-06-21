# Guidelines

Any violations must be flagged and fixed

- Follow DDD principles
- Follow SOLID principles
- Code must be modular
- Follow clean code principles
- Functions should read like a table of contents
- Higher level functions or classes should be defined before lower level ones, for example at the top of the file
- Any function greater than 50 lines should be broken down into smaller functions
- Always `uv` to add or remove libraries
- Avoid using `Any` or `object` as a type hint if possible.
- Avoid hacks, workarounds, and temporary fixes. If you find yourself writing one, stop, take a step back, and propose a more architecturally sound solution. Refactoring is a normal part of the development process.
- The general development approach must be to get to an end-to-end MVP with absolutely minimal features and as simple as possible.
- After changes run
  - `make format` to format the code
  - `make type_check` to check for type errors
  - `make test` to run tests
- Add documentation and docstrings to functions, classes, modules, and variables.

## Skill Loop

- ONLY when asked, run the following skills in order. Load only one skill at a time, and wait for its results before running the next skill.
  - Run DDD SOLID Enforcer to review code for SOLID principles and modular structure.
  - Run Clean Architecture Review Skill to ensure proper layering and separation of concerns.
  - Run Code Reviewer Skill to find potential bugs, code smells, and areas for improvement.
  - Run Test Coverage Auditor Skill to identify untested code paths.
  - Run Code Simplifier to simplify complex code sections for readability and maintainability.
  - Run Code Beautifier to improve consistency with formatting standards.
  - If flow diagrams exist, check whether they are still accurate or need updating.
