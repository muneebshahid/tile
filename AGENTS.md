# Guidelines

Any violations must be flagged and fixed

- Follow DDD principles
- Follow SOLID principles
- Code must be modular
- Functions should read like a table of contents
- Higher level functions or classes should be defined before lower level ones i.e. at the top of the file
- Any function greater than 50 lines should be broken down into smaller functions
- Always `uv` to add or remove libraries
- Avoid using `Any` or `object` as a type hint if possible.
- Avoid hacks, workarounds, and temporary fixes. If you find yourself writing one, stop take a step back and instead propose a more architecturally sound solution. Refactoring is to be considered a normal part of the development process.
- The general development approach must be to get to an end to end mvp with absolutely minimal features and as simple as possible.
- After changes run
  - `make format` to format the code
  - `make type_check` to check for type errors
  - `make test` to run tests
