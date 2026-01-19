# Contributing to V6rge

Thank you for your interest in contributing! Here's how you can help.

## 🐛 Reporting Bugs

1. Check existing [Issues](https://github.com/YourUsername/V6rge/issues) first.
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Your system info (GPU, OS, VRAM)
   - Console logs if available

## 💡 Suggesting Features

Open an issue with the `enhancement` label describing:
- What problem it solves
- Proposed solution
- Alternative approaches you considered

## 🔧 Pull Requests

### Getting Started

1. Fork the repo and clone your fork.
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes.
4. Test thoroughly.
5. Commit with a descriptive message.
6. Push and open a PR.

### Code Guidelines

**Python (Backend)**
- Follow PEP8.
- Use `black` for formatting.
- Add docstrings to new functions.
- Keep services modular (one file per feature).

**JavaScript (Frontend)**
- Use ES6+ syntax.
- No semicolons.
- Use `const`/`let`, avoid `var`.

### Testing

- Test with both CPU and GPU modes if possible.
- Test with low VRAM systems (simulate with smaller models).
- Verify no regressions in existing features.

## 🏷️ Issue Labels

| Label | Description |
|-------|-------------|
| `bug` | Something isn't working |
| `enhancement` | New feature request |
| `help wanted` | We need community help |
| `good first issue` | Great for newcomers |
| `hardware-required` | Needs specific hardware to test |

## 🙏 Thank You

Every contribution helps make V6rge better for everyone!
