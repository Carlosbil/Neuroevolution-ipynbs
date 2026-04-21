# Squad Team

> Neuroevolution-ipynbs

## Coordinator

| Name | Role | Notes |
|------|------|-------|
| Squad | Coordinator | Routes work, enforces handoffs and reviewer gates. |

## Members

| Name | Role | Charter | Status |
|------|------|---------|--------|
| Ripley | Lead Neuroevolution Architect | `.squad/agents/ripley/charter.md` | ✅ Active |
| Bishop | Evolution Engine Developer | `.squad/agents/bishop/charter.md` | ✅ Active |
| Hicks | Training Pipeline Engineer | `.squad/agents/hicks/charter.md` | ✅ Active |
| Vasquez | Tester & Reviewer | `.squad/agents/vasquez/charter.md` | ✅ Active |
| Scribe | Session Logger | `.squad/agents/scribe/charter.md` | 📋 Silent |
| Ralph | Work Monitor | — | 🔄 Monitor |

## Coding Agent

<!-- copilot-auto-assign: false -->

| Name | Role | Charter | Status |
|------|------|---------|--------|
| @copilot | Coding Agent | — | 🤖 Coding Agent |

### Capabilities

**🟢 Good fit — auto-route when enabled:**
- Bug fixes with clear reproduction steps
- Test coverage additions and isolated test refactors
- Small notebook/script maintenance tasks with clear acceptance criteria

**🟡 Needs review — route to @copilot with squad review:**
- Medium notebook refactors with explicit specs
- Feature additions following established neuroevolution patterns

**🔴 Not suitable — route to squad member instead:**
- Search-space redesign and architecture strategy
- Experimental methodology changes affecting scientific validity
- Security- or data-governance-sensitive changes

## Project Context

- **Project:** Neuroevolution-ipynbs
- **Owner:** Carlosbil
- **Stack:** Python, PyTorch, Jupyter notebooks, NumPy, scikit-learn
- **Description:** Parkinson voice classification with hybrid neuroevolution over Conv1D architectures.
- **Created:** 2026-04-21
