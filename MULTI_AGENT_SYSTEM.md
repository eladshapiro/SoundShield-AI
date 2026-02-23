# SoundShield-AI Multi-Agent System

## Overview

This project now includes a comprehensive multi-agent AI system designed to improve development workflow, code quality, and project coordination.

## What Has Been Created

### 1. Agent Skills (`.cursor/skills/`)

Four specialized AI agent roles have been created:

#### **Manager Agent** (`agent-manager/SKILL.md`)
- **Role:** Project coordination and task delegation
- **Use when:** Planning features, coordinating work, making architectural decisions
- **Key responsibilities:**
  - Break down complex tasks
  - Delegate to specialized agents
  - Track progress
  - Ensure quality standards
  - Make strategic decisions

#### **Researcher Agent** (`agent-researcher/SKILL.md`)
- **Role:** Technical investigation and recommendations
- **Use when:** Exploring ML models, benchmarking approaches, investigating solutions
- **Key responsibilities:**
  - Research state-of-the-art models
  - Evaluate libraries and frameworks
  - Benchmark performance
  - Provide evidence-based recommendations
  - Create proof-of-concepts

#### **Code Writer Agent** (`agent-code-writer/SKILL.md`)
- **Role:** Implementation specialist
- **Use when:** Writing code, implementing features, fixing bugs
- **Key responsibilities:**
  - Write clean, maintainable code
  - Implement features following standards
  - Fix bugs with tests
  - Optimize performance
  - Document code properly

#### **Reviewer Agent** (`agent-reviewer/SKILL.md`)
- **Role:** Quality assurance and security review
- **Use when:** Reviewing code, security audits, quality checks
- **Key responsibilities:**
  - Review code quality
  - Identify security vulnerabilities
  - Check performance issues
  - Validate test coverage
  - Ensure standards compliance

### 2. Comprehensive Documentation

#### **CLAUDE.md** (Project root)
The main AI agent system guide containing:
- Complete project overview
- Multi-agent system architecture
- Project structure and modules
- Coding standards (PEP 8, type hints, docstrings)
- Development workflows
- Security & privacy guidelines
- Testing guidelines
- Documentation standards
- Common patterns
- Troubleshooting guide
- Performance optimization
- Deployment checklist

#### **AGENTS_QUICKREF.md** (`.cursor/skills/`)
Quick reference guide for:
- Agent activation
- Agent selection criteria
- Typical workflows
- Agent communication format
- Quick commands
- Skills location

## How to Use the Multi-Agent System

### Basic Usage

When working with Claude in Cursor, you can now explicitly invoke agent roles:

```
"Acting as Manager Agent: Let me break down this feature into tasks..."
"Acting as Researcher Agent: I'll investigate better emotion detection models..."
"Acting as Code Writer Agent: I'll implement this feature with tests..."
"Acting as Reviewer Agent: Let me review this code for security issues..."
```

### Example Workflow

**User Request:** "Improve cry detection accuracy"

1. **Manager Agent** coordinates:
   - Analyzes requirement
   - Breaks into phases (research → implement → review)
   - Delegates tasks

2. **Researcher Agent** investigates:
   - Researches cry detection models
   - Benchmarks alternatives
   - Recommends best approach

3. **Code Writer Agent** implements:
   - Writes new detection algorithm
   - Adds unit and integration tests
   - Documents the implementation

4. **Reviewer Agent** validates:
   - Reviews code quality
   - Checks security implications
   - Validates test coverage
   - Approves or requests changes

5. **Manager Agent** integrates:
   - Merges approved changes
   - Updates documentation
   - Verifies improvements

## File Structure

```
SoundShield-AI/
├── CLAUDE.md                          # Main AI agent guide
├── README.md                          # User documentation
├── .cursor/
│   └── skills/
│       ├── AGENTS_QUICKREF.md        # Quick reference
│       ├── agent-manager/
│       │   └── SKILL.md              # Manager agent guide
│       ├── agent-researcher/
│       │   └── SKILL.md              # Researcher agent guide
│       ├── agent-code-writer/
│       │   └── SKILL.md              # Code writer agent guide
│       └── agent-reviewer/
│           └── SKILL.md              # Reviewer agent guide
└── [project files...]
```

## Key Features

### 1. Specialized Expertise
Each agent has deep expertise in their domain:
- **Manager:** Project planning and coordination
- **Researcher:** Technical investigation and ML
- **Code Writer:** Python development and algorithms
- **Reviewer:** Quality assurance and security

### 2. Clear Workflows
Predefined workflows for:
- Feature development
- Bug fixes
- Research tasks
- Code reviews
- Security audits

### 3. Comprehensive Standards
Detailed guidelines for:
- Python coding style (PEP 8)
- Type hints and docstrings
- Error handling
- Security practices
- Testing (unit, integration)
- Documentation
- Performance optimization

### 4. SoundShield-AI Context
All agents understand:
- Project mission (child protection)
- Technical stack (PyTorch, Whisper, librosa)
- Module structure
- Security/privacy requirements
- Multilingual support (English, Hebrew)

## Benefits

1. **Better Code Quality:** Specialized review process catches issues early
2. **Faster Development:** Clear workflows and standards reduce ambiguity
3. **Improved Security:** Dedicated security review for this sensitive system
4. **Knowledge Retention:** Documentation captures decisions and patterns
5. **Consistent Standards:** All code follows same style and conventions
6. **Better Coordination:** Manager agent orchestrates complex changes

## Getting Started

### For Users

1. Read `AGENTS_QUICKREF.md` for quick start
2. Review `CLAUDE.md` for comprehensive guidance
3. Use appropriate agent role for your task

### For Claude (AI)

1. When activated, check which agent role is most appropriate
2. Read the corresponding skill file from `.cursor/skills/`
3. Follow the guidelines in that skill and CLAUDE.md
4. Coordinate with other agents as needed

## Important Notes

### Security & Privacy

This system handles sensitive child protection data. All agents must:
- Never log personal information
- Clean up temporary files
- Follow secure coding practices
- Validate all inputs
- Use encryption when appropriate

### Code Standards

All code must:
- Follow PEP 8 (Python style guide)
- Include type hints
- Have comprehensive docstrings
- Handle errors gracefully
- Include tests (≥80% coverage)
- Be reviewed before merging

### Testing Requirements

All changes require:
- Unit tests for individual functions
- Integration tests for workflows
- Manual testing with sample audio
- Security review for sensitive code
- Performance benchmarking for critical paths

## Maintenance

### Updating Skills

If agent roles or responsibilities change:
1. Update the relevant `SKILL.md` file in `.cursor/skills/`
2. Update `CLAUDE.md` if architectural changes
3. Update `AGENTS_QUICKREF.md` if workflow changes

### Adding New Agents

To add new specialized agents:
1. Create `.cursor/skills/agent-[name]/SKILL.md`
2. Follow the skill creation guidelines
3. Update this README
4. Update CLAUDE.md multi-agent section
5. Update AGENTS_QUICKREF.md

## Resources

- **Full Guide:** `CLAUDE.md` - Comprehensive documentation
- **Quick Reference:** `.cursor/skills/AGENTS_QUICKREF.md`
- **Manager Skills:** `.cursor/skills/agent-manager/SKILL.md`
- **Researcher Skills:** `.cursor/skills/agent-researcher/SKILL.md`
- **Code Writer Skills:** `.cursor/skills/agent-code-writer/SKILL.md`
- **Reviewer Skills:** `.cursor/skills/agent-reviewer/SKILL.md`

## Version

**Multi-Agent System Version:** 1.0  
**Created:** February 23, 2026  
**Last Updated:** February 23, 2026

---

**Questions?** 
- Consult CLAUDE.md for detailed guidance
- Check AGENTS_QUICKREF.md for quick answers
- Ask the Manager Agent to coordinate complex tasks
