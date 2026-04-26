# Governance

## Project Governance

Voice-to-Text is an open-source project managed by its maintainers and community.

## Project Lead

- **Shahadath Hossain** (@shahadathhs)
  - Final decision authority
  - Project direction and roadmap
  - Release management
  - Community coordination

## Maintainers

Maintainers are contributors who have shown:

- Consistent, high-quality contributions
- Good understanding of the codebase
- Active participation in reviews and discussions
- Helpful engagement with the community

### Current Maintainers

- **Shahadath Hossain** (@shahadathhs) - Project Lead

### Becoming a Maintainer

Maintainers are invited by the project lead based on:

1. **Track Record**: Consistent contributions over time
2. **Code Quality**: High-quality, well-tested code
3. **Review Participation**: Active in PR reviews
4. **Community Engagement**: Helpful and responsive
5. **Alignment**: Agreement with project goals and values

There is no fixed timeline or requirement. When the project lead identifies someone who would make a good maintainer, they will be invited.

### Maintainer Responsibilities

- Review and merge pull requests
- Respond to issues and questions
- Guide contributors
- Enforce code standards
- Participate in decision-making
- Mentor new contributors
- Release management

### Maintainer Removal

Maintainers may be removed for:
- Inactivity for 6+ months without notice
- Repeated violation of [Code of Conduct](CODE_OF_CONDUCT.md)
- Consistent disregard for project standards
- Other serious issues as determined by project lead

## Decision Making

### Types of Decisions

**Consensus Decisions** (Require agreement):
- Code of Conduct changes
- Major breaking changes
- Project direction changes
- Adding/removing maintainers

**Maintainer Decisions** (Simple majority):
- PR reviews and merges
- Feature acceptance/rejection
- Code standards updates
- Dependency updates

**Project Lead Decisions** (Final authority):
- Emergency decisions
- Tie-breakers
- Release timing
- Critical bug fixes

### Decision Process

1. **Proposal**: Create an issue or discussion
2. **Discussion**: Allow time for community input (minimum 7 days for major decisions)
3. **Decision**: Maintainers vote or project lead decides
4. **Communication**: Decision is announced and documented

## Release Management

### Versioning

- **Semantic Versioning**: Follows [SemVer](https://semver.org/)
- **Automated Releases**: Uses python-semantic-release
- **Version Bumps**: Based on commit messages

### Release Process

1. **Development**: Features developed on feature branches
2. **Testing**: Thoroughly tested before merge
3. **Merge**: Merged to `main` branch
4. **Release**: Automatic release via CI/CD
5. **Announcement**: Release notes published

### Release Types

- **Major (X.0.0)**: Breaking changes
- **Minor (0.X.0)**: New features
- **Patch (0.0.X)**: Bug fixes

## Code Review

### Review Requirements

All changes must be reviewed by at least one maintainer.

### Review Criteria

- **Code Quality**: Follows project standards
- **Testing**: Adequately tested
- **Documentation**: Updated as needed
- **Performance**: No significant regressions
- **Security**: No security vulnerabilities
- **Alignment**: Fits project goals

### Review Timeline

- **Response Time**: Maintainers aim to respond within 7 days
- **Complex Changes**: May take longer
- **Urgent Fixes**: Fast-tracked

## Community Guidelines

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas
- **Pull Requests**: Code contributions
- **Code Reviews**: Technical discussions

### Expected Behavior

- **Respectful**: Treat everyone with respect
- **Constructive**: Focus on what's best for the community
- **Collaborative**: Work together to solve problems
- **Inclusive**: Welcome diverse perspectives

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.

## Conflict Resolution

### Steps

1. **Discussion**: Try to resolve directly
2. **Mediation**: Ask another maintainer for help
3. **Escalation**: Bring to project lead
4. **Final Decision**: Project lead decides

### Code of Conduct Enforcement

- **Warning**: For first or minor violations
- **Temporary Ban**: For repeated or serious violations
- **Permanent Ban**: For severe violations
- **Appeal Process**: Contact project lead

## Project Scope

### In Scope

- Core transcription functionality
- API and CLI interfaces
- Documentation and examples
- Bug fixes and performance
- Security updates

### Out of Scope

- Commercial support
- Custom development
- Unrelated features
- Feature creep

### Scope Changes

Adding new major features requires:
- Community discussion
- Maintainer agreement
- Resource assessment
- Documentation plan

## Intellectual Property

### License

- **License**: MIT License
- **Copyright**: Contributors retain copyright
- **Contributions**: Licensed under MIT

### Trademarks

- Project name and logo are project assets
- Third-party use requires permission

### Contributor License Agreement

By contributing, you agree that:
- Your contributions are your original work
- You have rights to contribute the code
- Your contributions are licensed under MIT

## Financial Aspects

### Funding

- **No Commercial Funding**: Project is not commercially funded
- **Donations**: Not currently accepted
- **Sponsorship**: Open to discussion

### Expenses

- **Hosting**: GitHub (free tier)
- **CI/CD**: GitHub Actions (free tier)
- **Domain**: Self-funded by maintainer
- **Other Costs**: Self-funded by contributors

## Amendments

### Changing Governance

Governance changes require:
- Proposal from maintainer
- Community discussion (14 days)
- Maintainer super-majority (2/3)
- Document update

## Contact

For governance questions:
- **Project Lead**: @shahadathhs
- **Issues**: Use GitHub issues with `governance` tag
- **Discussions**: Use GitHub Discussions

---

**Last Updated**: 2026-04-26

This governance document is a living document and may evolve as the project grows.
