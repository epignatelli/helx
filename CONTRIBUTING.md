# Contributing to helx
If you are reading this, it is likely that your are interested in contributing -- and time is precious, so thank you for your time.
This guidelines are designed to guide you through how to submit commits to `helx`.


## Code of conduct
First and foremost, please be respectuful to others.
Please, read and follow our [Code of conduct](https://github.com/epignatelli/helx/CODE_OF_CONDUCT.md).
The code applies to both humans and bots.


## Committing code
Contributing to the codebase or to other any operational support material happens through Pull Requests (PRs).
If you have already a PR in mind, please open an issue first, explaining the underlying motivation for the change.
To commit new code, please:
1. Open a [new issue](https://github.com/epignatelli/helx/issues/new/choose) and assign yourself to it
2. Create and work on a new branch -- you can do it [directly from the issue page](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-a-branch-for-an-issue)
3. When you are ready to merge, or you you have a proof of concept to discuss, open a PR, assign yourself to it, and ask review from one of the team


## Development environment

### Installation
You can install `helx` in your development environment using `conda` and `pip`.
From the repository root, you can install the package in development mode (`-e`) in a pre-baked environment:
```bash
conda env create -f environment.yml
pip install -e .
```

### Testing
We use `pytest` for testing.
You can run the tests with:
```bash
pytest test/
```

## Standards
### Python code
We adopt the [PEP8 Python style guide](https://peps.python.org/pep-0008/), and a big handful of good old common sense.
Please, keep your code as modular as possible.

### Documentation
We adopt the [Goole Python docstrings style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

### Versioning
We adopt the [semantic versioning](https://semver.org/) standard.
No prefixes, no suffixes are allowed, and no alpha/beta/rc versions are allowed.

### Commit messages
We adopt the [angular convention](https://github.com/angular/angular/blob/68a6a07/CONTRIBUTING.md#commit) for commit messages.
The commit message consists of a header, a body, and a footer:
```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```
The type is one of the following:
* **build**: Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)
* **ci**: Changes to our CI configuration files and scripts (examples: CircleCi, SauceLabs)
* **docs**: Documentation only changes
* **feat**: A new feature
* **fix**: A bug fix
* **perf**: A code change that improves performance
* **refactor**: A code change that neither fixes a bug nor adds a feature
* **test**: Adding missing tests or correcting existing tests

The scope is the name of the package affected (as perceived by the person reading the changelog generated from commit messages).

### Branch names and PR titles
It is recommended to create a new branch [directly from the issue page](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-a-branch-for-an-issue) on GitHub.
In this case, branch names and PR titles are generated automatically.
If you are targeting multiple issues, please include all issue numbers in the PR title, separated by a comma.


## Pull requests (PRs)
Contributing to the codebase or any operational support material must happen through PRs.
If you have already a PR in mind, there must be an underlying motivation for the change.
Please open an appropriate issue that explains it.

We follow a couple of standards to make our life easier and develop with more agility and less debt:
- Please use one of the [issue forms]() to submit a new issue.
- If you want to ask a question, please use the [discussions]() area, instead.
- When you open a PR, the title, the label, and the milestone can be of your choice.
- Please assign yourself to the PR you opened, and request a review from @epignatelli.
- The commits must follow the standards detailed in the [next section](./#Commit-standards)
- The code must follow the standards detailed in the [next section](./#Code-standards)


## Commit standards
We employ the [conventional commit](https://github.com/conventional-changelog/commitlint/tree/master/%40commitlint/config-conventional) standard for commit messages.
These standards automatically trigger the next semantic versioning, and its corresponding release.
Check the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/#how-does-this-relate-to-semver) description for more.


## Code standards
### - Python code
We follow the [PEP8 Python style guide](https://peps.python.org/pep-0008/), and a big handful of good old common sense.
Please, keep your code:
- DRY
- Modular
- Readable
- Clear of typos

The code should be [`black`](https://github.com/psf/black) formatted, and is usually [pylint-ed](https://pypi.org/project/pylint/).


### - Documentation


## New releases
A new version is release at every new commit to the main branch.
When one or more new commits are added to the main branch, the following happens:
- Their respective messages are analyses
- Based on the messages, the CD calculates the next version number (see [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/#how-does-this-relate-to-semver) for more)
- The CD pushes a new tag, whose name is the new version number
- The CD creataes a corresponding release, and pushes it to mypi
