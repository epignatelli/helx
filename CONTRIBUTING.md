# Contributing to helx
If you are reading this, your are interested in contributing -- and time is precious, so thank you very much for your time.
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



## Development install
You can install `helx` in your development environment using `conda` and `pip`.
From the repository root, you can install the package in development mode (`-e`) in a pre-baked environment:
```bash
git clone https://github.com/epignatelli/helx
cd helx
conda env create -f environment.yml
pip install -e .
```

## Testing
We use `pytest` for testing.
You can run the tests with:
```bash
pytest test/
```

---
## Standards
### Python code
We adopt the [PEP8 Python style guide](https://peps.python.org/pep-0008/), and a big handful of good old common sense.
Please, keep your code as modular as possible.

### Documentation
We adopt the [Goole Python docstrings style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

### Versioning
We adopt the [semantic versioning](https://semver.org/) standard.
No prefixes, no suffixes are allowed, and no alpha/beta/rc versions are allowed.
The version number is automatically calculated by the CD, based on the commit messages (see [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/#how-does-this-relate-to-semver) for more, and read further for more details).

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
