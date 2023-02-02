This guidelines are inspired at those from [@angular/angular](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md).


# Contributing to helx
If you are reading this, it is likely that your are interested in contributing -- and time is precious, so thank you for your time.
This guidelines are designed to guide you through how to submit commits to `helx`.


## Code of conduct
First and foremost, please be respectuful to others.   
Please, read and follow our [Code of conduct](https://github.com/epignatelli/helx/CODE_OF_CONDUCT.md).   
This applied to both humans and bots.   


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
Following this format allows automating the deployment of new releases.

## Code standards



### Release
A new version is release at every PR merge to the main branch.
When new commits are added to the main branch, their respective messages are analyses, and can lead to either a new patch, a new minor or a new major version.
