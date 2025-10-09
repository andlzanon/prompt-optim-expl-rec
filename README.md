# Prompt Otimization for Better Explanations in Recommender Systems

This project has the main objective to use prompt otimization techniques to generate explanations for recommender systems 

## 📋 Reproduction

### Environment 

To install the libraries used in this project, use the command: 
    
    pip install requirements

Or create a conda enviroment with the following command:

    conda env create --n <env_name> --f requirements.txt

After this step it is necessary to install the CaseRecommender library with the command:
    
    pip install -U git+git://github.com/caserec/CaseRecommender.git

You can use used [Anaconda](https://www.anaconda.com/) to run the experiments.

## 📝 Project Organization

## 🤝 Project's Development Process

We will develop our code on the dev branch, that represents a paper we are developing. Once we finish the code to this paper, we will merge dev with main. 

Therefore, to develop a feature use the following steps:

1. Create a branch from dev

Switch to the dev branch and pull the latest changes:

    git checkout dev

    git pull origin dev


Create a new branch from dev for your changes:

    git checkout -b my-new-feature


Use a descriptive branch name

2. Make your changes

Edit, add, or remove files as needed. Stage and commit your changes:

    git add .

    git commit -m "Add: short description of your change"

3. Push your branch to your fork
    git push origin my-new-feature

4. Open a Pull Request

    - Go to the repo on GitHub.

    - Click Compare & pull request next to your branch.

    - Set the base repository to the original repo’s dev branch.

    - Add a clear title and description for your changes.

    - Click Create pull request.

5. Review and merge

The maintainers will review your PR. Once approved, it will be merged into dev.

✅ Tips

Keep your branch up-to-date with dev:

git fetch origin
git merge origin/dev

Make small, focused commits with clear messages.

Follow the project’s coding style and conventions.