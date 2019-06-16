## Implemented papers

This folder contains subfolders where in each one we try to reproduce the experiments and results obtained in the 
literature using the *PyRoboLearn* framework.

Each subfolder contains 
- a `README` file that:
    - summarizes the paper and provides the BibTex reference with a link to the original paper
    - states the possible differences between their experiment setup and ours and the obtained results
    - provides the current status (if it has been implemented yet, if the reproduction was successful or not), and 
    describes what needs to be improved.
    - describes which variables the user can play with
- an environment (similar to gym environments) in the `env.py` file (that initializes the world, robots, rewards, 
states, actions, and so on) which allows to be used in other codes.
- a main file (`main.py`) that initializes the learning task (environment, policy, and so on), and provides an 
example on how to use the environment defined in `env.py`.
- a possible `figures` directory containing figures; this can include a picture of the environment, training plots, 
etc.
- a possible `meshes` directory containing meshes for the different objects that are present in the environment.
- a possible `data` folder containing the parameters/hyperparameters of a pre-trained learning model, or collected 
data that can be used for imitation learning for instance.


### Structure

The name of each folder is `<author_last_name><year><first_non_stop_word_in_title>`, this is the same as returned by 
[Google Scholar](https://scholar.google.com/), when clicking on the `BibTex` button:

```
@article{<1st author's last name><year><1st non stop word in title>,
  title={<title>},
  author={<1st author's last name>, 1st author's first name and ...},
  journal={...},
  year={<year>}
}
```

*Note*: Stop words are commonly used words such as 'the', 'an', 'a', 'in', and others that search engines usually 
ignore.


### Further notes

- It might be that in the future, this folder will be moved to its own repository, and each paper will have its own 
repository and will be loaded as submodules recursively (using `git clone --recursive`)
- I will often focus more on implementing the environments than training the various policies using the different 
algorithms
- Any comments, help, pull requests are appreciated. 
