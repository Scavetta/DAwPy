<<<<<<< HEAD
# Data Analysis with Python Workshop

## Workshop structure

1. Basic operations (addition, substraction ...)
2. Data structures (lists, dicts ...)
3. Control flow and loops (if else, for ...)
4. Functions and packages
5. Dataframe Basics and visualisations
6. Next steps

## Tips for using Jupyter notebooks in Jupyter lab

* use arrows to navigate between cells, ENTER to enter
* "a" creates a cell below
* "b" creates a cell above
* "x" cuts/delets cell
* CTRL + ENTER runs cell
* "m" convert cell to markdown
* "y" convert cell to code

## Environment Setup

First figure out which Python you have (python2 or 3), you can do it with:

```
which python
```

In general the steps are roughly the same for python2 and 3, just the naming of some of them might be different, i.e. pip3 instead of pip.
After this you need `pip` (this is the python package manager, you use it to install other packages). Installation instructions are here: https://pip.pypa.io/en/stable/installing/
After pip is installed and working (check with typing `pip` in the terminal), install `virtualenv`: https://virtualenv.pypa.io/en/stable/installation/. The purpose of virtialenv is very similar to "projects" in RStudio. Every directory with a virtualenv will have a collection with project specific libraries.

I recommend having one global virtualenv first, and later you can create different ones if you want. So the next steps:

```
virtualenv ~/ds_venv
source ~/ds_venv/bin/activate
```

Those two steps will create a virtualenv in that directory and activate it (your prompt should change to include the venv name)
Now you can install all libraries you need and they will be added to this virtualenv). The cool part is you can't mess this up, so if you feel like removing it you should just remove the directory.
To deactivate temporarily you can type `deactivate`.

While your global venv is activated install the important data science packages:

```
pip install pandas
pip install sklearn
pip install scipy
pip install jupyter
```

After those are completed:

```
jupyter notebook
```

This should open a notebook where you can do your work.
If you still prefer working with scripts (I do), install the atom editor (https://atom.io/) + the hydrogen package (https://atom.io/packages/hydrogen).
Then go to a directory (make sure venv is active) and open it with `atom .`. Then you can create a script `script.py` or open an existing one, and evaluate inline within Atom (this is how I work).

You can also try jupyter lab (its a bit similar to Rstudio, maybe you like it):

```
pip install jupyterlab
jupyter lab
```

## Additional materials:

* [pandas cheat sheet](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)
* [python cheat sheet](https://www.pythonsheets.com/notes/python-basic.html)
=======
# DAwPy
Data for Data Analysis with Python Workshop
>>>>>>> 1c0861fae85dbe03e474eba73b26e17c4dffb7d8
