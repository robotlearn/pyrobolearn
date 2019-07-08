#!/usr/bin/env bash 

# . venv/bin/activate 

rm -Rf build
rm -Rf source/docstring

sphinx-apidoc -f -o source/docstring/ ../pyrobolearn
make html

