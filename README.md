# HMMC

A Hidden Markov Model approach for segmenting 3D genome compartments from Hi-C datasets stored in the [cooler](https://github.com/open2c/cooler) format.

HMMC operates on eigenvectors computed [using cooltools](https://github.com/open2c/cooltools) in a pandas dataframe format as an input. HMMC has so far been applied to human and mouse coolers binned at 50kb-1Mb resolution.

Currently installation is done locally with pip: lone the repository to the desired directory, navigate there and install:
~~~
pip install -e ./
~~~

Examples are found in notebooks.
