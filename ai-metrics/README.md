AI Metrics
==========

This service monitors the performance of AI classification algorithms.

Dependencies
------------

    virtualenv venv
    source venv/bin/activate

    # Production Dependencies
    pip install -r requirements.txt

    # Development Dependencies
    pip install -r requirements/local.txt

See deployment notes below.  Scipy dependencies might take additional effort
the first time you install them.  

Running
-------
    
    python server.js

Deployment
----------

Scipy and Numpy make this a little finnicky to deploy, since they require
building from C/Fortran source and make use of the GPU if possible.  Because of
this, they follow a non-standard build process and can't be vendored the way
python packages normally can.

If deployed on Heroku, this needs to be built with 
[this buildpack](git@github.com:thenovices/heroku-buildpack-scipy.git). Note
that at the moment, this project's version numbers are incompatible with this
buildpack, so when it comes time for production deployment, a version migration
will have to happen. Currently, tweaking versions for a production deployment
isn't high enough of a priority to justify the time investment.

If deployed manually, you will need to be able to install the numpy and scipy
packages via pip.  Typically this means either installing them globally with
their dependencies, or installing the following packages:

    1. BLAS
    2. LAPACK
    3. ATLAS
    4. gcc-fortran

I've always thought the easiest method was to install the packages globally
along with gcc-fortran, and then reistall them locally with pip.  For whatever
reason, this is the only way I've been able to get it to work consistently.
Installing BLAS/LAPACK/ATLAS is just too tricky.  ATLAS punks out if you have
CPU throttling enabled in the BIOS, which almost all modern computers do.


Configuration
-------------

Configuration is stored in the project-level `conf.toml` file.  This file 
contains variables necessary for the ai-metrics server to communicate with
others. The server expects this file to be in the same pwd as its process.
`aimetrics.conf.get_conf` is a convenience method that will automatically load
the configuration file into a dictionary.

Notebooks
---------

A number of ipython notebooks are stored in the `notebooks` directory.  These
contain one-off performance analyses. Related data is stored in the `output`
directory.
