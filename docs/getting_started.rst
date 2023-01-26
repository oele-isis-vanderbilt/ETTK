Getting Started
===============

Below are the instructor for getting started using ETTK.

Installation
------------

You are able to install via PyPI by the following::

    pip install ettk

Or, you can also install from source::

    git clone https://github.com/oele-isis-vanderbilt/ETTK
    cd ETTK
    pip install .

Common Errors
-------------

Some errors that we have noticed when using this package are the following:

#. Having both ``opencv-python`` and ``opencv-contrib-python`` installed, causing errors. Do the following to make sure that everything is okay::

    pip uninstall opencv-python opencv-contrib-python av
    pip install opencv-contrib-python==4.6.0.66

#. OpenCV and PyAV can interfer with one another. Therefore, you want to use the following command::

    pip uninstall av
    pip install av --no-binary av
