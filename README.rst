ramsmod
========
A library for analysing reliability data that was developed
to support the RAMS modelling course at Örebro University.

E.g.
.. code-block:: python

    from ramsmod.rdata.datasets import load_right_censored
    from ramsmod.rdata.fitting import kaplan_meier_fit

    data = load_right_censored()
    km_table = kaplan_meier_fit(data['t'], data['d'])
    print(km_table)

Features
--------

- Visualising right-censored and interval-censored failure data.
- Fitting reliability curve to right-censored and interval-censored failure data using the Kaplan-Meier and Turnbull methods.
- Comparison of reliability between two groups of failure data using log-rank (right-censored data) and Mantel (interval-censored data) tests.

Installation
------------

Install ramsmod by running:

    pip install ramsmod

Bugs & Support
-----------------
If you have any questions please send an email to:
sean.reed@oru.se

If you think you have found a bug, please create an issue on
GitHub or send an email.

License
-------

The project is licensed under the MIT license - see LICENSE.rst