TimeFeatures add-on
===================

The **TimeFeatures** add-on streamlines end-to-end feature engineering
workflows inside Orange.  It lets you

1. **Construct temporal features** – create lags, rolling statistics and
   arbitrary expressions with the *Time Features Constructor* node.  
2. **Visualise variable dependencies** – build an interactive directed
   graph that shows which variables depend on which others and, for
   shifts, the exact time lag, via the *Variable Dependency Graph* node.  
3. **Persist datasets** – write any table (raw data, engineered features,
   target labels, …) to PostgreSQL with a single click using *Save to DB*.

.. toctree::
   :maxdepth: 3


Widgets
-------

.. toctree::
   :maxdepth: 3

   widgets/owtimefeaturesconstructor
   widgets/owvardependencygraph
   widgets/owsavetodb
