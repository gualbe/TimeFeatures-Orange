Time Features Constructor
==============================

The **Time Features Constructor** node builds time–based features
(shifts, rolling statistics, etc.) on top of any Orange data table.
It usually sits between the data-source node and the nodes that will
consume the enriched data set (for example *Variable Dependency Graph*,
*Save to DB* or any learner).

Overview
--------

* Receives a data table (**Data**) and, optionally, a
  *Variable Definitions* table.
* Lets you define new variables **manually** or by **importing** an
  existing Variable/Expression table.
* Outputs the updated data table and the updated
  *Variable Definitions* table.

Inputs and outputs
------------------

* **Input – Data**  
  Mandatory. Original data on which features will be computed.

* **Input – Variable Definitions**  
  Optional. A two-column table (Variable, Expression) that lists
  expressions to evaluate.

* **Output – Data**  
  The original data plus every generated feature.

* **Output – Variable Definitions**  
  A full list of variables (original + derived) with their expressions.

Variable Definitions format
---------------------------

* **Variable** – Name of the variable, original or derived.  
* **Expression** – Formula used to compute the variable.  
  Original variables carry the value ``nan``.

Typical workflow
----------------

1. **Connect data** – Attach any data-providing node to **Data**.  
   The data set must have a file/stream name so that new columns are
   tagged properly.

2. **Create features**

   *Automatic mode* – Send a pre-built Variable/Expression table into
   the *Variable Definitions* input.  
   The node evaluates every expression automatically.

   *Manual mode*

   • Choose **New → Numeric** to insert an empty row into
     *Variables to generate*.  
   • Type the expression or use the drop-downs:

     – Variable chooser (original or already generated variables).  
     – Operators ``+  -  *  /  %`` and parentheses.  
     – Basic functions ``abs  int  float  pow``.  
     – Time functions listed below.

   • Edit, duplicate or delete pending variables as needed.  
   • Press **Send** to evaluate every pending expression.

3. **Outputs** – Pressing **Send** (or supplying an input table)
   triggers feature computation, table update and the emission of both
   outputs.

Time functions
--------------

* ``shift(var, lag)`` – Shift *var* **lag** steps (negative or positive).
* ``sum(var, inf, sup)`` – Sum of *var* from *t – inf* to *t – sup*.
* ``min(var, inf, sup)`` – Minimum in the window.
* ``max(var, inf, sup)`` – Maximum in the window.
* ``sd(var, inf, sup)`` – Standard deviation in the window.
* ``mean(var, inf, sup)`` – Arithmetic mean in the window.
* ``count(var, inf, sup)`` – Number of non-null values in the window.
* ``shiftBatch(var, n)`` – Convenience macro that generates *n*
  consecutive **backward** lags for *var*: ``t-1 … t-n``.
  For example ``shiftBatch(x, 3)`` automatically creates

  * ``x_t-1``  (same as ``shift(x, 1)``)
  * ``x_t-2``  (``shift(x, 2)``)
  * ``x_t-3``  (``shift(x, 3)``)

  Positive *n* therefore means “look *n* steps into the past”.


For all window functions *inf* and *sup* may be positive or negative;
*inf* must be the smaller (earlier) offset.

Important notes
---------------

* **No nested window functions** – You cannot write
  ``shift(mean(var,0,1),2)``.  First create ``mean(var,0,1)``,
  then shift the new variable.
* **Duplicate names** – The node auto-renames duplicates by appending
  ``_1``, ``_2``…
* **Invalid expressions** – Highlighted in red and blocked until fixed.
* **Reset** – Clears every pending and generated variable.

Keyboard shortcuts
------------------

* **Enter** – Confirm expression editing.  
* **Delete / Backspace** – Remove the selected pending variable.

Summary
-------

* Requires **Data**; *Variable Definitions* is optional.  
* Generates shifts and rolling statistics (including ``shiftBatch``).  
* Outputs an enriched data table and an updated definition table.  
* Manual editor with helper drop-downs and validation.  
* Press **Send** to compute all pending features.
