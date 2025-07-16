Variable Dependency Graph
=========================

The **Variable Dependency Graph** widget converts a *Variable / Expression*
table (produced by *Time Features Constructor*) into an interactive directed
network that highlights how every generated feature depends on the rest of
the data set.

Overview
--------

* **Input** – a single Orange `Table` whose  
  first attribute column is **Variable** and whose first meta column is
  **Expression** (each row comes straight from *Time Features Constructor*).

* **Output** – a `Network` object (orange-network package) whose  
  nodes represent variables and whose edges represent dependencies
  discovered in the expressions.

* **Typical use-case** – plug the table emitted by
  *Time Features Constructor* into *Variable Dependency Graph* and then feed
  the resulting network to *Network Explorer* for visual analysis.

What the widget does
--------------------

1. **Parse every expression**  
   Each row is scanned for references to other variables
   (exact matches only – whitespace and dashes are normalised).

2. **Detect temporal shifts**  
   If an expression contains ``shift(var, ±k)`` the edge between *var* and the
   derived feature is annotated with the lag *k*.  Positive values are shown
   as *1, 2, …* (no sign).

3. **Build a sparse adjacency matrix**  
   Edges carry the numerical lag (default = ``1`` when no shift is present).

4. **Create the `Network` object**  

   * **edge_labels** – string version of the lag (``"1"``, ``"3"``, …),
     ready for display in *Network Explorer* (turn on *Show edge weights*).
   * **node meta fields**
     - ``var_name``  → the column name displayed as the node label.  
     - ``var_type``  → *Derived* or *Original* (can be mapped to colour).

User interface
--------------

* **Generate** button – forces a refresh (table changes are detected
  automatically, but you can click to be sure).
* No additional settings: the widget is intentionally minimal – all styling,
  filtering and layout happen in *Network Explorer*.

Practical tips
--------------

* When you open the network in *Network Explorer*:
  - Set **Label → var_name** to show readable names.  
  - Set **Colour → var_type** to distinguish originals (orange) from derived features (blue).
  - Enable *Show edge weights* to see the temporal lags on the arrows.

* The widget ignores expressions that reference variables not present in the
  table – useful when you experiment with placeholder names.

