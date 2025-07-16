Save to DB
====================

The **Save to DB** node stores any data table that flows through a
workflow into a PostgreSQL database and keeps a master catalogue of the
datasets that have been uploaded.  It is the final step in the example
workflow, after feature engineering and graph generation, and can be
re-used in any pipeline that needs persistent storage.

What the node expects
---------------------

* **Input — Data**  
  A regular Orange data table.  
  The input is mandatory; without it the node is idle.

What the node does
------------------

1. Opens a connection to the target PostgreSQL instance.  
2. Creates a new table whose name you specify, and writes the entire
   dataset into it (columns become fields, each row becomes a record).  
3. Updates an internal *datasets* catalogue table so that every stored
   data set can be tracked later (name, schema, upload time, user, …).  
4. (Optional) Sends an e-mail notification summarising the operation
   once the upload succeeds.

How to configure it
-------------------

#. Drag the node into the canvas and connect the *Data* output of the
   previous step.  The **Save** button becomes enabled as soon as a
   table is received.

#. Fill in the **database credentials** (performed once per session):  

   * **Server** → ``localhost`` (or your remote host)  
   * **Username** → the PostgreSQL user, e.g. ``postgres``  
   * **Password** → the password you entered during the installation

   Press **Connect**.  The status line turns green when the login
   succeeds.

#. Choose a **Table name** for the new data set.  Pick something unique;
   the node will complain if a table with the same name already exists.

#. (Optional) Type an **e-mail address**.  A short confirmation message
   with basic statistics and the destination table is sent after the
   upload.

#. Press **Save**.  A progress bar appears while the data are streamed
   into the database.  Once finished, you can verify the result in
   *pgAdmin* or any SQL client — you should see:

   * a new table with the chosen name containing every column of the
     Orange table;
   * an updated **datasets** master table with a new entry that points
     to the freshly created data set.

Tips & best practices
---------------------

* **Class field** If the data set has a class variable, select its type
  (*Categorical* or *Numeric*) in the drop-down so the field is stored
  with the right SQL data type.

* **Versioning policy** If you need to keep historical snapshots, use a
  consistent naming convention such as  
  ``sales_2025_q2`` or ``sensorlog_2025-07-16``.

* **Large uploads** Network hiccups may break the transaction.  When
  working with millions of rows consider increasing the PostgreSQL
  *statement_timeout* or using a local instance and later dumping the
  table to the production server.

* **Automation** Because the node writes a canonical entry to the
  *datasets* catalogue, you can build dashboards or downstream Orange
  workflows that query that index and load any stored data set on
  demand.

Limitations
-----------

* The current implementation supports **PostgreSQL only**.  To use
  another DBMS you must adapt the connection code.

* Table names must obey PostgreSQL rules (start with a letter or
  underscore, contain only alphanumerics and underscores, and be
  case-insensitive unless quoted).

* Binary columns (images, complex objects) are not yet supported; they
  must be serialised beforehand.

