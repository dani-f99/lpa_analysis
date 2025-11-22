-----------------------------
------Daniel-Fridman---------
--- System-Immunology-Lab --- 
------Haifa-University ------
-----------2024--------------
-----------------------------
-----------------------------
--- General-Information -----
This program prepares input tables for LPA analysis. It first pulls datasets from ImmuneDB tables on a MySQL 
server (note: the connector may need adjustment if a different server is used). The data is then divided by 
metadata, processed, merged, and modified. The final output is a dataframe, exported as a CSV, that can be used as LPA input
-----------------------------
-----------------------------
-----------------------------
-----------------------------
--- Required-Python-Modules -
1. natsort
2. seaborn
3. matplotlib
4. numpy
5. sklearn
6. scipy
7. altair
-----------------------------
-----------------------------
-----------------------------
-----------------------------
--- Guide ---
1. Use the lpa_preprocessing script to create correct input for this program OR use your own input according to the
   LPA requirements (see LPA section in this readme).
2. Verify that all the needed packages are installed.
3. Create input folder which contains the documents and data_sizes datasets (see input_example folder) - see cell line no.3.
4. Change the 'input_name' variable string to your input folder.
5. Run the code.
6. Figures will be saved in output folder (output folder will be created when cell no.2 will be executed).
-----------------------------
-----------------------------
-----------------------------
-----------------------------
---Functions-----------------
functions.name_cmap -> Custom function that create hex-name list of colors from a cmap.
functions.do_pca -> Function that applies PCA on table.
functions.mask_df -> Create masking dataframe for seaborn heatmap.
functions.domain_heatmap -> Heatmap plot: KLDe distances of each dataset from the shared domain
functions.pca_scatter -> Custom function that plot scatter plot of pca dataframe data
function.get_scatter -> Scatterplot of two dataframes
-----------------------------
-----------------------------
-----------------------------
-----------------------------
-----------------------------
---LPA-----------------------
Publication: https://link.springer.com/article/10.1007/s11257-021-09295-7
GitHub: https://github.com/ScanLab-ossi/LPA
-----------------------------
