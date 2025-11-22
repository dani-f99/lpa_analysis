-----------------------------
------Daniel-Fridman---------
--- System-Immunology-Lab --- 
------Haifa-University ------
-----------2024--------------
-----------------------------
-----------------------------
--- General-Information -----
This program aim is to prepare input table which can be used as input for the LPA analysis program. 
The program first pulls the datasets from ImmuneDB tables located in MySQL server (if different server is in use, may need to change connector),
then the datasets are divided by metadata, processed, merged and modified. The final output dataframe (exported as csv) that can be used as LPA input. 
-----------------------------
-----------------------------
-----------------------------
-----------------------------
--- Required-Python-Modules -
1.  natsort
2.  math
3.  seaborn
4.  matplotlib
5.  numpy
6.  sklearn
7.  scipy
8.  typing
9.  pathlib
10. itertools
11. copy
12. altair
-----------------------------
-----------------------------
-----------------------------
-----------------------------
--- Guide ---
0. Use the lpa_preprocessing script to create correct input for this program OR use your own input according to the
   LPA requirements (see LPA section in this readme).
1. Verify that all the needed packages are installed.
2. Create input folder which contains the documents and data_sizes datasets (see input_example folder) - see cell line no.3.
3. Change the 'input_name' variable string to your input folder.
4. Run the code.
5. Figures will be saved in output folder (output folder will be created when cell no.2 will be executed).
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
