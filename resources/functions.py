# Required Modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer


# Custom function that create hex-name list of colors from a cmap
def name_cmap(cmap_name:str) -> list:
    """
    cmap_name : str -> the name of the heatmap we want to get list of string from (plt heatmaps).
    """
    cmap_hex = []
    for color in plt.get_cmap(cmap_name).colors:
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))
        cmap_hex.append(hex_color)
    return cmap_hex


# Custom function that applies PCA on table
def do_pca(df_input : pd.DataFrame,
           plot : bool = False):
    """
    df_input : pd.DataFrame -> input matrix, signature matrix output from the LPA analysis.
    plot : bool -> if True, will plot bar chart of the explained varaiance of the PC.
    """
    pca_object = PCA(n_components=0.95, random_state=0)
    pca_object.fit(df_input)
    pca_output = pca_object.transform(df_input)
    pca_evar = pca_object.explained_variance_ratio_
    pca_df = pd.DataFrame(data = pca_output,
                          index = df_input.index,
                          columns = [f"PC{i+1}" for i in range(0,pca_output.shape[1])])
    
    pca_var = pd.DataFrame({"PC":pca_df.columns, "Variance":pca_evar})
    
    if plot:
       plt.bar(x=pca_df.columns, height=pca_evar )
       plt.xlabel("PC")
       plt.ylabel("Explaned Variance")
       plt.show()
        
    return pca_df, pca_var


# Create masking dataframe for seaborn heatmap
def mask_df(heatmap_matrix : pd.DataFrame,
            dropped_matrix : pd.DataFrame) -> pd.DataFrame:
    """
    heatmap_matrix : pd.DataFrame -> Original matrix.
    dropped_matrix : pd.DataFrame -> The matrix section (derived from the original matrix) to be masked.
    """
    
    masked_outout = pd.DataFrame(np.full(heatmap_matrix.shape, False), 
                                 index=heatmap_matrix.index,
                                 columns=heatmap_matrix.columns)

    for i in dropped_matrix.itertuples():
        temp_dataset = i[1]
        temp_word = i[2]
        masked_outout.loc[masked_outout.index == temp_word ,temp_dataset] = True
        
    return masked_outout


# Heatmap plot: KLDe distances of each dataset from the shared domain
def domain_heatmap(domain_df : pd.DataFrame,
                   mask_input : pd.DataFrame = None,
                   save_fig : bool = False) -> plt.Figure:
    """
    domain_df : pd.DataFrame -> LPA signature matrix (of each document from the domain).
    mask_input : pd.DataFrame = None -> mask dataframe (output of masked_outout fucntion), if wanted.
    save_fig : bool = False -> path for saving the figure outout.
    """
    
    fhight = domain_df.shape[1]/5
    fig, ax = plt.subplots(1,1, figsize=(20,fhight))


    if mask_input is None:
        mask_df = np.full(domain_df.shape,False)

    try:
        cond_shape = (mask_input.shape == domain_df.shape)
        if cond_shape:
            mask_df = mask_input
        else:
            mask_df = np.full(domain_df.shape,False)
    except:
        mask_df = np.full(domain_df.shape,False)
     
    sns.heatmap(data=domain_df.T,
    cbar_kws={'label': 'KLDe Distance'},
    cmap="turbo",
    xticklabels=True,
    yticklabels=True,
    mask=mask_df.T,
    ax=ax)
    
    cbar_ax = ax.figure.axes[-1]
    cbar_ax.tick_params(labelsize=11)
    cbar_ax.yaxis.get_label().set_fontsize(15)

    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlabel("Position", fontsize=15)
    ax.set_ylabel("Repertoire", fontsize=15)

    range_all = range(1,105) #FR1, CDR1, FR2, CDR2, FR3
    range_cdr = list(range(27,39)) + list(range(56,66)) #CDR1, CDR2
    range_fr = list(range(1,27))+list(range(39,56))+list(range(66,105)) #FR1,FR2,FR3
    for xtick, xcolor in zip(ax.get_xticklabels(), ["red" if i in range_cdr else "blue" for i in domain_df.T.columns]):
        xtick.set_color(xcolor)

    if save_fig == True:
        time = datetime.now().strftime("[%d.%m.%y-%H;%M]")
        fig.savefig('output\\{} new_covid_vaccine_LPA.png'.format(time), bbox_inches='tight')


# Custom function that plot scatter plot of pca dataframe data
def pca_scatter(pca_df : pd.DataFrame,
                range_pc : range = range(1,4),
                cmap_name : str = "tab10",
                save_fig : bool = False,
                color_index: int = -1) -> plt.Figure:
    """
    pca_df : pd.DataFrame -> PCA output matrix (output of do_pca).
    range_pc : range -> Range of PC we want to present.
    cmap_name : str -> Name of camp used to color the plot dots.
    save_fig : bool -> If True will save figure.
    color_index: int -> Which metadata value the colors will mark.
    """
    
    range_pc = range_pc
    fig, axs = plt.subplots(len(range_pc),len(range_pc),figsize=(len(range_pc)*7,len(range_pc)*7))
    unique_subj = np.unique(np.array([i.split(".")[color_index] for i in pca_df.index]))
    unique_target = np.unique(np.array([i.split(".")[0] for i in pca_df.index]))
    print("Color marks unique: ", unique_subj)
    print("Shape marks unique: ", unique_target)

    cmap = name_cmap(cmap_name)
    dict_marker = {i:j for i,j in zip(unique_target,["*","^"])}
    dict_subjects = {i:j for i,j in zip(list(unique_subj), cmap)}
    labels_axs = []

    for j_target in unique_target:
        marker = dict_marker[j_target]

        for i_subj in unique_subj:
            color = dict_subjects[i_subj]
            label = ".".join([j_target,i_subj])
            ij_cond = [i for i in pca_df.index if (i.split(".")[color_index]==i_subj) and (i.split(".")[0]==j_target)]
            pca_input = pca_df.loc[ij_cond,:]

            for i,j in [[i,j] for i in range_pc for j in range_pc]:
                x_input = "PC{}".format(i)
                y_input = "PC{}".format(j)

                plot = axs[i-1,j-1].scatter(pca_input[x_input], 
                                    pca_input[y_input],
                                    label = label,
                                    marker= marker,
                                    c = color,
                                    s = 150,
                                    alpha = 0.6,
                                    edgecolor="black",
                                    lw = 1)

                axs[i-1,j-1].set_xlabel(x_input)
                axs[i-1,j-1].set_ylabel(y_input)
                axs[i-1,j-1].legend().set_visible(False)
                labels_axs.append(label)
    
    labels_axs = np.unique(np.array(labels_axs))
    plt.legend(labels=labels_axs, loc="upper left", fontsize=20, bbox_to_anchor=(1.1, 3.4), ncol=1)
            
    if save_fig:
        time = datetime.now().strftime("[%d.%m.%y-%H;%M]")
        fig.savefig('output\\{} new_covid_vaccine_pca.png'.format(time), bbox_inches='tight')

    plt.show()

# Scatterplot of two dataframes side by side presenting their median KLDe distances from the domain (visual comparision)
def get_scatter(input_df_1 : pd.DataFrame,
                input_df_2 : pd.DataFrame,
                method : str,
                to_scale : str = None,
                same_yticks : bool = False,
                types : list = None,
                save_figure : bool = False) -> plt.Figure:
    """
    input_df_1 : pd.DataFrame -> Dataframe 1 of KLDe distances from the domain (top side).
    input_df_2 : pd.DataFrame -> Dataframe 2 of KLDe distances from the domain (bottom side).
    method : str -> Which method of analysis was performed, in this case (in the notebook - medians)
    to_scal : str -> name of the scaling techique to be used, can be minmax, standard, robust or normalizer.
    same_yticks : bool -> If True the yticks of both sub-plots will be simillar.
    types : list -> mark the subplots datasource type.
    save_figure : bool -> If True will save the figure into the output folder.
    """
    
    # index must be "str" type array
    input_df_1.index = input_df_1.index.astype("str")
    input_df_2.index = input_df_2.index.astype("str")
     
    scaler_dict = {"minmax": MinMaxScaler((-1,1)).set_output(transform="pandas"),
                   "standard": StandardScaler(with_mean=True, with_std=True).set_output(transform="pandas"),
                   "robust": RobustScaler().set_output(transform="pandas"),
                   "normalizer": Normalizer().set_output(transform="pandas"),
                   "":None}

    if to_scale in scaler_dict.keys():
        input_1T = scaler_dict[to_scale].fit_transform(input_df_1)
        input_2T = scaler_dict[to_scale].fit_transform(input_df_2)
    else:
        to_sacle = "noscaling"
        input_1T = input_df_1
        input_2T = input_df_2

    fig, ax = plt.subplots(2,1, figsize=(15,15))
    for i,j in zip([0,1],[input_1T,input_2T]):
        for col in j.columns:
            ax[i].scatter(x=j.index,
                          y=j[col].values,
                          label=col.split(".")[-1],
                          alpha=0.6,
                          edgecolor="black",
                          linewidth =1)
            
        ##ax[i].set_xticks(j.index.astype("str"))##
        #ax[i].set_xticks(ticks=range(0,len(j.index)), labels=j.index.astype("str"))
        ax[i].tick_params(axis="x", rotation=90)
        ax[i].set_xlim(left=-1, right=len(j.index)+1)

         
        for tick in ax[i].get_xticklabels():
            tick = tick.get_text()
            if j.loc[j.index == tick,:].isnull().values.all():
                line_color = "red"
                alpha = 0.9
                lw = 0.7
            else:
                line_color = "black"
                alpha = 0.65
                lw = 0.4

            ax[i].axvline(x=tick, zorder=0, color=line_color, ls="dotted", alpha=alpha, lw=lw)
  
        ax[i].axhline(y=0, zorder=0, color="black", ls="--", alpha=1, lw=1.3)

        range_cdr = list(range(27,39)) + list(range(56,66)) #CDR1, CDR2
        for xtick, xcolor in zip(ax[i].get_xticklabels(), ["tab:red" if i in range_cdr else "black" for i in input_df_1.index.astype("int")]):
            xtick.set_color(xcolor)

        ax[i].tick_params(axis="both", labelsize=13)
        #ax[i].set_title(types[i], size=15)
        ax[i].text(s=types[i], x=0, y=0.0275, fontsize=15)
        ax[i].set_xlabel("Amino Acid Position", size=17)
        ax[i].set_ylabel(f"{to_scale.capitalize()} {method.capitalize()} of Subject KLDe Distance", size=17)
        ax[i].legend(title="Subjects:", bbox_to_anchor=(1, 1))

    if same_yticks:
        ax_ticks = np.concatenate((ax[0].get_yticks(),ax[1].get_yticks()))
        ax_tminmax = (ax_ticks.min(),ax_ticks.max())
        ax0_tminmax = (ax[0].get_yticks().min(), ax[0].get_yticks().max())
        ax1_tminmax = (ax[1].get_yticks().min(), ax[1].get_yticks().max())
        
        if (ax0_tminmax[0] == ax_tminmax[0]) & (ax0_tminmax[1] == ax_tminmax[1]):
            dom_axis = ax[0].get_yticks()
        elif (ax1_tminmax[0] == ax_tminmax[0]) & (ax1_tminmax[1] == ax_tminmax[1]):
            dom_axis = ax[1].get_yticks()
        else:
            dom_axis = np.unique(ax_ticks)
        
        ticks_array = dom_axis
        tick_uniques = np.unique(np.abs(ticks_array))
        ticks_final = np.unique(np.concatenate((tick_uniques,-tick_uniques)))

        for i in [0,1]:
            ax[i].set_yticks(ticks_final)

            for tick in ax[i].get_yticks():
                ax[i].axhline(y=tick, zorder=0, color="grey", ls="dotted", alpha=0.75, lw=0.4)
        
    if save_figure:
        fig.savefig(f'figures_new\\{method}_{to_scale}_subj.png', bbox_inches='tight')


##################################################################
# [Archived] revision of get_scatter - getting medians scatterplot
def scatter_medians(input_1 : pd.DataFrame,
                    input_2 : pd.DataFrame,
                    scaler : "str" = None,
                    save_fig : bool = False) -> plt.Figure:
    
    if len(input_1.columns) == len(input_2.columns):
        fig, ax = plt.subplots(2,1, figsize=(15,15))
        for i,j in zip([0,1],[input_1,input_2]):
            for col in j.columns:
                ax[i].scatter(x=j.index,
                            y=j[col].values,
                            label=col.split(".")[-1],
                            alpha=0.6,
                            edgecolor="black",
                            linewidth =0.4)
                
            ax[i].set_xticks(j.index.astype("int"))
            ax[i].tick_params(axis="x", rotation=90)

            for tick in ax[i].get_yticks():
                ax[i].axhline(y=tick, zorder=0, color="grey", ls="dotted", alpha=0.75, lw=0.4)

            for tick in ax[i].get_xticks():
                ax[i].axvline(x=tick, zorder=0, color="grey", ls="dotted", alpha=0.75, lw=0.4)

            ax[i].axhline(y=0, zorder=0, color="black", ls="--", alpha=1, lw=1.3)

            range_cdr = list(range(27,39)) + list(range(56,66)) #CDR1, CDR2
            for xtick, xcolor in zip(ax[i].get_xticklabels(), ["tab:red" if i in range_cdr else "black" for i in input_1.index.astype("int")]):
                xtick.set_color(xcolor)

            #ax[i].set_title(types[i], size=15)
            ax[i].set_xlabel("Amino Acid Position", size=12)
            #ax[i].set_ylabel(f"{to_scale.capitalize()} {method.capitalize()} of Subject KLDe Distance", size=12)
            ax[i].legend(title="Subjects:", bbox_to_anchor=(1, 1))

            if save_fig:
                fig.savefig(f'figures_new\\{method}_{to_scale}_subj.png', bbox_inches='tight')

        plt.show()
        
    else:
        raise ValueError('Missmatch between shape of df_1 and df_2.')


##################################################################################
# [Archived] Filling zeros in dataframe according to index+column value [Archived]
def fillzeros(raw_matrix : pd.DataFrame,
              zeros_matrix : pd.DataFrame) -> pd.DataFrame:
    
    for i in docuemnts_zero.itertuples():
        cond_doc = raw_matrix["document"] == i[1]
        cond_elem = raw_matrix["element"] == i[2]
        raw_matrix.loc[(cond_doc) & (cond_elem), "frequency_in_document"] = 0

    return raw_matrix

#######################################################
# [Archived] Place np.nan values in dataframe [Archived]
def place_nan(df_input : pd.DataFrame,
              sizes_df : pd.DataFrame,
              treshold : int = 10) -> pd.DataFrame:
    
    sizes_df["pos_aa"]= sizes_df["pos_aa"].astype("int")
    to_drop = sizes_df.loc[sizes_df.n_mut < 10, ["dataset","pos_aa"]]

    df_naned = df_input
    index_series = df_naned.index.to_series()
  
    for i in df_input.columns:
        aa_drop = to_drop.loc[to_drop["dataset"]==i, "pos_aa"].values.astype("str")
        df_naned.loc[index_series.isin(aa_drop), i] = np.nan

    return df_naned