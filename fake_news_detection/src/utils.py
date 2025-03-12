import os
import logging
import yaml  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import gensim

from nltk.corpus import stopwords
from typing import List
from wordcloud import WordCloud


def setup_logging(save_dir, log_name):
    """Setup logging to save logs in the same directory as the model."""
    log_file = os.path.join(save_dir, log_name)
    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    return logging.getLogger()


def load_config(config_path):
    """Loads YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_conf_matrix(conf_matrix, save_dir):
    """Save formatted copy of confusion matrix """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    conf_matrix_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(conf_matrix_path)


def missing_values_analysis(df: pd.DataFrame) -> None:
    """
    Analyse missing values in dataframe attributes.

    Args:
        df: Input dataframe.
    """
    for column in df.columns:
        missing_count = len(df[column][df[column].isna()])
        missing_percentage = round(missing_count / len(df) * 100, 2)
        if missing_count > 0:
            print(f'{column}: {missing_count} ({missing_percentage}%)')

    print('\nMissing values plot (inverse logic, plot is showing how many values are not NaN):')
    msno.bar(df)


def one_value_attributes_analysis(df: pd.DataFrame) -> None:
    """
    Check which attributes contain only one value.

    Args:
        df: Input dataframe.
    """
    for col in df.columns:
        if len(df.loc[:, col].dropna().unique()) == 1:
            print(col)


def analyse_numerical_attributes(df: pd.DataFrame, label_column: str, columns: list = None) -> None:
    """
    Analysis of numerical attributes.

    Args:
        df: Input dataframe.
        label_column: Column representing label.
        columns: List of columns to use, if None, all columns are used.    
    """
    columns = columns if columns is not None else list(df.columns)

    if columns and columns != [label_column]:
        for col in columns:
            if col == label_column:
                continue
            print(f'\n\nAnalysis of attribute "{col}"')
            boxplot_per_classes(df=df, attribute=col, groupby=label_column,
                                title=f'Boxplot of attribute {col} per class')
            kdeplot_per_classes(df=df, attribute=col, groupby=label_column,
                                title=f'Kdeplot of attribute {col} per class')
    else:
        print('There are no attributes to be analysed.')


def kdeplot_per_classes(
    df: pd.DataFrame,
    attribute: str,
    groupby: str,
    title: str = None,
    ticks_rotation: int = 0
) -> None:
    """
    Draw kdeplot of attribute per each class.
    
    Args:
        df: Dataframe with data to be drawn on kdeplot.
        attribute: Name of attribute to be drawn on kdeplot.
        groupby: Name of attribute with predicted classes.
        title: Title of plot.
        ticks_rotation: Rotation of x-ticks (labels).
    """
    for x in df[groupby].unique():
        q75, q25 = np.percentile(df[df[groupby] == x][attribute], [75 ,25])
        iqr = q75 - q25
        if iqr == 0:
            return
        sns.kdeplot(df[df[groupby] == x][attribute], label=x, shade=True, shade_lowest=False)
    plt.title(title)
    plt.xticks(rotation=ticks_rotation)
    plt.show()


def boxplot_per_classes(
    df: pd.DataFrame,
    attribute: str,
    groupby: str,
    title: str = None,
    ticks_rotation: int = 0
) -> None:
    """
    Draw boxplot of attribute per each class.
    
    Args:
        df: Dataframe with data to be drawn on boxplot.
        attribute: Name of attribute to be drawn on boxplot.
        groupby: Name of attribute with predicted classes.
        title: Title of plot.
        ticks_rotation: Rotation of x-ticks (labels).
    """
    sns.boxplot(x=groupby, y=attribute, data=df)
    plt.title(title)
    plt.xticks(rotation=ticks_rotation)
    plt.show()


def analyse_categorical_attributes(df: pd.DataFrame, label_column: str, columns: list = None) -> None:
    """
    Analysis of categorical attributes.

    Args:
        df: Input dataframe.
        label_column: Column representing label.
        columns: List of columns to use, if None, all columns are used.
    """
    columns = columns if columns is not None else list(df.columns)
    
    if columns and columns != [label_column]:
        for col in columns:
            if col == label_column:
                continue
            print(f'\n\nAnalysis of attribute "{col}"')
            barplot_per_classes(df=df, attribute=col, groupby=label_column,
                                title=f'Barplot of attribute {col} per class',
                                ticks_rotation=90, topn=15)
    else:
        print('There are no attributes to be analysed.')


def barplot_per_classes(
    df: pd.DataFrame,
    attribute: str,
    groupby: str,
    title: str = None,
    ticks_rotation: int = 0,
    topn: int = None
) -> None:
    """
    Draw barplot of attribute per each class.
    
    Args:
        df: Dataframe with data to be drawn on barplot.
        attribute: Name of attribute to be drawn on barplot.
        groupby: Name of attribute with predicted classes.
        title: Title of plot.
        ticks_rotation: Rotation of x-ticks (labels).
        topn: Top n classes to be drawn on the plot.
    """
    uniq_values = df[attribute].value_counts().head(topn).index
    df = df[df[attribute].isin(uniq_values)]
    data = df.groupby(groupby)[attribute] \
        .value_counts(normalize=True) \
        .rename('percentage') \
        .mul(100) \
        .reset_index()
    sns.barplot(x=attribute, y='percentage', hue=groupby, data=data)
    plt.xticks(rotation=ticks_rotation)
    plt.title(title)
    plt.show()


def analyse_textual_attributes(df: pd.DataFrame, columns: list = None) -> None:
    """
    Analysis of textual attributes.

    Args:
        df: Input dataframe.
        columns: List of columns to use, if None, all columns are used.
    """
    columns = columns if columns is not None else list(df.columns)
    
    if columns:
        df = text_preprocessing(df=df.copy(), columns=columns)
        for col in columns:
            print(f'\n\nAnalysis of attribute "{col}"')
            word_count_histogram(df=df.copy(), column=col)
            draw_word_cloud(texts=df[f'{col}_preprocessed'].values)
    else:
        print('There are no attributes to be analysed.')


def text_preprocessing(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Text preprocessing helper.

    Args:
        df: Input dataframe.
        columns: List of columns to use, if None, all columns are used.
    
    Returns:
        Dataframe after text preprocessing.
    """
    stop_words = set(stopwords.words('english'))
    for col in columns:
        new_col = f'{col}_preprocessed'
        df[new_col] = df[col].fillna('')
        df[new_col] = df[new_col].apply(lambda x: gensim.utils.simple_preprocess(str(x), deacc=True, min_len=3))
        df[new_col] = df[new_col].apply(lambda x: [word for word in x if word not in stop_words])
        df[new_col] = df[new_col].apply(lambda x: ' '.join(x))
    return df


def word_count_histogram(df: pd.DataFrame, column: str) -> None:
    """
    Draw histograms of number of sentences and words.

    Args:
        df: Input dataframe.
    """
    df.dropna(subset=[column], inplace=True)
    df[f'{column}_wc'] = df[f'{column}'].apply(lambda x: len(x.split()))
    df[f'{column}_wc_preprocessed'] = df[f'{column}_preprocessed'].apply(lambda x: len(x.split()))
    
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f'Word count histograms for attribute {column}', fontsize=16)
    ax = plt.subplot(121)
    ax.set_title('Original text')
    ax.hist(df[f'{column}_wc'], bins=30)

    ax = plt.subplot(122)
    ax.set_title('Preprocessed text')
    ax.hist(df[f'{column}_wc_preprocessed'], bins=30)

    plt.show()


def draw_word_cloud(texts: List[str]) -> None:
    """
    Draw word cloud from text of given dataframe.

    Args:
        texts: List of all texts, building corpora.
    """
    word_cloud = WordCloud(width=1000, height=500,
                          background_color='white',
                          stopwords=set(stopwords.words('english')),
                          min_font_size=10)
    word_cloud.generate(' '.join(texts))
  
    plt.figure(figsize=(10, 5), facecolor=None) 
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.tight_layout(pad=0) 
    plt.show()


def check_correlations(df: pd.DataFrame, columns: list = None) -> None:
    """
    Check correlations of numerical attributes.

    Args:
        df: Input dataframe.
        columns: List of columns to use, if None, all columns are used.
    """
    columns = columns if columns is not None else list(df.columns)

    if columns:
        _, _ = plt.subplots(figsize=(10,10))
        sns.heatmap(df[columns].corr(), annot=True, fmt=".2f", vmin=-1.0, vmax=1.0, square=True);
    else:
        print('There are no attributes to be analysed.')