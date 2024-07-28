import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Tuple

def split_data(raw_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the raw DataFrame into training and validation sets.
    
    Parameters:
    raw_df (pd.DataFrame): The raw input DataFrame.
    target_col (str): The column name to be used as the target.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: The training and validation DataFrames.
    """
    train_df, val_df = train_test_split(raw_df, test_size=0.2, random_state=12, stratify=raw_df[target_col])
    return train_df, val_df

def del_cols(col_list: List[str], delete_cols: List[str]) -> List[str]:
    """
    Delete specified columns from a list of columns.
    
    Parameters:
    col_list (List[str]): The original list of columns.
    delete_cols (List[str]): List of columns to delete.
    
    Returns:
    List[str]: The updated list of columns.
    """
    for col in delete_cols:
        if col in col_list:
            col_list.remove(col)
    return col_list

def define_columns(df: pd.DataFrame, delete_cols: List[str]) -> Tuple[List[str], str, List[str], List[str]]:
    """
    Define input, target, numeric, and categorical columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    delete_cols (List[str]): List of columns to delete.
    
    Returns:
    Tuple[List[str], str, List[str], List[str]]: Input columns, target column, numeric columns, and categorical columns.
    """
    # Define input, target, numeric, categorical columns
    input_cols = df.columns.tolist()[:-1]
    target_col = df.columns.tolist()[-1]

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:-1]
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # Remove specified columns from column lists
    input_cols = del_cols(input_cols, delete_cols)
    numeric_cols = del_cols(numeric_cols, delete_cols)
    categorical_cols = del_cols(categorical_cols, delete_cols)

    return input_cols, target_col, numeric_cols, categorical_cols

def create_inputs_targets(df: pd.DataFrame, input_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create input and target DataFrames from the given DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    input_cols (List[str]): List of column names to be used as inputs.
    target_col (str): The column name to be used as the target.
    
    Returns:
    Tuple[pd.DataFrame, pd.Series]: The inputs and targets.
    """
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    return inputs, targets

def preprocess_numeric(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, numeric_cols: List[str], scaler: MinMaxScaler) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scale numeric columns using MinMaxScaler.
    
    Parameters:
    train_inputs (pd.DataFrame): Training inputs.
    val_inputs (pd.DataFrame): Validation inputs.
    numeric_cols (List[str]): List of numeric columns.
    scaler (MinMaxScaler): Scaler instance.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]: Scaled training and validation inputs, and the scaler used.
    """
    scaler.fit(train_inputs[numeric_cols])
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    return train_inputs, val_inputs, scaler

def preprocess_categorical(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, categorical_cols: List[str], encoder: OneHotEncoder) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], OneHotEncoder]:
    """
    Encode categorical columns using OneHotEncoder.
    
    Parameters:
    train_inputs (pd.DataFrame): Training inputs.
    val_inputs (pd.DataFrame): Validation inputs.
    categorical_cols (List[str]): List of categorical columns.
    encoder (OneHotEncoder): Encoder instance.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, List[str], OneHotEncoder]: Encoded training and validation inputs, list of encoded column names, and the encoder used.
    """
    encoder.fit(train_inputs[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
    return train_inputs, val_inputs, encoded_cols, encoder

def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool, delete_cols: List[str]) -> Tuple[Dict[str, Any], MinMaxScaler, OneHotEncoder, List[str]]:
    """
    Preprocess the raw DataFrame for machine learning.
    
    Parameters:
    raw_df (pd.DataFrame): The raw input DataFrame.
    scaler_numeric (bool): Whether to apply MinMaxScaler to numeric columns.
    delete_cols (List[str]): List of columns to delete from the input features.
    
    Returns:
    Tuple[Dict[str, Any], MinMaxScaler, OneHotEncoder, List[str]]: A dictionary containing preprocessed training and validation data,
    the scaler used, the encoder used, and the list of input columns.
    """
    # Split the data into training and validation sets
    train_df, val_df = split_data(raw_df, 'Exited')
    
    # Define input, target, numeric, categorical columns
    input_cols, target_col, numeric_cols, categorical_cols = define_columns(train_df, delete_cols)
    
    # Separate inputs and targets
    train_inputs, train_targets = create_inputs_targets(train_df, input_cols, target_col)
    val_inputs, val_targets = create_inputs_targets(val_df, input_cols, target_col)
        
    # Scale numeric columns if specified
    if scaler_numeric:
        scaler = MinMaxScaler()
        train_inputs, val_inputs, scaler = preprocess_numeric(train_inputs, val_inputs, numeric_cols, scaler)
    else:
        scaler = None
            
    # Encode categorical columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    train_inputs, val_inputs, encoded_cols, encoder = preprocess_categorical(train_inputs, val_inputs, categorical_cols, encoder)
    
    # Prepare final training and validation sets
    X_train = train_inputs[numeric_cols + encoded_cols]
    X_val = val_inputs[numeric_cols + encoded_cols]
    
    return {
        'train_X': X_train,
        'train_y': train_targets,
        'val_X': X_val,
        'val_y': val_targets
    }, scaler, encoder, input_cols

def preprocess_new_data(new_df: pd.DataFrame, scaler: MinMaxScaler, encoder: OneHotEncoder, scaler_numeric: bool, delete_cols: List[str]) -> Dict[str, Any]:
    """
    Preprocess new data for predictions.
    
    Parameters:
    new_df (pd.DataFrame): The new input DataFrame.
    scaler (MinMaxScaler): The scaler used for numeric columns.
    encoder (OneHotEncoder): The encoder used for categorical columns.
    scaler_numeric (bool): Whether to apply MinMaxScaler to numeric columns.
    delete_cols (List[str]): List of columns to delete from the input features.
    
    Returns:
    Dict[str, Any]: A dictionary containing preprocessed new inputs and targets.
    """
    input_cols, target_col, numeric_cols, categorical_cols = define_columns(new_df, delete_cols)
    new_inputs, new_targets = create_inputs_targets(new_df, input_cols, target_col)

    if scaler_numeric:
        new_inputs[numeric_cols] = scaler.transform(new_inputs[numeric_cols])

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    new_inputs[encoded_cols] = encoder.transform(new_inputs[categorical_cols])

    # Prepare final training and validation sets
    X_new = new_inputs[numeric_cols + encoded_cols]

    return {
        'new_X': X_new,
        'new_y': new_targets
    }