import transformers
import pandas as pd
import transformers.models as models
import torch
import numpy as np
from typing import Tuple
from . import constants as constants

def load_model_and_tokenizer(
    model_name = constants.MODEL,
    device = constants.DEVICE,
    model_kwargs = {
        'torch_dtype': torch.bfloat16
    },
    tokenizer_kwargs={
        'padding_side': 'left'
    }
) -> Tuple[models.llama.modeling_llama.LlamaForCausalLM, models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast]:  # NOTE: Change the type hint depending on the model_name
    # Load the model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    return model, tokenizer


def optimize_dataframe(df: pd.DataFrame, sort_for_compression: bool=True):
    """
    Optimize DataFrame for space efficiency by:
    1. First downcasting numbers to smallest possible type
    2. Then converting to categorical if it saves space
    3. Optional sorting for better Parquet compression
    """
    df_opt = df.copy()
    
    for col in df_opt.columns:
        col_type = df_opt[col].dtype
        
        # Stage 1: Numeric Optimization
        if pd.api.types.is_integer_dtype(col_type):
            # Get min and max to determine smallest possible int type
            downcast_type = 'unsigned' if df_opt[col].min() >= 0 else 'integer'  
            df_opt[col] = pd.to_numeric(df_opt[col], downcast=downcast_type)
        
        elif pd.api.types.is_float_dtype(col_type):
            original_values = df_opt[col].values.copy()
            df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')
            if not np.allclose(original_values, df_opt[col].values, equal_nan=True):    # type: ignore
                df_opt[col] = original_values  # Revert if precision loss
        
        # Stage 2: Categorical Check (for all non-categorical columns)
        if not isinstance(df_opt[col], pd.CategoricalDtype):
            current_size = df_opt[col].memory_usage(deep=True)
            cat_size = df_opt[col].astype('category').memory_usage(deep=True)
            if cat_size < current_size:
                df_opt[col] = df_opt[col].astype('category')
    
    # Optional sorting for Parquet compression
    if sort_for_compression:
        cardinality = {col: df_opt[col].nunique() for col in df_opt.columns}
        df_opt = df_opt.sort_values(by=sorted(cardinality.keys(), key=lambda x: cardinality[x]))
    
    return df_opt


if __name__ == '__main__':
    DEVICE = 'cuda'
    model, tokenizer = load_model_and_tokenizer(device=DEVICE)