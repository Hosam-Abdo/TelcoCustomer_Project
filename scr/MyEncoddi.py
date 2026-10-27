import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def encoddi(data_frame, verbose=False, return_encoders=False,drop_first=True):
    df = data_frame.copy()

    bool_feat    = [col for col in df.columns if len(df[col].unique()) == 2]
    
    nonbool_feat = [col for col in df.columns if len(df[col].unique()) > 2  
                    and not is_numeric_dtype(df[col])]
    
    numeric_feat = [col for col in df.columns if df[col].nunique() > 2 
                    and is_numeric_dtype(df[col])]

    for elem in bool_feat:
        oe = LabelEncoder()
        df[elem] = oe.fit_transform(df[elem])
    
    encoders = {}
    for elem in nonbool_feat:
        ohe = OneHotEncoder(drop="first" if drop_first else None,
                            sparse_output=False, dtype=int)
        encoded = ohe.fit_transform(df[[elem]])  

        encoded_df = pd.DataFrame(encoded,
                                  columns = ohe.get_feature_names_out([elem]), 
                                  index = df.index                        
                                 )
        df = pd.concat([df.drop(columns=elem), encoded_df], axis=1)
        encoders[elem] = ohe
    
    if verbose:
        print("bool_feat:    ", bool_feat)
        print("nonbool_feat: ", nonbool_feat)
        print("numeric_feat: ", numeric_feat)
        print(encoders)
        
    return df, encoders