# estimators.py

# We define the clean functions to run the models with the linearmodels library

from linearmodels import IV2SLS
from linearmodels.panel import PanelOLS, PooledOLS

def estimate_models(df):
    # Estiates the model using FE and Pooled OLS
    # Returns the coefficients estimated for y_{it-1} and the GMM Anderson-Hsiao (IV)

    # Model 1: FE
    mod_fe = PanelOLS(df.y, df.y_lag, entity_effects=True)
    gamma_fe = mod_fe.fit(cov_type='unadjusted').params.iloc[0]

    # Model 2: Pooled OLS (ignores the FE)
    mod_pool = PooledOLS(df.y, df.y_lag)
    gamma_pool = mod_pool.fit(cov_type='unadjusted').params.iloc[0]

    # Anderson-Hsiao (IV) estimator
    # Calculate the differences
    df_ah = df.copy()
    df_ah['dy'] = df_ah.groupby('id')['y'].diff()
    df_ah['dy_lag'] = df_ah.groupby('id')['y_lag'].diff()

    # Create the instrument (lagged differences)
    df_ah['y_lag2'] = df_ah.groupby('id')['y'].shift(2)

    # Drop missing values
    df_ah = df_ah.dropna()

    # Estimate the model using the instrument
    mod_iv = IV2SLS(
        dependent=df_ah['dy'], 
        exog=None,
        endog=df_ah['dy_lag'], 
        instruments=df_ah['y_lag2']
    )
    
    cluster_ids = df_ah.index.get_level_values('id')
    gamma_iv = mod_iv.fit(cov_type='clustered', clusters=cluster_ids).params['dy_lag']  

    return gamma_fe, gamma_pool, gamma_iv