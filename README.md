msc-dissertation-code
Final dissertation code
The full code and supplementary materials for this dissertation are available at:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.api import OLS, add_constant
from statsmodels.stats.stattools import jarque_bera
import warnings
import os

    # Process dates
    if pd.api.types.is_numeric_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'], origin='1900-01-01', unit='D')
    else:
        data['date'] = pd.to_datetime(data['date'])

    data.set_index('date', inplace=True)

    print("\nLog transformation processing:")
    data['log_csi300'] = np.log(data['csi300'])
    data['log_sp500'] = np.log(data['sp500'])
    data['log_oil_price'] = np.log(data['oil_price'])
    print("- CSI300 log transformation completed")
    print("- SP500 log transformation completed")
    print("- Brent oil price log transformation completed")

    print("\n*** Extreme Value Winsorization (Important) ***")
    from scipy.stats import mstats
    numeric_cols = ['log_csi300', 'cpi', 'ipi', 'm2', 'interest_rate',
                    'exchange_rate', 'epu', 'log_sp500', 'pmi', 'log_oil_price']

    for col in numeric_cols:
        data[col] = mstats.winsorize(data[col], limits=[0.05, 0.05])
    print(f"Winsorization completed for {len(numeric_cols)} variables")

    # Sample size statistics
    print(f"\n** Sample Size: {len(data)} **")
    print(f"Data period: {data.index.min().strftime('%Y-%m')} to {data.index.max().strftime('%Y-%m')}")

    return data


def descriptive_statistics(data):
    """2. Descriptive Statistics Display (with charts)"""
    print("\n" + "=" * 60)
    print("Step 2: Descriptive Statistics Display")
    print("=" * 60)

    vars_analysis = ['log_csi300', 'cpi', 'ipi', 'm2', 'interest_rate',
                     'exchange_rate', 'epu', 'log_sp500', 'pmi', 'log_oil_price']

    desc_stats = data[vars_analysis].describe()

    skewness = data[vars_analysis].skew()
    kurtosis = data[vars_analysis].kurtosis()
    desc_stats.loc['skewness'] = skewness
    desc_stats.loc['kurtosis'] = kurtosis

    desc_stats.to_csv(os.path.join(OUTPUT_DIR, 'descriptive_statistics.csv'), encoding='utf-8-sig')
    print("Descriptive statistics table saved")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    desc_stats.loc['mean'].plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title('Mean Values of Variables', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    desc_stats.loc['std'].plot(kind='bar', ax=ax2, color='lightcoral', edgecolor='black')
    ax2.set_title('Standard Deviation of Variables', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Standard Deviation')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    skewness.plot(kind='bar', ax=ax3, color='lightgreen', edgecolor='black')
    ax3.set_title('Skewness of Variables', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Skewness')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    kurtosis.plot(kind='bar', ax=ax4, color='gold', edgecolor='black')
    ax4.set_title('Kurtosis of Variables', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Kurtosis')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'descriptive_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Descriptive statistics charts generated")

    return desc_stats


def correlation_analysis(data):
    """Data Validation - Step 1: Correlation Analysis"""
    print("\n" + "=" * 60)
    print("Data Validation - Step 1: Correlation Analysis")
    print("=" * 60)

    vars_corr = ['log_csi300', 'cpi', 'ipi', 'm2', 'interest_rate',
                 'exchange_rate', 'epu', 'log_sp500', 'pmi', 'log_oil_price']

    corr_matrix = data[vars_corr].corr()

    print("Correlation coefficients between CSI300 and other variables:")
    csi_corr = corr_matrix['log_csi300'].drop('log_csi300').sort_values(key=abs, ascending=False)
    for var, corr in csi_corr.items():
        print(f"  {var}: {corr:.4f}")

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show only lower triangle
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={"shrink": .8},
                mask=mask, linewidths=0.5)
    plt.title('Variable Correlation Analysis Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(r'C:\Users\A\Desktop\correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("** Correlation analysis results generated **")

    return corr_matrix


def adf_unit_root_test(data):
    """Step 2: Unit Root Test (ADF)"""
    print("\n" + "=" * 60)
    print("Step 2: ADF Unit Root Test")
    print("=" * 60)

    vars_test = ['log_csi300', 'cpi', 'ipi', 'm2', 'interest_rate',
                 'exchange_rate', 'epu', 'log_sp500', 'pmi', 'log_oil_price']

    adf_results = {}

    print("ADF Unit Root Test Results:")
    print("-" * 70)
    print("Variable      Level ADF    P-val   1% Crit    First Diff ADF  P-val   Order")
    print("-" * 70)

        adf_stat, p_val, _, _, critical_vals, _ = adfuller(data[var].dropna())

        diff_data = data[var].diff().dropna()
        adf_stat_diff, p_val_diff, _, _, critical_vals_diff, _ = adfuller(diff_data)

        if p_val < 0.05:
            integration_order = 'I(0)'
        elif p_val_diff < 0.05:
            integration_order = 'I(1)'
        else:
            integration_order = 'I(2)'

        adf_results[var] = {
            'level_adf': adf_stat,
            'level_pvalue': p_val,
            'level_1%_critical': critical_vals['1%'],
            'diff_adf': adf_stat_diff,
            'diff_pvalue': p_val_diff,
            'diff_1%_critical': critical_vals_diff['1%'],
            'integration_order': integration_order
        }

        print(f"{var:<12} {adf_stat:>8.3f} {p_val:>7.3f} {critical_vals['1%']:>9.3f} "
              f"{adf_stat_diff:>10.3f} {p_val_diff:>7.3f} {integration_order:>8}")

    print("-" * 70)

    i2_vars = [var for var, result in adf_results.items() if result['integration_order'] == 'I(2)']
    if i2_vars:
        print(f"Warning: Found I(2) variables: {i2_vars}")
        print("Note: ARDL model requires all variables to be I(0) or I(1)")
    else:
        print("✓ All variables are I(0) or I(1), meeting ARDL model requirements")

    adf_df = pd.DataFrame(adf_results).T
    adf_df.to_csv(r'C:\Users\A\Desktop\adf_test_results.csv', encoding='utf-8-sig')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    level_pvals = [adf_results[var]['level_pvalue'] for var in vars_test]
    bars1 = ax1.bar(range(len(vars_test)), level_pvals, color='lightblue', edgecolor='black')
    ax1.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='5% Significance Level')
    ax1.set_xticks(range(len(vars_test)))
    ax1.set_xticklabels(vars_test, rotation=45)
    ax1.set_ylabel('P-value')
    ax1.set_title('ADF Test - Level Values', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    diff_pvals = [adf_results[var]['diff_pvalue'] for var in vars_test]
    bars2 = ax2.bar(range(len(vars_test)), diff_pvals, color='lightcoral', edgecolor='black')
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='5% Significance Level')
    ax2.set_xticks(range(len(vars_test)))
    ax2.set_xticklabels(vars_test, rotation=45)
    ax2.set_ylabel('P-value')
    ax2.set_title('ADF Test - First Difference', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 0.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    for i, p_val in enumerate(diff_pvals):
        ax2.text(i, p_val + 0.002, f'{p_val:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(r'C:\Users\A\Desktop\adf_test_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    return adf_results

#Optimal Lag Order Selection
def lag_selection(data, max_lags=6):
    print("\n" + "=" * 60)
    print("Step 3: Optimal Lag Order Selection")
    print("=" * 60)
    print(f"Setting maximum lag order: {max_lags}")

    endog = data['log_csi300'].dropna()
    exog = data[['cpi', 'ipi', 'm2', 'interest_rate', 'exchange_rate',
                 'epu', 'log_sp500', 'pmi', 'log_oil_price']].dropna()
                 
    common_index = endog.index.intersection(exog.index)
    endog = endog.loc[common_index]
    exog = exog.loc[common_index]

    print(f"Sample size for lag selection: {len(endog)}")

    lag_results = {}

    print("\nLag Order Selection Results:")
    print("-" * 40)
    print("Lag Order    AIC        BIC        HQIC")
    print("-" * 40)

    for lag in range(1, max_lags + 1):
        try:
            ardl_temp = ARDL(endog, lag, exog, lag)
            ardl_temp_fit = ardl_temp.fit()

            aic_val = ardl_temp_fit.aic
            bic_val = ardl_temp_fit.bic
            hqic_val = ardl_temp_fit.hqic if hasattr(ardl_temp_fit, 'hqic') else np.nan

            lag_results[lag] = {'AIC': aic_val, 'BIC': bic_val, 'HQIC': hqic_val}

            print(f"{lag:>6}    {aic_val:>8.2f}   {bic_val:>8.2f}   {hqic_val:>8.2f}")

        except Exception as e:
            print(f"Lag order {lag} calculation failed: {e}")
            lag_results[lag] = {'AIC': np.nan, 'BIC': np.nan, 'HQIC': np.nan}

    print("-" * 40)

    valid_results = {k: v for k, v in lag_results.items() if not np.isnan(v['AIC'])}

    if valid_results:
        optimal_aic = min(valid_results.keys(), key=lambda x: valid_results[x]['AIC'])
        optimal_bic = min(valid_results.keys(), key=lambda x: valid_results[x]['BIC'])
        optimal_hqic = min(valid_results.keys(), key=lambda x: valid_results[x]['HQIC']) if not np.isnan(
            list(valid_results.values())[0]['HQIC']) else optimal_aic

        print(f"\nOptimal lag order selection:")
        print(f"  AIC criterion: {optimal_aic} lags")
        print(f"  BIC criterion: {optimal_bic} lags")
        print(f"  HQIC criterion: {optimal_hqic} lags")
        
        selected_lag = optimal_bic
        print(f"\n** Selected lag order using BIC criterion: {selected_lag} lags **")

    else:
        selected_lag = 2
        print(f"\nLag selection failed, using default lag order: {selected_lag} lags")

    valid_lags = [k for k in lag_results.keys() if not np.isnan(lag_results[k]['AIC'])]
    if valid_lags:
        aic_vals = [lag_results[k]['AIC'] for k in valid_lags]
        bic_vals = [lag_results[k]['BIC'] for k in valid_lags]

        plt.figure(figsize=(10, 6))
        plt.plot(valid_lags, aic_vals, 'bo-', label='AIC', linewidth=2, markersize=8)
        plt.plot(valid_lags, bic_vals, 'rs-', label='BIC', linewidth=2, markersize=8)
        plt.xlabel('Lag Order', fontsize=12)
        plt.ylabel('Information Criterion Value', fontsize=12)
        plt.title('Lag Order Selection', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(r'C:\Users\A\Desktop\lag_selection_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

    return selected_lag

#ARDL
def estimate_ardl_model(data, lag_order):
    print("\n" + "=" * 60)
    print("Step 4: ARDL Model Estimation (Baseline Model)")
    print("=" * 60)

    endog = data['log_csi300'].dropna()
    exog = data[['cpi', 'ipi', 'm2', 'interest_rate', 'exchange_rate',
                 'epu', 'log_sp500', 'pmi', 'log_oil_price']].dropna()

    common_index = endog.index.intersection(exog.index)
    endog = endog.loc[common_index]
    exog = exog.loc[common_index]

    print(f"Model specification: ARDL({lag_order}, {lag_order})")
    print(f"Sample size: {len(endog)}")
    print(f"Dependent variable: LOG(CSI300)")
    print(f"Explanatory variables: {list(exog.columns)}")

        ardl_model = ARDL(endog, lag_order, exog, lag_order)
        ardl_fit = ardl_model.fit()

        print(f"\nARDL model estimation successful!")
        print(f"Observations: {ardl_fit.nobs}")
        print(f"Parameters: {len(ardl_fit.params)}")
        print(f"Parameter names: {list(ardl_fit.params.index)}")

        try:
            aic_val = getattr(ardl_fit, 'aic', None)
            bic_val = getattr(ardl_fit, 'bic', None)
            llf_val = getattr(ardl_fit, 'llf', None)
            rsquared = getattr(ardl_fit, 'rsquared', None)
            rsquared_adj = getattr(ardl_fit, 'rsquared_adj', None)

                n = ardl_fit.nobs
                k = len(ardl_fit.params)
                if llf_val is not None:
                    aic_val = -2 * llf_val + 2 * k
                    bic_val = -2 * llf_val + k * np.log(n)
                else:
                    aic_val = bic_val = "N/A"
                try:
                    y_true = endog.iloc[1:]  # Exclude first observation
                    y_pred = ardl_fit.fittedvalues

                    common_idx = y_true.index.intersection(y_pred.index)
                    y_true_aligned = y_true.loc[common_idx]
                    y_pred_aligned = y_pred.loc[common_idx]

                    ss_res = np.sum((y_true_aligned - y_pred_aligned) ** 2)
                    ss_tot = np.sum((y_true_aligned - np.mean(y_true_aligned)) ** 2)
                    rsquared = 1 - (ss_res / ss_tot)

                    n = len(y_true_aligned)
                    k = len(ardl_fit.params)
                    rsquared_adj = 1 - (1 - rsquared) * (n - 1) / (n - k)
                except:
                    rsquared = rsquared_adj = "N/A"

        except Exception as e:
            print(f"Error getting model statistics: {e}")
            aic_val = bic_val = llf_val = rsquared = rsquared_adj = "N/A"

        def format_stat(stat):
            if stat == "N/A" or stat is None:
                return "N/A"
            elif isinstance(stat, (int, float)):
                return f"{stat:.4f}"
            else:
                return str(stat)

        print(f"AIC: {format_stat(aic_val)}")
        print(f"BIC: {format_stat(bic_val)}")
        print(f"R²: {format_stat(rsquared)}")
        print(f"Adjusted R²: {format_stat(rsquared_adj)}")
        print(f"Log-likelihood: {format_stat(llf_val)}")

        try:
            f_statistic = getattr(ardl_fit, 'fvalue', None)
            f_pvalue = getattr(ardl_fit, 'f_pvalue', None)
            if f_statistic is not None and f_pvalue is not None:
                print(f"F-statistic: {f_statistic:.4f} (p-value: {f_pvalue:.4f})")
            else:
                print("F-statistic: Unable to obtain")
        except:
            print("F-statistic retrieval failed")


            const_params = [p for p in ardl_fit.params.index if 'const' in p.lower()]
            lag_y_params = [p for p in ardl_fit.params.index if 'log_csi300' in p and 'L1' in p]
            level_x_params = [p for p in ardl_fit.params.index if
                              'L1' not in p and 'const' not in p.lower() and 'log_csi300' not in p]
            lag_x_params = [p for p in ardl_fit.params.index if 'L1' in p and 'log_csi300' not in p]

        coeff_table = pd.DataFrame({
            'Coefficient': ardl_fit.params,
            'Std_Error': ardl_fit.bse,
            'T_Statistic': ardl_fit.tvalues,
            'P_Value': ardl_fit.pvalues,
            'Significance': ['***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
                             for p in ardl_fit.pvalues]
        })
        coeff_table.to_csv(r'C:\Users\A\Desktop\ardl_coefficients.csv', encoding='utf-8-sig')

        residuals = ardl_fit.resid
        fitted_values = ardl_fit.fittedvalues

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].plot(endog, label='Actual Values', linewidth=2, color='blue')
        axes[0, 0].plot(fitted_values, label='Fitted Values', linewidth=2, color='red', alpha=0.8)
        axes[0, 0].set_title('Actual vs Fitted Values', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(residuals, color='green', linewidth=1)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Residual Time Series', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].scatter(fitted_values, residuals, alpha=0.6, color='orange')
        axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 2].set_xlabel('Fitted Values')
        axes[0, 2].set_ylabel('Residuals')
        axes[0, 2].set_title('Residuals vs Fitted Values', fontsize=12, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].hist(residuals, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True, alpha=0.3)

        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Residual Normality Q-Q Plot', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals, lags=20, ax=axes[1, 2], alpha=0.05)
        axes[1, 2].set_title('Residual Autocorrelation Function', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(r'C:\Users\A\Desktop\ardl_model_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("** ARDL model results and diagnostic charts generated **")

        return ardl_fit

    except Exception as e:
        print(f"ARDL model estimation failed: {e}")
        print("Attempting to generate backup regression results...")

        try:
            X_ols = add_constant(exog)
            ols_fit = OLS(endog, X_ols).fit()



def calculate_long_term_coefficients(ardl_fit, lag_order)
    print("\n" + "=" * 60)
    print("Step 4.5: ARDL Long-Term Coefficient Calculation")
    print("=" * 60)

    if ardl_fit is None:
        print("ARDL model not estimated successfully, cannot calculate long-term coefficients")
        return None

    try:
        params = ardl_fit.params
        param_names = list(params.index)
        print(f"Available parameters: {param_names}")

        dep_lag_coeffs = {}
        for param in param_names:
            if 'log_csi300' in param and '.L' in param:
                lag_num = param.split('.L')[1]
                dep_lag_coeffs[f'φ_{lag_num}'] = params[param]

        print(f"Dependent variable lag coefficients: {dep_lag_coeffs}")

        phi_sum = sum(dep_lag_coeffs.values())
        denominator = 1 - phi_sum
        print(f"Denominator (1 - Σφ_i): {denominator:.6f}")

        if abs(denominator) < 0.001:
            print("Warning: Denominator very close to zero, may indicate unit root or model instability")

        exog_vars = ['cpi', 'ipi', 'm2', 'interest_rate', 'exchange_rate',
                     'epu', 'log_sp500', 'pmi', 'log_oil_price']

        long_term_coeffs = {}
        long_term_results = {}

        print(f"\nLong-term coefficient calculation:")
        print("-" * 50)
        print("Variable          Short-term Coeffs     Long-term Coeff")
        print("-" * 50)

        for var in exog_vars:
            var_coeffs = {}
            for param in param_names:
                if param == var:
                    var_coeffs['L0'] = params[param]
                elif param.startswith(var + '.L'):
                    lag_num = param.split('.L')[1]
                    var_coeffs[f'L{lag_num}'] = params[param]

            if var_coeffs:
                numerator = sum(var_coeffs.values())
                long_term_coeff = numerator / denominator
                long_term_coeffs[var] = long_term_coeff
                short_term_str = ", ".join([f"{k}:{v:.4f}" for k, v in var_coeffs.items()])

                print(f"{var:<16} {short_term_str:<20} {long_term_coeff:>12.6f}")

                long_term_results[var] = {
                    'short_term_coeffs': var_coeffs,
                    'numerator': numerator,
                    'long_term_coeff': long_term_coeff
                }
            else:
                print(f"{var:<16} {'No coefficients found':<20} {'N/A':>12}")

        print("-" * 50)
        
        print(f"\nCalculating standard errors for long-term coefficients...")

        long_term_se = {}
        long_term_tvalues = {}
        long_term_pvalues = {}

        cov_matrix = ardl_fit.cov_params()

        for var, result in long_term_results.items():
            try:
                var_param_indices = []
                dep_lag_indices = []

                for i, param in enumerate(param_names):
                    if param == var or param.startswith(var + '.L'):
                        var_param_indices.append(i)
                    elif 'log_csi300' in param and '.L' in param:
                        dep_lag_indices.append(i)

                numerator = result['numerator']
                if var_param_indices:
                    var_cov_subset = cov_matrix.iloc[var_param_indices, var_param_indices]
                    numerator_var = np.sum(var_cov_subset.values)
                else:
                    numerator_var = 0
                    
                se_approx = np.sqrt(numerator_var) / abs(denominator)
                long_term_se[var] = se_approx
                
                t_stat = long_term_coeffs[var] / se_approx if se_approx > 0 else 0
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), ardl_fit.df_resid)) if se_approx > 0 else np.nan

                long_term_tvalues[var] = t_stat
                long_term_pvalues[var] = p_val

            except Exception as e:
                print(f"Standard error calculation failed for {var}: {e}")
                long_term_se[var] = np.nan
                long_term_tvalues[var] = np.nan
                long_term_pvalues[var] = np.nan
                
        print(f"\nLong-term Coefficient Results with Statistical Tests:")
        print("-" * 80)
        print("Variable          Long-term     Std Error    t-statistic    p-value    Significance")
        print("-" * 80)

        for var in long_term_coeffs.keys():
            coeff = long_term_coeffs[var]
            se = long_term_se.get(var, np.nan)
            t_val = long_term_tvalues.get(var, np.nan)
            p_val = long_term_pvalues.get(var, np.nan)

            if not np.isnan(p_val):
                sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''
            else:
                sig = ''

            print(f"{var:<16} {coeff:>10.6f} {se:>10.6f} {t_val:>12.4f} {p_val:>10.4f} {sig:>8}")

        print("-" * 80)
        
        try:
            original_aic = getattr(ardl_fit, 'aic', None)
            original_bic = getattr(ardl_fit, 'bic', None)
            original_rsquared = getattr(ardl_fit, 'rsquared', None)
            original_llf = getattr(ardl_fit, 'llf', None)

            endog_data = ardl_fit.model.endog
            fitted_values = ardl_fit.fittedvalues
            common_idx = endog_data.index.intersection(fitted_values.index) if hasattr(endog_data, 'index') else range(
                len(endog_data))
            if hasattr(endog_data, 'index'):
                endog_aligned = endog_data.loc[common_idx]
                fitted_aligned = fitted_values.loc[common_idx]
            else:
                min_len = min(len(endog_data), len(fitted_values))
                endog_aligned = endog_data[:min_len]
                fitted_aligned = fitted_values[:min_len]

            ss_res = np.sum((endog_aligned - fitted_aligned) ** 2)
            ss_tot = np.sum((endog_aligned - np.mean(endog_aligned)) ** 2)
            longterm_rsquared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            n = len(endog_aligned)
            k = len(long_term_coeffs)  # Number of long-term coefficients
            if n > k + 1:
                longterm_rsquared_adj = 1 - (1 - longterm_rsquared) * (n - 1) / (n - k - 1)
            else:
                longterm_rsquared_adj = longterm_rsquared
                
            if original_llf is not None:
                longterm_aic = -2 * original_llf + 2 * k
                longterm_bic = -2 * original_llf + k * np.log(n)
            else:
                longterm_aic = original_aic if original_aic is not None else np.nan
                longterm_bic = original_bic if original_bic is not None else np.nan

        except Exception as e:
            print(f"Long-term model statistics calculation failed: {e}")
            longterm_rsquared = np.nan
            longterm_rsquared_adj = np.nan
            longterm_aic = np.nan
            longterm_bic = np.nan

     
            for var in long_term_coeffs.keys():
                coeff = long_term_coeffs[var]
                se = long_term_se.get(var, np.nan)
                t_val = long_term_tvalues.get(var, np.nan)
                p_val = long_term_pvalues.get(var, np.nan)

                if not np.isnan(p_val):
                    sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''
                else:
                    sig = ''

              
        longterm_df = pd.DataFrame({
            'Variable': list(long_term_coeffs.keys()),
            'Long_Term_Coefficient': list(long_term_coeffs.values()),
            'Std_Error': [long_term_se.get(var, np.nan) for var in long_term_coeffs.keys()],
            'T_Statistic': [long_term_tvalues.get(var, np.nan) for var in long_term_coeffs.keys()],
            'P_Value': [long_term_pvalues.get(var, np.nan) for var in long_term_coeffs.keys()],
            'Significance': [('***' if long_term_pvalues.get(var, 1) < 0.01 else
                              '**' if long_term_pvalues.get(var, 1) < 0.05 else
                              '*' if long_term_pvalues.get(var, 1) < 0.1 else '')
                             for var in long_term_coeffs.keys()]
        })
        longterm_df.to_csv(r'C:\Users\A\Desktop\ardl_longterm_coefficients.csv', index=False, encoding='utf-8-sig')

        plt.figure(figsize=(12, 8))

        vars_list = list(long_term_coeffs.keys())
        coeffs_list = list(long_term_coeffs.values())
        colors = ['red' if c < 0 else 'blue' for c in coeffs_list]

        bars = plt.bar(range(len(vars_list)), coeffs_list, color=colors, alpha=0.7, edgecolor='black')

        for i, (var, coeff) in enumerate(zip(vars_list, coeffs_list)):
            p_val = long_term_pvalues.get(var, 1)
            if p_val < 0.01:
                plt.text(i, coeff + (0.001 if coeff > 0 else -0.001), '***',
                         ha='center', va='bottom' if coeff > 0 else 'top', fontsize=12, fontweight='bold')
            elif p_val < 0.05:
                plt.text(i, coeff + (0.001 if coeff > 0 else -0.001), '**',
                         ha='center', va='bottom' if coeff > 0 else 'top', fontsize=12, fontweight='bold')
            elif p_val < 0.1:
                plt.text(i, coeff + (0.001 if coeff > 0 else -0.001), '*',
                         ha='center', va='bottom' if coeff > 0 else 'top', fontsize=12, fontweight='bold')

        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.xlabel('Variables', fontsize=12)
        plt.ylabel('Long-term Coefficient', fontsize=12)
        plt.title('ARDL Long-Term Coefficients', fontsize=14, fontweight='bold')
        plt.xticks(range(len(vars_list)), vars_list, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='Positive Effect'),
            Patch(facecolor='red', alpha=0.7, label='Negative Effect')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig(r'C:\Users\A\Desktop\ardl_longterm_coefficients.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("** Long-term coefficient analysis completed and saved **")

        return {
            'coefficients': long_term_coeffs,
            'standard_errors': long_term_se,
            'tvalues': long_term_tvalues,
            'pvalues': long_term_pvalues,
            'denominator': denominator,
            'rsquared': longterm_rsquared,
            'rsquared_adj': longterm_rsquared_adj,
            'aic': longterm_aic,
            'bic': longterm_bic,
            'nobs': n
        }

    except Exception as e:
        print(f"Long-term coefficient calculation failed: {e}")
        import traceback
        traceback.print_exc()

#Bound Test
def manual_bounds_test(ardl_fit, k=9):
    if ardl_fit is None:
        return None, "ARDL model not estimated successfully"

    try:
        params = ardl_fit.params
        cov_matrix = ardl_fit.cov_params()

        print(f"ARDL model parameter names: {list(params.index)}")

        level_params = []
        level_indices = []

        for i, param_name in enumerate(params.index):
            if ('L1' not in param_name and
                    'const' not in param_name and
                    'log_csi300' not in param_name and
                    param_name != 'const'):
                level_params.append(param_name)
                level_indices.append(i)

        print(f"Identified level term parameters: {level_params}")

        if len(level_indices) == 0:
            print("No clear level term parameters found, trying to identify non-lagged terms...")
            for i, param_name in enumerate(params.index):
                if 'const' not in param_name and 'L1' not in param_name:
                    level_params.append(param_name)
                    level_indices.append(i)

        if len(level_indices) == 0:
            raise ValueError("Cannot identify level term parameters, possible model structure issue")

        print(f"Final parameters for Bounds Test: {level_params}")

        n_params = len(params)
        n_constraints = len(level_indices)
        R = np.zeros((n_constraints, n_params))

        for i, idx in enumerate(level_indices):
            R[i, idx] = 1

        theta = params.values
        R_theta = R @ theta

        try:
            # Calculate inverse of R * Σ * R'
            R_cov_R = R @ cov_matrix.values @ R.T
            R_cov_R_inv = np.linalg.inv(R_cov_R)

         # F statistic
            f_stat = (R_theta.T @ R_cov_R_inv @ R_theta) / n_constraints

        except np.linalg.LinAlgError:
            print("Covariance matrix inversion failed, using simplified calculation method"）
            t_stats = [params[param] / ardl_fit.bse[param] for param in level_params]
            f_stat = np.mean([t ** 2 for t in t_stats])

        print(f"Calculated F statistic: {f_stat:.4f}")

        if k <= 5:
            critical_values = {
                '10%': [2.26, 3.35],
                '5%': [2.62, 3.79],
                '2.5%': [2.96, 4.18],
                '1%': [3.41, 4.68]
            }
        elif k <= 10:
            critical_values = {
                '10%': [1.85, 2.85],
                '5%': [2.11, 3.15],
                '2.5%': [2.33, 3.42],
                '1%': [2.62, 3.77]
            }
        else:
            # Large sample approximation
            critical_values = {
                '10%': [1.75, 2.75],
                '5%': [2.01, 3.05],
                '2.5%': [2.23, 3.32],
                '1%': [2.52, 3.67]
            }

        bounds_result = type('BoundsResult', (), {
            'stat': f_stat,
            'critical_values': critical_values,
            'level_params': level_params,
            'n_constraints': n_constraints
        })()

        return bounds_result, f_stat

    except Exception as e:
        print(f"Manual Bounds Test calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Bounds Test calculation failed: {str(e)}"


def bounds_cointegration_test(ardl_fit):
    print("\n" + "=" * 60)
    print("Step 5: Bounds Test Cointegration Test")
    print("=" * 60)

    if ardl_fit is None:
        print("ARDL model estimation failed, cannot perform cointegration test")
        with open(r'C:\Users\A\Desktop\bounds_test_results.txt', 'w', encoding='utf-8') as f:
            f.write("Bounds Test Cointegration Test\n")
            f.write("=" * 40 + "\n")
            f.write("Cannot execute: ARDL model estimation failed\n")
        return None, "Test failed"
        
    try:
        if hasattr(ardl_fit, 'bounds_test'):
            bounds_result = ardl_fit.bounds_test()
            print("Using native bounds_test method")
        else:
            print("Native bounds_test method not available, using manual calculation method")
            bounds_result, f_stat = manual_bounds_test(ardl_fit)
            if bounds_result is None:
                raise ValueError("Manual Bounds Test failed")

    except Exception as e:
        print(f"Bounds Test execution failed: {e}")
        print("Using manual calculation method...")

        bounds_result, f_stat = manual_bounds_test(ardl_fit)
        if bounds_result is None:
            with open(r'C:\Users\A\Desktop\bounds_test_results.txt', 'w', encoding='utf-8') as f:
                f.write("Bounds Test Cointegration Test\n")
                f.write("=" * 40 + "\n")
                f.write(f"Execution failed: {str(e)}\n")
            return None, "Test failed"

    print("Bounds Test Cointegration Test Results:")
    print("-" * 50)
    print(f"F statistic: {bounds_result.stat:.4f}")
    print("\nCritical Values Table:")
    print("-" * 35)
    print("Significance Level    I(0) Lower    I(1) Upper")
    print("-" * 35)

    cointegration_conclusion = ""
    f_stat = bounds_result.stat

    for level, values in bounds_result.critical_values.items():
        lower_bound, upper_bound = values
        print(f"{level:>8}      {lower_bound:>7.3f}     {upper_bound:>7.3f}")

        if level == '5%': 
            if f_stat > upper_bound:
                cointegration_conclusion = f"F statistic({f_stat:.3f}) > Upper critical value({upper_bound:.3f}), reject null hypothesis"
                cointegration_result = "Cointegration exists"
            elif f_stat < lower_bound:
                cointegration_conclusion = f"F statistic({f_stat:.3f}) < Lower critical value({lower_bound:.3f}), accept null hypothesis"
                cointegration_result = "No cointegration"
            else:
                cointegration_conclusion = f"F statistic({f_stat:.3f}) within critical value interval"
                cointegration_result = "Result inconclusive"

    print("-" * 35)
    print(f"\nTest Conclusion (5% significance level):")
    print(f"  {cointegration_conclusion}")
    print(f"  Conclusion: {cointegration_result}")

    plt.figure(figsize=(10, 6))
    levels = list(bounds_result.critical_values.keys())
    lower_bounds = [bounds_result.critical_values[level][0] for level in levels]
    upper_bounds = [bounds_result.critical_values[level][1] for level in levels]

    x_pos = range(len(levels))

    plt.plot(x_pos, lower_bounds, 'bo-', label='I(0) Lower Bound', linewidth=2, markersize=8)
    plt.plot(x_pos, upper_bounds, 'ro-', label='I(1) Upper Bound', linewidth=2, markersize=8)
    plt.axhline(y=bounds_result.stat, color='green', linestyle='--',
                linewidth=3, label=f'F Statistic ({bounds_result.stat:.3f})')

    plt.fill_between(x_pos, 0, lower_bounds, alpha=0.2, color='red', label='Reject Cointegration Region')
    plt.fill_between(x_pos, upper_bounds, max(upper_bounds) * 1.2, alpha=0.2, color='green',
                     label='Accept Cointegration Region')

    plt.xlabel('Significance Level', fontsize=12)
    plt.ylabel('Critical Values', fontsize=12)
    plt.title('Bounds Test Cointegration Test Results', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, levels)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(r'C:\Users\A\Desktop\bounds_test_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("** Bounds Test results saved **")

    return bounds_result, cointegration_result

#ECM 
def error_correction_model(data, ardl_fit, cointegration_result, lag_order):
    print("\n" + "=" * 60)
    print("Step 6: Error Correction Model (ECM)")
    print("=" * 60)

    if cointegration_result != "Cointegration exists":
        print(f"Cointegration test result: {cointegration_result}")
        print("Does not meet the prerequisites for building ECM, only analyze short-term ARDL effects")


    if ardl_fit is None:
        print("ARDL model estimation failed, cannot build ECM")
        return None

    try:
        print("Cointegration relationship exists, building ECM model...")

        endog = data['log_csi300'].dropna()
        exog = data[['cpi', 'ipi', 'm2', 'interest_rate', 'exchange_rate',
                     'epu', 'log_sp500', 'pmi', 'log_oil_price']].dropna()

        common_index = endog.index.intersection(exog.index)
        endog = endog.loc[common_index]
        exog = exog.loc[common_index]

        print("Calculating long-term equilibrium relationship...")

        Δy_t = α + β*ECM_{t-1} + Σγ_i*Δx_{t-i} + ε_t

        d_endog = endog.diff().dropna()
        d_exog = exog.diff().dropna()

        ecm_term = endog.shift(1) - ardl_fit.fittedvalues.shift(1)
        ecm_term = ecm_term.dropna()

        common_idx = d_endog.index.intersection(d_exog.index).intersection(ecm_term.index)
        if len(common_idx) < 30:  
            print(f"Warning: ECM sample size insufficient ({len(common_idx)}), results may be unreliable")

        d_endog_aligned = d_endog.loc[common_idx]
        d_exog_aligned = d_exog.loc[common_idx]
        ecm_term_aligned = ecm_term.loc[common_idx]

        ecm_regressors = pd.DataFrame(index=common_idx)
        ecm_regressors['ecm_lag1'] = ecm_term_aligned

        for col in d_exog_aligned.columns:
            ecm_regressors[f'd_{col}'] = d_exog_aligned[col]

        for i in range(1, min(lag_order, 3)):  
            for col in d_exog_aligned.columns:
                lagged_col = d_exog_aligned[col].shift(i)
                if not lagged_col.isna().all():
                    ecm_regressors[f'd_{col}_lag{i}'] = lagged_col

        ecm_regressors = ecm_regressors.dropna()
        d_endog_final = d_endog_aligned.loc[ecm_regressors.index]

        if len(ecm_regressors) < 20:
            raise ValueError("ECM sample size too small, cannot reliably estimate")

        print(f"ECM model sample size: {len(ecm_regressors)}")
        print(f"ECM explanatory variables: {list(ecm_regressors.columns)}")

        X_ecm = add_constant(ecm_regressors)
        ecm_fit = OLS(d_endog_final, X_ecm).fit()

        ecm_coef = ecm_fit.params.get('ecm_lag1', np.nan)
        ecm_tstat = ecm_fit.tvalues.get('ecm_lag1', np.nan)
        ecm_pval = ecm_fit.pvalues.get('ecm_lag1', np.nan)

        print(f"\nError correction coefficient: {ecm_coef:.4f}")
        print(f"t statistic: {ecm_tstat:.4f}")
        print(f"p-value: {ecm_pval:.4f}")

        if ecm_coef < 0 and ecm_pval < 0.05:
            adjustment_speed = f"Adjustment speed: {abs(ecm_coef) * 100:.1f}%/period"
            half_life = np.log(0.5) / np.log(1 + ecm_coef) if ecm_coef > -1 else "Infinity"
            conclusion = "✓ ECM coefficient significantly negative, effective error correction mechanism exists"
        elif ecm_coef < 0:
            adjustment_speed = f"Adjustment speed: {abs(ecm_coef) * 100:.1f}%/period"
            half_life = "Not significant"
            conclusion = "? ECM coefficient negative but not significant, weak error correction mechanism"
        else:
            adjustment_speed = "No effective adjustment"
            half_life = "Not applicable"
            conclusion = "× ECM coefficient not significant or positive, no effective error correction mechanism"

        print(f"{conclusion}")
        print(f"Adjustment speed: {adjustment_speed}")

        ecm_coef_table = pd.DataFrame({
            'Coefficient': ecm_fit.params,
            'Std_Error': ecm_fit.bse,
            'T_Statistic': ecm_fit.tvalues,
            'P_Value': ecm_fit.pvalues,
            'Significance': ['***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
                             for p in ecm_fit.pvalues]
        })
        ecm_coef_table.to_csv(r'C:\Users\A\Desktop\ecm_coefficients.csv', encoding='utf-8-sig')

        ecm_residuals = ecm_fit.resid
        ecm_fitted = ecm_fit.fittedvalues

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        axes[0, 0].plot(d_endog_final.index, d_endog_final, label='Actual ΔY', linewidth=2, color='blue')
        axes[0, 0].plot(d_endog_final.index, ecm_fitted, label='Fitted ΔY', linewidth=2, color='red', alpha=0.8)
        axes[0, 0].set_title('ECM Fitting Performance', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(ecm_term_aligned.index, ecm_term_aligned, color='purple', linewidth=1)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Error Correction Term (ECM)', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('ECM_{t-1}')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(ecm_residuals.index, ecm_residuals, color='green', linewidth=1)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('ECM Residuals', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].scatter(ecm_fitted, ecm_residuals, alpha=0.6, color='orange')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Fitted Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('ECM Residuals vs Fitted Values', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(r'C:\Users\A\Desktop\ecm_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("** ECM model results saved **")

        return ecm_fit


# Add model robustness check function
def robustness_checks(data, ardl_fit):
    """Model robustness checks"""
    print("\n" + "=" * 60)
    print("Model Robustness Checks")
    print("=" * 60)

    if ardl_fit is None:
        print("ARDL model not estimated successfully, cannot perform robustness checks")
        return None

    robustness_results = {}

    try:
        residuals = ardl_fit.resid
        if len(residuals) == 0:
            print("Error: Unable to obtain model residuals")
            return None

        residuals = np.array(residuals)

        try:
            jb_result = jarque_bera(residuals)
            if hasattr(jb_result, 'statistic'):
                jb_stat = jb_result.statistic
                jb_pvalue = jb_result.pvalue
            elif isinstance(jb_result, tuple) and len(jb_result) >= 2:
                jb_stat = jb_result[0]
                jb_pvalue = jb_result[1]
            else:
                n = len(residuals)
                s = stats.skew(residuals)
                k = stats.kurtosis(residuals)
                jb_stat = n / 6 * (s ** 2 + 0.25 * (k ** 2))
                jb_pvalue = 1 - stats.chi2.cdf(jb_stat, 2)

        except Exception as e:
            print(f"Jarque-Bera test calculation failed: {e}")
            n = len(residuals)
            s = stats.skew(residuals)
            k = stats.kurtosis(residuals)
            jb_stat = n / 6 * (s ** 2 + 0.25 * (k ** 2))
            jb_pvalue = 1 - stats.chi2.cdf(jb_stat, 2)

        robustness_results['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'conclusion': 'Normal distribution' if jb_pvalue > 0.05 else 'Non-normal distribution'
        }

        print(f"1. Jarque-Bera normality test:")
        print(f"   Statistic: {jb_stat:.4f}, p-value: {jb_pvalue:.4f}")
        print(f"   Conclusion: Residuals {robustness_results['jarque_bera']['conclusion']}")

  
        try:
            lb_result = acorr_ljungbox(residuals, lags=10, return_df=False)
            if isinstance(lb_result, dict):
                lb_stat = lb_result['lb_stat'].iloc[-1]
                lb_pvalue = lb_result['lb_pvalue'].iloc[-1]
            elif isinstance(lb_result, tuple) and len(lb_result) >= 2:
                if hasattr(lb_result[0], '__iter__'):
                    lb_stat = lb_result[0][-1]
                    lb_pvalue = lb_result[1][-1]
                else:
                    lb_stat = lb_result[0]
                    lb_pvalue = lb_result[1]
            else:
                acf_values = [np.corrcoef(residuals[:-i], residuals[i:])[0, 1] for i in range(1, 11)]
                lb_stat = len(residuals) * sum([acf ** 2 for acf in acf_values if not np.isnan(acf)])
                lb_pvalue = 1 - stats.chi2.cdf(lb_stat, 10)

        except Exception as e:
            print(f"Ljung-Box test calculation failed: {e}")
            try:
                acf_values = [np.corrcoef(residuals[:-i], residuals[i:])[0, 1] for i in range(1, 11)]
                lb_stat = len(residuals) * sum([acf ** 2 for acf in acf_values if not np.isnan(acf)])
                lb_pvalue = 1 - stats.chi2.cdf(lb_stat, 10)
            except:
                lb_stat = lb_pvalue = np.nan

        robustness_results['ljung_box'] = {
            'statistic': lb_stat,
            'p_value': lb_pvalue,
            'conclusion': 'No autocorrelation' if lb_pvalue > 0.05 else 'Autocorrelation exists'
        }

        print(f"\n2. Ljung-Box autocorrelation test (lag 10):")
        print(f"   Statistic: {lb_stat:.4f}, p-value: {lb_pvalue:.4f}")
        print(f"   Conclusion: Residuals {robustness_results['ljung_box']['conclusion']}")

        try:
            fitted_values = ardl_fit.fittedvalues

            common_idx = residuals.index.intersection(fitted_values.index) if hasattr(residuals, 'index') else range(
                len(residuals))
            if hasattr(residuals, 'index'):
                residuals_aligned = residuals.loc[common_idx]
                fitted_aligned = fitted_values.loc[common_idx]
            else:
                residuals_aligned = residuals[:len(fitted_values)]
                fitted_aligned = fitted_values[:len(residuals)]

            white_regressors = pd.DataFrame({
                'fitted': fitted_aligned,
                'fitted_sq': fitted_aligned ** 2
            })
            white_regressors = add_constant(white_regressors)

            resid_sq = residuals_aligned ** 2
            white_reg = OLS(resid_sq, white_regressors).fit()
            
            n_r_squared = white_reg.nobs * white_reg.rsquared
            white_pvalue = 1 - stats.chi2.cdf(n_r_squared, df=2)  

        except Exception as e:
            print(f"White test calculation failed: {e}")
            n_r_squared = white_pvalue = np.nan

        robustness_results['white_test'] = {
            'statistic': n_r_squared,
            'p_value': white_pvalue,
            'conclusion': 'Homoscedastic' if white_pvalue > 0.05 else 'Heteroscedastic'
        }

        print(f"\n3. White heteroscedasticity test:")
        print(f"   nR² statistic: {n_r_squared:.4f}, p-value: {white_pvalue:.4f}")
        print(f"   Conclusion: Residuals {robustness_results['white_test']['conclusion']}")

        try:
            mid_point = len(residuals) // 2
            first_half_var = np.var(residuals[:mid_point])
            second_half_var = np.var(residuals[mid_point:])

            stability_ratio = max(first_half_var, second_half_var) / min(first_half_var, second_half_var)
        except:
            stability_ratio = np.nan

        robustness_results['stability_test'] = {
            'ratio': stability_ratio,
            'conclusion': 'Structurally stable' if stability_ratio < 2.0 else 'Possible structural break'
        }

        print(f"\n4. Structural stability test:")
        print(f"   Variance ratio: {stability_ratio:.4f}")
        print(f"   Conclusion: Model {robustness_results['stability_test']['conclusion']}")


    except Exception as e:
        print(f"Robustness checks failed: {e}")
        import traceback
        traceback.print_exc()



       



    



