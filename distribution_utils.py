import polars as pl
import numpy as np
import math
from typing import Literal, Dict, List
from scipy import stats

def analyze_distributions(
    df: pl.DataFrame,
    value_col: str = "numeric_value",
    group_col: str = "code",
    clip_min: float = 1.0,
    clip_max: float = 99.0,
    percentiles: List[float] = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
) -> pl.DataFrame:
    """
    Analyze which distribution (normal, lognormal, gamma, minmax) best normalizes each group.
    
    Uses normalization quality score: compares transformed percentiles to theoretical
    normal z-scores. Lower score = better normalization.
    
    Parameters:
    -----------
    df : pl.DataFrame
        Input dataframe
    value_col : str
        Column name containing numeric values
    group_col : str
        Column name to group by
    clip_min, clip_max : float
        Percentile bounds for winsorization (1-99 recommended)
    top_n : int, optional
        Only analyze top N groups by sample size
    percentiles : List[float]
        Percentiles to evaluate (default: [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9])
        
    Returns:
    --------
    pl.DataFrame with columns for each distribution's parameters, 
    transformed percentile values, normalization scores, and recommendation
    """
    
    # Get theoretical z-scores for standard normal at each percentile
    theoretical_z = {p: stats.norm.ppf(p/100) for p in percentiles}
    
    # Helper functions for scaling
    def get_param_funcs():
        return {
            "normal": lambda vals: {"mean": np.mean(vals), "var": np.var(vals)},
            "lognormal": lambda vals: {
                "mu": np.mean(np.log(vals[vals > 0])) if (vals > 0).any() else 0,
                "sigma2": np.var(np.log(vals[vals > 0])) if (vals > 0).any() else 1
            },
            "gamma": lambda vals: {
                "alpha": (np.mean(vals) ** 2) / np.var(vals) if np.var(vals) > 0 else 1,
                "beta": np.var(vals) / np.mean(vals) if np.mean(vals) > 0 else 1
            },
            "minmax": lambda vals: {"min": np.min(vals), "max": np.max(vals)}
        }
    
    def get_scaling_funcs():
        eps = 1e-8
        return {
            "normal": lambda p, x: (x - p["mean"]) / math.sqrt(max(p["var"], eps)),
            "lognormal": lambda p, x: (math.log(x) - p["mu"]) / math.sqrt(max(p["sigma2"], eps)) if x > 0 else None,
            "gamma": lambda p, x: (x - p["alpha"] * p["beta"]) / math.sqrt(max(p["alpha"] * p["beta"]**2, eps)),
            "minmax": lambda p, x: (x - p["min"]) / max(p["max"] - p["min"], eps)
        }
    
    def calculate_normalization_score(scaled_values, percentiles_list):
        """
        Calculate how well the scaled values match standard normal z-scores.
        Returns RMSE - lower is better.
        """
        errors = []
        for i, p in enumerate(percentiles_list):
            if scaled_values[i] is not None:
                expected_z = theoretical_z[p]
                actual_z = scaled_values[i]
                errors.append((actual_z - expected_z) ** 2)
        
        if not errors:
            return float('inf')
        
        return math.sqrt(np.mean(errors))
    
    param_funcs = get_param_funcs()
    scaling_funcs = get_scaling_funcs()
    
    # Winsorize within groups
    p_min = clip_min / 100.0
    p_max = clip_max / 100.0
    
    df_clean = (
        df
        .filter(pl.col(value_col).is_not_null())
        .with_columns(
            pl.col(value_col).quantile(p_min).over(group_col).alias("lower_bound"),
            pl.col(value_col).quantile(p_max).over(group_col).alias("upper_bound")
        )
        .with_columns(
            pl.col(value_col)
            .clip(pl.col("lower_bound"), pl.col("upper_bound"))
            .alias(value_col)
        )
        .drop(["lower_bound", "upper_bound"])
    )
    
    # Get basic stats per group
    group_stats = (
        df_clean
        .group_by(group_col)
        .agg([
            pl.col(value_col).count().alias("count"),
            pl.col(value_col).mean().alias("mean"),
            pl.col(value_col).std().alias("std"),
            pl.col(value_col).min().alias("min_val"),
            pl.col(value_col).max().alias("max_val"),
            # Approximate skewness
            (
                (pl.col(value_col) - pl.col(value_col).mean()).pow(3).mean() / 
                pl.col(value_col).std().pow(3)
            ).alias("skewness"),
            # Get percentiles for later
            *[pl.col(value_col).quantile(p/100).alias(f"p{p}") for p in percentiles]
        ])
        .filter(pl.col("count") >= 30)  # Need enough samples for fitting
        .sort("count", descending=True)
    )
    
    
    # Fit distributions for each group
    results = []
    
    for row in group_stats.iter_rows(named=True):
        code = row[group_col]
        
        # Get values for this group
        values = (
            df_clean
            .filter(pl.col(group_col) == code)
            .select(value_col)
            .to_series()
            .to_numpy()
        )
        
        has_negatives = (values < 0).any()
        range_val = row['max_val'] - row['min_val']
        
        # Calculate parameters for each distribution
        params = {}
        for dist_name, param_func in param_funcs.items():
            try:
                if dist_name in ["lognormal", "gamma"] and (has_negatives or not (values > 0).all()):
                    params[dist_name] = None
                else:
                    params[dist_name] = param_func(values)
            except:
                params[dist_name] = None
        
        # Get percentile values
        pct_values = {f"p{p}": row[f"p{p}"] for p in percentiles}
        
        # Transform percentiles with each distribution and calculate normalization scores
        transformed = {dist: {} for dist in ["normal", "lognormal", "gamma", "minmax"]}
        norm_scores = {}
        
        for dist_name in ["normal", "lognormal", "gamma", "minmax"]:
            if params[dist_name] is not None:
                scale_func = scaling_funcs[dist_name]
                scaled_values = []
                
                for p in percentiles:
                    try:
                        val = pct_values[f"p{p}"]
                        scaled = scale_func(params[dist_name], val)
                        transformed[dist_name][f"p{p}_scaled"] = round(scaled, 3) if scaled is not None else None
                        scaled_values.append(scaled)
                    except:
                        transformed[dist_name][f"p{p}_scaled"] = None
                        scaled_values.append(None)
                
                # Calculate normalization score
                norm_scores[dist_name] = calculate_normalization_score(scaled_values, percentiles)
            else:
                for p in percentiles:
                    transformed[dist_name][f"p{p}_scaled"] = None
                norm_scores[dist_name] = float('inf')
        
        # Determine best distribution
        # Filter out invalid scores
        valid_scores = {k: v for k, v in norm_scores.items() if v != float('inf')}
        
        if not valid_scores:
            best_dist = 'normal'  # Fallback
        else:
            # Additional heuristics
            is_bounded = range_val <= 100 and row['max_val'] <= 100 and row['min_val'] >= 0
            is_clustered_high = is_bounded and pct_values['p90'] > (row['min_val'] + 0.8 * range_val)
            
            if is_clustered_high and norm_scores['minmax'] < 2.0:
                # Bounded data clustered near max (e.g., SpO2) - prefer minmax if reasonable
                best_dist = 'minmax'
            elif has_negatives:
                # Must use normal
                best_dist = 'normal'
            else:
                # Choose distribution with best normalization score
                best_dist = min(valid_scores, key=valid_scores.get)
        
        # Build result row
        result = {
            group_col: code,
            'count': row['count'],
            'recommendation': best_dist,
            'skewness': round(row['skewness'], 4),
            'range': round(range_val, 2),
            'min_val': round(row['min_val'], 2),
            'max_val': round(row['max_val'], 2),
            'has_negatives': has_negatives,
            'is_clustered_high': is_clustered_high
        }
        
        # Add normalization scores
        for dist_name in ["normal", "lognormal", "gamma", "minmax"]:
            score = norm_scores[dist_name]
            result[f"{dist_name}_norm_score"] = round(score, 4) if score != float('inf') else None
        
        # Add parameters for each distribution
        for dist_name in ["normal", "lognormal", "gamma", "minmax"]:
            if params[dist_name] is not None:
                for param_name, param_val in params[dist_name].items():
                    result[f"{dist_name}_{param_name}"] = round(param_val, 4)
        
        # Add key transformed percentiles (not all to keep output manageable)
        key_percentiles = [1, 10, 50, 90, 99]
        for dist_name in ["normal", "lognormal", "gamma", "minmax"]:
            for p in key_percentiles:
                col_name = f"{dist_name}_p{p}"
                result[col_name] = transformed[dist_name].get(f"p{p}_scaled")
        
        # Add theoretical z-scores for reference
        for p in key_percentiles:
            result[f"theory_z_p{p}"] = round(theoretical_z[p], 3)
        
        results.append(result)
    
    return pl.DataFrame(results).sort("count", descending=True)


def get_distribution_summary(analysis_df: pl.DataFrame) -> Dict:
    """Get summary statistics from distribution analysis."""
    total = len(analysis_df)
    
    dist_counts = (
        analysis_df
        .group_by("recommendation")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
    )
    
    return {
        "total_groups": total,
        "distribution_counts": dist_counts.to_dicts(),
        "percent_negative": (analysis_df["has_negatives"].sum() / total * 100),
        "median_skewness": analysis_df["skewness"].median(),
        "avg_best_norm_score": analysis_df.select([
            pl.when(pl.col("recommendation") == "normal").then(pl.col("normal_norm_score"))
            .when(pl.col("recommendation") == "lognormal").then(pl.col("lognormal_norm_score"))
            .when(pl.col("recommendation") == "gamma").then(pl.col("gamma_norm_score"))
            .when(pl.col("recommendation") == "minmax").then(pl.col("minmax_norm_score"))
            .alias("best_score")
        ]).get_column("best_score").mean(),
    }