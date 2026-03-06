import polars as pl
import numpy as np
import math
from typing import List, Optional, Dict, Callable
from .text_tokenizer import Tokenizer as text_tokenizer
import warnings
import time

# Single tokenizer class that handles multiple ways of tokenizing data

# numeric type handling:
#   discrete: distinct token for each level
#   continuous: single token that combines a class and value embedding

# numeric sequence location:
#   factored: numeric quantity is separate from any other class
#   fused: numeric quanity is part of class

# code handling:
#   concept: each item delimited by // is its own token
#   bpe: all items after first part of code are encoded with bpe
 
# there are two stages to any tokenizer:
# 1) training: get the breaks, scaling, bpe vocab
# 2) encoding: return ids and vals for a new data set IF (1) already exists

# Important quantization note if you have LAB//hemoglobin, 12.3 and
# OTHER//hemoblobin, 12.3, discrete fused quantization will lead to <LAB>,
# <OTHER>, hemoglobin_Qx, hemoglobin_Qy which may potentially be different
# because quantiles are computed at the code level


def _scaling_formulas():
    eps = 1e-8
    return {
        "normal": {
            "scale": lambda p, x: (x - p["mean"]) / math.sqrt(max(p["var"], eps)),
            "unscale": lambda p, x: x * math.sqrt(max(p["var"], eps)) + p["mean"]
        },
        "lognormal": {
            "scale": lambda p, x: (math.log(x) - p["mu"]) / math.sqrt(max(p["sigma2"], eps)),
            "unscale": lambda p, x: math.exp(x * math.sqrt(max(p["sigma2"], eps)) + p["mu"])
        },
        "gamma": {
            "scale": lambda p, x: (x - p["alpha"] * p["beta"]) / math.sqrt(max(p["alpha"] * p["beta"]**2, eps)),
            "unscale": lambda p, x: x * math.sqrt(max(p["alpha"] * p["beta"]**2, eps)) + p["alpha"] * p["beta"]
        },
        "minmax": {
            "scale": lambda p, x: (x - p["min"]) / max(p["max"] - p["min"], eps),
            "unscale": lambda p, x: x * max(p["max"] - p["min"], eps) + p["min"]
        }
    }


class Tokenizer:

    def __init__(self,code_mode='concept',num_type='discrete',num_seq='fused',final_vocab_size=4096,split_pattern=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""", bpe_training_sample: int | float |None = None, bpe_training_seed: int | None = 42,bin_clip_min=1.0, bin_clip_max=99.0, continuous_clip_min=1.0, continuous_clip_max=99.0,distributions=None):
        
        if code_mode=='bpe' and num_seq == 'fused':
            raise ValueError("Only 'factored' numeric sequence handling is allowed for 'bpe' code handling")
        
        self.scaling_params = {}
        self.bin_params = {}
        self.ttoi = {}
        self.itot = {}
        self.code_mode = code_mode
        self.num_type = num_type
        self.num_seq = num_seq
        self.tt = None
        self.split_pattern = split_pattern
        self.final_vocab_size = final_vocab_size
        self.scaling_dist = {}
        self.n_bins = 10    
        self.bpe_training_sample = bpe_training_sample
        self.bpe_training_seed = bpe_training_seed
        self.bin_clip_min = bin_clip_min
        self.bin_clip_max = bin_clip_max
        self.continuous_clip_min = continuous_clip_min
        self.continuous_clip_max = continuous_clip_max
        self.distributions = distributions # dict {distribution: [code1, code2,...]}

    def train(self,df):
        # convenience function for tokenizer training
        print("learning encoding for numeric values...")
        if self.num_type == 'continuous':
            if self.distributions is not None:
                for dist,code_list in self.distributions.items():
                    self.get_scaling_params(df,distribution=dist,code_list=code_list,clip_min=self.continuous_clip_min,clip_max=self.continuous_clip_max)
            else:
                self.get_scaling_params(df,clip_min=self.continuous_clip_min,clip_max=self.continuous_clip_max)
            self.get_time_delta_scaling_params(df,clip_min=self.continuous_clip_min,clip_max=self.continuous_clip_max)
            n = len(self.scaling_params)
            print(f"\tunique numeric codes: {n}")
        else:
            self.get_bins(df,n_bins=self.n_bins,clip_min=self.bin_clip_min,clip_max=self.bin_clip_max)
            self.get_time_delta_bins(df,n_bins=self.n_bins,clip_min=self.bin_clip_min,clip_max=self.bin_clip_max)
            n = len(self.bin_params)
            print(f"\tunique numeric codes: {n}")

        print("learning code tokenization...")
        self.get_encoding(df)
        if self.code_mode == 'concept':
            n = len(self.ttoi)
            print(f"\tvocabulary size: {n}")
            self.final_vocab_size = n
        else:
            nst = len(self.tt.special_tokens)
            print(f"\tvocabulary size: {self.final_vocab_size} with {nst} special tokens.")
            

    def encode(self,df):
        # tokenize a dataframe
        last_id = None
        last_time = None
        last_class = None
        ids = []
        vals = []
        i = 0
        if self.code_mode == 'bpe':
            self.build_bpe_cache(df)

        start = time.perf_counter()
        for row in df.iter_rows(named=True):
            if i % 100_000 == 0 and i > 0:
                elapsed = time.perf_counter() - start
                print(f"{i} | {100_000/elapsed:.2f} rps")
                start = time.perf_counter()  # reset for next interval
            # start a new seq if the person is new
            if row['subject_id'] != last_id:
                ids.append(self.ttoi['<sos>'] if self.code_mode == "concept" else self.sos_token)
                if self.num_type == 'continuous':
                    vals.append(np.nan)
                last_id,last_time,last_class = row['subject_id'], None, None
            # if time elapsed, add the time tokens
            # flag this to move to its own function
            if row['time'] is not None and last_time is not None:
                if row['time'] != last_time:
                    dt = (row['time']-last_time).total_seconds()
                    if self.num_seq == 'fused': # there is no way we can be doing bpe if fused.
                        if self.num_type == 'continuous':
                            ids.append(self.ttoi['<TIME>']) 
                            vals.append(self.scale('<TIME>',dt))
                        else:
                            time_q = self.digitize('<TIME>',dt)
                            ids.append(self.ttoi['<TIME>_'+str(time_q)])
                    if self.num_seq == 'factored':
                        if self.num_type == 'continuous':
                            ids.append(self.ttoi['<TIME>'] if self.code_mode == "concept" else self.time_token)
                            ids.append(self.ttoi['<NUM>'] if self.code_mode == "concept" else self.num_token)
                            vals.extend([np.nan,self.scale('<TIME>',dt)])
                        else:
                            ids.append(self.ttoi['<TIME>'] if self.code_mode == "concept" else self.time_token)
                            time_q = self.digitize('<TIME>',dt)
                            ids.append(self.ttoi['Q'+str(time_q)] if self.code_mode == "concept" else self.discrete_tokens[time_q])
                    last_class = '<TIME>'
            if row['time'] is not None:
                last_time = row['time']
            # add the tokens for the actual data row. if the first token is the class we're on, skip it
            n_ids,n_vals = self.tokenize_row(row,i)
            if n_ids[0] == last_class:
                ids.extend(n_ids[1:])
                if vals:
                    vals.extend(n_vals[1:])
            else:
                ids.extend(n_ids)
                if vals:
                    vals.extend(n_vals)
                last_class = n_ids[0]
            i+=1

        return ids,vals
    

    def encode_debug(self, df):
        # just like encode but returns a df to see which tokens come from which row
        if self.code_mode == 'bpe':
            self.build_bpe_cache(df)

        last_id = None
        last_time = None
        last_class = None
        tokens_col = []

        i = 0
        start = time.perf_counter()

        for row in df.iter_rows(named=True):
            if i % 100_000 == 0 and i > 0:
                elapsed = time.perf_counter() - start
                print(f"{i} | {100_000/elapsed:.2f} rps")
                start = time.perf_counter()

            row_tokens = []  # collect all tokens relevant to *this* row

            # new sequence (new patient)
            if row['subject_id'] != last_id:
                sos_id = self.ttoi['<sos>'] if self.code_mode == "concept" else self.sos_token
                if self.code_mode == 'bpe':
                    row_tokens.append(self.tt.decode([sos_id]))
                else:
                    row_tokens.append(self.itot[sos_id])
                last_id, last_time, last_class = row['subject_id'], None, None

            # add time tokens if needed
            if row['time'] is not None and last_time is not None and row['time'] != last_time:
                dt = (row['time'] - last_time).total_seconds()
                if self.num_seq == 'fused':
                    if self.num_type == 'continuous':
                        row_tokens.append('<TIME>')
                    else:
                        time_q = self.digitize('<TIME>', dt)
                        row_tokens.append(f'<TIME>_{time_q}')
                elif self.num_seq == 'factored':
                    if self.num_type == 'continuous':
                        row_tokens.extend(['<TIME>', '<NUM>'])
                    else:
                        time_q = self.digitize('<TIME>', dt)
                        row_tokens.extend(['<TIME>', f'Q{time_q}'])
                last_time = row['time']

            # add actual row tokens
            n_ids, n_vals = self.tokenize_row(row, i)
            if n_ids[0] == last_class:
                n_ids = n_ids[1:]
            else:
                last_class = n_ids[0]

            if self.code_mode == 'bpe':
                decoded = [self.tt.decode([n]) for n in n_ids]
            else:
                decoded = [self.itot[iid] for iid in n_ids]

            row_tokens.extend(decoded)

            # append exactly one entry per row
            tokens_col.append(row_tokens)
            i += 1

        # Ensure length matches DataFrame
        assert len(tokens_col) == len(df), f"Token column length {len(tokens_col)} != DataFrame height {len(df)}"

        df = df.with_columns(pl.Series("tokens", tokens_col))
        return df


            

    def tokenize_row(self,row,row_idx):
        # right now this only tokenizes the code and the numeric value
        code = row['code']
        numeric_value = row['numeric_value']
        parts = code.split('//')

        if self.code_mode == 'concept':
            if numeric_value is not None and not np.isnan(numeric_value):
                if self.num_seq == 'fused':
                    if self.num_type == 'continuous':
                        vals = np.full(len(parts), np.nan)
                        vals[-1] = self.scale(code,numeric_value)
                    elif self.num_type == 'discrete':
                        parts[-1] = parts[-1]+"_"+str(self.digitize(code,numeric_value))
                        vals = None
                elif self.num_seq == 'factored':
                    if self.num_type == 'continuous':
                        vals = np.full(len(parts)+1, np.nan)
                        vals[-1] = self.scale(code,numeric_value)
                        parts.append('<NUM>')
                    else:
                        parts.append("Q"+str(self.digitize(code,numeric_value)))
                        vals = None
            else:
                if self.num_type == 'continuous':
                    vals = np.full(len(parts), np.nan)
                else:
                    vals = None
            parts[0]='<'+parts[0]+'>' # special tag to mark data types
            ids = [self.ttoi[p] for p in parts]

        elif self.code_mode == 'bpe':
            # parts[0]='<'+parts[0]+'>' # special tag to mark data types
            # ids = [idx for p in parts for idx in self.tt.encode(p)]

            # parts[0] = '<' + parts[0] + '>'
            # joined = "//".join(parts)
            # encoded = self.tt.encode(joined)
            # ids = [i for i in encoded if i != sep_idx]
            ids = self.bpe_cache[row_idx]  # pre-encoded tokens for this row


            if numeric_value is not None and not np.isnan(numeric_value):
                if self.num_type == 'continuous':
                    # vals = np.full(len(ids)+1, np.nan)
                    vals = [np.nan] * (len(ids) + 1)
                    vals[-1] = self.scale(code,numeric_value)
                    ids.append(self.num_token)
                else:
                    # t = "Q"+str(self.digitize(code,numeric_value))
                    ids.append(self.discrete_tokens[self.digitize(code,numeric_value)])
                    # ids.append(self.tt.encode(t))
                    vals = None
            else:
                if self.num_type == 'continuous':
                    # vals = np.full(len(ids), np.nan)
                    vals = [np.nan] * len(ids)

                else:
                    vals = None
        return ids,vals

    # ---------- multivariate tokenization stuff ---------- #
    def get_scaling_params( self, df, distribution = "normal", code_list = None,clip_min=1.0,clip_max=99.0):
        # compute scaling params per code. default is normal distribution and
        # applies to all codes if code_list is None by default performs
        # winsorization at 1% and 99% levels (i.e. values more extreme than
        # clip_min and clip_max percentiles are clipped to those levels)
        param_funcs = {
            "normal": lambda col: {"mean": col.mean(), "var": col.var()},
            "lognormal": lambda col: {"mu": col.log().mean(), "sigma2": col.log().var()},
            "gamma": lambda col: {
                "alpha": (col.mean() ** 2) / col.var(),
                "beta": col.var() / col.mean()
            },
            "minmax": lambda col: {"min": col.min(), "max": col.max()}
        }

        if distribution not in param_funcs:
            raise ValueError(f"Unsupported distribution: {distribution}")

        filtered = df.filter(pl.col("numeric_value").is_not_null() & (~pl.col("numeric_value").is_nan()))
        if code_list is not None:
            filtered = filtered.filter(pl.col("code").is_in(code_list))

        p_min = clip_min / 100.0
        p_max = clip_max / 100.0
        
        thresholds = (
            filtered
            .group_by("code")
            .agg([
                pl.col("numeric_value").quantile(p_min).alias("min_threshold"),
                pl.col("numeric_value").quantile(p_max).alias("max_threshold"),
            ])
        )
        threshold_dict = {row["code"]: (row["min_threshold"], row["max_threshold"]) 
                          for row in thresholds.iter_rows(named=True)}

        if clip_min >0.0 or clip_max <100.0:
            if clip_min > 0.0 or clip_max < 100.0:
                p_min = clip_min / 100.0
                p_max = clip_max / 100.0
                
                filtered = filtered.with_columns(
                    pl.col("numeric_value")
                    .quantile(p_min).over("code").alias("lower_bound"),
                    pl.col("numeric_value")
                    .quantile(p_max).over("code").alias("upper_bound")
                ).with_columns(
                    pl.col("numeric_value")
                    .clip(pl.col("lower_bound"), pl.col("upper_bound"))
                    .alias("numeric_value")
                ).drop(["lower_bound", "upper_bound"])

        exprs = param_funcs[distribution](pl.col("numeric_value"))
        stats = filtered.group_by("code").agg([v.alias(k) for k, v in exprs.items()])

        for row in stats.iter_rows(named=True):
            # Replace None or NaN with 1 for variance-like fields
            safe_values = {}
            for k in exprs:
                val = row[k]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 1.0
                    warnings.warn(
                        f"Code '{row['code']}' generated a distribution fit with a null parameter; "
                        f"defaulting '{k}' to 1.0 for {distribution} distribution."
                        f"this is likely caused by a single unique numeric value for the code"
                    )
                safe_values[k] = val

            min_thresh, max_thresh = threshold_dict.get(row["code"], (None, None))
            self.scaling_params[row["code"]] = {
                "distribution": distribution,
                "min_threshold": min_thresh,
                "max_threshold": max_thresh,
                **safe_values
            }
         
    def get_time_delta_scaling_params(self,df,distribution="gamma",clip_min=1.0,clip_max=99.0):
        param_funcs = {
            "normal": lambda col: {"mean": col.mean(), "var": col.var()},
            "lognormal": lambda col: {"mu": col.log().mean(), "sigma2": col.log().var()},
            "gamma": lambda col: {
                "alpha": (col.mean() ** 2) / col.var(),
                "beta": col.var() / col.mean(),
            },
        }

        if distribution not in param_funcs:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        with_deltas = (
            df
            .with_columns(((pl.col("time") - pl.col("time").shift(1)).over("subject_id")).dt.total_seconds().alias("time_delta"))
            .filter(pl.col("time_delta").is_not_null() & (pl.col("time_delta") > 0))
        )

        p_min = clip_min / 100.0
        p_max = clip_max / 100.0
        min_threshold = float(with_deltas["time_delta"].quantile(p_min))
        max_threshold = float(with_deltas["time_delta"].quantile(p_max))

        # Apply winsorization if needed
        if clip_min > 0.0 or clip_max < 100.0:
            p_min = clip_min / 100.0
            p_max = clip_max / 100.0
            
            with_deltas = with_deltas.with_columns(
                pl.col("time_delta").quantile(p_min).alias("lower_bound"),
                pl.col("time_delta").quantile(p_max).alias("upper_bound")
            ).with_columns(
                pl.col("time_delta")
                .clip(pl.col("lower_bound"), pl.col("upper_bound"))
                .alias("time_delta")
            ).drop(["lower_bound", "upper_bound"])

        exprs = param_funcs[distribution](pl.col("time_delta"))
        stats = with_deltas.select([v.alias(k) for k, v in exprs.items()]).row(0)
        self.scaling_params["<TIME>"] = {
            "distribution": distribution,
            "min_threshold": min_threshold,
            "max_threshold": max_threshold,
            **{k: stats[i] for i, k in enumerate(exprs.keys())},
        }
        
    def scale(self, code, x):
        if code not in self.scaling_params:
            return x
        p = self.scaling_params[code]
     
        if p.get("min_threshold") is not None and p.get("max_threshold") is not None:
            x = max(p["min_threshold"], min(x, p["max_threshold"]))
        
        formulas = _scaling_formulas().get(p["distribution"])
        if formulas is None:
            return x
        return formulas["scale"](p, x)

    def unscale(self, code, x_scaled):
        if code not in self.scaling_params:
            return x_scaled
        p = self.scaling_params[code]
        formulas = _scaling_formulas().get(p["distribution"])
        if formulas is None:
            return x_scaled
        return formulas["unscale"](p, x_scaled)
    
    #-------------- discrete tokenization stuff ---------------#
    def get_bins(self,df: pl.DataFrame,n_bins: int = 5,clip_min: float = 1.0,clip_max: float = 99.0):
        """
        Compute quantile-based bins per code within [clip_min, clip_max].
        Stores results in self.bin_params as:
          code -> {edges: list of bin edges}
        """
        filtered = df.filter(pl.col("numeric_value").is_not_null() & (~pl.col("numeric_value").is_nan()))
        p_edges = list(np.linspace(clip_min/100,clip_max/100,n_bins-1))

        for code in filtered["code"].unique().to_list():
            col_vals = filtered.filter(pl.col("code") == code)["numeric_value"].to_list()
            if not col_vals:
                continue  # skip codes with no values
            edges = [float(pl.Series(col_vals).quantile(p)) for p in p_edges]
            self.bin_params[code] = {"edges": edges}

    def get_time_delta_bins(self,df: pl.DataFrame,n_bins: int = 5,clip_min: float = 1.0,clip_max: float = 99.0):

        with_deltas = (
            df
            .with_columns(
                ((pl.col("time") - pl.col("time").shift(1))
                .over("subject_id")).dt.total_seconds()
                .alias("time_delta")
            )
            .filter(pl.col("time_delta").is_not_null() & (pl.col("time_delta") > 0))
        )

        deltas = with_deltas["time_delta"]
        if deltas.is_empty():
            raise ValueError("No time deltas found to compute bins.")
        p_edges = list(np.linspace(clip_min/100,clip_max/100,n_bins-1))
        # Compute quantile-based bin edges
        edges = [float(deltas.quantile(p)) for p in p_edges]
        self.bin_params["<TIME>"] = {"edges": edges}

    def digitize(self, code: str, x: float) -> Optional[int]:
        """Map a numeric value to its bin index for a given code."""
        if code not in self.bin_params:
            return None
        edges = self.bin_params[code]["edges"]
        return np.digitize(x,edges,right=False)

    def dedigitize(self, code: str, idx: int) -> Optional[float]:
        """Map a bin index back to a representative value, consistent with np.digitize."""
        if code not in self.bin_params:
            return None
        edges = self.bin_params[code]["edges"]
        n_bins = len(edges) - 1

        if idx == 0:
            return edges[0]  # below first bin
        elif idx >= n_bins:
            return edges[-1]  # above last bin
        else:
            return (edges[idx - 1] + edges[idx]) / 2  # bin center


    #--------- BPE Stuff ---------------------#
    def get_encoding(self,df):
        # df = df.with_columns(pl.col("code").str.replace(r"^([^/]+)", r"<$1>"))
        df = tag_first(df)
        df = df.with_columns(pl.col("code").str.split("//").alias("code_split"))

        if self.code_mode == "concept":
            if self.num_seq == 'fused':
                if self.num_type == 'continuous':
                    exploded_parts = df.select(pl.col("code_split").explode()).to_series().to_list()
                    unique_parts = list(set(exploded_parts))
                    unique_parts.append('<sos>')
                    unique_parts.append('<TIME>')
                else:
                    # last part of code has fused value.
                    # get all concepts from code split except these, then add levels from bin_param keys
                    exploded_parts = get_concept_parts_fused(df)
                    unique_parts = list(set(exploded_parts))
                    unique_parts.extend(['<sos>','<TIME>'])
                    num_tokens = [k.split("//")[-1] for k in self.bin_params.keys()]
                    num_tokens = list(set(num_tokens))
                    unique_parts.extend([f"{n}_{i}" for n in num_tokens for i in range(self.n_bins)])
            elif self.num_seq =='factored':
                exploded_parts = df.select(pl.col("code_split").explode()).to_series().to_list()
                unique_parts = list(set(exploded_parts))
                unique_parts.append('<sos>')
                if self.num_type == 'continuous':
                    unique_parts.extend(['<NUM>','<TIME>'])
                else:
                    unique_parts.extend([f"Q{i}" for i in range(self.n_bins)])
                    unique_parts.append('<TIME>')
            self.ttoi = {t:i for i,t in enumerate(unique_parts)}
            self.itot = {i:t for i,t in enumerate(unique_parts)}

        elif self.code_mode == "bpe":
            
            # Determine target row count
            target_rows = df.height
            if isinstance(self.bpe_training_sample, float):
                # If float between 0 and 1, treat as percentage
                if 0.0 < self.bpe_training_sample <= 1.0:
                    target_rows = int(df.height * self.bpe_training_sample)
                else:
                    raise ValueError("Float value for bpe_training_sample must be between 0.0 and 1.0")
            elif isinstance(self.bpe_training_sample, int):
                target_rows = self.bpe_training_sample

            # downsample if needed for learning BPE
            if self.bpe_training_sample is not None and df.height > target_rows:
                subject_counts = df.group_by("subject_id").len()
                subject_counts = subject_counts.sort("subject_id")
                subject_counts = subject_counts.sample(
                    fraction=1.0, 
                    shuffle=True, 
                    seed=self.bpe_training_seed
                )
                
                subject_counts = subject_counts.with_columns(
                    pl.col("len").cum_sum().alias("cum_rows")
                )
                
                selected_subjects = subject_counts.filter(
                    pl.col("cum_rows") <= target_rows
                ).select("subject_id")

                df_bpe = df.join(selected_subjects, on="subject_id", how="inner")
                print(f'Training on {df_bpe.height} out of {df.height} total rows')
            
            else:
                df_bpe = df
                
            exploded_parts = df_bpe.select(pl.col("code_split").explode()).to_series().to_list()
            first_parts = df_bpe["code_split"].list.first().alias("first_item")
            st = list(set(first_parts))
            st.extend(["//", "<sos>", "<TIME>", "<NUM>","<ROW>"])
            st.extend([f"Q{i}" for i in range(self.n_bins + 1)])
            self.tt = text_tokenizer(st,self.split_pattern,final_vocab_size=self.final_vocab_size)
            text = "//".join(exploded_parts)
            self.tt.train_bpe(text)
            # save off some fixed token ids for reference later
            self.discrete_tokens = tuple(self.tt.encode(f"Q{i}")[0] for i in range(self.n_bins))
            self.time_token = self.tt.encode('<TIME>')[0]
            self.num_token = self.tt.encode('<NUM>')[0]
            self.sos_token = self.tt.encode('<sos>')[0]


    def build_bpe_cache(self, df):
        """
        Pre-encode the full dataframe using BPE and build a per-row cache.
        Each row is joined by "//", and rows are separated by "<ROW>".
        Separator "//" tokens are removed from each row.
        Result: self.bpe_cache[row_idx] is a list of token IDs for that row.
        """
        df = tag_first(df)
        df = df.with_columns(pl.col("code").str.split("//").alias("code_split"))

        # Join each row by "//" and append row separator
        ROW_SEP = "<ROW>"
        row_texts = ["//".join(row) + ROW_SEP for row in df["code_split"].to_list()]
        full_text = "".join(row_texts)

        # Encode the full text once
        full_encoded = self.tt.encode(full_text)

        # Get token IDs for separators
        row_sep_idx = self.tt.encode(ROW_SEP)[0]   # single-token row separator
        part_sep_idx = self.tt.encode("//")[0]     # single-token part separator

        # Split encoded list into per-row lists, removing part separators
        self.bpe_cache = []
        start = 0
        for i, token in enumerate(full_encoded):
            if token == row_sep_idx:
                row_ids = [t for t in full_encoded[start:i] if t != part_sep_idx]
                self.bpe_cache.append(row_ids)
                start = i + 1

        # Add last row (if any)
        if start < len(full_encoded):
            row_ids = [t for t in full_encoded[start:] if t != part_sep_idx]
            self.bpe_cache.append(row_ids)




def tag_first(df):
    # tag the first entry in code with <this>
    # LAB//xxxx -> <LAB>//xxxx
    df = df.with_columns(
        code = pl.format(
            "<{}>{}",
            pl.col("code").str.slice(
                0, 
                pl.col("code").str.find("//").fill_null(pl.col("code").str.len_bytes())
            ),
            pl.col("code").str.slice(
                pl.col("code").str.find("//").fill_null(pl.col("code").str.len_bytes()),
                pl.col("code").str.len_bytes() - pl.col("code").str.find("//").fill_null(pl.col("code").str.len_bytes())
            )
        )
    )
    return df


def get_concept_parts_fused(df):
    # Exploded parts where numeric_value is null
    exploded_null = (
        df.filter(pl.col("numeric_value").is_null())
        .select(pl.col("code_split").explode())
        .to_series()
        .to_list()
    )

    # Exploded parts where numeric_value is not null, except the last part
    exploded_notnull_except_last = (
        df.filter(pl.col("numeric_value").is_not_null())
        .select(pl.col("code_split").list.slice(0, -1).explode())
        .to_series()
        .to_list()
    )

    # Combine
    exploded_parts = exploded_null + exploded_notnull_except_last
    return exploded_parts


