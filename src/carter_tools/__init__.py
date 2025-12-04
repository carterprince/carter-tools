import glob
import numpy as np
import os
import pandas as pd
import plotly.express as px
import re
import shlex
import subprocess
import sys

from collections import Counter
from typing import List, Union

_FUNCTION_WORDS = [
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "he", 
    "for", "it", "with", "as", "his", "on", "be", "at", "by", "i", 
    "this", "had", "not", "are", "but", "from", "or", "have", "an", 
    "they", "which", "one", "you", "were", "her", "all", "she", 
    "there", "would", "their", "we", "him", "been", "has", "when", 
    "who", "will", "more", "no", "if", "out", "so", "said", "what", 
    "up", "its", "about", "into", "than", "them", "can", "only", 
    "other", "new", "some", "could", "time", "these", "two", "may", 
    "then", "do", "first", "any", "my", "now", "such", "like", 
    "our", "over", "man", "me", "even", "most", "made", "after", "upon"
]

_PUNCTUATION_PATTERNS = {
    'period': r'\.', 'comma': r',', 'semicolon': r';', 'colon': r':',
    'exclamation': r'!', 'question': r'\?', 'hyphen': r'-',
    'em_dash': r'—|--', 'open_paren': r'\(', 'close_paren': r'\)',
    'digits': r'\d+'
}

def get_style_features(text):
    """
    Extracts stylometric features (function words, punctuation, sentence length)
    from a string. Returns a dictionary suitable for a Pandas DataFrame.
    """
    if not text or len(text.strip()) == 0:
        return None

    words = re.findall(r"\b[\w']+\b", text.lower())
    total_tokens = len(words)
    
    if total_tokens == 0:
        return None

    features = {}

    # Lexical Features
    counts = Counter(words)
    for word in _FUNCTION_WORDS:
        features[f'fw_{word}'] = (counts[word] / total_tokens) * 1000

    # Symbol Features
    for name, pattern in _PUNCTUATION_PATTERNS.items():
        count = len(re.findall(pattern, text))
        features[f'punct_{name}'] = (count / total_tokens) * 1000

    # Syntax / Structure
    word_lengths = [len(w) for w in words]
    features['avg_word_len'] = np.mean(word_lengths) if word_lengths else 0
    
    sentences = [s for s in re.split(r'[.!?]+', text) if len(s.strip()) > 0]
    sent_lens = [len(re.findall(r"\b[\w']+\b", s)) for s in sentences]
    
    features['avg_sent_len'] = np.mean(sent_lens) if sent_lens else 0
    features['std_sent_len'] = np.std(sent_lens) if sent_lens else 0
    
    return features


def print_colored_df(df, colorscale="RdBu", flip=False):
    """
    Manually prints a DataFrame to the console using ANSI escape codes.
    """
    import plotly.colors as pcolors
    
    # 1. Calculate Min/Max for numeric columns
    col_stats = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            col_stats[col] = (c_min, c_max)

    # 2. Calculate Column Widths
    col_widths = {}
    for col in df.columns:
        max_len = len(str(col))
        for val in df[col]:
            # Check if column is numeric (in stats) to decide formatting length
            if col in col_stats:
                val_len = len(f"{val:.3f}")
            else:
                val_len = len(str(val))
            
            if val_len > max_len:
                max_len = val_len
        col_widths[col] = max(max_len, 8)

    def get_ansi_color(value, c_min, c_max):
        if pd.isna(value) or c_min == c_max:
            return "" 

        try:
            t = (value - c_min) / (c_max - c_min)
            if flip:
                t = 1.0 - t
            # Ensure t is a standard float for Plotly
            color_str = pcolors.sample_colorscale(colorscale, [float(t)])[0]
        except Exception:
            return ""

        matches = re.findall(r'\d+', color_str)
        if len(matches) >= 3:
            r, g, b = map(int, matches[:3])
            return f"\033[38;2;{r};{g};{b}m"
        
        return ""

    RESET = "\033[0m"
    index_width = max([len(str(i)) for i in df.index] + [0])

    # Print Header
    print(f"{' ' * index_width} ", end="")
    for col in df.columns:
        print(f" {str(col).center(col_widths[col])} ", end="")
    print()

    # Print Rows
    for index, row in df.iterrows():
        print(f"{str(index).ljust(index_width)} ", end="")
        for col in df.columns:
            val = row[col]
            
            # Simple check: if we calculated stats for this column, treat it as a number
            if col in col_stats:
                c_min, c_max = col_stats[col]
                color = get_ansi_color(val, c_min, c_max)
                formatted_val = f"{val:.3f}"
            else:
                color = ""
                formatted_val = str(val)
                
            print(f" {color}{formatted_val.rjust(col_widths[col])}{RESET} ", end="")
        print()


def geomaps_to_df(path=".", pattern="geoMap *.csv", columns=None):
    """
    Reads multiple Google Trends CSVs matching a pattern and merges them.
    Expects Google Trends 'DMA' export format.
    """
    search_path = os.path.join(path, pattern)
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"No files found matching: {search_path}")
        return pd.DataFrame()

    all_dfs = []
    for f in files:
        df = pd.read_csv(f, skiprows=2)
        if len(df.columns) < 2: continue
        
        raw_col = df.columns[1]
        new_name = '_'.join([re.sub(r':\s\(.*?\)', '', raw_col).strip()])
        
        processed = df[['DMA', raw_col]].set_index('DMA')
        processed.columns = [new_name]
        
        processed[new_name] = pd.to_numeric(
            processed[new_name].astype(str).str.replace('%', '').str.replace('<1', '0.5'), 
            errors='coerce'
        )
        all_dfs.append(processed)

    if not all_dfs:
        return pd.DataFrame()

    result_df = pd.concat(all_dfs, axis=1).dropna().astype(int).reset_index()

    if columns:
        keep_cols = columns if "DMA" in columns else ["DMA"] + columns
        existing_cols = [c for c in keep_cols if c in result_df.columns]
        return result_df[existing_cols]
    else:
        return result_df


def usa_dmas_choropleth(df, color, colorscale="RdBu", **kwargs):
    """
    Generates a Plotly Choropleth map for US DMAs.
    Requires a 'DMA' column in the dataframe.
    """
    if 'DMA' not in df.columns:
        raise ValueError("DataFrame must contain a 'DMA' column.")

    fig_map = px.choropleth(
        df,
        geojson="https://raw.githubusercontent.com/carterprince/maps/refs/heads/main/dma-google-trends.json",
        locations='DMA',
        featureidkey="properties.dma1",
        color=color,
        hover_name='DMA',
        projection="orthographic",
        color_continuous_scale=colorscale,
        **kwargs
    )
    
    fig_map.update_geos(
        showland=False, 
        showcoastlines=False, 
        showlakes=False, 
        projection_rotation={"lat": 40, "lon": -98, "roll": 0}, 
        projection_scale=2.5
    )
    
    return fig_map

def view_images(paths: Union[str, List[str]], log: bool = False):
    """
    Opens images using the system default viewer.
    
    On Linux, it parses the default .desktop file to determine if the 
    viewer supports multiple files (slideshow) or just one.
    
    Args:
        paths: A single file path or a list of file paths.
        log: If True, prints status messages to stdout.
    """
    # 1. Normalize input
    if isinstance(paths, str):
        paths = [paths]
    
    valid_paths = [os.path.abspath(p) for p in paths if os.path.exists(p)]
    
    if not valid_paths:
        if log:
            print("Error: No valid file paths found.")
        return

    # --- LINUX LOGIC ---
    if sys.platform.startswith("linux"):
        try:
            # A. Ask the OS for the default app ID
            app_id = subprocess.check_output(
                ["xdg-mime", "query", "default", "image/png"], 
                stderr=subprocess.DEVNULL
            ).decode().strip()

            if app_id:
                # B. Find the actual .desktop file in standard paths
                search_dirs = [
                    os.path.expanduser("~/.local/share/applications"),
                    "/usr/local/share/applications",
                    "/usr/share/applications",
                    "/var/lib/snapd/desktop/applications"
                ]
                
                exec_cmd = []
                is_multi = False
                
                for directory in search_dirs:
                    desktop_path = os.path.join(directory, app_id)
                    if os.path.exists(desktop_path):
                        # C. Parse the file for the Exec command
                        with open(desktop_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for line in f:
                                if line.startswith("Exec=") and not line.startswith("Exec=env"):
                                    # Check for multi-file markers (%F or %U)
                                    is_multi = "%F" in line or "%U" in line
                                    
                                    # Clean command: remove 'Exec=' and % placeholders
                                    raw_cmd = line.split("=", 1)[1].strip()
                                    exec_cmd = [x for x in shlex.split(raw_cmd) if not x.startswith("%")]
                                    break
                        if exec_cmd: break

                # D. Launch based on capabilities
                if exec_cmd:
                    if is_multi:
                        if log:
                            print(f"Opening {len(valid_paths)} images with {app_id}...")
                        subprocess.Popen(exec_cmd + valid_paths)
                    else:
                        if log:
                            print(f"Warning: {app_id} only supports single files.")
                            print(f"Opening first file: {valid_paths[0]}")
                        subprocess.Popen(exec_cmd + [valid_paths[0]])
                    return

        except Exception:
            pass # Fallback to generic xdg-open if anything fails

        # Fallback for Linux
        if log:
            print("Warning: Could not detect viewer capabilities. Using xdg-open on first file.")
        subprocess.Popen(["xdg-open", valid_paths[0]])

    # --- MACOS LOGIC ---
    elif sys.platform == "darwin":
        if log:
            print(f"Opening {len(valid_paths)} images with Preview...")
        subprocess.Popen(["open"] + valid_paths)

    # --- WINDOWS LOGIC ---
    elif sys.platform == "win32":
        if log and len(valid_paths) > 1:
            print("Opening first file only (Windows default behavior).")
        os.startfile(valid_paths[0])


def load_data(filepath: str, **kwargs):
    """
    Universal data loader. Infers the file type based on extension
    and passes **kwargs to the underlying pandas read function.
    
    Usage:
        df = load_data("data.csv", sep="|")
        df = load_data("data.parquet", columns=["id", "val"])
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    loaders = {
        ".csv": pd.read_csv,
        ".tsv": lambda f, **k: pd.read_csv(f, sep="\t", **k),
        ".parquet": pd.read_parquet,
        ".xls": pd.read_excel,
        ".xlsx": pd.read_excel,
        ".json": pd.read_json,
        ".pkl": pd.read_pickle,
        ".pickle": pd.read_pickle,
        ".feather": pd.read_feather
    }

    if ext not in loaders:
        raise ValueError(f"Unsupported file extension: {ext}")

    return loaders[ext](filepath, **kwargs)
