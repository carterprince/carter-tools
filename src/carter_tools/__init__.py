import os
import re
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.colors as pcolors
from collections import Counter

# ==========================================
# 1. STYLOMETRY (Text Analysis)
# ==========================================

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


# ==========================================
# 2. VISUALIZATION (Console)
# ==========================================

def print_colored_df(df, colorscale="RdBu", flip=False):
    """
    Manually prints a DataFrame to the console using ANSI escape codes.
    
    Colors are normalized PER COLUMN based on the min/max of that column.
    Uses Plotly colorscales to determine the color.
    
    Args:
        df (pd.DataFrame): The DataFrame to print.
        colorscale (str): Any valid Plotly colorscale name (e.g., 'Viridis', 'Plasma', 'RdBu').
        flip (bool): If True, inverts the colorscale mapping.
    """
    
    # 1. Calculate Min/Max for numeric columns
    col_stats = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            col_stats[col] = (c_min, c_max)

    # 2. Calculate Column Widths (Header vs Data)
    col_widths = {}
    for col in df.columns:
        # Start with header length
        max_len = len(str(col))
        
        # Check length of all values in this column
        # We must replicate the formatting logic to get the true length
        for val in df[col]:
            if col in col_stats and isinstance(val, (int, float, np.number)):
                # Numeric format is .3f
                val_len = len(f"{val:.3f}")
            else:
                # String format
                val_len = len(str(val))
            
            if val_len > max_len:
                max_len = val_len
        
        # Add a tiny bit of padding (min width 8)
        col_widths[col] = max(max_len, 8)

    def get_ansi_color(value, c_min, c_max):
        """Maps value to ANSI RGB string using Plotly's sampler."""
        if pd.isna(value) or c_min == c_max:
            return "" 

        # Normalize value to 0.0 - 1.0
        try:
            t = (value - c_min) / (c_max - c_min)
        except ZeroDivisionError:
            return ""

        if flip:
            t = 1.0 - t

        # Sample color from Plotly
        try:
            color_str = pcolors.sample_colorscale(colorscale, [t])[0]
        except Exception:
            return ""

        # Parse "rgb(r, g, b)"
        matches = re.findall(r'\d+', color_str)
        if len(matches) >= 3:
            r, g, b = map(int, matches[:3])
            return f"\033[38;2;{r};{g};{b}m"
        
        return ""

    RESET = "\033[0m"
    
    # Calculate index width
    index_width = max([len(str(i)) for i in df.index] + [0])

    # Print Header
    print(f"{' ' * index_width} ", end="")
    for col in df.columns:
        # Center the header in the calculated width
        print(f" {str(col).center(col_widths[col])} ", end="")
    print()

    # Print Rows
    for index, row in df.iterrows():
        print(f"{str(index).ljust(index_width)} ", end="")
        for col in df.columns:
            val = row[col]
            
            # Determine color and format
            if col in col_stats and isinstance(val, (int, float)):
                c_min, c_max = col_stats[col]
                color = get_ansi_color(val, c_min, c_max)
                formatted_val = f"{val:.3f}"
            else:
                color = ""
                formatted_val = str(val)
                
            # Right justify the value within the column width
            print(f" {color}{formatted_val.rjust(col_widths[col])}{RESET} ", end="")
        print()


# ==========================================
# 3. DATA LOADING (Google Trends / Maps)
# ==========================================

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


# ==========================================
# 4. PLOTTING (Plotly Maps)
# ==========================================

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
