# carter-tools

Personal utility library for terminal visualizations, Google Trends DMA mapping, and text stylometry.

## Install

```bash
uv pip install git+https://github.com/carterprince/carter-tools
```

## Usage

### Terminal Heatmaps
Print Pandas DataFrames directly to the console with color gradients.

```python
import pandas as pd
import numpy as np
from carter_tools import print_colored_df

df = pd.DataFrame({
    "col_a": np.random.normal(loc=0, scale=1, size=20),
    "col_b": np.random.normal(loc=500, scale=100, size=20)
})

print_colored_df(df, colorscale="RdBu")
```

<img height="400" alt="image" src="https://github.com/user-attachments/assets/ce399196-6b70-4a4e-89ad-fbc8fb7e1cab" />

### Google Trends & Mapping
Clean Google Trends CSVs and map them to US DMAs (Designated Market Areas).

```python
from carter_tools import geomaps_to_df, usa_dmas_choropleth

# Load and merge all "geoMap*.csv" files in current dir
df = geomaps_to_df(columns=["data science", "machine learning"])

# Generate Plotly map
fig = usa_dmas_choropleth(df, color="machine learning", colorscale="RdYlBu_r")
fig.show()
```

<img height="400" alt="image" src="https://github.com/user-attachments/assets/bb5543a2-ce52-487d-83e9-a6b1d74c02ab" />

### Image Viewing
Launch the system's default image viewer for generated plots. Automatically handles multi-file slideshows on Linux (detecting `.desktop` capabilities) and macOS.

```python
from carter_tools import view_images

# Opens images in a single window/slideshow if supported
view_images(["output/plot_a.png", "output/plot_b.png"])
```

### Stylometry
Extract linguistic features (function words, punctuation usage, sentence length) for text analysis.

```python
from carter_tools import get_style_features

text = "To be, or not to be..."
features = get_style_features(text)
# Returns dict: {'fw_the': 50.0, 'punct_comma': 100.0, 'avg_word_len': 2.5...}
```
