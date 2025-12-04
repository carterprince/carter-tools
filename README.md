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
import pandas as pd
from pathlib import Path
from carter_tools import get_style_features, print_colored_df
from tabpfn import TabPFNClassifier

data = []
for p in Path("FedPapersCorpus").glob("*.txt"):
    text = p.read_text()
    feats = get_style_features(text)
    data.append({**feats, 'label': p.name.split('_')[0], 'filename': p.name})

df = pd.DataFrame(data).set_index('filename')
train = df[df['label'] != 'dispt']
pred  = df[df['label'] == 'dispt']

pfn = TabPFNClassifier(n_estimators=8, device='cuda') 
pfn.fit(train.drop(columns='label'), train['label'])
results = pd.DataFrame(pfn.predict_proba(pred.drop(columns='label')), 
                       columns=pfn.classes_, 
                       index=pred.index)
results['Prediction'] = results.idxmax(axis=1)
print_colored_df(results)
```

<img height="400" alt="image" src="https://github.com/user-attachments/assets/971d2428-60e0-4c49-b492-57843fcef18b" />
