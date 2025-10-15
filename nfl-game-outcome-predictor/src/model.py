import pandas as pd

# Load the play-by-play data
df = pd.read_csv("data/raw/NFL Play by Play 2009-2016.csv", low_memory=False)
print(df.shape)
df.head()
