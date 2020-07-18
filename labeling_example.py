import pandas as pd
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier

SKIP=0
FOOTBALL=1
BASEBALL=2

def import_data():
  comments_df = pd.read_csv("data/data.csv",encoding='iso-8859–1')
  return comments_df

@labeling_function()
def bat(x):
  return BASEBALL if "bat" in x.text.lower() else SKIP

@labeling_function()
def touchdown(x):
  return FOOTBALL if "touchdown" in x.text.lower() else SKIP

sample_df = import_data()
lfs = [bat, touchdown]
applier=PandasLFApplier(lfs=lfs)
train = applier.apply(df=sample_df)

coverage_bat, coverage_touchdown = (train != SKIP).mean(axis=0)
print(f"baseball coverage: {coverage_bat * 100:.1f}%")
print(f"football coverage: {coverage_touchdown * 100:.1f}%")


