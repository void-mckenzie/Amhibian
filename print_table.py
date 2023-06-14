import pandas as pd
from ast import literal_eval
import os

folder = "amphibian_model_outputs"

df_list = []
for fname in os.listdir(folder):
	print(fname)
	df = pd.read_csv(f"{folder}/{fname}")
	df_list.append(df)

master_df = pd.concat(df_list)[['weights', 'target','val_loss', 'val_iou', 'test_loss', 'test_iou']]
master_df["weights"] = master_df['weights'].apply(lambda x: x.split("/")[1][:-4])
master_df['target'] = master_df['target'].apply(lambda x: literal_eval(x)[0])
print(master_df)
print(master_df.to_latex(index=False, float_format="{:0.4f}".format))

