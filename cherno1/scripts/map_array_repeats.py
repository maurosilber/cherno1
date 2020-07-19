import numpy as np
import pandas as pd
import tqdm

arrays = pd.read_csv(
    "../data/internal_array_map", sep=" ", usecols=(3, 4), dtype=int
).rename(columns={"abs.init.row": "start", "abs.fin.row": "end"})

arrays.start -= 1
repeats_map = np.empty(arrays.end.iloc[-1], dtype=np.int32)
for i, a in tqdm.tqdm(arrays.iterrows(), total=len(arrays)):
    repeats_map[a.start : a.end] = i

np.save("../data/repeats_map.npy", repeats_map)
