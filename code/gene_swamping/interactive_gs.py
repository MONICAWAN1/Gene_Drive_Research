#%%
import pickle

with open("se_alpha_mapped.pickle", "rb") as f:
    data = pickle.load(f)

# Print the type and maybe the first few entries
print(f"Type: {type(data)}")

if isinstance(data, dict):
    for k in list(data)[:5]:
        print(f"{k}: {data[k]}")
elif isinstance(data, list):
    for item in data[:5]:
        print(item)
else:
    print(data)


#%%
import numpy as np
#%%
# your grid definitions
s_vals     = np.arange(0.01,   1.01, 0.01)
c_vals     = np.arange(0.01,   1.01, 0.01)
h_vals     = np.arange(0.01,   1.01, 0.1)
m_vals     = np.linspace(0.01, 1.0, 100)
alpha_vals = np.linspace(-0.01, -2.0, 100)

output_file = "gene_swamping_results.txt"

with open("param.txt", "w") as fout:
    for s in s_vals:
        for c in c_vals:
            for h in h_vals:
                for m in m_vals:
                    for alpha in alpha_vals:
                        fout.write(f"{s:.3f} {c:.3f} {h:.3f} {m:.6f} {alpha:.6f} {output_file}\n")
#%%