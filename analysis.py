# %%
import glob
from model import Model, gini
import pickle
import gc
import pandas
import seaborn
import statsmodels.formula.api as smf
import numpy
import matplotlib.pyplot as plt

# %%
res = []
for f in glob.glob("*.pkl"):
    seed = f.split(">")[1][1:-4]
    model = f.split(" ")[1]
    data = pickle.load(open(f, "rb"))
    for t in range(0, len(data)):
        res += [
            (
                model,
                seed,
                t,
                data[t].GDP,
                gini([h.M for h in data[t].H]),
                gini([h.W for h in data[t].H]),
                sum([f.G for f in data[t].FC]),
                (
                    sum([sum(f.C.values()) for f in data[t].FC])
                    + sum([f.G for f in data[t].FC])
                ),
                data[t].u,
                data[t].i,
                data[t].G.B,
                data[t].G.deficit,
            )
        ]
    del data
    gc.collect()

# %%
df = pandas.DataFrame(
    res,
    columns=[
        "model",
        "seed",
        "t",
        "GDP",
        "MGini",
        "WGini",
        "GvtConsumption",
        "Y",
        "u",
        "i",
        "Debt",
        "deficit",
    ],
)
df["GvtConsumptionShare"] = df.GvtConsumption / df.Y
df["DebtShare"] = df.Debt / df.GDP
df["deficitShare"] = df.deficit / df.GDP
df["logGDP"] = numpy.log(df.GDP)

df.to_pickle("res.pickle")
# %%
df = pandas.read_pickle("res.pickle")
# %%
df2 = df[df.model.str.contains('base') & (~df.model.str.contains('WTtop')) & ((df.t<300) & (df.t>50))]
# %%
seaborn.lineplot(x="t", y="GDP", hue="model", data=df2)
plt.savefig("GDP.pdf")
# %%
seaborn.lineplot(x="t", y="i", hue="model", data=df2)
plt.savefig("i.pdf")
#%%
seaborn.lineplot(x="t", y="u", hue="model", data=df2)
plt.savefig("u.pdf")
#%%
seaborn.lineplot(x="t", y="GvtConsumptionShare", hue="model", data=df2)
plt.savefig("Gvt.pdf")
#%%
seaborn.lineplot(x="t", y="WGini", hue="model", data=df2)
plt.savefig("WGini.pdf")
# %%
seaborn.lineplot(x="t", y="MGini", hue="model", data=df2)
plt.savefig("MGini.pdf")
# %%
data = df2[(df2.t>200)]
# %%
for m in data.model.unique():
    print(f"### {m}")
    print(smf.ols(f"u ~ C(model, Treatment('{m}'))", data=data).fit().summary())
# %%
for m in data.model.unique():
    print(f"### {m}")
    print(smf.ols(f"Y ~ C(model, Treatment('{m}'))", data=data).fit().summary())
# %%
for m in data.model.unique():
    print(f"### {m}")
    print(smf.ols(f"WGini ~ C(model, Treatment('{m}'))", data=data).fit().summary())
for m in data.model.unique():
    print(f"### {m}")
    print(smf.ols(f"MGini ~ C(model, Treatment('{m}'))", data=data).fit().summary())
# %%
s
# %%
