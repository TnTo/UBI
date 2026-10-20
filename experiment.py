from multiprocessing import Pool
from model import Model
from copy import deepcopy
import pickle
import gc


def base(m):
    return m


def constrained(m):
    m.EC = True
    return m


def base_UBI(m):
    m.UBI = True
    return m


def constrained_UBI(m):
    m.EC = True
    m.UBI = True
    return m


def base_WT(m):
    m.tM = 0.02
    return m


def constrained_WT(m):
    m.EC = True
    m.tM = 0.02
    return m


def base_PT(m):
    m.tP = 0.4
    return m


def constrained_PT(m):
    m.EC = True
    m.tP = 0.4
    return m


def base_WTtop(m):
    m.tM = 0.02
    m.tMquintile = 0.05
    return m


def constrained_WTtop(m):
    m.EC = True
    m.tM = 0.02
    m.tMquintile = 0.05
    return m


def base_WT_UBI(m):
    m.UBI = True
    m.tM = 0.02
    return m


def constrained_WT_UBI(m):
    m.EC = True
    m.UBI = True
    m.tM = 0.02
    return m


def base_PT_UBI(m):
    m.tP = 0.4
    m.UBI = True
    return m


def constrained_PT_UBI(m):
    m.EC = True
    m.UBI = True
    m.tP = 0.4
    return m


def base_WTtop_UBI(m):
    m.UBI = True
    m.tM = 0.02
    m.tMquintile = 0.05
    return m


def constrained_WTtop_UBI(m):
    m.EC = True
    m.UBI = True
    m.tM = 0.02
    m.tMquintile = 0.05
    return m


def iteration(s):
    for f in [
        base,
        constrained,
        base_UBI,
        constrained_UBI,
        base_WT,
        constrained_WT,
        base_PT,
        constrained_PT,
        base_WTtop,
        constrained_WTtop,
        base_WT_UBI,
        constrained_WT_UBI,
        base_PT_UBI,
        constrained_PT_UBI,
        base_WTtop_UBI,
        constrained_WTtop_UBI,
    ]:
        m = f(Model())
        data = []
        for _ in range(m.TMAX):
            m.step()
            data += [deepcopy(m)]
        pickle.dump(data, open(f"data_{f}_{s}.pkl", "wb"))
        del data
        gc.collect()

if __name__ == '__main__':
    with Pool(10) as pool:
        pool.map(iteration, range(1,11))
