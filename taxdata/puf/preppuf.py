"""
Scripts to clean up the raw PUF before matching
"""

import numpy as np

# RECIDs for aggregate variables by PUF year
AGG_VARS = {
    2009: [999999],
    2010: [999998, 999999],
    2011: [999996, 999997, 999998, 999999],
    2015: [999996, 999997, 999998, 999999],
}


def preppuf(puf, year):
    """Prepares the PUF for mathcing

    Args:
        puf (DataFrame): the raw PUF file
    """
    puf.columns = map(str.lower, puf.columns)
    # drop aggregate variables
    puf = puf[~puf["recid"].isin(AGG_VARS[year])].copy()
    puf["filer"] = 1
    puf["depne"] = puf[["xocah", "xocawh", "xoodep", "xopar"]].sum(axis=1)

    adjust = (
        puf["e03150"]
        + puf["e03210"]
        + puf["e03220"]
        + puf["e03230"]
        + puf["e03260"]
        + puf["e03270"]
        + puf["e03240"]
        + puf["e03290"]
        + puf["e03300"]
        + puf["e03400"]
        + puf["e03500"]
    )
    puf["totincx"] = puf["e00100"] + adjust

    puf["sequence"] = puf.index + 1
    puf["soiseq"] = puf.index + 1
    puf["s006"] /= 100
    puf["s006"] *= 1.03

    puf["dep_stat"] = puf["dsi"]
    puf["agede"] = np.where(puf["e02400"] > 0, 1, 0)

    return puf
