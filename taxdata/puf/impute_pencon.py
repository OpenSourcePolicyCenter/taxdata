"""
Impute DC pension contributions in PUF using aggregate data from IRS Form W-2.

The imputation strategy has two steps.  In the first step, use
aggregate W-2 data on the number of individuals with positive pension
contributions for each of 16 positive wage groups in each of 8 age
groups (that is, for each of 128 age-wage cells) to impute
stochastically whether or not an individual with earnings makes an
elective deferral to a defined-contribution (DC) pension plan.  Then
in the second step, use aggregate W-2 data on the dollar amount of
elective DC deferrals among those with a positive deferral for each of
16 positive wage groups in each of 8 age groups (that is, for the same
128 age-wage cells) to impute the pension contribution (that is, elective
deferral) amount.

This strategy would be straight forward to implement if the W-2 sample
was the same as the PUF sample.  But because the W-2 sample is larger
(at least for the high-wage groups), there is a need to make a minor
adjustment among higer-wage individuals (that is, among individuals
with wages over $200,000).  For details on the logic of the adjustment
and the results with and without the adjustment, see the comments
below on the HIWAGE parameters, which have been calibrated by hand.

Another tactical complication is that the amount of each individual's
pension contribution was legally limited to $16,500 during 2011.  The
second imputation stage uses an iterative procedure to cap individual
pension contributions in a way that maintains the age-wage cell's
aggregate pension contribution.

And finally, individuals cannot defer more than their gross earnings,
which in PUF records is equal to the sum of net earnings and pension
contributions, which is (e00200p + pencon_p) for the filing-unit head
and (e00200s + pencon_s) for the filing-unit spouse, if present.  For
CPS records in the puf.csv file, gross earnings is equal to e00200p for
the filing-unit head and e00200s for the filing-unit spouse, if present.
Gross earnings and pension contributions out of gross earnings are
calculated for the PUF records by conducting the whole imputation
process twice.  See the impute_pension_contributions() function code
for details.
"""

from __future__ import print_function
import numpy as np
import pandas as pd
from pathlib import Path


CURPATH = Path(__file__).resolve().parent
DUMP0 = False
DUMP1 = False
DUMP2 = False


def targets(year):
    """
    Return a DataFrame containing number of taxpayers & spouses with earnings
    that make a pension contribution (in millions of people) and a DataFrame
    containing the aggregate amount of the pension contributions (in billions
    of dollars) by the wage and age categories in "SOI Tax Stats - Individual
    Information Return Form W-2 Statistics," Table 2.B.1 for 2011.  Note that
    in the orginal table, the three cnt and amt cells (for wages in the range
    of $500K to $2M) are combined into one cell for the bottom age range,
    u26, and for the top age range, 75plus.  In each of these four cases,
    the three-cell total is evenly spread across the three cells in the
    revised data specified here.  Also, the top two wage groups (5u10M and
    10u30M) are combined into a single group (5Mplus).
    """
    cnt_df = pd.read_csv(Path(CURPATH, f"dcpentargetcnt{year}.csv"), index_col=0)
    amt_df = pd.read_csv(Path(CURPATH, f"dcpentargetamt{year}.csv"), index_col=0)
    return cnt_df, amt_df


# end of targets() function


# specify top of age and wage brackets
UNDER_AGE = [26, 35, 45, 55, 60, 65, 75, 99]
UNDER_WAGE = [
    5e3,
    10e3,
    15e3,
    20e3,
    25e3,
    30e3,
    40e3,
    50e3,
    75e3,
    100e3,
    200e3,
    500e3,
    1e6,
    2e6,
    5e6,
    124e6,
]


def age_group(row):
    """
    Specify age group of individual.
    """
    if row["age"] == 0:
        return -1
    for grp, underage in enumerate(UNDER_AGE):
        if row["age"] < underage:
            return grp
    raise ValueError(f"illegal value of age: {row['age']}")


# end of age_group() function


def wage_group(row):
    """
    Specify wage group of individual.
    """
    if row["wage"] == 0:
        return -1
    for grp, underwage in enumerate(UNDER_WAGE):
        if row["wage"] < underwage:
            return grp
    raise ValueError(f"illegal value of wage: {row['wage']}")


# end of wage_group() function


# specify adjustment of having a positive pencon for high-wage groups
HIWAGE_PROB_SF = 1.47  # prob of positive pension contribution multiplied by SF
MIN_HIWAGE_GROUP = 11  # SF applied to wage groups no less than this MIN

# This adjustment to some cell-specific probabilities of having a
# positive pension contribution is being done because, in some
# high-wage cells, the W-2 data show more people with positive
# pension contributions than the PUF shows with positive earnings.
# Making no imputation probability adjustment (that is, setting
# HIWAGE_PROB_SF equal to one), implies the imputed head count and
# dollar amount of positive pension contributions are both roughly one
# percent below the total in the W-2 data.  Setting the HIWAGE_PROB_SF
# to the value above makes the head count of those with positive
# pension contributions equal to the total in the W-2 data and the
# imputed dollar amount is only about 0.6 of one percent below the
# total in the W-2 data.  The calibration of the HIWAGE_PROB_SF value
# was conducted so that the calibrated SF value produces an imputed count
# of individuals having a positive pension contribution equal to the
# target count of 46.98 million individuals.  SF values below the one
# specified above produce imputed counts that are less than 46.98 million
# and SF values above the one specified above produce imputed counts that
# are more than 46.98 million.  This calibration was done with DUMP0 set
# to True and by executing "python impute_pencon.py > impute_pencon.res"
# several times each with a different value of HIWAGE_PROB_SF.


# specify maximum legal elective deferral amount for DC pensions in each year
# the PUF is supported
MAX_PENCON_AMT = {2011: 16500, 2015: 1800}


def impute(idata, target_cnt, target_amt, year):
    """
    Impute idata[pencon] given other idata variables and targets.
    """
    if DUMP1 and HIWAGE_PROB_SF != 1.0:
        print("HIWAGE_PROB_SF= {:.4f}".format(HIWAGE_PROB_SF))
        print("MIN_HIWAGE_GROUP= {}".format(MIN_HIWAGE_GROUP))
    # loop through the positive-age and positive-wage cells
    for agrp in range(0, len(UNDER_AGE)):
        for wgrp in range(0, len(UNDER_WAGE)):
            # impute actual count of having positive pencon
            in_cell = (idata["agegrp"] == agrp) & (idata["wagegrp"] == wgrp)
            cell_idata = idata[in_cell].copy()
            wgt_num_earners = cell_idata["weight"].sum() * 1e-6
            if wgt_num_earners <= 0.0:
                msg = "agrp={};wgrp={} has wgt_num_earners={:.4f} <= 0"
                raise ValueError(msg.format(agrp, wgrp, wgt_num_earners))
            wgt_pos_pencon = target_cnt.iloc[wgrp, agrp]
            prob = wgt_pos_pencon / wgt_num_earners
            if wgrp >= MIN_HIWAGE_GROUP:
                prob *= HIWAGE_PROB_SF
            if DUMP1 and prob > 1.0:
                print("agrp={};wgrp={} ==> prob= {:.3f}".format(agrp, wgrp, prob))
            cell_idata["pencon"] = np.where(cell_idata["urn"] < prob, 1, 0)
            pos_pc = cell_idata["pencon"] > 0
            if pos_pc.sum() == 0:  # no positive pension contributions in cell
                if DUMP1:
                    print("agrp={};wgrp={} has zero pencon".format(agrp, wgrp))
                continue  # to next wgrp in cell loop
            # impute actual amount of each positive pension contribution
            # taking into account that pension contributions are legally capped
            wage = cell_idata["wage"]
            wgt = cell_idata["weight"]
            wgt_pos_pc_wages = (wage[pos_pc] * wgt[pos_pc]).sum() * 1e-9
            cell_target_amt = target_amt.iloc[wgrp, agrp]
            rate0 = min(1.0, cell_target_amt / wgt_pos_pc_wages)
            if DUMP2:
                print("agrp={};wgrp={} ==> rate0= {:.4f}".format(agrp, wgrp, rate0))
            # iteratively raise non-capped deferral rate to hit target_amt
            num_iterations = 10
            for itr in range(0, num_iterations):
                uncapped_amt = np.where(pos_pc, np.round(wage * rate0).astype(int), 0)
                capped_amt = np.minimum(uncapped_amt, MAX_PENCON_AMT[year])
                over_amt = uncapped_amt - capped_amt
                over_tot = (over_amt * wgt).sum() * 1e-9
                rate1 = min(1.0, (cell_target_amt + over_tot) / wgt_pos_pc_wages)
                if np.allclose([rate1], [rate0]):
                    if DUMP2 and itr > 0:
                        print("  iter={} ==> rate= {:.4f}".format(itr, rate0))
                    break  # out of iteration loop
                else:
                    if DUMP2 and itr == (num_iterations - 1):
                        print("  iter={} ==> rate= {:.4f}".format(itr, rate0))
                    rate0 = rate1
            cell_idata["pencon"] = capped_amt
            # store cell_idata['pencon'] in idata
            idata.loc[in_cell, "pencon"] = cell_idata["pencon"]
            del cell_idata


# end of impute() function


def impute_pension_contributions(alldata, year):
    """
    Main function in impute_pencon.py file.
    Argument: puf.csv DataFrame just before imputation is done.
    Returns: puf.csv DataFrame with imputed pension contribution amounts.
    """
    # specify target DataFrames with total column and total row removed
    target_cnt, target_amt = targets(year)
    target_cnt.drop(labels="total", axis="index", inplace=True)
    target_cnt.drop(labels="total", axis="columns", inplace=True)
    target_amt.drop(labels="total", axis="index", inplace=True)
    target_amt.drop(labels="total", axis="columns", inplace=True)
    if DUMP0:
        print(
            "target_cnt.shape={} and size={}".format(target_cnt.shape, target_cnt.size)
        )
        print(
            "target_amt.shape={} and size={}".format(target_amt.shape, target_amt.size)
        )
        print("len(UNDER_AGE)={}".format(len(UNDER_AGE)))
        print("len(UNDER_WAGE)={}".format(len(UNDER_WAGE)))
        cnt = target_cnt.values.sum()
        print("sum(target_cnt)= {:.4f}".format(cnt))
        amt = target_amt.values.sum()
        print("sum(target_amt)= {:.4f}".format(amt))
        avg = amt / cnt
        print("avg(target_amt)= {:.3f}".format(avg))
    # construct individual-level idata from filing-unit alldata
    # (note: PUF records have filer=1 and CPS records have filer=0)
    # ... construct _p DataFrame with renamed variables
    ivars = ["age_head", "e00200p", "filer"]
    idata_p = alldata[ivars].copy()
    idata_p["spouse"] = np.zeros(len(idata_p.index), dtype=np.int8)
    idata_p["weight"] = alldata["s006"] * 0.01
    idata_p.rename(
        {"age_head": "age", "e00200p": "e00200"}, axis="columns", inplace=True
    )
    # ... construct _s DataFrame with renamed variables
    ivars = ["age_spouse", "e00200s", "filer"]
    idata_s = alldata[ivars].copy()
    idata_s["spouse"] = np.ones(len(idata_s.index), dtype=np.int8)
    idata_s["weight"] = alldata["s006"] * 0.01
    idata_s.rename(
        {"age_spouse": "age", "e00200s": "e00200"}, axis="columns", inplace=True
    )
    # ... combine the _p and _s DataFrames
    idata = pd.concat([idata_p, idata_s], copy=True)
    del idata_p
    del idata_s
    # ... construct variables that never change over the imputation process
    idata["agegrp"] = idata.apply(age_group, axis=1)
    np.random.seed(111_111)
    idata["urn"] = np.random.uniform(size=len(idata.index))
    # ... initialize pension contributions to zero
    idata["pencon"] = np.zeros(len(idata.index), dtype=np.int64)
    if DUMP0:
        idata["wage"] = idata["e00200"]
        idata["wagegrp"] = idata.apply(wage_group, axis=1)
        print("raw_num_earners(#)=", (idata["wage"] > 0).sum())
        wgt_earners = idata["weight"][idata["wage"] > 0].sum()
        print("wgt_num_earners(#M)= {:.3f}".format(wgt_earners * 1e-6))
        wearnings = idata["weight"] * idata["wage"]
        wgt_earnings = wearnings.sum()
        print("wgt_earnings($B)= {:.3f}".format(wgt_earnings * 1e-9))
        wgt_puf_earnings = wearnings[idata["filer"] == 1].sum()
        print("wgt_PUF_earnings($B)= {:.3f}".format(wgt_puf_earnings * 1e-9))
        wgt_cps_earnings = wearnings[idata["filer"] == 0].sum()
        print("wgt_CPS_earnings($B)= {:.3f}".format(wgt_cps_earnings * 1e-9))
        print("min_agegrp=", idata["agegrp"].min())
        print("max_agegrp=", idata["agegrp"].max())
        print("min_wagegrp=", idata["wagegrp"].min())
        print("max_wagegrp=", idata["wagegrp"].max())
    # do two imputations to construct gross wages for PUF records
    idata["wage"] = idata["e00200"]
    idata["wagegrp"] = idata.apply(wage_group, axis=1)
    impute(idata, target_cnt, target_amt, year)
    idata["wage"] = np.where(
        idata["filer"] == 1, idata["e00200"] + idata["pencon"], idata["e00200"]
    )
    idata["wagegrp"] = idata.apply(wage_group, axis=1)  # gross wage group
    impute(idata, target_cnt, target_amt, year)
    if DUMP0:
        cnt = (idata["weight"] * (idata["pencon"] > 0)).sum() * 1e-6
        print("wgt_pencon_cnt(#M)= {:.3f}".format(cnt))
        amt = (idata["weight"] * idata["pencon"]).sum() * 1e-9
        print("wgt_pencon_amt($B)= {:.3f}".format(amt))
        avg = amt / cnt
        print("avg_pencon_amt($K)= {:.3f}".format(avg))
    # subtract pencon from e00200 for CPS records to make them net earnings
    idata["e00200"] = np.where(
        idata["filer"] == 1, idata["e00200"], idata["e00200"] - idata["pencon"]
    )
    if DUMP0:
        wearnings = idata["weight"] * idata["e00200"]
        wgt_earnings = wearnings.sum()
        print("wgt_earnings($B)= {:.3f}".format(wgt_earnings * 1e-9))
        wgt_puf_earnings = wearnings[idata["filer"] == 1].sum()
        print("wgt_PUF_earnings($B)= {:.3f}".format(wgt_puf_earnings * 1e-9))
        wgt_cps_earnings = wearnings[idata["filer"] == 0].sum()
        print("wgt_CPS_earnings($B)= {:.3f}".format(wgt_cps_earnings * 1e-9))
    # set alldata values of pencon_p and pencon_s and the e00200* variables
    alldata["pencon_p"] = idata["pencon"][idata["spouse"] == 0]
    alldata["pencon_s"] = idata["pencon"][idata["spouse"] == 1]
    alldata["e00200p"] = idata["e00200"][idata["spouse"] == 0]
    alldata["e00200s"] = idata["e00200"][idata["spouse"] == 1]
    alldata["e00200"] = alldata["e00200p"] + alldata["e00200s"]
    # return revised alldata
    return alldata


# end of impute_pension_contributions() function


if __name__ == "__main__":
    RAWDATA = pd.read_csv("puf.csv")
    REVDATA = impute_pension_contributions(RAWDATA)
    REVDATA.to_csv("puf-pc.csv", index=False)
