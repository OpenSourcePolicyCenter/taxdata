import numpy as np
import pandas as pd
import taxcalc as tc
from .impute_pencon import impute_pension_contributions
from .constants import UNUSED_READ_VARS
from pathlib import Path


CUR_PATH = Path(__file__).resolve().parent
USABLE_VARS = tc.Records(data=None).USABLE_READ_VARS
USABLE_VARS.add("filer")


def finalprep(data):
    """
    Contains all the logic of the puf_data/finalprep.py script.
    """
    # - Check the PUF year
    max_flpdyr = max(data["flpdyr"])
    if max_flpdyr == 2008:
        data = transform_2008_varnames_to_2009_varnames(data)
    else:  # if PUF year is 2009+
        data = age_consistency(data)

    # - Make recid variable be a unique integer key:
    data = create_new_recid(data)

    # - Make several variable names be uppercase as in SOI PUF:
    data = capitalize_varnames(data)

    # - Impute cmbtp variable to estimate income on Form 6251 but not in AGI:
    cmbtp_standard = data["e62100"] - data["e00100"] + data["e00700"]
    zero = np.zeros(len(data.index))
    medical_limit = np.maximum(
        zero, data["e17500"] - np.maximum(zero, data["e00100"]) * 0.075
    )
    med_adj = np.minimum(medical_limit, 0.025 * np.maximum(zero, data["e00100"]))
    stx_adj = np.maximum(zero, data["e18400"])
    cmbtp_itemizer = (
        cmbtp_standard
        + data["p04470"]
        + data["e21040"]
        - data["e18500"]
        - data["e20800"]
        - stx_adj
        - med_adj
    )
    cmbtp = np.where(data["FDED"] == 1, cmbtp_itemizer, cmbtp_standard)
    data["cmbtp"] = np.where(data["f6251"] == 1, cmbtp, 0.0)

    # - Split earnings variables into taxpayer (p) and spouse (s) amounts:
    data = split_earnings_variables(data, max_flpdyr)

    # - Add AGI bin indicator used for adjustment factors:
    data = add_agi_bin(data)

    # - Replace e20500 with g20500:
    data = replace_20500(data)

    data["s006"] = data["matched_weight"] * 100

    # - Remove variables not expected by Tax-Calculator:
    data = data.filter(USABLE_VARS, axis=1)

    data["blind_head"] = np.where(data["blind_head"] == 1, 1, 0)
    data["blind_spouse"] = np.where(data["blind_spouse"] == 1, 1, 0)

    # - Convert data to integers:
    data = data.round(0).astype("int64")

    # - Impute pension contributions:
    data = impute_pension_contributions(data.copy(), max_flpdyr)

    # - Rename 'filer' to 'data_source'
    data = data.rename(columns={"filer": "data_source"})

    # - Sort columns to ensure every PUF is the same
    data.sort_index(axis=1, inplace=True)

    return data


# end of main function code


def create_new_recid(data):
    """
    Construct unique recid.
    """
    data["recid"] = data.index + 1
    return data


def age_consistency(data):
    """
    Construct age_head from agerange if available; otherwise use CPS value.
    Construct age_spouse as a normally-distributed agediff from age_head.
    """
    # set random-number-generator seed so that always get same random numbers
    np.random.seed(seed=123456789)
    # generate random integers to smooth age distribution in agerange
    shape = data["age_head"].shape
    agefuzz8 = np.random.randint(0, 9, size=shape)
    agefuzz9 = np.random.randint(0, 10, size=shape)
    agefuzz10 = np.random.randint(0, 11, size=shape)
    agefuzz15 = np.random.randint(0, 16, size=shape)

    # assign age_head using agerange midpoint or CPS age if agerange absent
    data["age_head"] = np.where(
        data["agerange"] == 0,
        data["age_head"],
        (data["agerange"] + 1 - data["dsi"]) * 10,
    )

    # smooth the agerange-based age_head within each agerange
    data["age_head"] = np.where(
        np.logical_and(data["agerange"] == 1, data["dsi"] == 0),
        data["age_head"] - 3 + agefuzz9,
        data["age_head"],
    )
    data["age_head"] = np.where(
        np.logical_and(data["agerange"] == 2, data["dsi"] == 0),
        data["age_head"] - 4 + agefuzz9,
        data["age_head"],
    )
    data["age_head"] = np.where(
        np.logical_and(data["agerange"] == 3, data["dsi"] == 0),
        data["age_head"] - 5 + agefuzz10,
        data["age_head"],
    )
    data["age_head"] = np.where(
        np.logical_and(data["agerange"] == 4, data["dsi"] == 0),
        data["age_head"] - 5 + agefuzz10,
        data["age_head"],
    )
    data["age_head"] = np.where(
        np.logical_and(data["agerange"] == 5, data["dsi"] == 0),
        data["age_head"] - 5 + agefuzz10,
        data["age_head"],
    )
    data["age_head"] = np.where(
        np.logical_and(data["agerange"] == 6, data["dsi"] == 0),
        data["age_head"] - 5 + agefuzz15,
        data["age_head"],
    )
    data["age_head"] = np.where(
        np.logical_and(data["agerange"] == 1, data["dsi"] == 1),
        data["age_head"] - 0 + agefuzz8,
        data["age_head"],
    )
    data["age_head"] = np.where(
        np.logical_and(data["agerange"] == 2, data["dsi"] == 1),
        data["age_head"] - 2 + agefuzz8,
        data["age_head"],
    )
    data["age_head"] = np.where(
        np.logical_and(data["agerange"] == 3, data["dsi"] == 1),
        data["age_head"] - 4 + agefuzz10,
        data["age_head"],
    )

    # convert zero age_head to one
    data["age_head"] = np.where(data["age_head"] == 0, 1, data["age_head"])

    # assign age_spouse relative to age_head if married;
    # if head is not married, set age_spouse to zero;
    # if head is married but has unknown age, set age_spouse to one;
    # do not specify age_spouse values below 15
    adiff = np.random.normal(0.0, 4.0, size=shape)
    agediff = np.int_(adiff.round())
    age_sp = data["age_head"] + agediff
    age_spouse = np.where(age_sp < 15, 15, age_sp)
    data["age_spouse"] = np.where(
        data["mars"] == 2, np.where(data["age_head"] == 1, 1, age_spouse), 0
    )
    return data


def capitalize_varnames(data):
    """
    Capitalize some variable names.
    """
    renames = {
        "dsi": "DSI",
        "eic": "EIC",
        "fded": "FDED",
        "flpdyr": "FLPDYR",
        "mars": "MARS",
        "midr": "MIDR",
        "xtot": "XTOT",
        "recid": "RECID",
    }
    data = data.rename(columns=renames)
    return data


def transform_2008_varnames_to_2009_varnames(data):
    """
    Convert 2008 IRS-SOI PUF variable names into 2009 PUF variable names.
    """
    data["e18400"] = data["e18425"] + data["e18450"]

    # drop unused variables only existing in 2008 IRS-SOI PUF
    UNUSED = {
        "e18425",
        "e18450",
        "e25370",
        "e25380",
        "state",
        "e87500",
        "e87510",
        "e87520",
        "e87540",
    }
    data = data.drop(UNUSED, 1)

    # drop variables not expected by Tax-Calculator
    data = data.drop(UNUSED_READ_VARS, 1)
    return data


def split_earnings_variables(data, data_year):
    """
    Split earnings subject to FICA or SECA taxation between taxpayer and spouse
    """
    # split wage-and-salary earnings subject to FICA taxation
    # the two e00200x variables come from the CPS. We'll use them just for
    # the wage ratio that we split up the PUF wages from
    total = np.where(data["MARS"] == 2, data["e00200p"] + data["e00200s"], 0).astype(
        float
    )
    frac_p = np.where(total != 0, data["e00200p"] / total, 1.0)
    frac_s = 1.0 - frac_p
    data["e00200p"] = np.around(frac_p * data["e00200"], 2)
    data["e00200s"] = np.around(frac_s * data["e00200"], 2)
    # specify FICA-SECA maximum taxable earnings (mte) for data_year
    # TODO: add these to a parameter file
    if data_year == 2008:
        mte = 102000
    elif data_year == 2009:
        mte = 106800
    elif data_year == 2011:
        mte = 106800
    elif data_year == 2015:
        mte = 118500
    else:
        raise ValueError("illegal SOI PUF data year {}".format(data_year))
    # total self-employment earnings subject to SECA taxation
    # (minimum handles a few secatip values slightly over the mte cap)
    secatip = np.minimum(mte, data["e30400"])  # for taxpayer
    secatis = np.minimum(mte, data["e30500"])  # for spouse
    # split self-employment earnings subject to SECA taxation
    # ... compute secati?-derived frac_p and frac_s
    total = np.where(data["MARS"] == 2, secatip + secatis, 0).astype(float)
    frac_p = np.where(total != 0, secatip / total, 1.0)
    frac_s = 1.0 - frac_p
    # ... split e00900 (Schedule C) and e02100 (Schedule F) net earnings/loss
    data["e00900p"] = np.around(frac_p * data["e00900"], 2)
    data["e00900s"] = np.around(frac_s * data["e00900"], 2)
    data["e02100p"] = np.around(frac_p * data["e02100"], 2)
    data["e02100s"] = np.around(frac_s * data["e02100"], 2)
    # ... estimate Schedule K-1 box 14 self-employment earnings/loss
    # ...    Note: secati? values fall in the [0,mte] range.
    # ...    So, if sum of e00900? and e02100? is negative and secati? is
    # ...    zero, we make a conservative assumption and set box14 to zero
    # ...    (rather than to a positive number), but we allow the estimate
    # ...    of box 14 to be negative (that is, represent a loss).
    nonbox14 = data["e00900p"] + data["e02100p"]
    box14 = np.where(
        np.logical_and(nonbox14 <= 0, secatip <= 0), 0.0, secatip - nonbox14
    )
    data["k1bx14p"] = box14.round(2)
    nonbox14 = data["e00900s"] + data["e02100s"]
    box14 = np.where(
        np.logical_and(nonbox14 <= 0, secatis <= 0), 0.0, secatis - nonbox14
    )
    data["k1bx14s"] = box14.round(2)
    # ... check consistency of self-employment earnings estimates
    raw = data["e00900p"] + data["e02100p"] + data["k1bx14p"]
    estp = np.where(raw < 0, 0.0, np.where(raw > mte, mte, raw))
    raw = data["e00900s"] + data["e02100s"] + data["k1bx14s"]
    ests = np.where(raw < 0, 0.0, np.where(raw > mte, mte, raw))
    assert np.allclose(estp, secatip, rtol=0.0, atol=0.01)
    assert np.allclose(ests, secatis, rtol=0.0, atol=0.01)
    return data


def add_agi_bin(data):
    """
    Add an AGI bin indicator used in Tax-Calc to apply adjustment factors
    """
    agi = pd.Series([0] * len(data.e00100))
    agi[data.e00100 < 0] = 0
    agi[(data.e00100 >= 0) & (data.e00100 < 5000)] = 1
    agi[(data.e00100 >= 5000) & (data.e00100 < 10000)] = 2
    agi[(data.e00100 >= 10000) & (data.e00100 < 15000)] = 3
    agi[(data.e00100 >= 15000) & (data.e00100 < 20000)] = 4
    agi[(data.e00100 >= 20000) & (data.e00100 < 25000)] = 5
    agi[(data.e00100 >= 25000) & (data.e00100 < 30000)] = 6
    agi[(data.e00100 >= 30000) & (data.e00100 < 40000)] = 7
    agi[(data.e00100 >= 40000) & (data.e00100 < 50000)] = 8
    agi[(data.e00100 >= 50000) & (data.e00100 < 75000)] = 9
    agi[(data.e00100 >= 75000) & (data.e00100 < 100000)] = 10
    agi[(data.e00100 >= 100000) & (data.e00100 < 200000)] = 11
    agi[(data.e00100 >= 200000) & (data.e00100 < 500000)] = 12
    agi[(data.e00100 >= 500000) & (data.e00100 < 1e6)] = 13
    agi[(data.e00100 >= 1e6) & (data.e00100 < 1.5e6)] = 14
    agi[(data.e00100 >= 1.5e6) & (data.e00100 < 2e6)] = 15
    agi[(data.e00100 >= 2e6) & (data.e00100 < 5e6)] = 16
    agi[(data.e00100 >= 5e6) & (data.e00100 < 1e7)] = 17
    agi[(data.e00100 >= 1e7)] = 18

    data["agi_bin"] = agi

    return data


def replace_20500(data):
    """
    Replace e20500, net casualty losses, with g20500, gross casualty losses
    (gross loss values less than 10% AGI are unknown and assumed to be zero)
    """
    gross = np.where(
        data.e20500 > 0.0, data.e20500 + 0.10 * np.maximum(0.0, data.e00100), 0.0
    )
    data["g20500"] = np.int_(gross.round())
    return data
