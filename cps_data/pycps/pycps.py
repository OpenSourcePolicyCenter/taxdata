import pandas as pd
import numpy as np
from operator import itemgetter
from tqdm import tqdm
from taxunit import TaxUnit


INCOME_VARS = [
    "wsal_val", "int_val", "div_val", "semp_val",
    "alimony", "pensions_annuities", "frse_val",
    "uc_val"
]


def find_person(data: list, lineno: int) -> dict:
    """
    Function to find a person in a list of dictionaries representing
    individuals in the CPS
    """
    for person in data:
        if person["a_lineno"] == lineno:
            return person
    # raise an error if they're never found
    msg = (f"Person with line number {lineno} not found in"
           f"household.")
    raise ValueError(msg)


def eic_eligible(person: dict) -> int:
    """
    Function to determine if a dependent is an EIC eligible child
    df: DataFrame of just dependents
    https://www.irs.gov/credits-deductions/individuals/earned-income-tax-credit/qualifying-child-rules
    """
    # relationship test
    relationship = ((person["a_exprrp"] == 5) |
                    (person["a_exprrp"] == 7) |
                    (person["a_exprrp"] == 9) |
                    (person["a_exprrp"] == 11))
    # age test
    eic_max_age = 19
    if person["a_ftpt"] == 1:
        eic_max_age = 24
    # person is between 1 and the max age
    age = 0 <= person["a_age"] <= eic_max_age
    # assume they pass the residency test
    # assume they pass joint filer test for now

    # EIC eligible if they pass both tests
    eligible = int(relationship & age)

    return eligible


def find_claimer(claimerno: int, head_lineno: int, a_lineno: int, data: list) -> bool:
    """
    Determine if an individual is the dependent of the head of
    the tax unit

    Parameters
    ----------
    claimerno: line number of the person claiming the dependent
    head_lineno: line number of the head of the unit
    a_lineno: line number of person being evaluated
    data: list of all the people in the household
    """
    # claimer is also the head of the unit
    if claimerno == head_lineno:
        return True
    # see if person is dependent or spouse of head
    claimer = find_person(data, claimerno)
    spouse_dep = (claimer["a_spouse"] == claimerno |
                  claimer["dep_stat"] == claimerno)
    if spouse_dep:
        return True
    # follow any potential spouse/dependent trails to find the one
    if claimer["dep_stat"] != 0:
        claimer2 = find_person(data, claimer["dep_stat"])
        while True:
            if claimer2["a_lineno"] == claimerno:
                return True
            if claimer2["dep_stat"] != 0:
                claimer2 = find_person(data, claimer2["dep_stat"])
            else:
                break
    return False


def is_dependent(person, unit):
    # if they're a spouse, tax unit, or claimed, return false
    flagged = person["p_flag"] or person["s_flag"] or person["d_flag"]
    if flagged:
        return False
    # qualifying child
    if person["a_parent"] == unit.a_lineno:
        age_req = 19
        # age requirement increases for full time students
        if person["a_ftpt"] == 1:
            age_req = 24
        if person["a_age"] > age_req:
            return False
        if person["a_age"] > unit.age_head:
            return False
        # assume that they do live with you
        # financial support
        total_support = unit.tot_inc + person["ptotval"]
        try:
            pct_support = person["ptotval"] / total_support
        except ZeroDivisionError:
            pct_support = 0.0
        if pct_support > 0.5:
            return False
        # only person claiming them
        if person["d_flag"]:
            return False
    # qualifying relative
    else:
        # assume they live with you
        # income test
        if person["ptotval"] > 4150:
            return False
        # only person claiming them
        if person["d_flag"]:
            return False
    # if you get to this point, they qualify as a dependent
    return True


def remove_dependent(tu, dependent):
    if dependent["a_age"] <= 17:
        tu.nu18 -= 1
    elif 18 <= dependent["a_age"] <= 20:
        tu.n1820 -= 1
    elif dependent["a_age"] >= 21:
        tu.n21 -= 1


def create_units(df, year, verbose=False):
    data = df.to_dict("records")
    # sort on familly position and family relationship so that
    # we can avoid making children their own units just because
    # they're listed ahead of their parents
    data = sorted(data, key=itemgetter("ffpos", "a_famrel"))
    units = {}
    dependents = []
    for person in data:
        flagged = person["p_flag"] or person["s_flag"] or person["d_flag"]
        if verbose:
            print(person["a_lineno"], flagged)
        if not flagged:
            # make them a tax unit
            if verbose:
                print("making unit", person["a_lineno"])
            person["p_flag"] = True
            tu = TaxUnit(person, year)
            # loop through the rest of the household for
            # spouses and dependents
            if person["a_spouse"] != 0:
                spouse = find_person(data, person["a_spouse"])
                if verbose:
                    print("adding spouse", spouse["a_lineno"])
                tu.add_spouse(spouse)
            for _person in data:
                # only allow dependents in the same family
                if person["ffpos"] == _person["ffpos"]:
                    is_dep = is_dependent(_person, tu)
                    if is_dep:
                        if verbose:
                            print("adding dependent", _person["a_lineno"])
                        tu.add_dependent(_person)
                        dependents.append(_person)
            units[person["a_lineno"]] = tu

    # check and see if any dependents must file
    # https://turbotax.intuit.com/tax-tips/family/should-i-include-a-dependents-income-on-my-tax-return/L60Hf4Rsg
    for person in dependents:
        filer = False
        if person["earned_inc"] >= 12000:
            filer = True
        elif person["unearned_inc"] >= 1050:
            filer = True
        elif person["ptotval"] >= max(1050, person["earned_inc"] + 350):
            filer = True
        # if a dependent says they filed, we'll believe them
        if person["filestat"] != 6:
            filer = True
        if filer:
            if verbose:
                print("dep filer", person["a_lineno"])
            tu = TaxUnit(person, year, dep_status=True)
            remove_dependent(units[person["claimer"]], person)
            units[person["a_lineno"]] = tu
    return [unit.output() for unit in units.values()]


def pycps(cps: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Core code for iterating through the households
    Parameters
    ----------
    cps: Pandas DataFrame that contains the CPS
    """
    print("Calculating pensions and annuities")
    if year == 2015:
        alimony = np.where(cps["oi_off"] == 20,
                           cps["oi_val"], 0.)
    else:
        alimony = cps["alm_val"]
    oi_pensions = np.where(cps["oi_off"] == 2, cps["oi_val"], 0.)
    ret_pensions1 = np.where(cps["ret_sc1"] == 1,
                             cps["ret_val1"], 0.)
    ret_pensions2 = np.where(cps["ret_sc2"] == 1,
                             cps["ret_val2"], 0.)
    annuities1 = np.where(cps["ret_sc1"] == 7,
                          cps["ret_val1"], 0.)
    annuities2 = np.where(cps["ret_sc2"] == 7,
                          cps["ret_val2"], 0.)
    oi_annuities = np.where(cps["oi_off"] == 13,
                            cps["oi_val"], 0.)
    pensions_annuities = (oi_pensions + ret_pensions1 + ret_pensions2 +
                          annuities1 + annuities2 + oi_annuities)
    cps["alimony"] = alimony
    cps["pensions_annuities"] = pensions_annuities

    # flags used when creating a unit
    cps["p_flag"] = False  # primary flag
    cps["s_flag"] = False  # spouse flag
    cps["d_flag"] = False  # dependent flag
    cps["hhid"] = cps["h_seq"]
    # calculate earned and unearned income
    earned_inc_vars = [
        "wsal_val", "semp_val", "frse_val"
    ]
    unearned_inc_vars = [
        "int_val", "div_val", "rtm_val",
        "alimony", "uc_val"
    ]
    cps["earned_inc"] = cps[earned_inc_vars].sum(axis=1)
    cps["unearned_inc"] = cps[unearned_inc_vars].sum(axis=1)
    # add apply create_unit with a progress bar
    tqdm.pandas(desc=str(year))
    units = cps.groupby("h_seq").progress_apply(create_units, year=year - 1)
    TAX_UNITS = []
    for u in units:
        TAX_UNITS += u

    # create a DataFrame of tax units with the new
    tax_units_df = pd.DataFrame(TAX_UNITS)

    return tax_units_df


if __name__ == "__main__":
    print("Reading 2013 CPS")
    cps13 = pd.read_csv("data/cpsmar2013.csv")
    units13 = pycps(cps13, 2013)
    del cps13
    print("Reading 2014 CPS")
    cps14 = pd.read_csv("data/cpsmar2014.csv")
    units14 = pycps(cps14, 2014)
    del cps14
    print("Reading 2015 CPS")
    cps15 = pd.read_csv("data/cpsmar2015.csv")
    units15 = pycps(cps15, 2015)
    del cps15
    df = pd.concat([units13, units14, units15])
    df["s006"] = df["s006"] / 3
    df.to_csv("raw_cps.csv", index=False)
