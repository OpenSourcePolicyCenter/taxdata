import pandas as pd
from operator import itemgetter
from tqdm import tqdm
from .taxunit import TaxUnit
from .helpers import filingparams, cps_yr_idx
from .cps_meta import C_TAM_YEARS


INCOME_VARS = [
    "wsal_val",
    "int_val",
    "div_val",
    "semp_val",
    "alimony",
    "pensions_annuities",
    "frse_val",
    "uc_val",
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
    msg = f"Person with line number {lineno} not found in" f"household."
    raise ValueError(msg)


def eic_eligible(person: dict, age_head: int, age_spouse: int, mars: int) -> int:
    """
    Function to determine if a dependent is an EIC eligible child
    df: DataFrame of just dependents
    https://www.irs.gov/credits-deductions/individuals/earned-income-tax-credit/qualifying-child-rules
    """
    # relationship test
    relationship = person["a_exprrp"] in [5, 7, 9, 11]
    # age test
    eic_max_age = filingparams.eic_child_age[cps_yr_idx]
    if person["a_ftpt"] == 1:
        eic_max_age = filingparams.eic_child_age_student[cps_yr_idx]
    # person is between 1 and the max age
    age = 0 <= person["a_age"] <= eic_max_age
    # person is younger than filer or their spouse
    if mars == 1:
        age *= person["a_age"] < age_head
    elif mars == 2:
        age *= person["a_age"] < age_head or person["a_age"] < age_spouse
    # assume they pass the residency test
    # assume they pass joint filer test for now

    # EIC eligible if they pass both tests
    eligible = int(relationship and age)

    # make sure eligible is 1 or 0
    assert eligible == 1 or eligible == 0

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
    spouse_dep = claimer["a_spouse"] == claimerno | claimer["dep_stat"] == claimerno
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


def is_dependent(person, unit, verbose=False):
    # if they're a spouse, tax unit, or claimed, return false
    flagged = person["p_flag"] or person["s_flag"] or person["d_flag"]
    # only dependents can't have dependents
    if unit.dep_stat == 1:
        return False
    if flagged:
        return False
    # if they're already flagged as a dependent, give them to the person
    # if they're married, don't let them be claimed as a dependent
    if person["a_maritl"] in [1, 2, 3]:
        return False
    if person["dep_stat"] == unit.a_lineno:
        return True
    # if they were not claimed, don't let them be
    if person["dep_stat"] == 0:
        return False
    # qualifying child
    if person["a_parent"] == unit.a_lineno:
        if verbose:
            print("Determining if person is a dependent child")
        # only person claiming them
        if person["d_flag"]:
            if verbose:
                print("Already claimed as a dependent")
            return False
        age_req = filingparams.dependent_child_age[cps_yr_idx]
        # age requirement increases for full time students
        if person["a_ftpt"] == 1:
            age_req = filingparams.dependent_child_age_student[cps_yr_idx]
        if person["a_age"] > age_req:
            if verbose:
                print("Failed Age Test")
            return False
        if person["a_age"] > unit.age_head:
            if verbose:
                print("Failed Age Test")
            return False
        # assume that they do live with you
        # financial support
        total_support = unit.tot_inc + person["tot_inc"]
        try:
            pct_support = person["tot_inc"] / total_support
        except ZeroDivisionError:
            pct_support = 0.0
        if pct_support > 0.5:
            if verbose:
                print("Failed Income Support Test")
            return False
    # qualifying relative
    else:
        if verbose:
            print("Determining if person is a qualifying relative")
        # assume they live with you
        # income test
        if person["ptotval"] > 4150:  # TODO: Add this to filing rules JSON
            if verbose:
                print("Failed Income Test")
            return False
        total_support = person["tot_inc"] + unit.tot_inc
        try:
            pct_support = person["tot_inc"] / total_support
        except ZeroDivisionError:
            pct_support = 0.0
        if pct_support > 0.5:
            if verbose:
                print("Failed Income Test")
            return False
        # qualifying relationship
        if person["a_exprrp"] not in [5, 7, 8, 9, 11]:
            if verbose:
                print("Failed Relationship Test")
            return False
        # only person claiming them
        if person["d_flag"]:
            return False
    # if you get to this point, they qualify as a dependent
    return True


def create_units(data, year, verbose=False, ctam_benefits=True):
    """
    Logic for iterating through households and creating tax units
    """
    # sort on familly position and family relationship so that
    # we can avoid making children their own units just because
    # they're listed ahead of their parents
    data = sorted(data, key=itemgetter("ffpos", "a_famrel"))
    hh_inc = 0  # sum up total income in the household. Used for HH status
    for person in data:
        hh_inc += person["tot_inc"]
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
            tu = TaxUnit(person, year, hh_inc, ctam_benefits=ctam_benefits)
            # loop through the rest of the household for
            # spouses and dependents
            if person["a_spouse"] != 0:
                spouse = find_person(data, person["a_spouse"])
                if verbose:
                    print("adding spouse", spouse["a_lineno"])
                tu.add_spouse(spouse)
            for _person in data:
                if verbose:
                    print("Searching for dependents")
                # only allow dependents in the same family
                if person["ffpos"] == _person["ffpos"]:
                    is_dep = is_dependent(_person, tu)
                    if is_dep:
                        if verbose:
                            print("adding dependent", _person["a_lineno"])
                        _eic = eic_eligible(
                            _person, tu.age_head, tu.age_spouse, tu.mars
                        )
                        tu.add_dependent(_person, _eic)
                        dependents.append(_person)
            units[person["a_lineno"]] = tu

    # check and see if any dependents must file
    # https://turbotax.intuit.com/tax-tips/family/should-i-include-a-dependents-income-on-my-tax-return/L60Hf4Rsg
    for person in dependents:
        filer = False
        midx = 0  # marital staus indicator
        aidx = 0  # age indicator
        if person["a_age"] >= filingparams.elderly_age[cps_yr_idx]:
            aidx = 1
        if person["a_spouse"] != 0:
            midx = 2
            spouse = find_person(data, person["a_spouse"])
            if spouse["a_age"] >= filingparams.elderly_age[cps_yr_idx]:
                aidx += 1
        # earned income filing threshold
        earn_thd = filingparams.dep_earned_inc_thd[cps_yr_idx][aidx][midx]
        unearn_thd = filingparams.dep_unearned_inc_thd[cps_yr_idx][aidx][midx]
        gross_thd = filingparams.dep_gross_inc_thd[cps_yr_idx][aidx][midx]
        if person["earned_inc"] >= earn_thd:
            filer = True
        elif person["unearned_inc"] >= unearn_thd:
            filer = True
        # TODO: set gross income threshold
        elif person["ptotval"] >= max(gross_thd, person["earned_inc"] + 350):
            filer = True
        # if a dependent says they didn't file, we'll believe them
        if person["filestat"] == 6:
            filer = False
        if filer:
            if verbose:
                print("dep filer", person["a_lineno"])
            tu = TaxUnit(person, year, dep_status=True, ctam_benefits=ctam_benefits)
            # remove dependent from person claiming them
            units[person["claimer"]].remove_dependent(person)
            if verbose:
                print(units[person["claimer"]].n24)
            units[person["a_lineno"]] = tu

    return [unit.output() for unit in units.values()]


def _create_units(data, year, verbose=False, ctam_benefits=False):
    """
    Logic for iterating through households and creating tax units
    This is a more complex function for creating the tax units. It actively
    searches through the household trying to form the tax unit, rather than
    rely on just what the CPS reports. It's unused, but I'm leaving the code
    here just in case it's needed at a later date.
    """
    # sort on familly position and family relationship so that
    # we can avoid making children their own units just because
    # they're listed ahead of their parents
    data = sorted(data, key=itemgetter("ffpos", "a_famrel"))
    hh_inc = 0  # sum up total income in the household. Used for HH status
    for person in data:
        hh_inc += person["tot_inc"]
    units = {}
    dependents = []
    for person in data:
        # if they're not a dependent or already claimed as a spouse, unit!
        if person["dep_stat"] == 0 and not person["s_flag"]:
            # make them a tax unit
            if verbose:
                print("making unit", person["a_lineno"])
            person["p_flag"] = True
            tu = TaxUnit(person, year, hh_inc, ctam_benefits=ctam_benefits)
            # loop through the rest of the household for
            # spouses and dependents
            if person["a_spouse"] != 0:
                spouse = find_person(data, person["a_spouse"])
                if verbose:
                    print("adding spouse", spouse["a_lineno"])
                tu.add_spouse(spouse)
            for _person in data:
                if verbose:
                    print("Searching for dependents")
                # only allow dependents in the same family
                if person["a_lineno"] == _person["dep_stat"]:
                    if verbose:
                        print("adding dependent", _person["a_lineno"])
                    _eic = eic_eligible(_person, tu.age_head, tu.age_spouse, tu.mars)
                    tu.add_dependent(_person, _eic)
                    dependents.append(_person)
            units[person["a_lineno"]] = tu

    # check and see if any dependents must file
    # https://turbotax.intuit.com/tax-tips/family/should-i-include-a-dependents-income-on-my-tax-return/L60Hf4Rsg
    for person in dependents:
        if person["filestat"] != 6:
            if verbose:
                print("dep filer", person["a_lineno"])
            tu = TaxUnit(person, year, dep_status=True, ctam_benefits=ctam_benefits)
            # remove dependent from person claiming them
            units[person["claimer"]].remove_dependent(person)
            if verbose:
                print(units[person["claimer"]].n24)
            units[person["a_lineno"]] = tu

    return [unit.output() for unit in units.values()]


def pycps(cps: list, year: int, ctam_benefits: bool, verbose: bool) -> pd.DataFrame:
    """
    Core code for iterating through the households
    Parameters
    ----------
    cps: List where each element is a household in the CPS
    year: CPS year to use
    ctam_benefits: If true, attach C-TAM benefits to the CPS
    verbose
    """
    tax_units = []
    if year not in C_TAM_YEARS and ctam_benefits:
        raise ValueError(f'C-TAM Benefits not available for year {year}')
    for hh in tqdm(cps):
        tax_units += create_units(hh, year - 1, ctam_benefits=ctam_benefits, verbose=verbose)
    # create a DataFrame of tax units with the new
    tax_units_df = pd.DataFrame(tax_units)

    return tax_units_df
