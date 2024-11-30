import pandas as pd
import pickle
from collections import OrderedDict
from tqdm import tqdm
from pathlib import Path
from .helpers import read_benefits


CUR_PATH = Path(__file__).resolve().parent
DATA_PATH = Path(CUR_PATH, "data")


def person_details(record, benefits, h_seq, fhseq, ffpos, year):
    """
    Add additonal details for person records
    """
    if year >= 2015:
        record["alimony"] = 0.0
        if record["oi_off"] == 20:
            record["alimony"] = record["oi_val"]
    else:
        record["alimony"] = record["alm_val"]
    # Calculate pensions and annuities
    pensions_annuities = (
        ((record["oi_off"] == 2) * record["oi_val"])
        + ((record["ret_sc1"] == 1) * record["ret_val1"])
        + ((record["ret_sc2"] == 1) * record["ret_val2"])
        + ((record["ret_sc1"] == 7) * record["ret_val1"])
        + ((record["ret_sc2"] == 7) * record["ret_val2"])
        + ((record["oi_off"] == 13) * record["oi_val"])
    )
    record["pensions_annuities"] = pensions_annuities
    # flags used in tax unit creation
    record["p_flag"] = False
    record["s_flag"] = False
    record["d_flag"] = False
    record["hhid"] = h_seq

    # calculate earned and unearned income
    EARNED_INC_VARS = ["wsal_val", "semp_val", "frse_val", "rnt_val"]
    UNEARNED_INC_VARS = ["int_val", "div_val", "rtm_val", "alimony", "uc_val", "ss_val"]
    record["earned_inc"] = sum([record[var] for var in EARNED_INC_VARS])
    record["unearned_inc"] = sum([record[var] for var in UNEARNED_INC_VARS])
    record["tot_inc"] = record["earned_inc"] + record["unearned_inc"]

    if benefits:
        # add benefit variables from the CPS
        global MCAID, MCARE, VB, SNAP, SSI, SS, HOUSING, TANF, UI, WIC
        perid_ben = [
            (MCAID, "MedicaidX"),
            (MCARE, "MedicareX"),
            (VB, "vb_impute"),
            (SSI, "ssi_impute"),
            (SS, "ss_impute"),
            (TANF, "tanf_impute"),
            (UI, "UI_impute"),
            (WIC, "wic_impute"),
        ]
        for data, var_name in perid_ben:
            # C-TAM data only includes those who receive benefits so catch
            # error for those that do not receive them
            try:
                record[var_name] = data[str(record["peridnum"])][var_name]
            except KeyError:
                record[var_name] = 0.0
        record["housing_impute"] = HOUSING[f"{fhseq}{ffpos}"]["housing_impute"]
        # C-TAM SNAP imputations only contain records for households receiving
        # benefits. Catch the error for those that don't.
        try:
            record["snap_impute"] = SNAP[h_seq]["snap_impute"]
        except KeyError:
            record["snap_impute"] = 0.0
        # replace values of unemployment and social security from original CPS
        record["unearned_inc"] -= record["ss_val"]
        record["unearned_inc"] -= record["uc_val"]
        record["unearned_inc"] += record["UI_impute"]
        record["unearned_inc"] += record["ss_impute"]
        record["tot_inc"] -= record["ss_val"]
        record["tot_inc"] -= record["uc_val"]
        record["tot_inc"] += record["UI_impute"]
        record["tot_inc"] += record["ss_impute"]
    else:
        # calculate benefits in CPS where possible
        record["tanf_val"] = 0.0
        if record["paw_yn"] == 1:
            record["tanf_val"] = record["paw_val"]
        if year >= 2016:
            record["housing_val"] = 0.0
        else:
            record["housing_val"] = record["fhoussub"]
    return record


def parse(rec, parse_dict):
    """
    Function for parsing lines of the CPS
    """
    record = OrderedDict()

    for var in parse_dict.keys():
        start, end, decimals = parse_dict[var]
        value = int(rec[start:end])
        if decimals != 0:
            value /= int("1" + ("0" * decimals))
        record[var] = value

    return record


def create_cps(
    dat_file,
    year,
    parsing_dict,
    benefits=True,
    exportpkl=True,
    exportcsv=True,
    datapath=None,
):
    """
    Read the .DAT CPS file and convert it to a list of dictionaries that
    to later be converted to tax units. Optionally export that list as a
    pickle file or export the full CPS as a CSV
    Parameters
    ----------
    dat_file: Path to the .DAT version of the CPS downloaded from NBER
    year: year of the CPS being converted
    parsing_dict: dictionary with information
    benefits: Set to true to include C-TAM imputed benefits in the CPS
    exportpkl: Set to true to export a pickled list of households in the CPS
    exportcsv: Set to true to export a CSV version of the CPS
    datapath: base export path
    """
    if (exportpkl or exportcsv) and not datapath:
        msg = "A value for `datapath` must be specified when `exportpkl` or `exportcsv` is true"
        raise ValueError(msg)
    # read in file
    print("Reading DAT file")
    with Path(dat_file).open("r") as f:
        cps = [line.strip() for line in f.readlines()]

    # Read in benefits
    if benefits:
        global MCAID, MCARE, VB, SNAP, SSI, SS, HOUSING, TANF, UI, WIC
        ben = read_benefits(year)
        MCAID, MCARE, VB, SNAP, SSI, SS, HOUSING, TANF, UI, WIC = ben

    record_list = []  # list to hold individual records
    cps_list = []  # list to hold list of households
    household = []  # list to hold members of household
    print("Creating Records")
    for record in tqdm(cps, desc=str(year)):
        rec_type = record[0]
        # household records
        if rec_type == "1":
            if household:
                cps_list.append(household)
                household = []
            house = parse(record, parsing_dict["household"])
        # family record
        elif rec_type == "2":
            family = parse(record, parsing_dict["family"])
        # person record
        elif rec_type == "3":
            person = parse(record, parsing_dict["person"])
            # add housing subsidy to person record because it's needed in person_details
            if year < 2016:
                person['fhoussub'] = family['fhoussub']
            person = person_details(
                person,
                benefits,
                house["h_seq"],
                family["fh_seq"],
                family["ffpos"],
                year,
            )
            full_rec = {**house, **family, **person}
            household.append(full_rec)
            if exportcsv:
                record_list.append(full_rec)

    # add the last household to the cps list
    cps_list.append(household)

    if exportcsv:
        print("Converting to DataFrame")
        cpsmar = pd.DataFrame(record_list).fillna(0)
        print("Exporting CSV")
        export_path = Path(datapath, f"cpsmar{year}.csv")
        cpsmar.to_csv(export_path, index=False)

    if exportpkl:
        print("Pickling File")
        with Path(datapath, f"cpsmar{year}.pkl").open("wb") as f:
            pickle.dump(cps_list, f)

    return cps_list
