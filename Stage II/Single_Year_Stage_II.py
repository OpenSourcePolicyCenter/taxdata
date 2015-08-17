
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray, CyLPModel

def Single_Year_Stage_II(puf, Stage_I_factors, Stage_II_targets, year, tol):


    length = len(puf.s006)


    print("Preparing coefficient matrix...")

    # the first half of this function (with all of the np.where statements)
    # might be amenable to refactoring to a numba function, could improve
    # readability

    s006 = np.where(puf.e02400>0,
                    puf.s006*Stage_I_factors[year]["APOPSNR"]/100,
                    puf.s006*Stage_I_factors[year]["ARETS"]/100)



    single_return = np.where(puf.mars==1, s006, 0)
    joint_return = np.where((puf.mars==2)|(puf.mars==3), s006, 0)
    hh_return = np.where(puf.mars==4,s006,0)
    return_w_SS = np.where(puf.e02400>0,s006,0)

    dependent_exempt_num = (puf.xocah+puf.xocawh+puf.xoodep+puf.xopar)*s006
    interest = puf.e00300*s006
    dividend = puf.e00600*s006
    biz_income = np.where(puf.e00900>0, puf.e00900, 0)*s006
    biz_loss = np.where(puf.e00900<0, -puf.e00900, 0)*s006
    cap_gain = np.where(puf.e01000>0, puf.e01000, 0)*s006
    annuity_pension = puf.e01700*s006
    sch_e_income = np.where(puf.e02000>0, puf.e02000, 0)*s006
    sch_e_loss = np.where(puf.e02000<0, -puf.e02000, 0)*s006
    ss_income = puf.e02400*s006
    unemployment_comp = puf.e02300*s006


    # Wage distribution

    wage_1 = np.where(puf.e00100<=0, puf.e00200,0)*s006
    wage_2 = np.where((puf.e00100>0)&(puf.e00100<=10000), puf.e00200,0)*s006
    wage_3 = np.where((puf.e00100>10000)&(puf.e00100<=20000), puf.e00200,0)*s006
    wage_4 = np.where((puf.e00100>20000)&(puf.e00100<=30000), puf.e00200,0)*s006
    wage_5 = np.where((puf.e00100>30000)&(puf.e00100<=40000), puf.e00200,0)*s006
    wage_6 = np.where((puf.e00100>40000)&(puf.e00100<=50000), puf.e00200,0)*s006
    wage_7 = np.where((puf.e00100>50000)&(puf.e00100<=75000), puf.e00200,0)*s006
    wage_8 = np.where((puf.e00100>75000)&(puf.e00100<=100000), puf.e00200,0)*s006
    wage_9 = np.where((puf.e00100>100000)&(puf.e00100<=200000), puf.e00200,0)*s006
    wage_10 = np.where((puf.e00100>200000)&(puf.e00100<=500000), puf.e00200,0)*s006
    wage_11 = np.where((puf.e00100>500000)&(puf.e00100<=1000000), puf.e00200,0)*s006
    wage_12 = np.where((puf.e00100>1000000), puf.e00200,0)*s006


    # Set up the matrix
    One_half_LHS = np.vstack((single_return, joint_return, hh_return, return_w_SS,
                              dependent_exempt_num, interest, dividend,
                              biz_income,biz_loss, cap_gain, annuity_pension,
                              sch_e_income, sch_e_loss, ss_income, unemployment_comp,
                              wage_1, wage_2, wage_3, wage_4, wage_5, wage_6,
                              wage_7, wage_8, wage_9, wage_10, wage_11, wage_12))


    # Coefficients for r and s
    A1 = np.matrix(One_half_LHS)
    A2 = np.matrix(-One_half_LHS)


    print("Preparing targets for ", year)

    APOPN = Stage_I_factors[year]["APOPN"]

    b = []

    b.append(Stage_II_targets[year]['Single']-single_return.sum())
    b.append(Stage_II_targets[year]['Joint']-joint_return.sum())
    b.append(Stage_II_targets[year]['HH']-hh_return.sum())
    b.append(Stage_II_targets[year]['SS_return']-return_w_SS.sum())

    b.append(Stage_II_targets[year]['Dep_return'] -  dependent_exempt_num.sum())

    AINTS = Stage_I_factors[year]["AINTS"]
    INTEREST = Stage_II_targets[year]['INTS']*APOPN/AINTS*1000-interest.sum()

    ADIVS = Stage_I_factors[year]["ADIVS"]
    DIVIDEND = Stage_II_targets[year]['DIVS']*APOPN/ADIVS*1000 - dividend.sum()

    ASCHCI = Stage_I_factors[year]["ASCHCI"]
    BIZ_INCOME = Stage_II_targets[year]['SCHCI']*APOPN/ASCHCI*1000 - biz_income.sum()


    ASCHCL = Stage_I_factors[year]["ASCHCL"]
    BIZ_LOSS = Stage_II_targets[year]['SCHCL']*APOPN/ASCHCL*1000 - biz_loss.sum()

    ACGNS = Stage_I_factors[year]["ACGNS"]
    CAP_GAIN = Stage_II_targets[year]['CGNS']*APOPN/ACGNS*1000 - cap_gain.sum()

    ATXPY = Stage_I_factors[year]["ATXPY"]
    ANNUITY_PENSION = Stage_II_targets[year]['Pension']*APOPN/ATXPY*1000 - annuity_pension.sum()

    ASCHEI = Stage_I_factors[year]["ASCHEI"]
    SCH_E_INCOME = Stage_II_targets[year]["SCHEI"]*APOPN/ASCHEI*1000 - sch_e_income.sum()

    ASCHEL = Stage_I_factors[year]["ASCHEL"]
    SCH_E_LOSS = Stage_II_targets[year]["SCHEL"]*APOPN/ASCHEL*1000 - sch_e_loss.sum()

    ASOCSEC = Stage_I_factors[year]["ASOCSEC"]
    APOPSNR = Stage_I_factors[year]["APOPSNR"]
    SS_INCOME = Stage_II_targets[year]["SS"]*APOPSNR/ASOCSEC*1000 - ss_income.sum()

    AUCOMP = Stage_I_factors[year]["AUCOMP"]
    UNEMPLOYMENT_COMP = Stage_II_targets[year]["UCOMP"]*APOPN/AUCOMP*1000 - unemployment_comp.sum()

    AWAGE = Stage_I_factors[year]["AWAGE"]
    WAGE_1 = Stage_II_targets[year]["WAGE_1"]*APOPN/AWAGE*1000 - wage_1.sum()
    WAGE_2 = Stage_II_targets[year]["WAGE_2"]*APOPN/AWAGE*1000 - wage_2.sum()
    WAGE_3 = Stage_II_targets[year]["WAGE_3"]*APOPN/AWAGE*1000 - wage_3.sum()
    WAGE_4 = Stage_II_targets[year]["WAGE_4"]*APOPN/AWAGE*1000 - wage_4.sum()
    WAGE_5 = Stage_II_targets[year]["WAGE_5"]*APOPN/AWAGE*1000 - wage_5.sum()
    WAGE_6 = Stage_II_targets[year]["WAGE_6"]*APOPN/AWAGE*1000 - wage_6.sum()
    WAGE_7 = Stage_II_targets[year]["WAGE_7"]*APOPN/AWAGE*1000 - wage_7.sum()
    WAGE_8 = Stage_II_targets[year]["WAGE_8"]*APOPN/AWAGE*1000 - wage_8.sum()
    WAGE_9 = Stage_II_targets[year]["WAGE_9"]*APOPN/AWAGE*1000 - wage_9.sum()
    WAGE_10 = Stage_II_targets[year]["WAGE_10"]*APOPN/AWAGE*1000 - wage_10.sum()
    WAGE_11 = Stage_II_targets[year]["WAGE_11"]*APOPN/AWAGE*1000 - wage_11.sum()
    WAGE_12 = Stage_II_targets[year]["WAGE_12"]*APOPN/AWAGE*1000 - wage_12.sum()



    temp = [INTEREST,DIVIDEND, BIZ_INCOME, BIZ_LOSS, CAP_GAIN, ANNUITY_PENSION, SCH_E_INCOME, SCH_E_LOSS, SS_INCOME, UNEMPLOYMENT_COMP,
            WAGE_1,WAGE_2, WAGE_3,WAGE_4, WAGE_5, WAGE_6, WAGE_7,WAGE_8,WAGE_9, WAGE_10, WAGE_11, WAGE_12]

    # how is 'b' different from 'temp'?
    # could also do  'b = list(temp)'
    for m in temp:
        b.append(m)

    targets = CyLPArray(b)
    print("Targets for year ", year, " is ", targets)

    LP = CyLPModel()

    r = LP.addVariable('r', length)
    s = LP.addVariable('s', length)

    print("Adding constraints")
    LP.addConstraint(r >=0, "positive r")
    LP.addConstraint(s >=0, "positive s")
    LP.addConstraint(r + s <= tol, "abs upperbound")

    c = CyLPArray((np.ones(length)))
    LP.objective = c * r + c * s




    LP.addConstraint(A1 * r + A2 * s == targets, "Aggregates")

    print("Setting up the LP model")
    model = CyClpSimplex(LP)


    print("Solving LP......")
    model.initialSolve()

    print("DONE!!")
    z = np.empty([length])
    z = (1+model.primalVariableSolution['r'] - model.primalVariableSolution['s'])*s006
    return z



