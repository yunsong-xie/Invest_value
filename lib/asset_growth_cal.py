import pandas as pd
import numpy as np

def get_after_tax_earn(income):
    if income <= 80800:
        after_tax_earn = income
    elif income <= 501600:
        after_tax_earn = 80800 + (income - 80800) * 0.85
    else:
        after_tax_earn = 80800 + (501600 - 80800) * 0.85 + (income - 501600) * 0.8
    return after_tax_earn


def get_margin_interest(fund, margin):
    margin_fund = fund / margin - fund
    margin_interest = 0
    if margin_fund <= 24999:
        margin_interest += margin_fund * 0.08325
        return margin_interest

    margin_interest += 24999 * 0.08325
    if margin_fund <= 49999:
        margin_interest += (margin_fund - 24999) * 0.07825
        return margin_interest

    margin_interest += (49999 - 24999) * 0.07825
    if margin_fund <= 99999:
        margin_interest += (margin_fund - 49999) * 0.06875
        return margin_interest

    margin_interest += (99999 - 49999) * 0.06875
    if margin_fund <= 249999:
        margin_interest += (margin_fund - 99999) * 0.06825
        return margin_interest

    margin_interest += (249999 - 99999) * 0.06825
    if margin_fund <= 499999:
        margin_interest += (margin_fund - 249999) * 0.06575
        return margin_interest

    margin_interest += (499999 - 249999) * 0.06575
    if margin_fund <= 999999:
        margin_interest += (margin_fund - 499999) * 0.04250
        return margin_interest

    margin_interest += (999999 - 499999) * 0.06825
    margin_interest += (margin_fund - 999999) * 0.04
    return margin_interest


margin = 0.75
earning_rate = 0.265
n_year_contribution_cutoff = 6
n_year = 15
fund = 122000
contribution = 50000
dict_sum = {'Fund': [], 'Earn': [], 'EPI': [], 'EPIT': [], 'EPITC': []}
for i_year in range(n_year):
    fund_ori = fund
    fund_post_earn = (fund / margin) * (1 + earning_rate) - ((fund / margin) - fund)

    margin_interest = get_margin_interest(fund, margin)
    fund_post_earn_interest = fund_post_earn - margin_interest
    if i_year >= n_year_contribution_cutoff:
        fund_post_earn_interest = fund_post_earn_interest - contribution
    fund_post_earn_interest_tax = fund + get_after_tax_earn(fund_post_earn_interest - fund)
    if i_year < n_year_contribution_cutoff:
        fund_post_earn_interest_tax_contribution = fund_post_earn_interest_tax + contribution
    else:
        fund_post_earn_interest_tax_contribution = fund_post_earn_interest_tax
    fund = fund_post_earn_interest_tax_contribution
    dict_sum['Fund'].append(round(fund / 10 ** 4))
    dict_sum['Earn'].append(round((fund_post_earn - fund_ori) / 10 ** 4))
    dict_sum['EPI'].append(round((fund_post_earn_interest - fund_ori) / 10 ** 4))
    dict_sum['EPIT'].append(round((fund_post_earn_interest_tax - fund_ori) / 10 ** 4))
    dict_sum['EPITC'].append(round((fund_post_earn_interest_tax_contribution - fund_ori) / 10 ** 4))
pd_sum = pd.DataFrame(dict_sum)
pd_sum.index = np.arange(len(pd_sum)) + 1
print(pd_sum)
