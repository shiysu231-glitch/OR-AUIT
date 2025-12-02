import numpy as np


def ICC(cse, typ, dat):
    k = dat.shape[1]  # number of raters/ratings
    n = dat.shape[0]  # number of targets

    mpt = np.mean(dat, axis=1)  # mean per target
    mpr = np.mean(dat, axis=0)  # mean per rater/rating
    tm = np.mean(mpt)  # total mean

    WSS = np.sum(np.sum((dat - mpt[:, np.newaxis]) ** 2))  # within target sum sqrs
    WMS = WSS / (n * (k - 1))  # within target mean sqrs

    RSS = np.sum((mpr - tm) ** 2) * n  # between rater sum sqrs
    RMS = RSS / (k - 1)  # between rater mean sqrs

    BSS = np.sum((mpt - tm) ** 2) * k  # between target sum sqrs
    BMS = BSS / (n - 1)  # between target mean squares

    ESS = WSS - RSS  # residual sum of squares
    EMS = ESS / ((k - 1) * (n - 1))  # residual mean sqrs

    if cse == 1:
        if typ == 'single':
            out = (BMS - WMS) / (BMS + (k - 1) * WMS)
        elif typ == 'k':
            out = (BMS - WMS) / BMS
        else:
            raise ValueError('Wrong value for input typ')
    elif cse == 2:
        if typ == 'single':
            out = (BMS - EMS) / (BMS + (k - 1) * EMS + k * (RMS - EMS) / n)
        elif typ == 'k':
            out = (BMS - EMS) / (BMS + (RMS - EMS) / n)
        else:
            raise ValueError('Wrong value for input typ')
    elif cse == 3:
        if typ == 'single':
            out = (BMS - EMS) / (BMS + (k - 1) * EMS)
        elif typ == 'k':
            out = (BMS - EMS) / BMS
        else:
            raise ValueError('Wrong value for input typ')
    else:
        raise ValueError('Wrong value for input cse')

    return out

# prePa = '0.21243589433530966pre.txt'  # 修改为你的文件路径
# acPa = 'act12.txt'  # 修改为你的文件路径
# a = np.loadtxt(prePa)
# b = np.loadtxt(acPa)
# iclist=[]
# for i in range(12):
#     a_first_column = a[:, i]
#     b_first_column = b[:, i]
#     c = np.column_stack((a_first_column, b_first_column))
#     icva=ICC(3,'single',c)
#     iclist.append(np.around(icva, decimals=3))
# print(iclist)
# print(np.mean(iclist))
