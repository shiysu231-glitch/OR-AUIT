import numpy as np


def calculate_icc(cse, typ, data):
    data = np.array(data)
    # number of raters/ratings
    k = data.shape[1]
    # number of targets
    n = data.shape[0]
    # mean per target
    mpt = np.mean(data, axis=1)
    # mean per rater/rating
    mpr = np.mean(data, axis=0)
    # get total mean
    tm = np.mean(mpt)
    # within target sum sqrs
    WSS = np.sum(np.sum((data - np.expand_dims(mpt, axis=1)) ** 2, axis=1))
    # within target mean sqrs
    WMS = WSS / (n * (k - 1))
    # between rater sum sqrs
    RSS = np.sum((mpr - tm) ** 2) * n
    # between rater mean sqrs
    RMS = RSS / (k - 1)
    # get total sum sqrs
    TSS = np.sum(np.sum((data - tm) ** 2))
    # between target sum sqrs
    BSS = np.sum((mpt - tm) ** 2) * k
    # between targets mean squares
    BMS = BSS / (n - 1)
    # residual sum of squares
    ESS = WSS - RSS
    # residual mean sqrs
    EMS = ESS / ((k - 1) * (n - 1))

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
    out = abs(out)
    return out

# def iccaaa(Y, icc_type='ICC(3,1)'):
#
#     [n, k] = Y.shape
#
#     # Degrees of Freedom
#     dfc = k - 1
#     dfe = (n - 1) * (k-1)
#     dfr = n - 1
#
#     # Sum Square Total
#     mean_Y = np.mean(Y)
#     SST = ((Y - mean_Y) ** 2).sum()
#
#     # create the design matrix for the different levels
#     x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
#     x0 = np.tile(np.eye(n), (k, 1))  # subjects
#     X = np.hstack([x, x0])
#
#     # Sum Square Error
#     predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))),
#                                 X.T), Y.flatten('F'))
#     residuals = Y.flatten('F') - predicted_Y
#     SSE = (residuals ** 2).sum()
#
#     MSE = SSE / dfe
#
#     # Sum square column effect - between colums
#     SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
#     MSC = SSC / dfc  # / n (without n in SPSS results)
#
#     # Sum Square subject effect - between rows/subjects
#     SSR = SST - SSC - SSE
#     MSR = SSR / dfr
#
#     if icc_type == 'ICC(3,k)':
#         k = 1
#     # t
#     ICC = (MSR - MSE) / (MSR + (k-1) * MSE+1e-12)
#
#     return ICC