import numpy as np


def calMAE(x, y):
    x = np.array(x)
    y = np.array(y)
    z = abs(x-y)
    out = z/len(z)
    return np.sum(out)

def calcICC(dat):
    # dat = np.transpose(dat)
    k = np.size(dat, 1)
    n = np.size(dat, 0)
    mpt = np.mean(dat, axis=1)
    mpr = np.mean(dat, axis=0)
    tm = np.mean(mpt)
    #新加代码
    a = len(dat[1])
    mpt_matrix = np.tile(mpt, (a, 1)).T
    ws = np.sum(np.square(dat - mpt_matrix))
    #
    # ws = sum(np.square(dat-mpt))
    WSS = np.sum(ws)

    rs = np.square(mpr - tm)
    RSS = np.sum(rs) * n

    bs = np.square(mpt - tm)
    BSS = np.sum(bs) * k

    BMS = BSS / (n - 1)
    ESS = WSS - RSS
    EMS = ESS / ((k - 1) * (n - 1))
    icc = (BMS - EMS) / (BMS + (k - 1) * EMS)
    # if icc <= 0.0:
    #     icc = 0.0
    return icc
# def calcICC(dat):
#     k = np.size(dat, 1)
#     n = np.size(dat, 0)
#     mpt = np.mean(dat, 1)
#     mpr = np.mean(dat, 0)
#     tm = np.mean(mpt)
#
#     ws = sum(np.square(dat - mpt))
#     WSS = np.sum(ws)
#
#     rs = np.square(mpr - tm)
#     RSS = np.sum(rs) * n
#
#     bs = np.square(mpt - tm)
#     BSS = np.sum(bs) * k
#
#     BMS = BSS / (n - 1)
#     ESS = WSS - RSS
#     EMS = ESS / ((k - 1) * (n - 1))
#     icc = (BMS - EMS) / (BMS + (k - 1) * EMS)
#
#     return icc




# def calcPearson(x, y):
#     x_mean, y_mean = calcMean(x, y)
#     n = len(x)
#     sumTop = 0.0
#     sumBottom = 0.0
#     x_pow = 0.0
#     y_pow = 0.0
#
#     for i in range(n):
#         sumTop += (x[i] - x_mean) * (y[i]-y_mean)
#
#     for i in range(n):
#         x_pow += math.pow(x[i] - x_mean, 2)
#
#     for i in range(n):
#         y_pow += math.pow(y[i]-y_mean, 2)
#
#     sumBottom = math.sqrt(x_pow * y_pow)
#
#     p = sumTop/sumBottom
#     return p