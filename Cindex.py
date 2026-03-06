import math 
import numpy as np

from scipy import stats
# 定义sign函数，等同于C代码中Rmath.h的sign函数
def sign(x):
    """
    计算x的符号
    参数:
        x (float): 输入值
    返回:
        int: 1 (x>0), -1 (x<0), 0 (x=0)
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

# 定义csign函数，等同于C代码中的csign函数
def csign(X_i, Censor_i, X_j, Censor_j):
    """
    根据生存时间和删失状态比较两个样本
    参数:
        X_i (float): 第i个样本的生存时间
        Censor_i (int): 第i个样本的删失状态 (1: 事件发生, 0: 删失)
        X_j (float): 第j个样本的生存时间
        Censor_j (int): 第j个样本的删失状态
    返回:
        int: 比较结果 (-1, 0, 1)
    """
    if X_i > X_j:
        if Censor_i == 1 and Censor_j == 1:
            return 1
        elif Censor_i == 1 and Censor_j == 0:
            return 0
        elif Censor_i == 0 and Censor_j == 1:
            return 1
        elif Censor_i == 0 and Censor_j == 0:
            return 0
    elif X_i < X_j:
        if Censor_i == 1 and Censor_j == 1:
            return -1
        elif Censor_i == 1 and Censor_j == 0:
            return -1
        elif Censor_i == 0 and Censor_j == 1:
            return 0
        elif Censor_i == 0 and Censor_j == 0:
            return 0
    else:  # X_i == X_j
        if Censor_i == 1 and Censor_j == 1:
            return 0
        elif Censor_i == 1 and Censor_j == 0:
            return -1
        elif Censor_i == 0 and Censor_j == 1:
            return 1
        elif Censor_i == 0 and Censor_j == 0:
            return 0
    return 0  # 默认返回0，对应C代码中的return 0

# 定义TauXX函数，计算生存时间自身的一致性统计量
def TauXX(timeX, statusX, nn):
    """
    计算TauXX统计量（生存时间自身的一致性）
    参数:
        timeX (list of float): 生存时间数组
        statusX (list of int): 删失状态数组 (1: 事件, 0: 删失)
        nn (int): 样本大小
    返回:
        float: TauXX统计量
    """
    nobs = nn
    est_tXX = 0.0
    for i in range(nobs):
        for j in range(nobs):
            if j == i:
                continue
            sign_val = csign(timeX[i], statusX[i], timeX[j], statusX[j])
            est_tXX += sign_val * sign_val  # 平方和
    output = est_tXX / nobs / (nobs - 1)
    return output

# 定义TauXY函数，计算生存时间与连续变量scoreY的一致性统计量
def TauXY(timeX, statusX, scoreY, nn):
    """
    计算TauXY统计量（生存时间与连续变量的一致性）
    参数:
        timeX (list of float): 生存时间数组
        statusX (list of int): 删失状态数组
        scoreY (list of float): 连续变量数组
        nn (int): 样本大小
    返回:
        float: TauXY统计量
    """
    nobs = nn
    est_tXY = 0.0
    for i in range(nobs):
        for j in range(nobs):
            if j == i:
                continue
            sign_val_x = csign(timeX[i], statusX[i], timeX[j], statusX[j])
            sign_val_y = sign(scoreY[i] - scoreY[j])
            est_tXY += sign_val_x * sign_val_y
    output = est_tXY / nobs / (nobs - 1)
    return output

# 定义VarTauXX函数，计算TauXX的方差
def VarTauXX(timeX, statusX, nn):
    """
    计算TauXX统计量的方差
    参数:
        timeX (list of float): 生存时间数组
        statusX (list of int): 删失状态数组
        nn (int): 样本大小
    返回:
        float: TauXX的方差
    """
    nobs = nn
    temp_s1 = 0.0
    temp_s2 = 0.0
    temp_s3 = 0.0

    for i in range(nobs):
        temp_s1_j = 0.0
        temp_s3_j = 0.0
        for j in range(nobs):
            if j == i:
                continue
            temp = csign(timeX[i], statusX[i], timeX[j], statusX[j])
            temp = temp * temp  # 平方
            temp_s1_j += temp
            temp_s3_j += temp * temp  # 四次方
        temp_s1 += 4 * temp_s1_j * temp_s1_j
        temp_s3 += 2 * temp_s3_j
        temp_s2 += temp_s1_j

    temp_s2 = (temp_s2 / nobs / (nobs - 1)) * temp_s2 * (2 * nobs - 3) * (-2)
    output = (temp_s1 - temp_s3 + temp_s2) / nobs / (nobs - 1) / (nobs - 2) / (nobs - 3)
    return output

# 定义VarTauXY函数，计算TauXY的方差
def VarTauXY(timeX, statusX, scoreY, nn):
    """
    计算TauXY统计量的方差
    参数:
        timeX (list of float): 生存时间数组
        statusX (list of int): 删失状态数组
        scoreY (list of float): 连续变量数组
        nn (int): 样本大小
    返回:
        float: TauXY的方差
    """
    nobs = nn
    temp_s1 = 0.0
    temp_s2 = 0.0
    temp_s3 = 0.0

    for i in range(nobs):
        temp_s1_j = 0.0
        temp_s3_j = 0.0
        for j in range(nobs):
            if j == i:
                continue
            temp = csign(timeX[i], statusX[i], timeX[j], statusX[j]) * sign(scoreY[i] - scoreY[j])
            temp_s1_j += temp
            temp_s3_j += temp * temp
        temp_s1 += 4 * temp_s1_j * temp_s1_j
        temp_s3 += 2 * temp_s3_j
        temp_s2 += temp_s1_j

    temp_s2 = (temp_s2 / nobs / (nobs - 1)) * temp_s2 * (2 * nobs - 3) * (-2)
    output = (temp_s1 - temp_s3 + temp_s2) / nobs / (nobs - 1) / (nobs - 2) / (nobs - 3)
    return output

# 定义CovTauXXXY函数，计算TauXX和TauXY的协方差
def CovTauXXXY(timeX, statusX, scoreY, nn):
    """
    计算TauXX和TauXY统计量的协方差
    参数:
        timeX (list of float): 生存时间数组
        statusX (list of int): 删失状态数组
        scoreY (list of float): 连续变量数组
        nn (int): 样本大小
    返回:
        float: TauXX和TauXY的协方差
    """
    nobs = nn
    temp_s1 = 0.0
    temp_s2 = 0.0
    temp_s3 = 0.0
    temp_s4 = 0.0

    for i in range(nobs):
        temp_sXX_j = 0.0
        temp_sXY_j = 0.0
        temp_sXXXY_j = 0.0
        for j in range(nobs):
            if j == i:
                continue
            temp_XX = csign(timeX[i], statusX[i], timeX[j], statusX[j])
            temp_XX = temp_XX * temp_XX  # 平方
            temp_XY = csign(timeX[i], statusX[i], timeX[j], statusX[j]) * sign(scoreY[i] - scoreY[j])
            temp_sXX_j += temp_XX
            temp_sXY_j += temp_XY
            temp_sXXXY_j += temp_XX * temp_XY
        temp_s1 += 4 * temp_sXX_j * temp_sXY_j
        temp_s3 += 2 * temp_sXXXY_j
        temp_s2 += temp_sXX_j
        temp_s4 += temp_sXY_j

    output = (temp_s1 - temp_s3 + (2 * nobs - 3) * (-2) * temp_s2 * temp_s4 / nobs / (nobs - 1)) / nobs / (nobs - 1) / (nobs - 2) / (nobs - 3)
    return output

# 定义CovTauXYXZ函数，计算TauXY和TauXZ的协方差（针对两个不同连续变量）
def CovTauXYXZ(timeX, statusX, scoreY, scoreZ, nn):
    """
    计算TauXY和TauXZ统计量的协方差（针对两个不同连续变量）
    参数:
        timeX (list of float): 生存时间数组
        statusX (list of int): 删失状态数组
        scoreY (list of float): 第一个连续变量数组
        scoreZ (list of float): 第二个连续变量数组
        nn (int): 样本大小
    返回:
        float: TauXY和TauXZ的协方差
    """
    nobs = nn
    temp_s1 = 0.0
    temp_s2 = 0.0
    temp_s3 = 0.0
    temp_s4 = 0.0

    for i in range(nobs):
        temp_sXY_j = 0.0
        temp_sXZ_j = 0.0
        temp_sXYXZ_j = 0.0
        for j in range(nobs):
            if j == i:
                continue
            temp_XY = csign(timeX[i], statusX[i], timeX[j], statusX[j]) * sign(scoreY[i] - scoreY[j])
            temp_XZ = csign(timeX[i], statusX[i], timeX[j], statusX[j]) * sign(scoreZ[i] - scoreZ[j])
            temp_sXY_j += temp_XY
            temp_sXZ_j += temp_XZ
            temp_sXYXZ_j += temp_XY * temp_XZ
        temp_s1 += 4 * temp_sXY_j * temp_sXZ_j
        temp_s3 += 2 * temp_sXYXZ_j
        temp_s2 += temp_sXY_j
        temp_s4 += temp_sXZ_j

    output = (temp_s1 - temp_s3 + (2 * nobs - 3) * (-2) * temp_s2 * temp_s4 / nobs / (nobs - 1)) / nobs / (nobs - 1) / (nobs - 2) / (nobs - 3)
    return output


def estC(timeX,statusX,scoreY):
    Tau_XX=TauXX(timeX,statusX,len(timeX))
    Tau_XY=TauXY(timeX,statusX,scoreY,len(timeX))

    rel=(Tau_XY/Tau_XX+1)/2
    return rel

def vardiffC(timeX,statusX,scoreY,scoreZ):
    t11 = TauXX(timeX, statusX,len(timeX))
    t12 = TauXY(timeX, statusX, scoreY,len(timeX))
    t13 = TauXY(timeX, statusX, scoreZ,len(timeX))
    var_t11 = VarTauXX(timeX, statusX,len(timeX))
    var_t12 = VarTauXY(timeX, statusX, scoreY,len(timeX))
    var_t13 = VarTauXY(timeX, statusX, scoreZ,len(timeX))
    cov_t1112 = CovTauXXXY(timeX, statusX, scoreY,len(timeX))
    cov_t1113 = CovTauXXXY(timeX, statusX, scoreZ,len(timeX))
    cov_t1213 = CovTauXYXZ(timeX, statusX, scoreY, scoreZ,len(timeX))
    # print(t11,t12,t13,var_t11,var_t12,var_t13,cov_t1112,cov_t1113,cov_t1213)

    est_varCxy = 0.25 * np.array([1/t11, -t12/t11**2]) @ np.array([[var_t12, 
        cov_t1112],[cov_t1112, var_t11]]) @ np.array([1/t11, -t12/t11**2])
    
    est_varCxz = 1/4 * np.array([1/t11, -t13/t11**2]) @ np.array([[var_t13, 
        cov_t1113],[cov_t1113, var_t11]]) @ np.array([1/t11, -t13/t11**2])


    est_cov=1/4 * np.array([1/t11, -t12/t11**2]) @ np.array([[cov_t1213, 
        cov_t1113],[cov_t1112, var_t11]]) @ np.array([1/t11, -t13/t11**2])
    


    est_vardiff_c = est_varCxy + est_varCxz - 2 * est_cov

    rel={"est_vardiff_c":est_vardiff_c,"est_varCxy":est_varCxy,"est_varCxz":est_varCxz,"est_cov":est_cov}
    return rel


    
def compareC(timeX,statusX,scoreY,scoreZ):
    timeX=np.array(timeX)
    statusX=np.array(statusX)
    scoreY=np.array(scoreY)
    scoreZ=np.array(scoreZ)
    estY=estC(timeX,statusX,scoreY)
    estZ=estC(timeX,statusX,scoreZ)
    est_diffc=estY-estZ
    tmpout=vardiffC(timeX,statusX,scoreY,scoreZ)
    zscore=est_diffc/np.sqrt(tmpout["est_vardiff_c"])
    pval= 2 * norm.sf(abs(zscore))

    rel={
        "cindex1":estY,
        "cindex2":estZ,
        "cindex_diff":est_diffc,
        "cindex_diff_var":tmpout["est_vardiff_c"],
        "cindex_cov11":tmpout["est_varCxy"],
        "cindex_cov22":tmpout["est_varCxz"],
        "cindex_cov12":tmpout["est_cov"],
        "zcore":zscore,
        "pval":pval
    }
    return rel

def computeC(timeX,statusX,scoreY,ci=95):
    timeX=np.array(timeX)
    statusX=np.array(statusX)
    scoreY=np.array(scoreY) 
    scoreZ=np.array(scoreY)
    tmpout=vardiffC(timeX,statusX,scoreY,scoreZ)
    estY=estC(timeX,statusX,scoreY)
    var=tmpout["est_varCxy"]
    z_value = stats.norm.ppf(1 - (1 - ci/100)/2)
    rel={"cindex": estY,
         "cindex_lower":max(0,estY-z_value*var**0.5),
         "cindex_upper":min(estY+z_value*var**0.5,1)}
    return rel