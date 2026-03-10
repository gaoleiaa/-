from copy import deepcopy
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter 
from lifelines.statistics import multivariate_logrank_test
from lifelines.fitters.coxph_fitter import CoxPHFitter

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
from sklearn.calibration import calibration_curve


def get_plt_data(data,plt_mode):
    xlist=[]
    ylist=[]
    legendlist=[]
    if plt_mode=="roc":
        prob_dict=data["prob"]
        y_dict=data["y"]
        for modelname,probs in prob_dict.items():
            labels=y_dict[modelname]
            fpr, tpr, _ = roc_curve(labels, probs)
            auc=roc_auc_score(labels,probs)
            xlist.append(fpr)
            ylist.append(tpr)
            # lower_auc,upper_auc=auc_delong_ci(labels,probs,95)
            legend=f'{modelname} (AUC = {auc:.3f})'
            legendlist.append(legend)
        xlist.append([0,1])
        ylist.append([0,1])
        legendlist.append("Random Guess")

    elif plt_mode=="dca":
        thresholds=np.linspace(0, 1, 101)
        prob_dict=data["prob"]
        y_dict=data["y"]
        for modelname,probs in prob_dict.items():
            labels=y_dict[modelname]
            net_benefit = []
            N=len(labels)
            for threshold in thresholds:
                preds=(probs>threshold).astype(int)
                tp=np.sum((labels==1)&(preds==1))
                fp=np.sum((labels==0)&(preds==1))
                net_benefit_value=(tp/N)-(fp/N)*(threshold/(1-threshold))
                net_benefit.append(net_benefit_value)
            xlist.append(thresholds)
            ylist.append(net_benefit)
            legendlist.append(modelname)
        legendlist.extend(["All","None"])
        xlist.extend([thresholds,thresholds])
        tpall=(labels==1).sum()
        fpall=(labels==0).sum()
        ylist.extend([np.array([(tpall/N)-(fpall/N)*(threshold/(1-threshold)) for threshold in thresholds]),np.array([0 for x in range(101)])])

    elif plt_mode=="calibration":
        prob_dict=data["prob"]
        y_dict=data["y"]
        for modelname,probs in prob_dict.items():
            labels=y_dict[modelname]
            prob_true, prob_pred = calibration_curve(labels, probs, n_bins=4, strategy='quantile')
            xlist.append(prob_pred)
            ylist.append(prob_true)
            legendlist.append(modelname)
        xlist.append(np.array([0,1]))
        ylist.append(np.array([0,1]))
        legendlist.append("Perfectly Calibrated")
    return xlist,ylist,legendlist


def get_best_cutoff(y_true, prob, method='youden'):

    best_metric = -np.inf
    best_threshold = 0
    thresholds = sorted(list(set(prob)))# 生成0-1之间的101个阈值点
    for threshold in thresholds:
        y_pred = (prob >threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        current_metric = sensitivity + specificity - 1

        if current_metric > best_metric:
            best_metric = current_metric
            best_threshold = threshold
    return best_threshold


def get_cox_rel_at_timepoint(df,event_col,time_col,time_point,cph,feature_col_list=None):
    if feature_col_list:
        data=df[feature_col_list+[event_col,time_col]]
    else:
        data=df
    def inner(x):
        if int(x[event_col])==1 and x[time_col]<time_point:
            return 0
        elif x[time_col]>=time_point:
            return 1
        else:
            return -1
    if not cph:
        cph=CoxPHFitter()
        cph.fit(data,time_col,event_col)
    y_prob=np.array(cph.predict_survival_function(data,times=time_point).iloc[0,:])
    y=np.array(data.apply(lambda x:inner(x),axis=1))
    star=(y!=-1)

    return y_prob[star],y[star]


def find_index(L,x):
    left=0
    right=len(L)
    while right-left>1:
        middle=int((left+right)/2)
        if L[middle]<=x:
            left=middle
        else:
            right=middle
    return int(left)
    

def get_time_survival_info(time_list,survival_list,x,x_type):
    if x_type=="time":
        if x<time_list[0] or x>time_list[-1]:
            return None
        index=find_index(time_list,x)
        return round(survival_list[index],2)
    if x_type=="survival":
        if x>survival_list[0] or x<survival_list[-1]:
            return None
        index=find_index(survival_list[::-1],x)
        index=len(survival_list)-1-index
        return round(time_list[index],2)



def create_risk_table(ax, kmf_list, labels, time_points=None,color_list=None,time_type="days"):
    """
    在指定坐标轴上绘制风险集表
    """
    if time_points is None:
        # 自动选择有代表性的时间点
        all_times = []
        for kmf in kmf_list:
            all_times.extend(kmf.event_table.index.tolist())
        time_points = sorted(set(all_times))
        # 选择5-8个有代表性的时间点
        if len(time_points) > 8:
            time_points = np.linspace(min(time_points), max(time_points), 6).tolist()
    
    # 计算每个时间点的统计量
    table_data = []
    for time_point in time_points:
        row_data = {'时间': time_point}
        for i, kmf in enumerate(kmf_list):
            # 计算风险集人数
            at_risk = calculate_at_risk_at_time(kmf, time_point)
            row_data[labels[i]] = at_risk
        table_data.append(row_data)
    
    # 创建DataFrame
    df_table = pd.DataFrame(table_data)
    df_table = df_table.set_index('时间')
    
    df_table=df_table.transpose()
    df_table.insert(loc=0,value=labels,column=f'TIme ({time_type})')
    # 绘制表格
    table = ax.table(cellText=df_table.values,
                     rowLabels=None,
                     colLabels=df_table.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    # 美化表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # 调整表格大小
    
    # 设置表格样式
    for key, cell in table.get_celld().items():
        if key[1]==0:
            cell.set_facecolor('#4C72B0')
            cell.set_text_props(weight='bold', color=color_list[key[0]-1])           
        if key[0] == 0:  # 表头行
            cell.set_facecolor('#4C72B0')
            cell.set_text_props(weight='bold', color='white')
        elif key[0] % 2 == 1:  # 奇数行
            cell.set_facecolor('#f0f0f0')
        else:  # 偶数行
            cell.set_facecolor('white')
        
        cell.set_edgecolor('white')
        cell.set_height(0.1)
    
    ax.set_title('Number At Risk', fontsize=12, fontweight='bold', pad=3)

def calculate_at_risk_at_time(kmf, time_point):
    """计算指定时间点的风险集人数"""
    event_table = kmf.event_table
    # 找到小于等于指定时间的最后一行
    mask = event_table.index <= time_point
    if mask.any():
        last_row = event_table[mask].iloc[-1]
        return int(last_row['at_risk'] - last_row['removed'])
    return 0



def logrank_test_df(df,event_col,time_col,group_col):
    a=multivariate_logrank_test(df[time_col],df[group_col],df[event_col])
    rel={}
    return a.test_statistic,a.p_value


def plt_km_multigroup_df(df, event_col, time_col, var_col,confidence_level,time_type="days",save_path=None,if_p=1,p_location=[0.8,0.4],
                        if_table=0, ci_legend=0, ci_no_lines=1,
                        show_censors=1, censor_styles={"marker": "+", "ms": 6, "mew": 1}, at_risk_counts=True,
                        ci_alpha=0.05, ci_show=1,figsize=(12, 10), table_height_ratio=0.2, **kwargs):
    df=deepcopy(df)
    if time_type=="months":
        df[time_col]=df[time_col]/30
    if time_type=="years":
        df[time_col]=df[time_col]/365.25
    
    color_list = [ "blue", "green", "orange", "purple"]  # 扩展颜色列表
    
    i = -1
    kmf_list = []
    
    # 创建图形和坐标轴
    

    if if_table:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 1, height_ratios=[1-table_height_ratio, table_height_ratio])
        # 创建生存曲线图
        ax_km = fig.add_subplot(gs[0])
        
        # 创建风险集表
        ax_table = fig.add_subplot(gs[1])
    else:
        fig,ax_km= plt.subplots(figsize=figsize)
    i=-1
    if var_col:
        element_set = sorted(list(df[var_col].unique()))
        for element in element_set:
            i+=1
            subdf = df[df[var_col] == element]
            kmf = KaplanMeierFitter(alpha=1-confidence_level)
            kmf.fit(subdf[time_col], subdf[event_col])
            kmf_list.append(deepcopy(kmf))
            
            # 使用当前坐标轴绘制
            kmf.plot_survival_function(ax=ax_km, color=color_list[i % len(color_list)], 
                                    label=f"{var_col}={element}", at_risk_counts=0, 
                                    ci_legend=ci_legend, ci_no_lines=ci_no_lines,
                                    show_censors=show_censors, censor_styles=censor_styles, 
                                    ci_alpha=ci_alpha, ci_show=ci_show, **kwargs)
    else:
        element_set=["All patients"]
        kmf = KaplanMeierFitter(alpha=1-confidence_level)
        kmf.fit(df[time_col], df[event_col])
        kmf_list.append(deepcopy(kmf))
        
        # 使用当前坐标轴绘制
        kmf.plot_survival_function(ax=ax_km, color=color_list[i % len(color_list)], 
                                label=f"All patients", at_risk_counts=0, 
                                ci_legend=ci_legend, ci_no_lines=ci_no_lines,
                                show_censors=show_censors, censor_styles=censor_styles, 
                                ci_alpha=ci_alpha, ci_show=ci_show, **kwargs)       

    if if_p:
        _,pval=logrank_test_df(df,event_col,time_col,var_col)
        ax_km.text(p_location[0],p_location[1],f"Logrank Test: p={round(pval,2)}" if pval>0.001 else "Logrank Test: p<0.001" ,fontsize=12,fontweight="bold",color="red")
    ax_km.set_xlabel(f"Time ({time_type})",fontsize=12, fontweight='bold')
    ax_km.set_ylabel("Survival Rate",fontsize=12,  fontweight='bold')
    ax_km.legend(loc="best",fontsize=12)
    for i in ["top","right"]:
        ax_km.spines[i].set_visible(False)
    if if_table:
        create_risk_table(ax_table, kmf_list, element_set,time_points=[int(x) for x in list(ax_km.get_xticks())[1:-1]],color_list=color_list,time_type=time_type)
        ax_table.axis('off')
    plt.tight_layout()
    # if save_path:
    #     plt.savefig(save_path,dpi=300)
    return plt.gcf()