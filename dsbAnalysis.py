import csv
import numpy as np
from sklearn.metrics import  confusion_matrix 
from sklearn.metrics import mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import random


answerFile = "answers.csv"

###############
#   
#   Read in answers. Format is same as submission with single colume of volumes
#   replacing the CDF
#
fa = open("answers.csv","r")
readera = csv.reader(fa,delimiter=',')
i_line = 0
sv_true = []
dv_true = []
for line in readera:
    if i_line>0:
        label = line[0]
        if 'Systole' in label:
            sv_true.append(float(line[1]))
        else:
            dv_true.append(float(line[1]))
    i_line+=1
    
sv_true = np.array(sv_true)
dv_true = np.array(dv_true)
fa.close() 

#top ten
#subFName = ["show_me_the_money.csv","heart.csv","WeiDong_YuanfangGuan.csv","TenciaWoshialex.csv","RatherBeFishing.csv","kunsthart.csv","Plankton.csv","RyanMunion.csv","sinafz.csv","JuliandeWit.csv"]
#subTName = ["Show Me The Money","heart","Wei Dong Yuanfang Guan","Tencia Woshialex","RatherBeFishing","kunsthart","Plankton","Ryan Munion","sinafz","JuliandeWit"]
#medianNotMode = [0,1,0,0,0,0,0,1,0,0]

# top 4
path = "submissions/"
subTeamName = ["#1 Tencia & Woshialex","#2 kunsthart","#3 Julian de Wit","#4 show_me_the_money"]
subInFile = ["TenciaWoshialex.csv","kunsthart.csv","JuliandeWit.csv","show_me_the_money.csv"]
subOutFile = ["Tencia_n_Woshialex","kunsthart","Julian_de_Wit","show_me_the_money"]
medianNotMode = [0,1,1,0]
subInFile = [path+x for x in subInFile]

errorTable = np.zeros((len(subInFile),3 )) #table for avg error, stdev of error
accuracyTable = np.zeros((len(subInFile),2,2 )) #for each submission 2X2 diastolic

for i_sub in range(len(subInFile)):


    print "Processing %s" % subTeamName[i_sub]
    submissionNum = i_sub+1
    #########################
    #
    #   Read in the submission
    #

    fs = open(subInFile[i_sub],"r")
    readers = csv.reader(fs,delimiter=',')
    i_line = 0
    sv_cdf = []
    dv_cdf = []
    for line in readers:
        if i_line>0:
            label = line[0]
            if 'Systole' in label:
                sv_cdf.append([float(x) for x in line[1:]])
            else:
                dv_cdf.append([float(x) for x in line[1:]])
        i_line+=1

    ############
    #
    #  arrays containing all cdfs for each submission
    #
    sv_cdf = np.vstack(sv_cdf)    # index one indicates entry number
    dv_cdf = np.vstack(dv_cdf)    # index two is the CDF index
    n_cdf_bins = len(sv_cdf[0]) 
    fs.close() 


    n_entry = len(sv_cdf)
    dv_err = np.zeros(n_entry)
    sv_err = np.zeros(n_entry)
    dv_conf = np.zeros(n_entry)
    sv_conf = np.zeros(n_entry)
    dvsv_cov = np.zeros(n_entry)
    ef_pred = np.zeros(n_entry)
    ef_true = (dv_true - sv_true)/dv_true
    ef_x = np.arange(0.01,0.99,0.01) # points at which we'll calculate ejection fraction pdf
    #ef_pdf = np.zeros((n_entry,len(ef_x)))    leaving this out for now

    dv_maxiTemp = np.zeros(n_entry)
    sv_maxiTemp = np.zeros(n_entry)
    dv_meanTemp = np.zeros(n_entry)
    sv_meanTemp = np.zeros(n_entry)
    dv_pdfTemp = np.zeros((n_entry,len(dv_cdf[0])-1))
    sv_pdfTemp = np.zeros((n_entry,len(sv_cdf[0])-1))
    sv_pred_arr = np.zeros(n_entry)
    dv_pred_arr = np.zeros(n_entry)



    for N in range(n_entry):
        dv_pdf_max = 0.0
        dv_pdf_mean = 0.0
        dv_pdf_median = 0.0
        dv_pdf_median_i = 0
        dv_pdf_maxi = 0
        dv_pdf_var = 0.0
        sv_pdf_max = 0.0
        sv_pdf_mean = 0.0
        sv_pdf_maxi = 0
        sv_pdf_var = 0.0
        sv_pdf_median = 0.0
        sv_pdf_median_i = 0
        
        sv_pred = 0.0
        dv_pred = 0.0

        #"differentiating" the cdf
        dv_pdf = np.array([dv_cdf[N][i]-dv_cdf[N][i-1] for i in range(1,n_cdf_bins) ],dtype=np.float)
        sv_pdf = np.array([sv_cdf[N][i]-sv_cdf[N][i-1] for i in range(1,n_cdf_bins) ],dtype=np.float)

        #calculating expectation value for dv and sv based on pdf
        for i in range(len(dv_pdf)):
            dv_pdf_mean += dv_pdf[i]*float(i+0.5) # note : value of dv at index 0 is 1 ml
            sv_pdf_mean += sv_pdf[i]*float(i+0.5)
            if medianNotMode[i_sub]:
                dv_pdf_median += dv_pdf[i]
                sv_pdf_median += sv_pdf[i]
                if sv_pdf_median_i == 0.0 and sv_pdf_median >=0.5:
                    sv_pdf_median_i = i+0.5
                if dv_pdf_median_i == 0.0 and dv_pdf_median >=0.5:
                    dv_pdf_median_i = i+0.5                    
            else:
                if dv_pdf_max < dv_pdf[i]:
                    dv_pdf_max = dv_pdf[i]
                    dv_pdf_maxi = i+0.5                   #(remember volume is 1+index in pdf)
                if sv_pdf_max < sv_pdf[i]:
                    sv_pdf_max = sv_pdf[i]
                    sv_pdf_maxi = i+0.5
        if medianNotMode[i_sub]:
            sv_pdf_maxi = sv_pdf_median_i
            dv_pdf_maxi = dv_pdf_median_i
        dv_maxiTemp[N] = dv_pdf_maxi
        sv_maxiTemp[N] = sv_pdf_maxi
        dv_meanTemp[N] = dv_pdf_mean
        sv_meanTemp[N] = sv_pdf_mean
        dv_pdfTemp[N] = dv_pdf
        sv_pdfTemp[N] = sv_pdf

        #########
        #ommitting calculating the norm...
        #I'm assuming kaggle rejects submissions with non-normalized CDFs
        ##########
        #
        #  who knows what the prediction should be? we'll take the average of the
        #  pdf peak and the expectation value of the distribution
        #
        dv_pred = (float(dv_pdf_maxi) + dv_pdf_mean)/2.   
        sv_pred = (float(sv_pdf_maxi) + sv_pdf_mean)/2.
        dv_pred_arr[N] = dv_pred
        sv_pred_arr[N] = sv_pred
        ef_pred[N] = (dv_pred - sv_pred)/dv_pred
        dv_err[N] = dv_pred - dv_true[N]     #absolute error in volume
        sv_err[N] = sv_pred - sv_true[N]
        for i in range(len(dv_pdf)):    # loop for calculating variance of pdf around prediction
            dv_pdf_var += dv_pdf[i]*(float(i+0.5)-dv_pred)*(float(i+0.5)-dv_pred)
            sv_pdf_var += sv_pdf[i]*(float(i+0.5)-sv_pred)*(float(i+0.5)-sv_pred)        
        ##########
        #
        #  Defining the confidence of this prediction to be the stdev of pdf about the
        #  predicted value 
        #
        dv_conf[N] = np.sqrt(dv_pdf_var)
        sv_conf[N] = np.sqrt(sv_pdf_var)

    ############
    #
    #  counting the number of predictions within 10% of actual volume
    #  counting the number of predictions within 10ml of actual volume
    #
    dv_percErr = abs(dv_err)/dv_true 
    dv_perc_win_10perc = float(sum(dv_percErr<=0.10))/len(dv_percErr)
    dv_perc_win_10ml = float(sum(abs(dv_err)<=10))/len(dv_err)

    sv_percErr = abs(sv_err)/sv_true 
    sv_perc_win_10perc = float(sum(sv_percErr<=0.10))/len(sv_percErr)
    sv_perc_win_10ml = float(sum(abs(sv_err)<=10))/len(dv_err)

    accuracyTable[i_sub][0] = np.array([dv_perc_win_10perc,dv_perc_win_10ml])
    accuracyTable[i_sub][1] = np.array([sv_perc_win_10perc,sv_perc_win_10ml])


    
    ########################
    #
    #  generating a confusion matrix
    #  based on Ejection Fraction values on the DSB webpage
    #
    ef_clinic_bins = np.array([0.0,0.35,0.45,0.55,0.73,1.0])
    ef_clinic_bin_names = ["Sev. Abnorm.\n   <35%","Mod. Abnorm.\n 35% to 45%","Mild. Abnorm.\n 45% to 55%","Normal EF\n 55% to 73% ","Hyperdynamic\n   >73%"]
    ef_clinic_bin_index = [-1,0,1,2,3,4]
    ef_pred_class = np.zeros(n_entry,dtype=np.int)
    ef_true_class = np.zeros(n_entry,dtype=np.int)
    ###########
    #
    # looping over entries and assigning a class. -1 is out of bounds
    #
    for N in range(n_entry):
        found_bin = 0  # did we bin this ?
        for i in range(1,len(ef_clinic_bins)):
            if ef_pred[N] >= ef_clinic_bins[i-1] and ef_pred[N] < ef_clinic_bins[i]:
                ef_pred_class[N] = i-1
                found_bin = 1
            if ef_true[N] >= ef_clinic_bins[i-1] and ef_true[N] < ef_clinic_bins[i]:
                ef_true_class[N] = i-1
        if found_bin == 0:
            ef_pred_class[N] = -1

    cm_clinic = confusion_matrix(ef_true_class,ef_pred_class,ef_clinic_bin_index)

    ################################
    #
    #  filling in a table with RMS Error for the volumes and the Ejection Fraction
    #
    errorTable[i_sub][0] = np.sqrt(mean_squared_error(dv_true,dv_pred_arr))
    errorTable[i_sub][1] = np.sqrt(mean_squared_error(sv_true,sv_pred_arr))
    errorTable[i_sub][2] = np.sqrt(mean_squared_error(ef_true,(dv_pred_arr-sv_pred_arr)/dv_pred_arr))

    #########################################
    #
    #  Plotting all this stuff
    #

    #plotting prediction vs truth... both volumes
    slope, intercept, rvalue, pvalue, std_err = stats.linregress(np.append(dv_true,sv_true),np.append(dv_pred_arr,sv_pred_arr))
    plt.scatter(dv_true,dv_pred_arr,label='Diastolic Volume',marker='o',facecolors='none',edgecolors='r')
    plt.scatter(sv_true,sv_pred_arr,label='Systolic Volume',marker='o',facecolors='none',edgecolors='b')
    plt.xlabel("True Volume (mL)")
    plt.ylabel("Predicted Volume (mL)")
    plt.title("%s\nCorrelation of Volume Predictions with Test Values" % (subTeamName[i_sub]))
    x = np.linspace(0,500,10)
    plt.plot(x,x,color='k',label='guide y = x')
    plt.plot(x,x*slope+intercept,color = 'k',linestyle='--',label='y=%.2fx+%.2f\n$R^2=$%.3f p=%.2e' % (slope,intercept,rvalue**2,pvalue))
    plt.gca().set_xlim((0,500))
    plt.gca().set_ylim((0,500))
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig("%sCorrVols.png" % (subOutFile[i_sub]))
    plt.close()
    
    #plotting prediction vs truth... EF
    slope, intercept, rvalue, pvalue, std_err = stats.linregress(ef_true*100.0,ef_pred*100.0)
    plt.scatter(ef_true*100.0,ef_pred*100.0,marker='x',color='#990099',label="Ejection Fraction")
    x = np.linspace(0,90,10)
    plt.plot(x,x,color='k',label='guide y = x')
    plt.plot(x,x*slope+intercept,color = 'k',linestyle='--',label='y=%.2fx+%.2f\n$R^2=$%.3f p=%.2e' % (slope,intercept,rvalue**2,pvalue))
    plt.gca().set_xlim((0,90))
    plt.gca().set_ylim((0,90))
    plt.xlabel("True Ejection Fraction (%)")
    plt.ylabel("Predicted Ejection Fraction (%)")
    plt.title("%s\nCorrelation of EF Predictions with Test Values" % (subTeamName[i_sub]))
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig("%sCorrEF.png" % (subOutFile[i_sub]))
    plt.close()


    #plotting Bland-Altman for dv and sv
    avgVold = (dv_true+dv_pred_arr) /2.0
    difVold = (dv_true-dv_pred_arr)
    plt.scatter(avgVold,difVold,label='Diastolic Volume',marker='o',facecolors='none',edgecolors='r')
    avgVols = (sv_true+sv_pred_arr) /2.0
    difVols = (sv_true-sv_pred_arr)
    plt.scatter(avgVols,difVols,label='Systolic Volume',marker='o',facecolors='none',edgecolors='b')
    totalMean = np.mean(np.append(difVold,difVols))
    totalstd = np.std(np.append(difVold,difVols))
    plt.gca().set_xlim((0,500))
    plt.gca().set_ylim((-100.0,100.0))
    x = np.linspace(0,500,10)
    y = np.array([totalMean]*10)
    plt.plot(x,y,label="Mean = %2.3f" % totalMean)
    y = np.array([totalstd]*10)
    plt.plot(x,1.96*y,linestyle='--',label="1.96*StdDev = %2.3f" % (1.96*totalstd))
    plt.plot(x,-1.96*y,linestyle='--')
    plt.xlabel("Mean of Prediction and Test Value (mL)")
    plt.ylabel("Dif of Prediction and Test Value (mL)")
    plt.title("%s\nBland-Altman Plot for Volume Prediction" % (subTeamName[i_sub]))
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig("%sVolBlandAltman.png" % (subOutFile[i_sub]) )
    plt.close()

    #plotting Bland-Altman for dv and sv
    avgVol = (ef_true+ef_pred) /2.0
    difVol = (ef_true-ef_pred)
    plt.scatter(avgVol,difVol,label='Ejection Fraction',marker='o',color='m')
    totalMean = np.mean(difVol)
    totalstd = np.std(difVol)
    plt.gca().set_xlim((0.1,0.9))
    plt.gca().set_ylim((-0.6,0.6))
    x = np.linspace(0,1,10)
    y = np.array([totalMean]*10)
    plt.plot(x,y,label="Mean = %2.3f" % totalMean)
    y = np.array([totalstd]*10)
    plt.plot(x,1.96*y,linestyle='--',label="1.96*StdDev = %2.3f" % (1.96*totalstd))
    plt.plot(x,-1.96*y,linestyle='--')
    plt.xlabel("Mean of Prediction and Test Value (mL)")
    plt.ylabel("Dif of Prediction and Test Value (mL)")
    plt.title("%s\nBland-Altman Plot for Ejection Fractions" % (subTeamName[i_sub]))
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig("%sEFBlandAltman.png" % (subOutFile[i_sub]) )
    plt.close()

    
    # Confuction Matrix Plot
    cm_temp = cm_clinic[1:,1:] #(get rid of overflow bins)
    plt.imshow(cm_temp,interpolation='nearest',cmap=plt.cm.Reds)
    plt.title("%s\nConfusion Matrix" % (subTeamName[i_sub]))
    cbar=plt.colorbar()
    cbar.set_label("# of predictions")
    tick_marks = np.arange(len(ef_clinic_bin_names))
    plt.xticks(tick_marks, ef_clinic_bin_names, rotation=70)
    plt.yticks(tick_marks, ef_clinic_bin_names)
    plt.tight_layout()
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    width, height = cm_temp.shape
    for x in range(width):
        for y in range(height):
            plt.gca().annotate(str(cm_temp[x,y]),xy=(y,x),horizontalalignment = 'center',verticalalignment='center')
    plt.gcf().set_size_inches((8.0,6.5))
    plt.savefig("%scmClinicSub.png" % (subOutFile[i_sub]))
    plt.close()

    #A random pdf with true values
    figPDF,ax = plt.subplots(10,10,sharex=False,sharey=False,figsize=(20,20))
    figPDF.suptitle(" %s\nRandomly selected PDFS and asssociated test values\nRed : Systolic PDF        Blue : Diastolic PDF\ndashed line : True Systolic        solid line : True Diastolic" % (subTeamName[i_sub]),fontsize="20")
    random.seed()
    eventList = random.sample(range(440), 100)
    for i in range(10):
        for j in range(10):
            ax[i,j].tick_params(labelbottom="off",labelleft="off")
            N = eventList[j + i*10]
            ax[i,j].plot(range(599),sv_pdfTemp[N],color='r',label="Systolic Vol PDF")
            ax[i,j].plot(range(599),dv_pdfTemp[N],color='b',label="Diastolic Vol PDF")
            ylimits = ax[i,j].get_ylim()
            Y = np.linspace(ylimits[0],ylimits[1],10)
            X = np.array([sv_true[N]]*10)
            ax[i,j].plot(X,Y,color='k',linestyle='--',label='Systolic True')
            X = np.array([dv_true[N]]*10)
            ax[i,j].plot(X,Y,color='k',label='Diastolic True')
            ax[i,j].plot(range(599),sv_pdfTemp[N],color='r',label="Systolic Vol PDF")
            ax[i,j].plot(range(599),dv_pdfTemp[N],color='b',label="Diastolic Vol PDF")
            ax[i,j].set_title("index = %d" % N)
            moveX = list(ax[i,j].get_xlim())
            testX = np.max([sv_true[N],dv_true[N]])
            moveX[1] = np.min([testX+75,600])
            ax[i,j].set_xlim(moveX)
    figPDF.savefig("%s100PDFs.png" % (subOutFile[i_sub]))
    plt.close()    
    
    #    plt.title(" Example Pobability Density\n%s" % (subTeamName[i_sub]))
    #    plt.legend(loc="upper right")
    #    plt.xlabel("Volume in mL")


    
    #only do this for one submission
    if i_sub == 0:
        #########################
        #
        #  Pulling out the worst/best 5% in both absolute error and pdf stdev for
        #  both systolic and diastolic volumes
        #  plotting them on the same chart to see what the outliers look like
        #
    
        dv_err_25 = np.percentile(dv_err,2.5)
        dv_err_975 = np.percentile(dv_err,97.5)
        dv_err_worst_err = np.append(dv_err[dv_err <= dv_err_25] ,dv_err[dv_err >= dv_err_975] )
        dv_err_worst_conf = np.append(dv_conf[dv_err <= dv_err_25] ,dv_conf[dv_err >= dv_err_975] )

        dv_conf_5 = np.percentile(dv_conf,2.5)
        dv_conf_best_err = dv_err[dv_conf <= dv_conf_5]
        dv_conf_best_conf = dv_conf[dv_conf <= dv_conf_5]

        dv_err_475 = np.percentile(dv_err,47.5)
        dv_err_525 = np.percentile(dv_err,52.5)
        bool_array = np.logical_and(dv_err >= dv_err_475,dv_err <= dv_err_525)
        dv_err_best_err = dv_err[bool_array]
        dv_err_best_conf = dv_conf[bool_array] 
        
        dv_conf_95 = np.percentile(dv_conf,95)
        dv_conf_worst_err = dv_err[dv_conf >= dv_conf_95] 
        dv_conf_worst_conf = dv_conf[dv_conf >= dv_conf_95] 


        sv_err_25 = np.percentile(sv_err,2.5)
        sv_err_975 = np.percentile(sv_err,97.5)
        sv_err_worst_err = np.append(sv_err[sv_err <= sv_err_25] ,sv_err[sv_err >= sv_err_975] )
        sv_err_worst_conf = np.append(sv_conf[sv_err <= sv_err_25] ,sv_conf[sv_err >= sv_err_975] )

        sv_conf_5 = np.percentile(sv_conf,2.5)
        sv_conf_best_err = sv_err[sv_conf <= sv_conf_5]
        sv_conf_best_conf = sv_conf[sv_conf <= sv_conf_5]

        sv_err_475 = np.percentile(sv_err,47.5)
        sv_err_525 = np.percentile(sv_err,52.5)
        bool_array = np.logical_and(sv_err >= sv_err_475,sv_err <= sv_err_525)
        sv_err_best_err = sv_err[bool_array] 
        sv_err_best_conf = sv_conf[bool_array] 

        sv_conf_95 = np.percentile(sv_conf,95)
        sv_conf_worst_err = sv_err[sv_conf >= sv_conf_95] 
        sv_conf_worst_conf = sv_conf[sv_conf >= sv_conf_95] 

        figScat = plt.figure(figsize=(14,7.5))
        axDV = figScat.add_subplot(121)
        axSV = figScat.add_subplot(122)
        figScat.suptitle(" %s  :  Individual prediction error and associated PDF StDev" % (subTeamName[i_sub]))
    
        axSV.scatter(abs(sv_err),sv_conf,color='#bfbfbf',s=6,marker='o',label='All Predictions')
        axSV.scatter(abs(sv_err_best_err),sv_err_best_conf,color='c',marker='*',s=55,label='5% Smallest Abs Error')
        axSV.scatter(abs(sv_conf_best_err),sv_conf_best_conf,color='b',marker='*',s=55,label= '5% Most Confident')
        axSV.scatter(abs(sv_err_worst_err),sv_err_worst_conf,color='#ff9900',marker='v',s=55, label = '5% Biggest Abs Error')
        axSV.scatter(abs(sv_conf_worst_err),sv_conf_worst_conf,color='#993300',marker='v',s=55, label = '5% Least Confident')
        axSV.set_xlabel("Absolute Value of Error (ml)")
        axSV.set_ylabel("StDev on PDF (ml)")
        axSV.set_title("Systolic Volume (ml)")
        axSV.legend(loc = 'lower right',markerscale = 1.2, prop = {'size':14} )
        axSV.set_xlim((0,80))
        axSV.set_ylim((0,25))
        axSV.grid()
        #    plt.savefig("%sSVScatter.png" % (subTName[i_sub]))
        #    plt.close()

        axDV.scatter(abs(dv_err),dv_conf,color='#bfbfbf',s=6,marker='o', label = "All Predictions")
        axDV.scatter(abs(dv_err_best_err),dv_err_best_conf,color='c',marker='*',s=55,label='5% Smallest Abs Error')
        axDV.scatter(abs(dv_conf_best_err),dv_conf_best_conf,color='b',marker='*',s=55,label= '5% Most Confident')
        axDV.scatter(abs(dv_err_worst_err),dv_err_worst_conf,color='#ff9900',marker='v',s=55, label = '5% Biggest Abs Error')
        axDV.scatter(abs(dv_conf_worst_err),dv_conf_worst_conf,color='#993300',marker='v',s=55, label = '5% Least Confident')
        axDV.set_xlabel("Absolute Value of Error (ml)")
        axDV.set_ylabel("StDev on PDF (ml)")
        axDV.set_title("Diastolic Volume (ml)")
        axDV.legend(loc = 'lower right',markerscale = 1.2, prop = {'size':14} )
        axDV.set_xlim((0,80))
        axDV.set_ylim((0,25))
        axDV.grid()
        #plt.savefig("%sDVScatter.png" % (subTName[i_sub]))
        #plt.close()

        figScat.savefig("%sAllScatter.png" %(subTeamName[i_sub]))
        plt.close()





    
            
#outputing accuracy and regression tables

fout = open("accuracyTable.csv","w")
fout.write("team name,D/S,% within 10% of truth,% within 10ml of truth\n")
for i in range(len(subTeamName)):
    fout.write("%s,Diastole,%.4f,%.4f\n" % (subTeamName[i],accuracyTable[i][0][0],accuracyTable[i][0][1]))
    fout.write("%s,Systole,%.4f,%.4f\n" % (subTeamName[i],accuracyTable[i][1][0],accuracyTable[i][1][1]))
fout.close()


fout = open("errorTable.csv","w")
fout.write("team name,RMS error for D/S/EF\n")
for i in range(len(subTeamName)):
    fout.write("%s,Diastole,%.4f\n" % (subTeamName[i],errorTable[i][0]))
    fout.write("%s,Systole,%.4f\n" % (subTeamName[i],errorTable[i][1]))
    fout.write("%s,E Fraction,%.2f \n" % (subTeamName[i],100.0*errorTable[i][2]))
fout.close()
