import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib

import pandas as pd
from argparse import ArgumentParser

# #############################################################################

#data = pd.read_excel("COPD_Scoring_Data1.xlsm")

op = ArgumentParser()

op.add_argument("--scoring_data", type=str, default="Scoring_Data.xlsm",
                help="Scoring Data File Name")

op.add_argument("--AUC_Test", type=str, default="False",
                help="This should be 'True' if scoring data has actual Flare_Up value to Test and build ROC AUC Curve")

opts = op.parse_args()

if opts.scoring_data is None:
    print("Please run with scoring data filepath. e.g. Enigma3_Flare_Up_Predict.py --scoring_data=<filename>")

if (opts.AUC_Test =="True"):
    print("Note: --AUC_Test is 'True' that means scoring data has actual Flare_Up value, Test is in progress to build ROC AUC Curve")

data = pd.read_excel(opts.scoring_data)

feature_names = [
	'Demo1', 		#	Demographics				Binary	 
	#'Demo2', 		#	Demographics				Continuous	 
	#'Demo3', 		#	Demographics				Continuous	 
	#'Demo4', 		#	Demographics				Continuous	 
	'Demo5',		#	Demographics				Binary	 
    'Demo6', 		#	Demographics				Continuous	 
    'DisHis1', 	#	Disease History				Binary					1 
    'DisHis1Times', #	Disease History				Continuous				1
    'DisHis2', 		#	Disease History				Binary					2 
    'DisHis2Times',	#	Disease History				Continuous				2
    'DisHis3', 		#	Disease History				Binary					3
    'DisHis3Times', #	Disease History				Continuous				3
    'DisHis4', 		#	Disease History				Binary	 
    'DisHis5', 		#	Disease History				Binary	 
    'DisHis6', 		#	Disease History				Continuous	 
    'DisHis7',		#	Disease History				Binary	 
    'DisStage1', 	#	Disease Stage				Ordered Categorical	 
    'DisStage2', 	#	Disease Stage				Categorical	 
    'LungFun1', 	#	Lung Function				Continuous	 
    #'LungFun2', 	#	Lung Function				Continuous	 
    #'LungFun3',		#	Lung Function				Continuous	 
    'LungFun4', 	#	Lung Function				Continuous	 
    'LungFun5', 	#	Lung Function				Continuous	 
    'LungFun6',		#	Lung Function				Continuous	 
    #'LungFun7', 	#	Lung Function				Continuous	 
    #'LungFun8', 	#	Lung Function				Continuous	 
    'LungFun9',		#	Lung Function				Continuous	 
    'LungFun10', 	#	Lung Function				Continuous	 
    #'LungFun11', 	#	Lung Function				Binary	 
    #'LungFun12', 	#	Lung Function				Continuous	 
    #'LungFun13', 	#	Lung Function				Continuous	 
    #'LungFun14',	#	Lung Function				Continuous	 
    #'LungFun15', 	#	Lung Function				Continuous	 
    #'LungFun16',	#	Lung Function				Continuous	 
    #'LungFun17', 	#	Lung Function				Continuous	 
    #'LungFun18', 	#	Lung Function				Continuous	 
    'LungFun19',	#	Lung Function				Continuous	 
    'LungFun20', 	#	Lung Function				Continuous	 
    #'Dis1', 		#	Other lung diseases			Tertiary				4
    'Dis1Treat', 	#	Other lung diseases			Binary					4
    #'Dis2', 		#	Other lung diseases			Tertiary				5
    'Dis2Times', 	#	Other lung diseases			Continuous				5
    'Dis3',			#	Other lung diseases			Tertiary				6
    'Dis3Times', 	#	Other lung diseases			Continuous				6
    'Dis4', 		#	Other lung diseases			Tertiary				7
    'Dis4Treat', 	#	Other lung diseases			Binary					7
    'Dis5', 		#	Other lung diseases			Tertiary				8
    'Dis5Treat', 	#	Other lung diseases			Binary					8
    'Dis6',			#	Other lung diseases			Tertiary				9
    'Dis6Treat', 	#	Other lung diseases			Binary					9 
    'Dis7', 		#	Other lung diseases			Binary	
    'RespQues1', 	#	Respiratory Questionnaire	Continuous	 
    'ResQues1a', 	#	Respiratory Questionnaire	Continuous	 
    'ResQues1b', 	#	Respiratory Questionnaire	Continuous	 
    'ResQues1c',	#	Respiratory Questionnaire	Continuous	 
    'ResQues2a', 	#	Respiratory Questionnaire	Continuous	 
    #'SmokHis1', 	#	Smoking History				Continuous	
    #'SmokHis2', 	#	Smoking History				Continuous	
    #'SmokHis3', 	#	Smoking History				Continuous	
    'SmokHis4'		#	Smoking History				Continuous	 
    ]

X = data[feature_names]

classifier = joblib.load('Enigma3_Model.pkl') 

probas_ = classifier.predict(X)

np.savetxt("Enigma_Predicted_Value.txt", probas_, fmt='%i',)

"""
Below Code is executed while Testing the score with actual value when AUC will be computed by the Evaluators.
python3 Enigma3_Flare_Up_Predict.py --scoring_data=<scoring data filename> --AUC_Test=True
"""
if(opts.AUC_Test == "True"):
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)
	degree = 10
	y = data['Flare_Up']
	fpr, tpr, thresholds = roc_curve(y, probas_)
    
	tprs.append(interp(mean_fpr, fpr, tpr))
	tprs[-1][0] = 0.0
	roc_auc = auc(fpr, tpr)
	aucs.append(roc_auc)
	plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold AUC = %0.2f' % (roc_auc))

	plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)
	plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='blue', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Predicting Condition Flare Up in Patients with Respiratory Disease')
	plt.legend(loc="lower right")
	plt.show()

# #############################################################################
