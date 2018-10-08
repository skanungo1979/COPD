import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import svm
from sklearn.externals import joblib

import pandas as pd
from argparse import ArgumentParser

# #############################################################################

op = ArgumentParser()

op.add_argument("--development_data", type=str, default="Development_Data.xlsm",
                help="Development Data File Name")

opts = op.parse_args()

if opts.development_data is None:
    print("Please run with development data filepath. e.g. python3 Enigma3_Model.py --development_data=<filename>")

data = pd.read_excel(opts.development_data)

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
y = data['Flare_Up']

#Best Model
classifier = GaussianNB()

#classifier = BernoulliNB()
#classifier = LinearDiscriminantAnalysis()
#classifier = RandomForestClassifier()
#classifier = svm.SVC(kernel='linear', probability=True, random_state=random_state)
#classifier = DecisionTreeClassifier()
#classifier = LogisticRegression()
#classifier = KNeighborsClassifier()
#classifier = RidgeClassifier(alpha=8.0, solver="sparse_cg")
#classifier = SGDClassifier(loss='log')

classifier.fit(X, y)
joblib.dump(classifier, 'Enigma3_Model.pkl')

# #############################################################################
