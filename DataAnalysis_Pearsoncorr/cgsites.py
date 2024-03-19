import pandas as pd
from functools import reduce


data1dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE20236.xlsx'
data2dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE20242.xlsx'
data3dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE23638.xlsx'
data4dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE27097.xlsx'
data5dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE30870.xlsx'
data6dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE32148_healthy.xlsx'
data7dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE32149_healthy.xlsx'
data8dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE36064.xlsx'
data9dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE40005_healthy.xlsx'
data10dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE40279.xlsx'
data11dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE41037_healthy.xlsx'
data12dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE41169_healthy.xlsx'
data13dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE42861_healthy.xlsx'
data14dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE53128.xlsx'
data15dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE58045.xlsx'
data16dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE65638.xlsx'
data17dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE19711.xlsx'
data18dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE20067.xlsx'
data19dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE28746.xlsx'
data20dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE32148_disease.xlsx'
data21dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE32149_disease.xlsx'
data22dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE32396.xlsx'
data23dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE34035.xlsx'
data24dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE40005_disease.xlsx'
data25dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE41037_disease.xlsx'
data26dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE41169_disease.xlsx'
data27dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE42042.xlsx'
data28dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE42861_disease.xlsx'
data29dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE49904.xlsx'
data30dir = 'file:///Users/sparshg/Desktop/DNAMethyl/Project/DataAnalysis/GSE57285.xlsx'


data1 = pd.read_excel(data1dir,
                    #header=None
                    )
data2 = pd.read_excel(data2dir,
                    #header=None
                    )
data3 = pd.read_excel(data3dir,
                    #header=None
                    )
data4 = pd.read_excel(data4dir,
                    #header=None
                    )
data5 = pd.read_excel(data5dir,
                    #header=None
                    )
data6 = pd.read_excel(data6dir,
                    #header=None
                    )
data7 = pd.read_excel(data7dir,
                    #header=None
                    )
data8 = pd.read_excel(data8dir,
                    #header=None
                    )
data9 = pd.read_excel(data9dir,
                    #header=None
                    )
data10 = pd.read_excel(data10dir,
                    #header=None
                    )
data11 = pd.read_excel(data11dir,
                    #header=None
                    )
data12 = pd.read_excel(data12dir,
                    #header=None
                    )
data13 = pd.read_excel(data13dir,
                    #header=None
                    )
data14 = pd.read_excel(data14dir,
                    #header=None
                    )
data15 = pd.read_excel(data15dir,
                    #header=None
                    )
data16 = pd.read_excel(data16dir,
                    #header=None
                    )
data17 = pd.read_excel(data17dir,
                    #header=None
                    )
data18 = pd.read_excel(data18dir,
                    #header=None
                    )
data19 = pd.read_excel(data19dir,
                    #header=None
                    )
data20 = pd.read_excel(data20dir,
                    #header=None
                    )
data21 = pd.read_excel(data21dir,
                    #header=None
                    )
data22 = pd.read_excel(data22dir,
                    #header=None
                    )
data23 = pd.read_excel(data23dir,
                    #header=None
                    )
data24 = pd.read_excel(data24dir,
                    #header=None
                    )
data25 = pd.read_excel(data25dir,
                    #header=None
                    )
data26 = pd.read_excel(data26dir,
                    #header=None
                    )
data27 = pd.read_excel(data27dir,
                    #header=None
                    )
data28 = pd.read_excel(data28dir,
                    #header=None
                    )
data29 = pd.read_excel(data29dir,
                    #header=None
                    )
data30 = pd.read_excel(data30dir,
                    #header=None
                    )


print(data1)
print(data2)

df =        [ data1,
                data2,
                data3,
                data4,
                data5,
                data6,
                data7,
                data8,
                data9,
                data10,
                data11,
                data12,
                data13,
                data14,
                data15,
                data16,
                data17,
                data18,
                data19,
                data20,
                data21,
                data22,
                data23,
                data24,
                data25,
                data26,
                data27,
                data28,
                data28,
                data29,
                data30]

#data = pd.merge(,
 #           how='inner', left_index=True, right_index=True)

data = reduce(lambda  left,right: pd.concat(left,right,
                                            join='inner',
                                           #on='CG', right_index=True
                                                     ), df)


print(data)

data.to_csv('cgsites.csv', index=False)

