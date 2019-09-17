import pandas as pd
import numpy as np

location_set = ['RIAI99PA','RIAL99MS','RIBA99PA','RIDM99MS','RIDO99MS','RIGU99MS','RIJE99PA','RIME99MS','RIPR99PA','RIRE99MS','RIRU99PA','RISA99PA','RISE99MS','RISI99PA','RIST99PA','RIVE99PA','RIZI99PA','RIZO99MS']

time = '06:00'
time_target = '23:00'

dataset = pd.read_csv("Avg_air_temperature_2004.04_2018.csv")
dataset1 = pd.read_csv("Wind_direction_2004.04_2018.csv")
dataset2 = pd.read_csv("Wind_speed_2004.04_2018.csv")

def create_values(dataset, time = '06:00'):
    res1 = np.zeros((5704,len(location_set)))
    res = []
    for i in range(len(location_set)):
        df1 = dataset.loc[(dataset['EG_GH_ID'] == location_set[i]) & (dataset['TIME'] == time)]
        df1 = df1.to_numpy()
        df1 = df1[:,6:38]
        for j in range(184):
            for k in range(31):
                df2 = df1[j,k]
                res.append(df2)
        res1[:,i] = np.concatenate((res), axis = None)
        res = []
    return res1

Temp = create_values(dataset, time = time)
TempTarget = create_values(dataset, time = time_target)

WindMag = create_values(dataset1, time = time)
WindDir = create_values(dataset2, time = time)
windX = WindMag*np.sin((WindDir/180)*np.pi)
windY = WindMag*np.cos((WindDir/180)*np.pi)


TrainingSet = np.hstack((Temp,windX,windY,TempTarget))
TrainingSet = pd.DataFrame(TrainingSet)

TrainingSetClean = pd.DataFrame.dropna(TrainingSet,axis = 0,how = 'any')

export_csv = TrainingSetClean.to_csv (r"C:\Users\Reinis\PycharmProjects\TF-test\Tset6-23.csv", index = None, header=False)
print('Training Set created from', time, 'to', time_target)