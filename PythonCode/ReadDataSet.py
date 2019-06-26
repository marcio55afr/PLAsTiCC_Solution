from Config import *

def getInitial_DataTraining(nrow=None):
    if(nrow is not None):
        return pandas.read_csv( DATA_PATH+"training_set.csv", dtype = DATA_TYPE)
    else:
        return pandas.read_csv( DATA_PATH+"training_set.csv", dtype = DATA_TYPE, nrows = nrow)

def getInitial_MetaDataTraining():
    return pandas.read_csv( DATA_PATH+"training_set_metadata.csv", dtype = METADATA_TYPE_TRAINING)

def getDataTraining_Features():
    return pandas.read_hdf( DATA_PATH+"training_features_set.h5", key = 'features')

def getDataTraining_Features_ByPassband():
    return pandas.read_hdf( DATA_PATH+"training_passband_features_set.h5", key = 'features')

def getDataTraining_Features_ByPassband_Grouped():
    return pandas.read_hdf( DATA_PATH+"training_passband_features_grouped_set.h5", key = 'features')

def getDataTraining_Features_ByPassband2():
    return pandas.read_hdf( DATA_PATH+"training_passband_features2_set.h5", key = 'features')


# test_passband_features_set = pandas.read_csv(DATA_PATH+"test_passband_features_set.csv", dtype = FEATURE_TYPE)
# print( test_passband_features_set.iloc[-1,:] )
# test_passband_features_set.to_hdf( DATA_PATH+"test_passband_features_set.h5", key = 'features' )

# testData = pandas.read_csv( DATA_PATH+"test_set.csv", names=DATA_COLUMNS, dtype = DATA_TYPE, skiprows=15900000,nrows=100000)
# print('testData: ', testData.object_id.drop_duplicates().count())
# x = np.sort(testData.object_id)
# print('testData: ', len(x) )
# print('testData: ', x[-1] )

# testData = pandas.read_csv( DATA_PATH+"test_set.csv", names=DATA_COLUMNS,dtype = DATA_TYPE, skiprows=10000000,nrows=10000000)
# metaData = pandas.read_csv( DATA_PATH+"test_set_metadata.csv", dtype = METADATA_TYPE_TEST)
# 3466261 rows of features
# print( metaData.object_id )

# start = time.time()
# objects = testData.object_id.unique()
# print('objects: ', objects)
# print( 'UNIQUE EXPENDS: ', time.time()-start)
# print(type(objects))


#line 2802023 and object_id=104894709
# metaData.set_index( 'object_id', inplace=True )
# print('testMetaData: ', metaData.loc[objects] )

#linha 88.000.000 do testeSet object_id = 127.439.467
#O maior object_id do metadado de teste é = 130.788.054 que o da ultima linha


# teste = pandas.read_csv( DATA_PATH+"test_set.csv", dtype = DATA_TYPE, skiprows=10*M, nrows=10*M)
# teste.columns = DATA_COLUMNS

# obj1 = teste[teste.object_id == 316571]
# obj2 = teste[teste.object_id == 3666774]

# for i in range(6):
#     print( 'count of passands ',i )
#     print( obj1[ obj1.passband == i ].count() )
#     print( obj2[ obj1.passband == i ].count() )
#     print(obj1[i*50:(i+1)*50])
#     print(obj2[i*50:(i+1)*50])




# Esvaziando os arquivos de prediçoes:

df = pandas.DataFrame([])

df.to_csv(DATA_PATH+'predict_passband_set11_214583.csv', mode='w')
df.to_csv(DATA_PATH+'predict_set1_276425.csv', mode='w')
df.to_csv(DATA_PATH+'predict_passband_set23_175707.csv', mode='w')
df.to_csv(DATA_PATH+'predict_passband_set24_165120.csv', mode='w')
df.to_csv(DATA_PATH+'predict_passband_set25_171448.csv', mode='w')
df.to_csv(DATA_PATH+'predict_passband_set26_166556.csv', mode='w')
df.to_csv(DATA_PATH+'predict_passband_set27_172152.csv', mode='w')
df.to_csv(DATA_PATH+'predict_passband_set28_172152.csv', mode='w')
df.to_csv(DATA_PATH+'predict_passband_set29_173213.csv', mode='w')
df.to_csv(DATA_PATH+'predict_passband_set30_173046.csv', mode='w')
df.to_csv(DATA_PATH+'predict_passband_set31_173046.csv', mode='w')
df.to_csv(DATA_PATH+'predict_passband_set32_165198.csv', mode='w')
df.to_csv(DATA_PATH+'predict_passband_set33_165231.csv', mode='w')
