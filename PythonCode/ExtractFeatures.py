from Config import *
from ReadDataSet import *
import matplotlib



########## FEATURES GROUPED ON PASSBANDS ##########

def extract_Features(dataSet, metaData):


    dataSet['flux_ratio_sq'] = np.power(dataSet['flux'] / dataSet['flux_err'], 2.0)
    dataSet['flux_by_flux_ratio_sq'] = dataSet['flux'] * dataSet['flux_ratio_sq']

    aggs = {
        'flux_ratio_sq': [np.sum],
        'flux_by_flux_ratio_sq': [np.sum],
        'passband': [np.nanmean, np.nanvar, np.nanstd ],
        'flux' : [ np.nanmin, np.nanmax, np.nanmean, np.nanmedian, np.nanvar, np.nanstd ],
        'flux_err' : [ np.nanmin, np.nanmax, np.nanmean, np.nanmedian, np.nanvar, np.nanstd ],
        'mjd' : [ np.nanmin, np.nanmax ],
        'detected' : [ np.nanmean, np.nansum ]
    }

    dataFeatures = dataSet.groupby('object_id').agg( aggs )
    del(dataSet)
    gc.collect()

    dataFeatures.columns = [
        "flux_ratio_sq_sum", "flux_by_flux_ratio_sq_sum",
        "passband_mean", "passband_var","passband_std",
        "flux_min","flux_max","flux_mean","flux_median","flux_var", "flux_std",
        "flux_err_min","flux_err_max","flux_err_mean","flux_err_median","flux_err_var", "flux_err_std",
        "mjd_max", "mjd_min",
        "detected_mean","detected_sum"
    ]
    dataFeatures['flux_range'] = dataFeatures.flux_max - dataFeatures.flux_min
    dataFeatures['flux_err_range'] = dataFeatures.flux_err_max - dataFeatures.flux_err_min
    dataFeatures['mjd_range'] = dataFeatures.mjd_max - dataFeatures.mjd_min  # De acordo com https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696 eu removi o max e min do mjd pela amplitude
    
    dataFeatures['flux_diff'] = dataFeatures['flux_max'] - dataFeatures['flux_min']
    dataFeatures['flux_dif2'] = (dataFeatures['flux_max'] - dataFeatures['flux_min']) / dataFeatures['flux_mean']
    dataFeatures['flux_w_mean'] = dataFeatures['flux_by_flux_ratio_sq_sum'] / dataFeatures['flux_ratio_sq_sum']
    dataFeatures['flux_dif3'] = (dataFeatures['flux_max'] - dataFeatures['flux_min']) / dataFeatures['flux_w_mean']

    dataFeatures.drop(['mjd_max','mjd_max'], axis=1,inplace=True)
    
    print(dataFeatures.columns)

    if( metaData.index.name != 'object_id' ):
        metaData.set_index('object_id', inplace=True)
    #metaData.drop('hostgal_specz',axis=1, inplace=True)

    # check null values on DataFrame
    nan_df = dataFeatures.isnull().sum()
    n_nan = nan_df.sum()
    if( n_nan > 0 ):
        print("\n\nERROR 01...\nNaN values not surpoted for this function")
        print("\n\nYou can find the NaN value on column with count of NaN's greater than 0:")
        print(nan_df)
        print("dataset:", dataSet)
        print("metaData:", metaData)
        exit()
    del(nan_df)
    del(n_nan)
    gc.collect()

    dataSetFeatures = metaData.join( dataFeatures )

    nanList_distmod = np.isnan(dataSetFeatures.distmod)
    dataSetFeatures.distmod = np.where( nanList_distmod, 0, dataSetFeatures.distmod)
    dataSetFeatures['galactic'] = np.where( nanList_distmod, 1, 0)

    del(nanList_distmod)
    del(metaData)
    gc.collect()

    return dataSetFeatures


def extract_DataTraining_Features():
    dataSet = getInitial_DataTraining()
    metaData = getInitial_MetaDataTraining()
    dataTrainingFeatures = extract_Features( dataSet,metaData )
    dataTrainingFeatures.to_hdf( DATA_PATH + "training_features.h5", key = 'features')
    print("\n\n Extract features on training done!")
    print("\n\nWrote in 'training_features.h5' ")

# Precisa separar um modelo para hostgal_specz e hostgal_photoz, este terá apenas o hostgal_photoz
def extract_DataTest_Features():

    lastobj = pandas.DataFrame()
    lastobj_id = -1

    #var inicialized
    chunksize = 11*M
    num_objs_processed = 1
    dataTest_features = pandas.DataFrame([])
    subset_lines = 0

    FullMetaData = pandas.read_csv( DATA_PATH+"test_set_metadata.csv", dtype = METADATA_TYPE_TEST )
    FullMetaData.set_index( 'object_id', inplace=True )

    print('\n\nEXTRATING STARTS!')


    for subset in pandas.read_csv( DATA_PATH + "test_set.csv", dtype=DATA_TYPE, iterator=True ,chunksize = chunksize):

        # id do primeiro objeto do chunck atual
        firstobj_id = subset.iloc[ 0,0 ]

        # verificar se um mesmo objeto não está presente em dois chunks ao mesmo tempo
        # se estiver retirar do chunck passado e adicionar no atual
        if( lastobj_id == firstobj_id ):
            dataTest_features.drop( lastobj_id ,inplace=True )
            num_objs_processed -= 1 # Já que foi retirando 1 objeto e ele precisará ser lido novamento no metaData
            subset = lastobj.append( subset, ignore_index=True ) # numero de objetos dos dados ainda será calculado

        # id do ultimo objeto processado
        lastobj_id = subset.iloc[ -1,0 ]
        lastobj = subset[ subset.object_id == lastobj_id ]

        objects_id = subset['object_id'].unique()
        metaData = FullMetaData.loc[objects_id,:]

        #Processa cada chunk como foi processado o dataSet de Treino
        chunckData_Features = extract_Features( subset, metaData )

        #infos
        num_obj = len(objects_id)
        print('numero de objetos: ',num_obj)
        subset_memory_usage = subset.memory_usage().sum()
        subset_lines += subset.shape[0]
        metaData_memory_usage = metaData.memory_usage().sum()
        del(subset)
        del(metaData)
        gc.collect()
        num_objs_processed += num_obj

        #results
        dataTest_features = dataTest_features.append( chunckData_Features )
        del(chunckData_Features)
        gc.collect()


        #Quanto foi concluido
        print("\n\n{0} millions of lines read and processed with {1} objects".format(subset_lines//M, num_objs_processed))
        print(subset_memory_usage / M, end=" Mb used by subset\n")
        print(metaData_memory_usage / 1000, end=" Kb used by metaData\n")
        print(dataTest_features.memory_usage().sum() / M, end=" Mb used by Data Test Features\n")

    print("\n\n Writing features...")
    dataTest_features.to_hdf( DATA_PATH+"test_features_set.h5", 'features')
    print(dataTest_features)
    print(dataTest_features.iloc[:,8:32].describe())
    print(dataTest_features.columns)

    print("\n\n Extract features on test done!")
    print("\n\nWrote in 'test_features_set.h5' ")

    del(dataTest_features)
    gc.collect()


########## FEATURES BY PASSBANDS ##########

def extract_Features_ByPassband(dataSet, metaData):

    means = dataSet.pivot_table(index='object_id', values=['flux','flux_err'], columns=['passband'], aggfunc=np.nanmean)
    medians = dataSet.pivot_table(index='object_id', values=['flux','flux_err'], columns=['passband'], aggfunc=np.nanmedian)
    var = dataSet.pivot_table(index='object_id', values=['flux','flux_err'], columns=['passband'], aggfunc=np.nanvar)
    std = dataSet.pivot_table(index='object_id', values=['flux','flux_err'], columns=['passband'], aggfunc=np.nanstd)
    fluxes_min = dataSet.pivot_table(index='object_id', values=['flux','flux_err'],  columns=['passband'], aggfunc=np.amin)
    fluxes_max = dataSet.pivot_table(index='object_id', values=['flux','flux_err'],  columns=['passband'], aggfunc=np.amax)
    fluxes_range = fluxes_max - fluxes_min
    mjd_min = dataSet.pivot_table(index='object_id', values=['mjd'],  columns=['passband'], aggfunc=np.amin).mean(axis=1).to_frame()
    mjd_max = dataSet.pivot_table(index='object_id', values=['mjd'],  columns=['passband'], aggfunc=np.amax).mean(axis=1).to_frame()
    detected_count = dataSet.pivot_table(index='object_id', values=['detected'],  columns=['passband'], aggfunc=np.sum)
    detected_mean = dataSet.pivot_table(index='object_id', values=['detected'],  columns=['passband'], aggfunc=np.mean)
    del(dataSet)
    gc.collect()
    means.columns = [
        "flux_mean_pass_0","flux_mean_pass_1","flux_mean_pass_2","flux_mean_pass_3","flux_mean_pass_4","flux_mean_pass_5",
        "flux_mean_pass_err_0","flux_mean_pass_err_1","flux_mean_pass_err_2","flux_mean_pass_err_3","flux_mean_pass_err_4","flux_mean_pass_err_5",
    ]
    medians.columns = [
        "flux_median_pass_0","flux_median_pass_1","flux_median_pass_2","flux_median_pass_3","flux_median_pass_4","flux_median_pass_5",
        "flux_median_pass_err_0","flux_median_pass_err_1","flux_median_pass_err_2","flux_median_pass_err_3","flux_median_pass_err_4","flux_median_pass_err_5",
    ]
    std.columns = [
        "flux_std_pass_0","flux_std_pass_1","flux_std_pass_2","flux_std_pass_3","flux_std_pass_4","flux_std_pass_5",
        "flux_std_pass_err_0","flux_std_pass_err_1","flux_std_pass_err_2","flux_std_pass_err_3","flux_std_pass_err_4","flux_std_pass_err_5",
    ]
    var.columns = [
        "flux_var_pass_0","flux_var_pass_1","flux_var_pass_2","flux_var_pass_3","flux_var_pass_4","flux_var_pass_5",
        "flux_var_pass_err_0","flux_var_pass_err_1","flux_var_pass_err_2","flux_var_pass_err_3","flux_var_pass_err_4","flux_var_pass_err_5",
    ]
    fluxes_min.columns = [
        "flux_min_pass_0","flux_min_pass_1","flux_min_pass_2","flux_min_pass_3","flux_min_pass_4","flux_min_pass_5",
        "flux_min_pass_err_0","flux_min_pass_err_1","flux_min_pass_err_2","flux_min_pass_err_3","flux_min_pass_err_4","flux_min_pass_err_5",
    ]
    fluxes_max.columns = [
        "flux_max_pass_0","flux_max_pass_1","flux_max_pass_2","flux_max_pass_3","flux_max_pass_4","flux_max_pass_5",
        "flux_max_pass_err_0","flux_max_pass_err_1","flux_max_pass_err_2","flux_max_pass_err_3","flux_max_pass_err_4","flux_max_pass_err_5",
    ]
    fluxes_range.columns = [
        "flux_range_pass_0","flux_range_pass_1","flux_range_pass_2","flux_range_pass_3","flux_range_pass_4","flux_range_pass_5",
        "flux_range_pass_err_0","flux_range_pass_err_1","flux_range_pass_err_2","flux_range_pass_err_3","flux_range_pass_err_4","flux_range_pass_err_5",
    ]
    detected_count.columns = [
        "detected_count_pass_0","detected_count_pass_1","detected_count_pass_2","detected_count_pass_3","detected_count_pass_4","detected_count_pass_5",
    ]
    detected_mean.columns = [
        "detected_mean_pass_0","detected_mean_pass_1","detected_mean_pass_2","detected_mean_pass_3","detected_mean_pass_4","detected_mean_pass_5",
    ]
    mjd_min.columns = ['mjd_min']
    mjd_max.columns = ['mjd_max']

    dataSetFeatures = means.join(medians)
    dataSetFeatures = dataSetFeatures.join(std)
    dataSetFeatures = dataSetFeatures.join(var)
    dataSetFeatures = dataSetFeatures.join(fluxes_min)
    dataSetFeatures = dataSetFeatures.join(fluxes_max)
    dataSetFeatures = dataSetFeatures.join(mjd_min)
    dataSetFeatures = dataSetFeatures.join(mjd_max)
    dataSetFeatures = dataSetFeatures.join(detected_count)
    del(means)
    del(medians)
    del(std)
    del(fluxes_min)
    del(fluxes_max)
    del(fluxes_range)
    del(mjd_min)
    del(mjd_max)
    del(detected_count)
    gc.collect()

    #check null values on features
    nan_df = dataSetFeatures.isnull().sum()
    n_nan = nan_df.sum()
    if( n_nan > 0 ):
        print("\n\WARNING 01...\nNaN values on features")
        print("\n\nYou can find the NaN value on column with count of NaN's greater than 0:")
        print( var )
        print( nan_df[ nan_df!=0 ] )
        if( n_nan == dataSetFeatures.shape[1] ):
            print("ERROR 02 - The object won't appears on datafeatures!!!")
            exit()
    del(nan_df)
    del(n_nan)
    del(var)
    gc.collect()

    #Ajuste dos metadados para juntar com as features calculadas.
    if( metaData.index.name != 'object_id' ):
        metaData.set_index('object_id', inplace=True)

    #Usar o hostgal_specz quando tiver a coluna é uma boa alternativa
    metaData.drop( 'hostgal_specz', axis=1, inplace=True)

    dataSetFeatures = metaData.join(dataSetFeatures)

    # Distmod nulo significa que o elemento é galáctico e a distancia do objeto é menor menor com relação aos
    # outros objetos porém não é 0
    # retirar a coluna e verificar o aprendizado...
    nanList_distmod = np.isnan(dataSetFeatures.distmod)
    dataSetFeatures.distmod = np.where( nanList_distmod, 0, dataSetFeatures.distmod)
    dataSetFeatures['galactic'] = np.where( nanList_distmod, 1, 0)

    del(nanList_distmod)
    del(metaData)
    gc.collect()
    return dataSetFeatures

def extract_DataTraining_Features_ByPassband():
    dataSet = getInitial_DataTraining()
    metaData = getInitial_MetaDataTraining()
    dataTrainingFeatures = extract_Features_ByPassband( dataSet,metaData )
    dataTrainingFeatures.to_hdf( DATA_PATH + "training_passband_features2_set.h5", key = 'features')
    print("\n\n Extract features on training done!")
    print("\n\nWrote in 'training_passband_features2_set.h5' ")

#NEED REVIEW AND MAYBE SOME FIXES!!!!
def extract_DataTest_Features_ByPassband():

    lastobj = pandas.DataFrame()
    lastobj_id = -1

    #var inicialized
    chunksize = 6*M
    Kb = 1000
    num_objs_processed = 1
    dataTest_features = pandas.DataFrame([])
    subset_lines = 0

    FullMetaData = pandas.read_csv( DATA_PATH+"test_set_metadata.csv", dtype = METADATA_TYPE_TEST )
    FullMetaData.set_index('object_id', inplace=True)

    for subset in pandas.read_csv( DATA_PATH + "test_set.csv", dtype=DATA_TYPE, iterator=True ,chunksize = chunksize):

        # id do primeiro objeto do chunck atual
        firstobj_id = subset.iloc[ 0,0 ]

        # verificar se um mesmo objeto não está presente em dois chunks ao mesmo tempo
        # se estiver retirar do chunck passado e adiciona no atual
        if( lastobj_id == firstobj_id ):
            dataTest_features.drop( lastobj_id ,inplace=True )
            num_objs_processed -= 1 # Já que foi retirando 1 objeto e ele precisará ser lido novamento no metaData
            subset = lastobj.append( subset, ignore_index=True ) # numero de objetos dos dados ainda será calculado

        # id do ultimo objeto processado
        lastobj_id = subset.iloc[ -1,0 ]
        lastobj = subset[ subset.object_id == lastobj_id ]

        objects_id = subset['object_id'].unique()
        metaData = FullMetaData.loc[ objects_id, : ]

        #Processa cada chunk como foi processado o dataSet de Treino
        chunckData_Features = extract_Features_ByPassband( subset, metaData )

        #infos
        num_obj = len(objects_id)
        subset_memory_usage = subset.memory_usage().sum()
        subset_lines += subset.shape[0]
        metaData_memory_usage = metaData.memory_usage().sum()
        num_objs_processed += num_obj
        del(subset)
        del(metaData)
        gc.collect()

        #concatena os objetos processado
        dataTest_features = dataTest_features.append( chunckData_Features )
        del(chunckData_Features)
        gc.collect()

        #Quanto foi concluido
        print("\n\n{0} millions of lines read and processed with {1} objects".format(subset_lines, num_objs_processed))
        print(subset_memory_usage / M, end=" Mb used by subset\n")
        print(metaData_memory_usage / Kb, end=" Kb used by metaData\n")
        print(dataTest_features.memory_usage().sum() / M, end=" Kb used by Data Test Features\n")

    print('\n\nWriting the features...')
    dataTest_features.to_hdf( DATA_PATH+"test_passband_features_set40.h5", key='features' )
    print(dataTest_features)
    print(dataTest_features.iloc[:,0:12].describe())
    print(dataTest_features.columns)

    print("\n\n Extract features by passaband on test done!")
    print("Wrote in 'test_passband_features_set40.h5' ")


# O score com novas features(mjd-max/min e count detected):
# O score da simulação para distmod NAN = 0 estava entre 0.6607218683651804 e 0.6760084925690021 aproximadamente
# O score da simulação para distmod NAN = 41.263961 (media dos distmod) estava entre 0.6607218683651804 e 0.6764331210191082
# com o test sendo 30% do dataSet de treino,
#
# As novas features subiram o score em cerca de 0.01 e 0.02
# A mudança do distmod não é tão significativa devido ao modo de preencimento dos distmods vazios
#
# Com o acréscimo do desvio padrão e da variança os resultados usando somente o dataSet de treino foi:
# O score da simulação estava entre 0.6492569002123142 e 0.6717622080679405 aproximadamente
# com o test sendo 30% do dataSet de treino.
#
# Agora com o test sendo bem maior que o treino como é o dataset de test do desafio.. cerca de 40 vezes maior
#
# Com o desvio e a variança o resultado min max rodando 100 vezes o experimento foi
# O score da simulação estava entre 0.5088865656037638 e 0.5380292733925771 aproximadamente
#
# Sem o desvio e a variança:
# O score da simulação estava entre 0.4869315211709357 e 0.528881338212232 aproximadamente
# É uma diferença significativa...
#
# Então vamo acrescentar mais algumas features descritivas:
# - maximo e minimo dos fluxos de cada objeto por passband
# - também a amplitude (talvez nao seja necessario ter maximo e minimo e a amplitude ao mesmo tempo..)
# - a mediana da mesma forma
# -  ...? percentil talvez
#
# Com as o minimo e o max, a mediana e a amplitude o resultado foi:
# O score da simulação estava entre  e  aproximadamente
#

def extract_DataTraining_Means_byPassband():
    dataSet = getInitial_DataTraining()
    metaData = getInitial_MetaDataTraining()


    means = dataSet.pivot_table(index='object_id', values=['flux','flux_err'], columns=['passband'], aggfunc=np.nanmean)
    means.columns = [
        "flux_mean_pass_0","flux_mean_pass_1","flux_mean_pass_2","flux_mean_pass_3","flux_mean_pass_4","flux_mean_pass_5",
        "flux_mean_pass_err_0","flux_mean_pass_err_1","flux_mean_pass_err_2","flux_mean_pass_err_3","flux_mean_pass_err_4","flux_mean_pass_err_5",
    ]

    metaData.set_index('object_id', inplace=True)
    metaData['target'] = metaData['target'].map( CLASSES.index )
    dataTraining_Means = metaData.join(means)
    dataTraining_Means['id'] = range(dataTraining_Means.shape[0])

    return dataTraining_Means


def extract_DataTraining_Means():
    dataSet = getInitial_DataTraining()
    metaData = getInitial_MetaDataTraining()


    means = dataSet.groupby('object_id')['flux','flux_err'].agg( [np.nanmean, np.nanmedian, np.nanstd] )
    print(means.columns)
    print(means)
    means.columns = [ 'flux_mean','flux_median','flux_std','flux_err_mean','flux_err_median','flux_err_std', ]

    metaData.set_index('object_id', inplace=True)
    metaData['target'] = metaData['target'].map( CLASSES.index )
    dataTraining_Means = metaData.join(means)
    dataTraining_Means['id'] = range(dataTraining_Means.shape[0])

    return dataTraining_Means


def extract_DataTraining_CountClasses():
    dataSet = getInitial_DataTraining()
    metaData = getInitial_MetaDataTraining()
    metaData['target'] = list( map( CLASSES.index,metaData['target'] ))

    metaData.set_index('object_id', inplace=True)
    dataSet = metaData.join(dataSet)
    dataSet['id'] = range(dataSet.shape[0])

    dataSet = dataSet.groupby(['passband','target']).count().unstack(fill_value=0)

    return dataSet





extract_DataTraining_Features()
