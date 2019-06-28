from Config import *
import matplotlib.pyplot as plt
import seaborn as sns
import ReadDataSet as readDS
from sklearn import metrics, tree
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
import graphviz
import sys


def LightBGM_Classifier():

    print("\n\nExtrating features and reshapping the training dataset...")
    start = time.time()

    dataTraining = readDS.getDataTraining_Features()

    Y_train = dataTraining.pop('target').to_frame()
    X_train = dataTraining
    print( X_train.dtypes )
    print("\nTable for training is Ready!\nSpend {0} seconds\n\n".format( time.time() - start ))

    print("Creating and training the LightGB Model...")
    start = time.time()

    #parametros para o lgbm
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 14,
        'metric': 'multi_logloss',
        'learning_rate': 0.01,
        'num_iterations': 150,
        'max_depth': 3
    }

    Y_train['target'] = list( map( CLASSES.index, Y_train['target'] ) )

    train_data = lgb.Dataset(X_train, label=Y_train)
    bst = lgb.train( params, train_data, 10 )
    del(X_train)
    del(Y_train)
    del(train_data)
    gc.collect()

    print("\nLGBM Ready!\nSpend {0} seconds\n\n".format( time.time() - start ))

    print("\nTable for test is already done so..")

    print("Now the predictions!")
    start = time.time()
    predictions = pandas.DataFrame([], columns = list(map( lambda x: 'class_'+str(x) ,CLASSES)))

    test_features = pandas.read_hdf( DATA_PATH+"test_features_set.h5", 'features' )
    print( test_features.columns )
    object_ids = test_features.index.values
    predictions = pandas.DataFrame( bst.predict(test_features), columns = list(map( lambda x: 'class_'+str(x) ,CLASSES)) )

    # in case memory error do:
    # for subset in pandas.read_csv(DATA_PATH+"test_features_set.csv", dtype = FEATURE_TYPE, iterator=True ,chunksize = 10*M):

    #     aux = pandas.DataFrame( bst.predict(subset), columns = CLASSES )
    #     print('aux: ', aux)
    #     print('aux: ', aux.sum( axis=1))
    #     print('aux_columns: ', aux.columns)

    #     exit()
    #     predictions = predictions.append( aux )
    # del(subset)
    # gc.collect()


    #Oliver's aproach
    # predict_99 = np.ones(predictions.shape[0])
    # for i in range(predictions.shape[1]):
    #     predict_99 *= (1 - predictions.iloc[:, i])

    # predictions['class_99'] = predict_99 #resultou em uma prababilidade de proximos a 25%, muito alto precisa ser normalizado
    # predictions['class_99']  = 0.14 * predict_99 / np.mean(predict_99)  #veja no comentário https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data#413383


    #Trotta's aproach
    predict_99 = 1 - predictions.max( axis=1 )
    predictions['class_99'] = predict_99 #resultou em uma probabilidade de cerca de 40%, também precisa ser normalizado
    predictions['class_99'] = 0.14 *  predictions['class_99'] / np.mean( predictions['class_99']) #resultou em uma probabilidade de cerca de 40%, também precisa ser normalizado


    #normalization, sum equals to 1
    # sum = predictions.sum( axis=1)
    # for i in range(predictions.shape[1]):
    #     predictions.iloc[:,i] = predictions.iloc[:,i]/sum


    print('object_ids: \n\n', object_ids)
    print('sums: ', predictions.sum( axis=1))

    predictions['object_id'] = object_ids
    predictions.sort_values( by='object_id', inplace=True )
    predictions.set_index('object_id', inplace=True)

    print(" Writing the predictions... ")
    predictions.to_csv( DATA_PATH+'predict_set.csv' )
    print(predictions)
    print("\n Predictions Ready! Right on 'predict_set.csv'\n\nSpend {0} seconds\n\n".format( time.time() - start ))


#NEED CHANGE READ FILES....
def LightBGM_Classifier_ByPassbands():

    print("\n\nExtrating features and reshapping the training dataset...")
    start = time.time()

    dataTraining = readDS.getDataTraining_Features_ByPassband2()

    Y_train = dataTraining.pop('target').to_frame()
    X_train = dataTraining
    print( X_train.dtypes )
    print("\nTable for training is Ready!\nSpend {0} seconds\n\n".format( time.time() - start ))

    print("Creating and training the LightGB Model...")
    start = time.time()

    #parametros para o lgbm
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 14,
        'metric': 'multi_logloss',
        'learning_rate': 0.03,
        'num_iterations': 230,
        'max_depth': 6
    }

    Y_train['target'] = list( map( CLASSES.index, Y_train['target'] ) )

    train_data = lgb.Dataset(X_train, label=Y_train)
    bst = lgb.train( params, train_data, 10 )
    del(X_train)
    del(Y_train)
    del(train_data)
    gc.collect()

    print("\nLGBM Ready!\nSpend {0} seconds\n\n".format( time.time() - start ))

    print("\nTable for test is already done so..")

    print("Now the predictions!")
    start = time.time()
    predictions = pandas.DataFrame([], columns = list(map( lambda x: 'class_'+str(x) ,CLASSES)))

    test_features = pandas.read_hdf( DATA_PATH+"test_passband_features2_set.h5", 'features' )
    object_ids = test_features.index.values

    #Tentar usar o imputer por coluna da tabela, ver InterativeImputer
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    #colocar Knn como Imputer para chavear mais pra frente
    imp = imp.fit(test_features)
    test_features_imp = imp.transform(test_features)

    predictions = pandas.DataFrame( bst.predict(test_features_imp), columns = list(map( lambda x: 'class_'+str(x) ,CLASSES)) )

    # in case memory error do:
    # for subset in pandas.read_csv(DATA_PATH+"test_features_set.csv", dtype = FEATURE_TYPE, iterator=True ,chunksize = 10*M):

    #     aux = pandas.DataFrame( bst.predict(subset), columns = CLASSES )
    #     print('aux: ', aux)
    #     print('aux: ', aux.sum( axis=1))
    #     print('aux_columns: ', aux.columns)



    #Oliver's aproach
    # predict_99 = np.ones(predictions.shape[0])
    # for i in range(predictions.shape[1]):
    #     predict_99 *= (1 - predictions.iloc[:, i])

    # predictions['class_99'] = predict_99 #resultou em uma prababilidade de proximos a 25%, muito alto precisa ser normalizado
    # predictions['class_99']  = 0.18 * predict_99 / np.mean(predict_99)  #veja no comentário https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data#413383


    #Trotta's aproach
    predictions['class_99'] = 1 - predictions.max( axis=1 ) #resultou em uma probabilidade de cerca de 40%, também precisa ser normalizado
    print("média das predições da classe 99", np.nanmean(predictions['class_99']))
    # predictions['class_99'] = 0.14 *  predictions['class_99'] / np.mean( predictions['class_99']) #resultou em uma probabilidade de cerca de 40%, também precisa ser normalizado

    # # #normalization, sum equals to 1
    # sum = predictions.sum( axis=1)
    # for i in range(predictions.shape[1]):
    #     predictions.iloc[:,i] = predictions.iloc[:,i]/sum


    print('object_ids: \n\n', object_ids)
    print('sums: ', predictions.sum( axis=1))

    predictions['object_id'] = object_ids
    predictions.sort_values( by='object_id', inplace=True )
    predictions.set_index('object_id', inplace=True)

    print(" Writing the predictions... ")
    predictions.to_csv( DATA_PATH+'predict_passband2_set.csv' )
    print(predictions)
    print("\n Predictions Ready! Right on 'predict_passband2_set.csv'\n\nSpend {0} seconds\n\n".format( time.time() - start ))


def Tree_Classifier():

    print("\n\nExtrating features and reshapping the training dataset...")
    start = time.time()

    dataTraining = readDS.getDataTraining_Features_ByPassband()

    Y_train = dataTraining.pop('target')
    X_train = dataTraining
    print("\nTable for training is Ready!\nSpend {0} seconds\n\n".format( time.time() - start ))

    print("Creating and training the Tree Decision Model...")
    start = time.time()

    print( X_train.head(), Y_train.head() )
    print('\n\n used on training')
    model = tree.DecisionTreeClassifier(  )
    model = model.fit(X_train, Y_train)
    del(X_train)
    del(Y_train)

    print("\nLGBM Ready!\nSpend {0} seconds\n\n".format( time.time() - start ))

    print("\n\nExtrating features and reshapping the training dataset...\nThis will take a lot more time")
    start = time.time()

    dataTest = pandas.read_hdf( DATA_PATH+"test_passband_features_set.h5", 'features' )
    object_ids = dataTest.index.values
    # Retirada do target pois eu estou usando o dataSet de treino apenas para testar
    # dataTest.pop('target')

    # dataTest.loc[615,'flux_mean_pass_0'] = np.nan
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    #colocar Knn como Imputer para chavear mais pra frente

    imp = imp.fit(dataTest)
    dataTest_imp = imp.transform(dataTest)
    print("\nTable for test is Ready!\nSpend {0} seconds\n\n".format( time.time() - start ))

    print( dataTest.head())
    print('\n\n used on test')
    print("Now the predictions...")
    predictions = model.predict(dataTest_imp)
    print('predictions: ', predictions)

    #Oliver's aproach
    # predict_99 = np.ones(predictions.shape[0])
    # for i in range(predictions.shape[1]):
    #     predict_99 *= (1 - predictions.iloc[:, i])

    # predictions['class_99'] = predict_99 #resultou em uma prababilidade de proximos a 30%, muito alto precisa ser normalizado
    # predictions['class_99']  = 0.14 * predict_99 / np.mean(predict_99)  #veja no comentário https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data#413383


    #Trotta's aproach
    predict_99 = 1 - predictions.max( axis=1 )
    predictions['class_99'] = predict_99 #resultou em uma probabilidade de cerca de 40%, também precisa ser normalizado

    predictions_df = pandas.DataFrame( {'object_id':object_ids,'predictions':predictions} )
    predictions_df.sort_values( by='object_id', inplace=True )
    predictions_df.set_index('object_id', inplace=True)

    print(predictions_df)
    print('Writing predictions')
    predictions_df.to_csv( DATA_PATH+'predict_TreeDecision_set.csv' )
    # print( metrics.classification_report(Y_test, predictions) )
    # print( metrics.confusion_matrix(Y_test, predictions) )
    # print( model.score(X_test, Y_test) )

# LightBGM_Classifier()
# Tree_Classifier()
LightBGM_Classifier_ByPassbands()

def DTree_using_TrainingDataSet():
    dataTraining = readDS.getDataTraining_Features_ByPassband()
    print(dataTraining.shape)

    #treino e test com o dataSet de treino
    Y = dataTraining.pop('target')
    X = dataTraining

    score_min = 1
    score_max = 0

    # for i in range(100):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.975, random_state=33)

    model = tree.DecisionTreeClassifier()
    model = model.fit(X_train, Y_train)

    predictions = model.predict(X_test)

        # score = model.score(X_test, Y_test)
        # if( score < score_min):
        #     score_min = score
        # if( score > score_max ):
        #     score_max = score

        # if(i%10 == 0):
        #     print("#")



    # Precisa rodar no linux e gerar o pdf!!!
    # ver como fazer
    #
    # graph_data = tree.export_graphviz(model, out_file=None)
    # graph = graphviz.Source(graph_data)
    # graph.render("dataTraining")


    print(predictions)
    print(len(predictions))

    print( metrics.classification_report(Y_test, predictions) )
    print( metrics.confusion_matrix(Y_test, predictions) )
    print( model.score(X_test, Y_test) )
    print( 'score min = {0} \n score max = {1}'.format(score_min,score_max))

    tree.plot_tree(model)
    plt.show()
    plt.close()


# dataTest = readDS.getDataTest_FluxMean()
# print(dataTest.shape)
# print(dataTest[-100:])