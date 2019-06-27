from Config import *
import ReadDataSet as readDS
import ExtractFeatures as ExFt
import matplotlib.pyplot as plt
import matplotlib
import pandas
import numpy as np
import seaborn as sns

def removeOutLiers_fromColumns( dataframe ):
    describe = dataframe.describe().loc[ [ 'mean', 'std' ] ]
    for i in range(6):
        column = 'flux_mean_pass_'+str(i)
        mean = describe.loc[ 'mean', column ]
        std = describe.loc[ 'std',  column ]
        top_bound = mean + std*3
        bottom_bound = mean - std*3
        dataframe[column] = dataframe[column].loc[ (dataframe[column] >= bottom_bound) & (dataframe[column] <= top_bound) ]
    return dataframe

def getDataTrain_MeanFluxes():
    datafeatures = ExFt.extract_DataTraining_Means()
    return datafeatures

def getDataTrain_Mean_PassbandFluxes():
    datafeatures = ExFt.extract_DataTraining_Means_byPassband()
    targets = datafeatures['target'].unique().tolist()
    datafeatures.target = datafeatures.target.astype('category')
    return datafeatures

def getDataTrain_LogMeanFluxes():
    dt_FluxMeanLog = ExFt.extract_DataTraining_Means()
    print(dt_FluxMeanLog.columns)

    dt_FluxMeanLog.iloc[:,12:24] = dt_FluxMeanLog.iloc[:,12:24].apply( np.log )

    return dt_FluxMeanLog

def getFigure( n=None ):
    return plt.figure( constrained_layout=False, num=n, figsize=[16,11], dpi=110 )



def DataAnalysisSeparetePassbands():
    dataTraining = getDataTrain_MeanFluxes()
    print(dataTraining.columns)
    #plot das medias dos fluxos de todos os objetos do grupo de dados

    fig = getFigure()
    ax = fig.add_subplot( 111, ylim=(-1000,1000))
    sc = plt.scatter( x=dataTraining['id'], y=dataTraining['flux_mean_pass_0'], s=15, cmap='gist_ncar' ,c=dataTraining['target'], label=CLASSES, alpha=.8 )
    legend = ax.legend(*sc.legend_elements(),loc="upper right", title="Classes")
    ax.add_artist(legend)
    plt.title("Flux mean by passband 0" )
    # fig.title("Média dos fluxos de todos os objetos de treino")
    plt.show()
    plt.close()
    #remoção dos outliers para melhor visualização da distribuição geral

    fig = getFigure()
    for i in range(6):
        fig.add_subplot( 3, 2, i+1)
        plt.scatter( x=dataTraining['id'], y=dataTraining['flux_mean_pass_'+str(i)], s=15, cmap='gist_ncar', c=dataTraining['target'], alpha=.8 )
        plt.title("Flux mean by passband "+PASSBANDS[i] )
    # fig.title("Média dos fluxos de todos os objetos de treino")
    plt.show()
    plt.close()
    #remoção dos outliers para melhor visualização da distribuição geral


    #como podemos ver a média dos fluxos estão muitos próximas umas as outras em relação ao todo
    #portanto veremos a nova media e sera retirado novamente os elementos com mais de 3 vezes
    #o desvio padrao do conjunto

    fig = getFigure()
    for i in range(6):
        fig.add_subplot( 3, 2, i+1, ylim=(-1000,1000))
        plt.scatter( x=dataTraining['id'], y=dataTraining['flux_mean_pass_'+str(i)], s=10, c=dataTraining['target'] )
        plt.title("Flux mean by passband "+PASSBANDS[i] )
    # fig.title("Média dos fluxos de todos os objetos de treino")
    plt.show()
    plt.close()
    #remoção dos outliers para melhor visualização da distribuição geral
    fig = getFigure()
    for i in range(6):
        fig.add_subplot( 3, 2, i+1, ylim=(-3000,-1000))
        plt.scatter( x=dataTraining['id'], y=dataTraining['flux_mean_pass_'+str(i)], s=50, c=dataTraining['target'] )
        plt.title("Flux mean by passband "+PASSBANDS[i] )
    # fig.title("Média dos fluxos de todos os objetos de treino")
    plt.show()
    plt.close()

    fig = getFigure()
    for i in range(6):
        fig.add_subplot( 3, 2, i+1, ylim=(1000,3000))
        plt.scatter( x=dataTraining['id'], y=dataTraining['flux_mean_pass_'+str(i)], s=50, c=dataTraining['target'] )
        plt.title("Flux mean by passband "+PASSBANDS[i] )
    # fig.title("Média dos fluxos de todos os objetos de treino")
    plt.show()
    plt.close()

    fig = getFigure()
    for i in range(6):
        fig.add_subplot( 3, 2, i+1, ylim=(-100,100))
        plt.scatter( x=dataTraining['id'], y=dataTraining['flux_mean_pass_'+str(i)] ,c=dataTraining['target'], s=10 , alpha=.6)
        plt.title("Flux mean by passband "+PASSBANDS[i] )
    plt.show()
    plt.close()


    fig = getFigure()
    dataTrainingDDF = dataTraining[ dataTraining.ddf > 0 ]
    for i in range(6):
        fig.add_subplot( 3, 2, i+1, ylim=(-10,10))
        plt.scatter( x=dataTrainingDDF['id'], y=dataTrainingDDF['flux_mean_pass_'+str(i)] ,c=dataTrainingDDF['target'], s=10 , alpha=.6)
        plt.title("Flux mean by passband "+PASSBANDS[i] )
    plt.show()
    plt.close()

    fig = getFigure()
    dataTrainingWDF = dataTraining[ dataTraining.ddf < 1 ]
    for i in range(6):
        fig.add_subplot( 3, 2, i+1, ylim=(-100,100))
        plt.scatter( x=dataTrainingWDF['id'], y=dataTrainingWDF['flux_mean_pass_'+str(i)], cmap='gist_ncar' ,c=dataTrainingWDF['target'], s=10 , alpha=.6)
        plt.title("Flux mean by passband "+PASSBANDS[i] )
    plt.show()
    plt.close()


    GAL_MASK = np.isnan(dataTraining.distmod)

    #Pegar as classes que aparecem somente no ambiente galactico

    cmap = matplotlib.cm.get_cmap('gist_ncar')
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=13.0)

    GAL_CLASSES = [0,2,5,8,12]
    EXGAL_CLASSES = [1,3,4,6,7,9,10,11,13]
    barColors = list(map (cmap, map(norm, range(14))))
    barColorsReverse = list(map (cmap, map(norm, range(13,-1,-1))))
    barColorsGal = list(map (cmap, map(norm, GAL_CLASSES)))
    barColorsExGal = list(map (cmap, map(norm, EXGAL_CLASSES)))

    dataTraining['color'] = list(map (cmap, map(norm, dataTraining['target'])))


    fig = getFigure()
    dataTrainingG = dataTraining[ GAL_MASK ]
    for i in range(6):
        fig.add_subplot( 3, 2, i+1,ylim = (-2000,2000))
        plt.scatter( x=dataTrainingG['id'], y=dataTrainingG['flux_mean_pass_'+str(i)], cmap='gist_ncar' ,c=dataTrainingG['target'], s=10 , alpha=1)
        plt.title("Flux mean by passband "+PASSBANDS[i] )
    plt.show()
    plt.close()

    fig = getFigure()
    dataTrainingEG = dataTraining[ ~GAL_MASK  ]
    for i in range(6):
        fig.add_subplot( 3, 2, i+1,ylim =(-2000,2000))
        plt.scatter( x=dataTrainingEG['id'], y=dataTrainingEG['flux_mean_pass_'+str(i)], cmap='gist_ncar'  ,c=dataTrainingEG['target'], s=10 , alpha=1)
        plt.title("Flux mean by passband "+PASSBANDS[i] )
    plt.show()
    plt.close()


    #uma visão melhor dos dados usando histograma
    #print(dataTraining.describe()['flux_mean_pass_0'])
    fig = getFigure()
    ax1 = fig.add_subplot(211, title='Histogram')
    ax2 = fig.add_subplot(212, title='Logs histogram')

    dataTrainingNorm = removeOutLiers_fromColumns( dataTraining )
    dataTrainingNorm = removeOutLiers_fromColumns( dataTrainingNorm )
    dataTrainingNorm = removeOutLiers_fromColumns( dataTrainingNorm )
    # ax1.title("Frequência das médias particionadas em 50 bins")
    dataTrainingNorm.loc[:,['flux_mean_pass_0']].plot.hist( bins=50, ax=ax1 )

    dtLog = dataTrainingNorm['flux_mean_pass_0'].apply(np.log)
    # ax2.title("Frequência do log das médias")
    dtLog.plot.hist( bins=50, ax=ax2)
    plt.show()
    plt.close()

    # dfClassPerPass = ExFt.extract_DataTraining_CountClasses()

    # dfFluxPerClassPerPass = dtLog[dtLog.id < 0]

    for i in range(2):

        fig = getFigure()
        fluxMeanPass = 'flux_mean_pass_'+str(i)
        ax1 = fig.add_subplot(211, title='Histogram on passband '+PASSBANDS[i])
        k = { 'linewidth' : 3 }
        pandas.DataFrame(
            {  '0': dataTrainingNorm.loc[ dataTrainingNorm.target==0,fluxMeanPass],   '1': dataTrainingNorm.loc[ dataTrainingNorm.target==1,fluxMeanPass],
               '2': dataTrainingNorm.loc[ dataTrainingNorm.target==2,fluxMeanPass],   '3': dataTrainingNorm.loc[ dataTrainingNorm.target==3,fluxMeanPass],
               '4': dataTrainingNorm.loc[ dataTrainingNorm.target==4,fluxMeanPass],   '5': dataTrainingNorm.loc[ dataTrainingNorm.target==5,fluxMeanPass],
               '6': dataTrainingNorm.loc[ dataTrainingNorm.target==6,fluxMeanPass],   '7': dataTrainingNorm.loc[ dataTrainingNorm.target==7,fluxMeanPass],
               '8': dataTrainingNorm.loc[ dataTrainingNorm.target==8,fluxMeanPass],   '9': dataTrainingNorm.loc[ dataTrainingNorm.target==9,fluxMeanPass],
               '10': dataTrainingNorm.loc[ dataTrainingNorm.target==10,fluxMeanPass], '11': dataTrainingNorm.loc[ dataTrainingNorm.target==11,fluxMeanPass],
               '12': dataTrainingNorm.loc[ dataTrainingNorm.target==12,fluxMeanPass], '13': dataTrainingNorm.loc[ dataTrainingNorm.target==13,fluxMeanPass]
            }
        ).plot.kde( ax=ax1, color=barColors , **k )
        ax1.legend(loc="upper right", title="Classes" )

        k['linewidth'] = 3
        ax2 = fig.add_subplot(212, title='Logs histogram on passband '+PASSBANDS[i])
        plt.hist(
            [
                dataTrainingNorm.loc[ dataTrainingNorm.target==13,fluxMeanPass],dataTrainingNorm.loc[ dataTrainingNorm.target==12,fluxMeanPass],
                dataTrainingNorm.loc[ dataTrainingNorm.target==11,fluxMeanPass],dataTrainingNorm.loc[ dataTrainingNorm.target==10,fluxMeanPass],
                dataTrainingNorm.loc[ dataTrainingNorm.target==9,fluxMeanPass],dataTrainingNorm.loc[ dataTrainingNorm.target==8,fluxMeanPass],
                dataTrainingNorm.loc[ dataTrainingNorm.target==7,fluxMeanPass],dataTrainingNorm.loc[ dataTrainingNorm.target==6,fluxMeanPass],
                dataTrainingNorm.loc[ dataTrainingNorm.target==5,fluxMeanPass],dataTrainingNorm.loc[ dataTrainingNorm.target==4,fluxMeanPass],
                dataTrainingNorm.loc[ dataTrainingNorm.target==3,fluxMeanPass],dataTrainingNorm.loc[ dataTrainingNorm.target==2,fluxMeanPass],
                dataTrainingNorm.loc[ dataTrainingNorm.target==1,fluxMeanPass],dataTrainingNorm.loc[ dataTrainingNorm.target==0,fluxMeanPass]
            ],
            bins=100, histtype='step' ,log=True, color= barColorsReverse, **k, label=range(13,-1,-1)
        )
        ax2.legend(loc="upper right", title="Classes" )
        plt.show()
        plt.close()

    for i in [0,2,5,8,12]:
        print('\n\n {}'.format(i))
        print('\n pass 4: \n')
        print(  ((dataTrainingNorm.loc[ (dataTrainingNorm.target==i)&(~np.isnan(dataTrainingNorm.flux_mean_pass_4)), 'flux_mean_pass_4' ])))
        print('\n pass 5: \n')
        print(  ((dataTrainingNorm.loc[ (dataTrainingNorm.target==i)&(~np.isnan(dataTrainingNorm.flux_mean_pass_5)), 'flux_mean_pass_5' ])) )


    for i in range(6):

        fig = getFigure()
        fluxMeanPass = 'flux_mean_pass_'+str(i)
        ax1 = fig.add_subplot(211, title='Histogram on passband '+PASSBANDS[i])
        k = { 'linewidth' : 3 }
        if(i<4):
            pandas.DataFrame(
                {  '0': dataTrainingNorm.loc[ dataTrainingNorm.target==0,fluxMeanPass],'2': dataTrainingNorm.loc[ dataTrainingNorm.target==2,fluxMeanPass],
                '5': dataTrainingNorm.loc[ dataTrainingNorm.target==5,fluxMeanPass],'8': dataTrainingNorm.loc[ dataTrainingNorm.target==8,fluxMeanPass],
                '12': dataTrainingNorm.loc[ dataTrainingNorm.target==12,fluxMeanPass]
                }
            ).plot.kde( ax=ax1, color=barColorsGal, **k )
            ax1.legend(loc="upper right", title="Classes" )

        ax2 = fig.add_subplot(212, title='Logs histogram on passband '+PASSBANDS[i])
        plt.hist(
            [
               dataTrainingNorm.loc[ dataTrainingNorm.target==0,fluxMeanPass], dataTrainingNorm.loc[ dataTrainingNorm.target==2,fluxMeanPass],
               dataTrainingNorm.loc[ dataTrainingNorm.target==5,fluxMeanPass], dataTrainingNorm.loc[ dataTrainingNorm.target==8,fluxMeanPass],
               dataTrainingNorm.loc[ dataTrainingNorm.target==12,fluxMeanPass]
            ],
            bins=30, histtype='bar',log=True, color= barColorsGal,label=GAL_CLASSES ,**k
        )
        ax2.legend(loc="upper right", title="Classes" )
        plt.show()
        plt.close()

    for i in range(6):

        fig = getFigure()
        fluxMeanPass = 'flux_mean_pass_'+str(i)
        print(i)
        ax1 = fig.add_subplot(211, title='Histogram on passband '+PASSBANDS[i])
        k = { 'linewidth' : 3 }
        pandas.DataFrame(
            {   '1': dataTrainingNorm.loc[ dataTrainingNorm.target==1,fluxMeanPass], '3': dataTrainingNorm.loc[ dataTrainingNorm.target==3,fluxMeanPass],
                '4': dataTrainingNorm.loc[ dataTrainingNorm.target==4,fluxMeanPass], '6': dataTrainingNorm.loc[ dataTrainingNorm.target==6,fluxMeanPass],
                '7': dataTrainingNorm.loc[ dataTrainingNorm.target==7,fluxMeanPass], '9': dataTrainingNorm.loc[ dataTrainingNorm.target==9,fluxMeanPass],
                '10': dataTrainingNorm.loc[ dataTrainingNorm.target==10,fluxMeanPass], '11': dataTrainingNorm.loc[ dataTrainingNorm.target==11,fluxMeanPass],
                '13': dataTrainingNorm.loc[ dataTrainingNorm.target==13,fluxMeanPass]
            }
        ).plot.kde( ax=ax1, color=barColorsExGal, **k )
        ax1.legend(loc="upper right", title="Classes" )

        ax2 = fig.add_subplot(212, title='Logs histogram on passband '+PASSBANDS[i])
        plt.hist(
            [
                dataTrainingNorm.loc[ dataTrainingNorm.target==1,fluxMeanPass],dataTrainingNorm.loc[ dataTrainingNorm.target==3,fluxMeanPass],
                dataTrainingNorm.loc[ dataTrainingNorm.target==4,fluxMeanPass],dataTrainingNorm.loc[ dataTrainingNorm.target==6,fluxMeanPass],
                dataTrainingNorm.loc[ dataTrainingNorm.target==7,fluxMeanPass],dataTrainingNorm.loc[ dataTrainingNorm.target==9,fluxMeanPass],
                dataTrainingNorm.loc[ dataTrainingNorm.target==10,fluxMeanPass],dataTrainingNorm.loc[ dataTrainingNorm.target==11,fluxMeanPass],
                dataTrainingNorm.loc[ dataTrainingNorm.target==13,fluxMeanPass]
            ],
            bins=20, histtype='bar', log=True, color= barColorsExGal, **k, label=EXGAL_CLASSES
        )
        ax2.legend(loc="upper right", title="Classes" )
        plt.show()
        plt.close()

#NEED TO GETOUT THE PASSBANDS
def DataAnalysisInitial():
    dataTraining = getDataTrain_MeanFluxes()
    dataTraining.loc[:,['flux_mean']].plot.hist( bins=50, alpha=0.5 )
    # plt.show()
    plt.close()

    #removing outliers
    fig = getFigure()
    ax1 = fig.add_subplot( 311, title='Média dos Fluxos de cada objeto' )
    ax2 = fig.add_subplot( 312, title='Mediana dos Fluxos de cada objeto' )
    ax3 = fig.add_subplot( 313, title='Desvio Padrão dos Fluxos de cada objeto' )

    dataTraining.loc[ (dataTraining.flux_mean < dataTraining['flux_mean'].quantile( 0.95 )) &
        (dataTraining.flux_mean > dataTraining['flux_mean'].quantile( 0.05 )),
    'flux_mean' ].plot.hist( bins=100, alpha=0.8, ax=ax1)

    dataTraining.loc[ (dataTraining.flux_median < dataTraining['flux_median'].quantile( 0.95 )) &
        (dataTraining.flux_median > dataTraining['flux_median'].quantile( 0.05 )),
    'flux_median' ].plot.hist( bins=100, alpha=0.8, ax=ax2 )

    dataTraining.loc[ (dataTraining.flux_std < dataTraining['flux_std'].quantile( 0.95 )) &
        (dataTraining.flux_std > dataTraining['flux_std'].quantile( 0.05 )),
    'flux_std' ].plot.hist( bins=100, alpha=0.8, ax=ax3 )
    plt.show()
    plt.close()
    #Mesmo refazendo esse processo os dados continuam agrupados e aplicar a função de log
    #parece ser a melhor opção, agora comparado a raiz dos fluxos
    #print(dataTraining.describe()['flux_mean_pass_0'])
    fig = getFigure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    dtLog = dataTraining.copy()
    dtLog['flux_mean_pass_0'] = dtLog['flux_mean_pass_0'].apply( np.log )
    dtLog['flux_mean_pass_0'].plot.hist( bins=20, ax=ax1 )

    dtSqr = dataTraining.copy()
    dtSqr['flux_mean_pass_0'] = dtSqr['flux_mean_pass_0'].apply( lambda x: x**(1/3) )
    dtSqr['flux_mean_pass_0'].plot.hist( bins=50, ax=ax2 )

    # plt.show()
    plt.close()

    #plotagem dos fluxos como pontos
    dtLog['id'] = range(dtLog.shape[0])
    dtLog.plot.scatter( x='id', y='flux_mean_pass_0')

    ax2 = fig.add_subplot(212)
    dtSqr['id'] = range(dtSqr.shape[0])
    dtSqr.plot.scatter( x='id', y='flux_mean_pass_0')
    # plt.show()
    plt.close()

    #a função log mantém melhor a mais distribuição original,
    #e há visivelmente uma diferença entre as médias a partir
    #do objeto de id 2200 aproximadamente

    #veremos algumas relações entre os atributos agora...

    dtLog.plot.scatter( x='id', y='flux_mean_pass_0', c='ddf', colormap='viridis', alpha=0.9 )
    # plt.show()
    plt.close()

    # é possível identificar como o atributo ddf aumenta a média dos fluxos das classes em geral
    # veremos agora cada classe separadamente para encontrar alguma distinção do comportamento das
    # médias

    fig = getFigure()
    categories = list( dataTraining.target.cat.categories.values )

    plots = []
    for ctgs in categories:
        plots = dataTraining[ dataTraining.target==ctgs ]

        # Para achar os limites maximos e minimos
        # print(plots['flux_mean_pass_0'].min())
        # print(plots['flux_mean_pass_0'].max())

        fig.add_subplot( 4, 4, categories.index(ctgs)+1, ylim=(-450,450))
        path = plt.scatter( x = plots['id'], y = plots['flux_mean_pass_0'], s = 1, c='green' )


    # plt.show()
    plt.close()


    # Usando o log das média aparenta ter uma distinção maior entre as classes...

    fig = plt.figure()

    plots_log = []
    for ctgs in categories:
        plots_log = dataTraining[ dataTraining.target==ctgs ]
        plots_log['flux_mean_pass_0'] = plots_log['flux_mean_pass_0'].apply( np.log )
        #print(plots_log['flux_mean_pass_0'].min())
        #print(plots_log['flux_mean_pass_0'].max())
        fig.add_subplot( 4, 4, categories.index(ctgs)+1,ylim=(-7,7))
        path = plt.scatter( x = plots_log['id'], y = plots_log['flux_mean_pass_0'], s = 1, c='green' )


    # plt.show()
    plt.close()

    # Ainda podemos ver como cada classe se comporta em todos os passabands
    # e tentar vem alguma separação entra elas

    limits = []
    plots = []
    for ctgs in categories:

        fig = plt.figure()
        plots = dataTraining[ dataTraining.target==ctgs ]
        print(plots['flux_mean_pass_1'].max())

        lim = [0,0]

        fig.add_subplot( 3, 2, 1)
        path = plt.scatter( x = plots['id'], y = plots['flux_mean_pass_0'], s = 1, c='darkviolet' )

        fig.add_subplot( 3, 2, 2)
        path = plt.scatter( x = plots['id'], y = plots['flux_mean_pass_1'], s = 1, c='green' )

        fig.add_subplot( 3, 2, 3)
        path = plt.scatter( x = plots['id'], y = plots['flux_mean_pass_2'], s = 1, c='orangered' )

        fig.add_subplot( 3, 2, 4)
        path = plt.scatter( x = plots['id'], y = plots['flux_mean_pass_3'], s = 1, c='firebrick' )

        fig.add_subplot( 3, 2, 5)
        path = plt.scatter( x = plots['id'], y = plots['flux_mean_pass_4'], s = 1, c='maroon' )

        fig.add_subplot( 3, 2, 6)
        path = plt.scatter( x = plots['id'], y = plots['flux_mean_pass_5'], s = 1, c='dimgrey' )

        # plt.show()

    plt.close()

    # Visualizar algum padrão específico desta maneira ainda é muito difícil
    # Posteriormente será feito outra análise com uma representação melhor dos fluxos

    # Por enquanto podemos ver outras relações entre as variáveis citadas pelos competidores no Kaggle
    # como a relação entre os gaps dos dados e a longitude celestial do objeto
    # e a posição geral de cada classe no céu

    # Ou então ver quais classes se sobrepoem no ddf já que possuem uma distinção grande das médias dos fluxos

    # Classes na região de ddf

    targetPerDDF = pandas.DataFrame( dataTraining.groupby(['ddf','target']).count()['id'] )
    targetPerDDF = targetPerDDF.reset_index()

    # print(targetPerDDF)

    sns.catplot(x="ddf", y="id", hue="target", kind="bar", data=targetPerDDF)
    # plt.show()
    plt.close()

    #Claramente existem mais elementos na WDF do que na DDF pela sua definição
    #vamos mostrar proporcionalmente então

    idsSum_ddf0 = targetPerDDF.loc[targetPerDDF.ddf==0, 'id'].sum()
    idsSum_ddf1 = targetPerDDF.loc[targetPerDDF.ddf==1, 'id'].sum()
    print(idsSum_ddf0,idsSum_ddf1)
    targetPerDDFRelative = targetPerDDF.copy()

    for lab,row in targetPerDDFRelative.iterrows():
        if(row.ddf):
            targetPerDDFRelative.loc[lab,'id'] = row.id/idsSum_ddf1
        else:
            targetPerDDFRelative.loc[lab,'id'] = row.id/idsSum_ddf0

    print(targetPerDDFRelative)

    sns.catplot(x="ddf", y="id", hue="target", kind="bar", data=targetPerDDFRelative)
    # plt.show()
    plt.close()

    # É possível ver que alguma classes são mais presentes em certa tipo de visualização
    # a classes 6,15,16,53, 64 estão mais presentes na wdf
    # enquanto as classes 52 e a 90 estão mais presentes na ddf
    # e as outras estão distribuidas aparentemente iguais

    n_rows = targetPerDDFRelative.shape[0]//2

    ClassNumber = pandas.DataFrame()
    ClassNumber['DDF0'] = targetPerDDFRelative.iloc[:n_rows,-1]
    ClassNumber['DDF1'] = targetPerDDFRelative.iloc[ n_rows:,-1].values
    ClassNumber['target'] = targetPerDDFRelative.target[:n_rows]
    ClassNumber['diff'] = [0]*n_rows
    ClassNumber['Pr_Class_In_DDF'] = [0]*n_rows
    ClassNumber['Pr_Class_In_WDF'] = [0]*n_rows

    for lab,row in ClassNumber.iterrows():
        if(row.DDF0 < row.DDF1):
            ClassNumber.loc[lab,'diff'] = (row.DDF1-row.DDF0)/row.DDF1
            ClassNumber.loc[lab,'DDF_has_more_class'] = row.target
        else:
            ClassNumber.loc[lab,'diff'] = (row.DDF0-row.DDF1)/row.DDF0
            ClassNumber.loc[lab,'WDF_has_more_class'] = row.target

        if( ClassNumber.loc[lab,'diff'] < .20 ):
            ClassNumber.loc[lab,'DDF_has_more_class'] = None
            ClassNumber.loc[lab,'WDF_has_more_class'] = None

        ClassNumber.loc[lab,'Pr_Class_In_DDF'] = row.DDF0/(row.DDF0+row.DDF1)
        ClassNumber.loc[lab,'Pr_Class_In_WDF'] = row.DDF1/(row.DDF0+row.DDF1)

    print(ClassNumber)



#Meta Data Training Analysis
def DataAnalysis():
    # Aqui veremos comos os objetos de cada classe estão dispostos no céu com base em sua latitude e longitude

    dataTraining = getDataTrain_MeanFluxes()

    fig = getFigure()
    plots = []
    for ctgs in range(14):
        plots = dataTraining[ dataTraining.target==ctgs ]

        ax= fig.add_subplot( 4, 4, ctgs+1)
        plt.scatter( x = plots['ra'], y = plots['decl'], s = 1, c='lightskyblue' )
        plt.scatter( x = plots['ra'], y = plots['decl'], s = 1, c='white' )
        ax.set_facecolor((.05,.05,.05,.9))

    plt.show()
    plt.close()

    print(dataTraining.columns)

    fig = getFigure()
    plots = []
    for ctgs in range(14):
        plots = dataTraining[ dataTraining.target==ctgs ]

        ax = fig.add_subplot( 4, 4, ctgs+1)
        plt.scatter( x = plots['gal_l'], y = plots['gal_b'], s = 1, c='lightskyblue' )
        ax.set_facecolor((.05,0,.05,.9))

    plt.show()
    plt.close()
    exit()

    # Foi descoberto que o atributo distmod possui valores nulos,
    # e como visto que quanto mais longe um objeto está, maior seu redshift,
    # pode ser que haja uma relação entre os atributos distmod e o hostgal_photoz,
    # para poder estimar um valor para os objetos que não o possuem

    print(dataTraining.columns)

    plt.scatter( x = dataTraining['hostgal_specz'], y = dataTraining['distmod'], s = 1, c='red' )

    plt.show()
    plt.close()

    # Adiante pode se usar uma GPM( Gaussian Processes Model) para estimar o valor do distmod
    # mas por enquanto iremos apenas usar uma função simples a fim de teste
    fig = getFigure()

    plt.scatter( x = (np.log2(dataTraining['hostgal_specz'])*1.8)+44 , y = dataTraining['distmod'], s = 1, c='red' )

    plt.show()
    plt.close()

    # Existe a relação porém todo elemento com distmod Nan possui 0 de hostgal_specz,
    # provável que distmod seja calculado a partir do hostgal_specz

    # É preciso encontrar uma relação entre distmod e alguma variável
    corr = dataTraining.corr()

    plt.matshow(corr)
    plt.show()
    plt.close()

    sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

    corr.style.background_gradient(cmap='coolwarm')
    plt.show()
    plt.close()


    plt.show()
    plt.close()


#Com base nessas análises podemos gerar algumas features diferentes para o classificador
#como getDataTrain_LogMeanFluxes() além do dataset inicial com as médias.

DataAnalysisInitial()
DataAnalysis()
DataAnalysisSeparetePassbands()

#println(dataTraining[dataTraining.object_id == 92])
#exClass.index.map( lambda id: dataTraining.query(expr = 'object_id == @id').plot.scatter(x = 'mjd', y = 'flux', c='passband', colormap='hsv') )

