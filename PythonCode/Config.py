import numpy as np
import pandas
import gc
import time


M = 1000000

DATA_PATH = "C:/Users/marcio.freitas/Desktop/MonografiaTCC/DataSet/"

DATA_TYPE = {'object_id': np.int32, 'mjd': np.float32, 'passband': np.int8, 'flux': np.float32, 'flux_err': np.float32, 'detected': np.int8}

METADATA_TYPE_TRAINING = {'object_id': np.int32, 'ra': np.float32, 'decl': np.float32, 'gal_l': np.float32, 'gal_b': np.float32,
'ddf': np.int8, 'hostgal_specz': np.float32, 'hostgal_photoz': np.float32, 'hostgal_photoz_err': np.float32, 'distmod': np.float32,
'mwebv': np.float32, 'target': np.int8}

METADATA_TYPE_TEST = {'object_id': np.int32, 'ra': np.float32, 'decl': np.float32, 'gal_l': np.float32, 'gal_b': np.float32,
'ddf': np.int8, 'hostgal_specz': np.float32, 'hostgal_photoz': np.float32, 'hostgal_photoz_err': np.float32, 'distmod': np.float32,
'mwebv': np.float32}

DATA_COLUMNS = ['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected']

METADATA_COLUMNS_TRAINING = ['object_id', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv', 'target']

METADATA_COLUMNS_TEST = ['object_id', 'ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv']

CLASSES = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]

PASSBANDS = { 0:'u', 1:'g', 2:'r', 3:'i', 4:'z', 5:'y'}

FEATURE_TYPE = {'object_id':	np.int32,
'ra':	np.float32,
'decl':	np.float32,
'gal_l':	np.float32,
'gal_b':	np.float32,
'ddf':	np.float16 ,
'hostgal_specz':	np.float32,
'hostgal_photoz':	np.float32,
'hostgal_photoz_err':	np.float32,
'distmod':	np.float32,
'mwebv':	np.float32,
'flux_mean_pass_0':	np.float32,
'flux_mean_pass_1':	np.float32,
'flux_mean_pass_2':	np.float32,
'flux_mean_pass_3':	np.float32,
'flux_mean_pass_4':	np.float32,
'flux_mean_pass_5':	np.float32,
'flux_mean_pass_err_0':	np.float32,
'flux_mean_pass_err_1':	np.float32,
'flux_mean_pass_err_2':	np.float32,
'flux_mean_pass_err_3':	np.float32,
'flux_mean_pass_err_4':	np.float32,
'flux_mean_pass_err_5':	np.float32,
'flux_median_pass_0':	np.float32,
'flux_median_pass_1':	np.float32,
'flux_median_pass_2':	np.float32,
'flux_median_pass_3':	np.float32,
'flux_median_pass_4':	np.float32,
'flux_median_pass_5':	np.float32,
'flux_median_pass_err_0':	np.float32,
'flux_median_pass_err_1':	np.float32,
'flux_median_pass_err_2':	np.float32,
'flux_median_pass_err_3':	np.float32,
'flux_median_pass_err_4':	np.float32,
'flux_median_pass_err_5':	np.float32,
'flux_std_pass_0':	np.float32,
'flux_std_pass_1':	np.float32,
'flux_std_pass_2':	np.float32,
'flux_std_pass_3':	np.float32,
'flux_std_pass_4':	np.float32,
'flux_std_pass_5':	np.float32,
'flux_std_pass_err_0':	np.float32,
'flux_std_pass_err_1':	np.float32,
'flux_std_pass_err_2':	np.float32,
'flux_std_pass_err_3':	np.float32,
'flux_std_pass_err_4':	np.float32,
'flux_std_pass_err_5':	np.float32,
'flux_var_pass_0':	np.float32,
'flux_var_pass_1':	np.float32,
'flux_var_pass_2':	np.float32,
'flux_var_pass_3':	np.float32,
'flux_var_pass_4':	np.float32,
'flux_var_pass_5':	np.float32,
'flux_var_pass_err_0':	np.float32,
'flux_var_pass_err_1':	np.float32,
'flux_var_pass_err_2':	np.float32,
'flux_var_pass_err_3':	np.float32,
'flux_var_pass_err_4':	np.float32,
'flux_var_pass_err_5':	np.float32,
'flux_min_pass_0':	np.float32,
'flux_min_pass_1':	np.float32,
'flux_min_pass_2':	np.float32,
'flux_min_pass_3':	np.float32,
'flux_min_pass_4':	np.float32,
'flux_min_pass_5':	np.float32,
'flux_min_pass_err_0':	np.float32,
'flux_min_pass_err_1':	np.float32,
'flux_min_pass_err_2':	np.float32,
'flux_min_pass_err_3':	np.float32,
'flux_min_pass_err_4':	np.float32,
'flux_min_pass_err_5':	np.float32,
'flux_max_pass_0':	np.float32,
'flux_max_pass_1':	np.float32,
'flux_max_pass_2':	np.float32,
'flux_max_pass_3':	np.float32,
'flux_max_pass_4':	np.float32,
'flux_max_pass_5':	np.float32,
'flux_max_pass_err_0':	np.float32,
'flux_max_pass_err_1':	np.float32,
'flux_max_pass_err_2':	np.float32,
'flux_max_pass_err_3':	np.float32,
'flux_max_pass_err_4':	np.float32,
'flux_max_pass_err_5':	np.float32,
'flux_range_pass_0':	np.float32,
'flux_range_pass_1':	np.float32,
'flux_range_pass_2':	np.float32,
'flux_range_pass_3':	np.float32,
'flux_range_pass_4':	np.float32,
'flux_range_pass_5':	np.float32,
'flux_range_pass_err_0':	np.float32,
'flux_range_pass_err_1':	np.float32,
'flux_range_pass_err_2':	np.float32,
'flux_range_pass_err_3':	np.float32,
'flux_range_pass_err_4':	np.float32,
'flux_range_pass_err_5':	np.float32,
'mjd_min':	np.float32,
'mjd_max':	np.float32,
'detected_count_pass_0':	np.float16,
'detected_count_pass_1':	np.float16,
'detected_count_pass_2':	np.float16,
'detected_count_pass_3':	np.float16,
'detected_count_pass_4':	np.float16,
'detected_count_pass_5':    np.float16 }
