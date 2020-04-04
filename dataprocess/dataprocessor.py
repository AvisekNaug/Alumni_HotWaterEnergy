"""
This script contains all the data processing activities that are needed
before it can be provided to any other object
"""


import os
import glob
from typing import Union


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


import pandas as pd
import swifter
from scipy import stats
from scipy.fftpack import fft
import scipy.signal as signal


from matplotlib import pyplot as plt


# sources of data for which processing can be done
DATASOURCE = ['BdX']
DATATYPE = ['.csv', '.xlsx', '.pkl']


from dataprocess.decorators import timer


# methods plugin dictionary
PLUGINS = dict()
def register(func):
	"""Register a function as a plug-in to call it via string
	
	Arguments:
		func {python function object} -- function to register
	
	Returns:
		python function object -- return same function
	"""
	global PLUGINS
	PLUGINS[func.__name__] = func

	return func


class readfolder():
	"""
	*List or read multiple files from a Directory or Folder ONLY
	*To read the files use file2df method after initializing
	*To return merged dataframe use bothe file2df and then mergerows
	**If specifying full file path use "readfile" class
	Reads data of same variables from raw sources of types mentioned in DATATYPE
	The individual files should have the same last(-1) dimensions. Note that, it
	will assume that there is a timeseries data column in the files
	"""
	def __init__(self, datadir, fileformat=None, timeformat = '%Y-%m-%d %H:%M:%S', 
	dateheading='Date'):
		"""Lists the files to read from the folder
		
		Arguments:
			datadir {str} -- Folder path to read from
		
		Keyword Arguments:
			fileformat {str} -- Provide the regex in case a selected number of files
			 with a certain pattern should be read. The expression to perform Unix 
			 style pathname pattern expansion. (default: {None})
			timeformat {str} -- Format for datetime parsing
			 (default: {'%Y-%m-%d %H:%M:%S'})
			dateheading {str} -- DateColumn Name in the raw data to read date from
			 (default: {'Date'})
		"""

		#---------ensure files to read are available and in required format------

		# make sure it is an existing directory
		assert os.path.isdir(datadir), "Directory not found"

		# attach forward slash if datadir does not have one
		if not datadir.endswith('/'):
			datadir += '/'

		# list the files in the directory
		if fileformat is None:  # read all files from the directory
			flist = glob.glob(os.path.join(datadir, '*'))
		else:
			flist = glob.glob(os.path.join(datadir, fileformat))

		# check whether list has any of the DATATYPEs that we can read from
		dtype_error = "None of the files in folder " + datadir + " is either of the data types "+\
			 " ".join(DATATYPE)
		# TODO: Remove this line below as it might be redundant
		assert any([fname.endswith(tuple(DATATYPE)) for fname in flist]), dtype_error

		# select files with the required extension
		readlist = [fname for fname in flist if fname.endswith(tuple(DATATYPE))]
		# check whether the is list has one element
		assert readlist, dtype_error
		
		#------------------------------------------------------------------------

		# assign path to read data from
		self.read_list = readlist

		# select time-format
		self.timeformat = timeformat
		self.dateheading = dateheading

		# read empty dataframes into lists
		self.dflist = []

		# create empty datframe variable
		self.df = None

	def return_df(self, processmethods: list = ['files2dflist',\
		 'datetime_parse_dflist']):

		"""Perform all the basic processes on the raw data and return processed
		dataframe in case user is not interested in doing all operations manually
		
		order of ops is persistent in the list: https://stackoverflow.com/questions\
			/13694034/is-a-python-list-guaranteed-to-have-its-elements-stay-\
				in-the-order-they-are-inse
		
		Keyword Arguments:
			processmethods {list} -- List of ops to do
		"""
		opcomplete = None
		for ops in processmethods:
			opcomplete = PLUGINS[ops](self,opcomplete)

		return opcomplete
		

	@register
	def files2dflist(self, *args):
		"""Just reads the raw dataframe in to a list w/o processing it
			This method stores the frames in a self.dflist for future computation
		Returns:
			list -- list of raw dataframes
		"""
		# Read the list of files of same variables from different file types
		for fname in self.read_list:
			self.dflist.append(PLUGINS['read'+
			 os.path.splitext(fname)[1][1:]](fname))

		return self.dflist

	@register
	def datetime_parse_dflist(self, dflist):
		"""Parse the list of dataframes for datetime and set it as index
		
		Returns:
			[type] -- [description]
		"""
		return [datetime_parse(dfiter, self.timeformat, self.dateheading) \
			for dfiter in dflist]

	@register
	def merge_dflist(self, dflist):
		return mergerows(dflist)


class readfile():
	"""
	*List or read a single file
	*To read the files use file2df method after initializing
	*To return a dataframe use bothe file2df and then mergerows
	**If specifying full file path use "readfile" class
	
	Returns:
		[type] -- [description]
	"""

	def __init__(self, filepath, timeformat='%Y-%m-%d %H:%M:%S', dateheading='Date'):

		#---------ensure file to read is available and in required format------

		# make sure it is an existing file
		assert os.path.isfile(filepath), "File not found or specified string is not a file"

		# check whether the file is of one of the readable types
		dtype_error = "File " + os.path.basename(filepath) + " is not of types: "+" ".join(DATATYPE)
		assert filepath.endswith(tuple(DATATYPE)), dtype_error
		
		#------------------------------------------------------------------------

		# assign path to read data from
		self.read_path = filepath

		# select time-format
		self.timeformat = timeformat
		self.dateheading = dateheading

		# create empty dataframe variable
		self.df = None

	def return_df(self, processmethods: list = ['file2df',\
		 'datetime_parse_df']):

		"""Perform all the basic processes on the raw data and return processed
		dataframe in case uesr is not interested in doing all operations manually
		
		order of ops is persistent in the list: https://stackoverflow.com/questions\
			/13694034/is-a-python-list-guaranteed-to-have-its-elements-stay-\
				in-the-order-they-are-inse
		
		Keyword Arguments:
			processmethods {list} -- List of ops to do
		"""
		opcomplete = None
		for ops in processmethods:
			opcomplete = PLUGINS[ops](self, opcomplete)

		return opcomplete

	@register
	def file2df(self, *args):
		"""Reads the raw dataframe w/o procesing it.
		
		Returns:
			[pd.DataFrame] -- raw Dataframe
		"""
		# Read the file
		self.df = PLUGINS['read'+ os.path.splitext(self.read_path)[1][1:]](self.read_path)

		return self.df

	@register
	def datetime_parse_df(self, df):
		return datetime_parse(df, self.timeformat, self.dateheading)


# Static Methods
@register
def readcsv(read_path):

	df = pd.read_csv(read_path)

	return df


@register
def readxlsx(read_path):
	
	df = pd.read_excel(read_path)

	return df


@register
def readpkl(read_path):

	df = pd.read_pickle(read_path)

	return df


def dropNaNrows(df):
	"""Drop rows with NaN in any column
	
	Arguments:
		df {[pd.DataFrame]} -- dataframe from which to drop NaN
	
	Returns:
		[pd.DataFrame] -- cleaned dataframe
	"""
	return df.dropna(axis=0, how='any')


def dropNaNcols(df, threshold = 0.95):
	"""Drop cols with NaN > (1-threshold) fraction in any column
	
	Arguments:
		df {[pd.DataFrame]} -- dataframe from which to drop NaN
	
	Returns:
		[pd.DataFrame] -- cleaned dataframe
	"""
	return df.dropna(axis=1, thresh=int(df.shape[0]*threshold))


def datetime_parse(df, timeformat, dateheading):
	"""Convert datetime column to index after parsing it
	
	Arguments:
		df {pd.DataFrame} -- dataframe to parse
		timeformat {str} -- Format for datetime parsing
		dateheading {str} -- DateColumn Name in the raw data to read date from

	Returns:
		[pd.DataFrame] -- parsed dataframe
	"""

	# prevent modifying original dataframe
	dfc = df.copy()

	# Parsing the Date column
	# TODO: parse with timezone information
	dfc.insert(loc=0, column='Time', value=pd.to_datetime(dfc[dateheading], \
		format=timeformat)) # + pd.DateOffset(hours=offset))

	# Drop the original "dateheading" column
	dfc = dfc.drop(dateheading, axis=1)

	# Set Time column as index
	dfc = dfc.set_index(['Time'], drop=True)

	# Dropping duplicated time points that may exist in the data
	dfc = dfc[~dfc.index.duplicated()]

	return dfc


def mergerows(dflist):
		"""Merge rows of dataframes sharing same columns but different time points
		Always Call merge_df_rows before calling merge_df_columns as time has
		not been set as index yet

		Arguments:
			dflist {list of pd.DataFrame} -- dataframe list to merge
		
		Returns:
			[pd.DataFrame] -- merged dataframe
		"""
		# Create Dataframe from the dlist files
		df = pd.concat(dflist, axis=0, join='outer', sort=False)

		# Sort the df based on the datetime index
		# df = df.sort_values(by=df.index)
		df = df.sort_index()

		# Dropping duplicated time points that may exist in the data
		df = df[~df.index.duplicated()]

		return df


def merge_df_columns(dlist):
	"""Merge dataframes  sharing same rows but different columns
	
	Arguments:
		dlist {[list]} -- list of dataframes to be along column axis
	
	Returns:
		[pd.DataFrame] -- concatenated dataframe
	"""
	df = pd.concat(dlist, axis=1, join='outer', sort=False)
	df = dropNaNrows(df)

	return df


@timer
def dataframeplot(df, lazy = True, style = 'b--', ylabel : str = 'Y-axis', xlabel : str = 'X-axis', legend = False):
	"""Inspects all the rows of data in one or separate plots
	
	Arguments:
		df {pd.DataFrame} -- The dataframe to plot
	
	Keyword Arguments:
		lazy {bool} -- If true, single plot object plots all columns. Preferably set to false for plotting
		 many columns(default: {True})
		style {str} -- type of line (default: {'b--'})
		ylabel {str} -- label for yaxis (default: {'Y-axis'})
		xlabel {str} -- label for xaxis (default: {'X-axis'})
		legend {bool} -- whether we want ot see legends. Turned off for many columns (default: {False})
	"""

	if not lazy:
		width, height = 20, int(df.shape[1]*4)
		plt.rcParams["figure.figsize"] = (width, height)
		_, ax = plt.subplots(nrows = df.shape[1], squeeze=False)
		for i,j in zip(df.columns,range(df.shape[1])):
			df.plot(y=[i],ax=ax[j][0],style=[style], legend=legend)
		ax[j][0].set_xlabel(xlabel)
		ax[j][0].set_ylabel(ylabel)
	else:
		ax = df.plot(y=df.columns, figsize=(20,7), legend=legend, style = [style]*df.shape[1])
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)


def constantvaluecols(df, limit : Union[float, np.ndarray] = 0.02):
	"""Drop columns which are constant values and may not controibute significant information
	
	Arguments:
		df {pd.DataFrame} -- The dataframe to drop constants from
	
	Keyword Arguments:
		limit {float or 1-d np.array} -- if std less than this limit treat it as constant (default: {0.2})
	"""

	statistics = df.describe()
	if isinstance(limit, np.ndarray):
		limit = limit.flatten()
		errmsg = "Shape mismatch. Flattened limit array has length {} and number of\
			 Dataframe columns is {}".format(limit.shape[0], df.shape[1])
		assert limit.shape[0] == df.shape[1], errmsg
		constantcols = np.where(statistics.loc['std'].to_numpy()<=limit)
	else:
		constantcols = np.where(statistics.loc['std'].to_numpy()<=limit*np.ones(df.shape[1], dtype=float))
	return df.drop(columns=df.columns[constantcols])


@timer
def removeoutliers(df, columns: list, **kwargs):
	"""Remove two types of outliers depending on "rmvtype" arguement
	*Remove outliers beyond z_thresh standard deviations in the data
	*Remove based on bounds
	
	Arguments:
		df {pd.DataFrame} -- the dataframe to remove outliers from
		columns {list} -- Columns to remove outliers from
	
	Keyword Arguments:
		Either -for statistical outlier removal
			z_thresh {int} -- How many standard deviations to consider for outlier removal
		Or -for bounds based outlier removal
			upperbound {float} -- upperbound for cutting off nonstatistical data
			lowerbound {float} -- lowererbound for cutting off nonstatistical data
	"""
	org_shape = df.shape[0]

	x = 'z_thresh' in kwargs.keys()
	y = 'upperbound' and 'lowerbound' in kwargs.keys()
	assert bool(~x&y | x&~y) , "Either z_thresh or both (upperbound and lowerbound) keyword arguments must be provided"


	if 'z_thresh' in kwargs.keys():  # do statistical outlier removal

		for column_name in columns:
		# Constrains will contain `True` or `False` depending on if it is a value below the threshold.
			constraints = abs(stats.zscore(df[column_name])) < kwargs['z_thresh']
			# Drop values set to be rejected
			df = df.drop(df.index[~constraints], axis = 0)

	else:  # do boundary based outlier removal

		upperbound = kwargs['upperbound']
		lowerbound = kwargs['lowerbound']
		# For every row apply threshold using bounds and see whether all columns for each row satisfy the bounds
		constraints = df.swifter.apply(lambda row: all([(cell < upperbound) and (cell > lowerbound) for cell in row[columns]]), axis=1)
		# Drop values set to be rejected
		df = df.drop(df.index[~constraints], axis = 0)

	print("Retaining {}% of the data".format(100*df.shape[0]/org_shape))

	return df


def frequencyplot(df, T: Union[int,float]= 300):
	"""plot the frequency spectrum of the columns of a timeseries dataframe

	Arguments:
		df {pd.DataFrame} -- the input datafrme
	
	Keyword Arguments:
		T {Union[int,float]} -- sampling period of the signal in seconds (default: {300})
	"""

	f_s = 1 / T
	N = df.shape[0]

	width, height = 7, int(2*df.shape[1])
	plt.rcParams["figure.figsize"] = (width, height)
	num_cols = len(df.columns)

	for idx, col in enumerate(df.columns):

		y_value = df[col].to_numpy()
		f_values, fft_values = get_fft_values(y_value, T, N, f_s)

		plt.subplot(num_cols, 1, idx + 1)
		plt.plot(f_values, fft_values, linestyle='-', color='blue', label = col)
		plt.xlabel('Frequency [Hz]', fontsize=5)
		plt.ylabel('Amplitude', fontsize=5)
		plt.title("Frequency domain of the signal "+col, fontsize=5)
		plt.legend()
	
	plt.subplots_adjust(hspace=0.7)
	plt.show()

	
def get_fft_values(y_value, T, N, f_s):
		f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
		fft_values_ = fft(y_value)
		fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
		return f_values, fft_values


def dfsmoothing(df, 
				column_names: list = [], 
				order: int = 5, 
				Wn: Union[list, float] = 0.015,
				T : Union[int,float]= 300):
	"""Smoothes the dataframe columns using butterworth smoothing
	
	Arguments:
		df {pd.DataFrame} -- the input datafrme
	
	Keyword Arguments:
		column_names {list} -- list of column names to be smoothed (default: {None})
		order {int} -- Order of the filter (default: {5})
		Wn {Union[list, float]} -- Cutoff frequency (default: {0.015})
		T {Union[int, float]} -- sampling period of the signal in seconds (default: {300})
	
	Returns:
		pd.DataFrame -- smoothed data frame
	"""

	if not column_names:

		return df

	else:

		# prevent modifying original dataframe
		df2 = df.copy()
		# sample rate
		fs = 1/T
		# nyquist frequency
		nyq = 0.5 * fs
		# make a list in case cutoff is scalar
		if not isinstance(Wn, list):
			Wn = [Wn] * len(column_names)
		# normal cutoff
		Wn = [cutoff / nyq for cutoff in Wn]
		# columnwise filtering
		for idx, i in enumerate(column_names):
			# First, design the Buterworth filter
			B, A = signal.butter(order, Wn[idx], btype='low', analog=False)
			# filter the signal
			df2[i] = signal.filtfilt(B, A, df2[i])
			# drop any NaN rows created
			df2 = dropNaNrows(df2)
		return df2


def sample_timeseries_df(df, period=1):
	"""
	resamples dataframe at "period" 5 minute time points
	:param df:
	:param period: number of 5 min time points
	:return: sampled dataframe
	"""
	
	timegap = period * 5

	return df[df.index.minute % timegap == 0]


def subsequencing(df, period=1):
	"""Divides temporally non-continuous dataframe into temporally continuous dataframes
	
	Arguments:
		df {pd.DataFrame} -- input dataframe
	
	Keyword Arguments:
		period {int} -- minimum 5*period minute resolution at which we check for continuity (default: {1})
	
	Returns:
		list(pd.DataFrame) -- list of continuous dataframes
	"""
	counter = 0
	timegap = period * 5
	dflist = []
	
	for i in range(len(df.index)):
		if df.index[i] + pd.DateOffset(minutes=timegap) not in df.index:
			dflist.append(df.iloc[counter:(i + 1), :])
			counter = i + 1

	return dflist


def old_req_seq_chunks(dflist,days=7,hours=0):
	"""Divide dataframes larger than minimum sequence length into minimum sequence length chunks
	
	Arguments:
		dflist {list(pd.DataFrame)} -- input list of dataframes with minimum sequence length
	
	Keyword Arguments:
		days {int} -- Number of days  (default: {7})
		hours {int} -- number of hours (default: {0})
	
	Returns:
		list(pd.DataFrame) --list of dataframes, each with minimum sequence length
	"""

	dfchunks = []

	for h in dflist:
		# number of possible min seq length chunks
		chunks = (h.index[-1]-h.index[0])//pd.Timedelta(str(days)+' days '+str(hours)+' hours')
		timeloc = h.index[0]
		for _ in range(chunks):
			dfchunks.append(h.loc[timeloc : timeloc + pd.DateOffset(days=days,hours=hours),:])
			timeloc = timeloc + pd.DateOffset(days=days,hours=hours,minutes = 5)

	return dfchunks


def req_seq_dfs(dflist,req_seq_length):
	"""Divide dataframes larger than "minimum sequence length" into "required sequence length" dataframes
	
	Arguments:
		dflist {list(pd.DataFrame)} -- input list of dataframes with "minimum sequence length"
		req_seq_length {int} -- required length of sequences 
	
	Returns:
		list(pd.DataFrame) --list of dataframes, each with required sequence length
	"""

	dfchunks = []

	for h in dflist:
		# number of possible req seq length dataframes
		no_dfs = h.shape[0]//req_seq_length
		idx = 0
		for _ in range(no_dfs):
			dfchunks.append(h.iloc[idx : idx + req_seq_length, :])
			idx = idx + req_seq_length

	return dfchunks


def df2dflist(df: Union[str, pd.DataFrame], *args, subsequence: bool = True, period: int = 1, days: int = 7, hours: int = 0):
	"""[summary]
	
	Arguments:
		df {Union[str, pd.DataFrame]} -- [description]
	
	Keyword Arguments:
		subsequence {bool} -- [description] (default: {True})
		period {int} -- [description] (default: {1})
		days {int} -- [description] (default: {7})
		hours {int} -- [description] (default: {0})
	
	Returns:
		[type] -- [description]
	"""
	
	if isinstance(df, str):

		dfclass = readfile(df)
		df = dfclass.return_df(processmethods=['file2df'])

	if subsequence:

		# divide the dataframe into contiguous samples
		dflist = subsequencing(df, period)

		#minimum sequence length on which we train
		# eg days = 7, period = 1, so we will select sequences from dflist of minimum sequence length 2016
		assert (hours==0)|(period<=12), "If min chunk size is not a multiple of 1 day, period must be less than or equal to 12"
		min_seq_length = days*int(1440/(5 * period)) +  hours*int(60/(5 * period))

		# remove dataframes whose length is smaller than minimum sequence length
		dflist = [item for item in dflist if item.shape[0] >= min_seq_length]

		# now we have dataframes whose length is atleast minimum sequence length
		# next we divide dataframes larger than minimum sequence length into required sequence length chunks
		if args:
			assert len(args)==1, "Function df2dflist takes at most two positional arguments. More than two passed."
			req_seq_length = args[0]
			assert len(req_seq_length)<=len(min_seq_length),\
			 "Req sequence length{} must be less than min sequence length{}".format(req_seq_length, min_seq_length)
		else:
			req_seq_length = min_seq_length
		dflist = req_seq_dfs(dflist, req_seq_length)

	else:

		dflist = [df]

	return dflist

def df2dflist_alt(df: Union[str, pd.DataFrame], *args, subsequence: bool = True, period: int = 1, days: int = 7, hours: int = 0):
	"""[summary]
	
	Arguments:
		df {Union[str, pd.DataFrame]} -- [description]
	
	Keyword Arguments:
		subsequence {bool} -- [description] (default: {True})
		period {int} -- [description] (default: {1})
		days {int} -- [description] (default: {7})
		hours {int} -- [description] (default: {0})
	
	Returns:
		[type] -- [description]
	"""
	
	if isinstance(df, str):

		dfclass = readfile(df)
		df = dfclass.return_df(processmethods=['file2df'])

	if subsequence:

		# divide the dataframe into contiguous samples
		# dflist = subsequencing(df, period)

		#minimum sequence length on which we train
		# eg days = 7, period = 1, so we will select sequences from dflist of minimum sequence length 2016
		assert (hours==0)|(period<=12), "If min chunk size is not a multiple of 1 day, period must be less than or equal to 12"
		min_seq_length = days*int(1440/(5 * period)) +  hours*int(60/(5 * period))

		# remove dataframes whose length is smaller than minimum sequence length
		# dflist = [item for item in dflist if item.shape[0] >= min_seq_length]

		# now we have dataframes whose length is atleast minimum sequence length
		# next we divide dataframes larger than minimum sequence length into required sequence length chunks
		if args:
			assert len(args)==1, "Function df2dflist takes at most two positional arguments. More than two passed."
			req_seq_length = args[0]
			assert len(req_seq_length)<=len(min_seq_length),\
			 "Req sequence length{} must be less than min sequence length{}".format(req_seq_length, min_seq_length)
		else:
			req_seq_length = min_seq_length
		# dflist = req_seq_dfs(dflist, req_seq_length)
		dflist = old_req_seq_chunks([df], days=days, hours=hours)

		# remove dataframes whose length is smaller than 10% of of min_seq_length
		dflist = [item for item in dflist if item.shape[0] >= 0.1*min_seq_length]

	else:

		dflist = [df]

	return dflist


def inputreshaper(X, input_timesteps=1, output_timesteps=1):
	"""Reshapes the input/ predictor array into (nsamples, input_timesteps, input dimension)
	
	Arguments:
		X {np.ndarray} -- The array to reshape
	
	Keyword Arguments:
		input_timesteps {int} -- The number of timesteps from past to present time to consider as part of input (default: {1})
		output_timesteps {int} -- The number of time steps into the future to predict (default: {1})
	
	Returns:
		np.ndarray -- [description]
	"""
	totalArray = []
	m, n = X.shape

	for i in range(input_timesteps):
		# copying
		temp = np.copy(X)
		# shifting
		temp = temp[i:m + i - input_timesteps + 1, :]
		# appending
		totalArray.append(temp)

	# reshaping collated array to (samples, input_timesteps, dimensions)
	collatedarray = np.concatenate(totalArray, axis=1)
	X_reshaped = collatedarray.reshape((collatedarray.shape[0], input_timesteps, n))
	if output_timesteps != 1:
		X_reshaped = X_reshaped[0:1 - output_timesteps, :, :]
	# ^^We are removing last output_timesteps-1 data due to predicting sequence of length output_timesteps

	return X_reshaped


def outputreshaper(y, output_timesteps=1, outputdim=1, input_timesteps=1):
	"""Reshapes the onput/ target array into (nsamples, output_timesteps, outputdim)
	
	Arguments:
		y {np.ndarray} -- The array to reshape
	
	Keyword Arguments:
		output_timesteps {int} -- The number of time steps into the future to predict (default: {1})
		outputdim {int} --  Number of output features (default: {1})
		input_timesteps {int} -- The number of timesteps from past to present time to consider as part of input (default: {1})
	
	Returns:
		[type] -- [description]
	"""
	N = output_timesteps
	totalArray = []
	if outputdim == 1:
		m = y.shape[0]  # since y is (m, )
	else:
		m, _ = y.shape

	for i in range(N):
		# copying
		temp = np.copy(y)
		# shifting
		temp = temp[i + input_timesteps - 1: m + i - N + 1]
		# appending
		totalArray.append(temp.reshape(-1, 1))  # reshaping needed for concatenating along axis=1

	# reshaping collated array to (samples, input_timesteps)
	collatedarray = np.concatenate(totalArray, axis=1)
	y_reshaped = collatedarray.reshape((collatedarray.shape[0], N, outputdim))
	return y_reshaped


def createlag(df, outputcols: list, lag: int = -1):
	"""Shift outputcols up by lag time points
	
	Arguments:
		df {pd.DataFrame} -- input dataframe
		outputcols {list} -- list of cols to shift
	
	Keyword Arguments:
		lag {int} -- Shift output columns by lag time points
	
	Returns:
		pd.DataFrame -- cleaned dataframe with lagged output cols
	"""

	# prevent modifying original dataframe
	df2 = df.copy()

	# shift
	df2[outputcols]  = df2[outputcols].shift(lag)
	
	# remove NaN rows created as a result
	df2 = dropNaNrows(df2)

	return df2


def df2arrays(df, split: float = 0.75, shuffle: bool = False, predictorcols : list = [], outputcols: list = [], \
 lag: int = -1, scaling : bool = False, feature_range = (0,1), reshaping : bool = True, input_timesteps: int = 1, output_timesteps: int = 1):
	"""
	0 Shift output columns up by lag time points
	1 Scales the arrays if needed based on MinMaxScaler
	2 Shuffle the dataframe rows if needed and divide the dataframe into train and test set numpy arrays
	 taking care of the predictor and target variables
	3 Rehapes the arrays if needed based on the requirements for the input to be a time sequence and output to be a time sequence
	
	Arguments:
		df {pd.DataFrame} -- Dataframe to create training and testing data from
	
	Keyword Arguments:
		split {float} -- Percentage of train data from the set] (default: {0.75})
		shuffle {bool} -- Shuffle the arrays if needed (default: {False})
		predictorcols {list} -- List of dataframe column names to consider as preditors (default: {[]})
		outputcols {list} -- List of dataframe column names to consider as targets (default: {[]})
		lag {int} -- Shift output columns by lag time points
		scaling {bool} -- Whether to scale the data using minmax scaler (default: {False})
		reshaping {bool} -- Whether to reshape the predictors and target (default: {True})
		input_timesteps {int} -- The number of timesteps from past to present time to consider as part of input (default: {1})
		output_timesteps {int} -- The number of time steps into the future to predict (default: {1})
	"""

	# 0 Shift output columns by lag time points
	df2 = createlag(df, outputcols = outputcols, lag = lag)

	# rearranging the dataframe in terms of [predictorcols]+[outputcols] for better management 
	#df = df[predictorcols+outputcols]

	# df to array
	X = df.loc[df2.index,:][predictorcols].to_numpy()
	y = df2[outputcols].to_numpy()

	# 1 Scales the arrays if needed based on MinMaxScaler	
	scaler = MinMaxScaler(feature_range=feature_range)
	X_scaler = scaler.fit(X)
	y_scaler = scaler.fit(y)
	if scaling:
		X = X_scaler.transform(X)
		y = y_scaler.transform(y)


	# 2 Shuffle the dataframe rows if needed before and divide the dataframe into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split, shuffle=shuffle)  
	"""
	X_train.shape =(samplesize, len(predictorcols))
	y_train.shape =(samplesize, len(outputcols))
	"""

	# 3 Rehapes the arrays if needed based on the requirements for the input to be a time sequence and \
	# output to be a time sequence
	if reshaping:  # NB: sample size will change due to reshaping and subsequent removal of NaN rows
		X_train = inputreshaper(X_train, input_timesteps, output_timesteps)  # (eg samplesize, 1, 4 or 5)
		y_train = outputreshaper(y_train, output_timesteps, len(outputcols), input_timesteps)  # (eg samplesize, 1, 1)
		X_test = inputreshaper(X_test, input_timesteps, output_timesteps)  # (eg samplesize, 1, 4 or 5)
		y_test = outputreshaper(y_test, output_timesteps, len(outputcols), input_timesteps)  # (eg samplesize, 1, 1)

	return [X_train, X_test, y_train, y_test, X_scaler, y_scaler]


def minmaxscaling(x: np.ndarray, feature_range=(0,1)):

	assert len(x.shape)<=2, "Cannot scale array with no of axis greater than 2. Array with {} axis passed".format(len(x.shape))
	if len(x.shape)==1:
		x = x.reshape(-1,1)

	scaler = MinMaxScaler(feature_range=feature_range)
	# Create scalar object
	x_sc = scaler.fit(x)

	return x_sc.transform(x), x_sc
