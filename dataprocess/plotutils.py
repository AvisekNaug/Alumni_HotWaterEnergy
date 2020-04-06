import matplotlib
from matplotlib import pyplot as plt

import numpy as np
from scipy.interpolate import make_interp_spline, BSpline


def pred_v_target_plot(timegap, outputdim, output_timesteps, preds, target,
 saveloc, scaling: bool, scaler, lag: int = -1, outputdim_names : list = [], typeofplot: str = 'train', Week: int = 0):

	if not outputdim_names:
		outputdim_names = ['Output']*outputdim

	plt.rcParams["figure.figsize"] = (15, 5*outputdim*output_timesteps)
	font = {'size':16}
	plt.rc('font',**font)
	plt.rc('legend',**{'fontsize':14})

	# Inerse scaling the data for each time step
	if scaling:
		for j in  range(output_timesteps):
			preds[:,j,:] = scaler.inverse_transform(preds[:,j,:])
			target[:,j,:] = scaler.inverse_transform(target[:,j,:])

	# attach forward slash if saveloc does not have one
	if not saveloc.endswith('/'):
			saveloc += '/'


	# training output
	fig, axs = plt.subplots(nrows = outputdim*output_timesteps, squeeze=False)
	for i in range(outputdim):
		for j in range(output_timesteps):
			# plot predicted
			axs[i+j, 0].plot(preds[:, j, i], 'r--', label='Predicted'+outputdim_names[i])
			# plot target
			axs[i+j, 0].plot(target[:, j, i], 'g--', label='Actual'+outputdim_names[i])
			# Plot Properties
			axs[i+j, 0].set_title('Predicted vs Actual at time = t + {} for {}'.format(-1*lag+j, outputdim_names[i]))
			axs[i+j, 0].set_xlabel('Time points at {} minute(s) intervals'.format(timegap))
			axs[i+j, 0].set_ylabel('Actual Energy')
			axs[i+j, 0].grid(which='both',alpha=100)
			axs[i+j, 0].legend()
			axs[i+j, 0].minorticks_on()
	fig.savefig(saveloc+str(timegap)+'_LSTM_'+typeofplot+'prediction-Week{}.pdf'.format(Week), bbox_inches='tight')
	plt.close(fig)

def pred_v_target_plot_transferlearning(timegap, outputdim, output_timesteps, preds, target,
 saveloc, scaling: bool, scaler, lag: int = -1, outputdim_names : list = [], typeofplot: str = 'train', Idx: int = 0):

	if not outputdim_names:
		outputdim_names = ['Output']*outputdim

	plt.rcParams["figure.figsize"] = (15, 5*outputdim*output_timesteps)
	font = {'size':16}
	plt.rc('font',**font)
	plt.rc('legend',**{'fontsize':14})

	# Inerse scaling the data for each time step
	if scaling:
		for j in  range(output_timesteps):
			preds[:,j,:] = scaler.inverse_transform(preds[:,j,:])
			target[:,j,:] = scaler.inverse_transform(target[:,j,:])

	# attach forward slash if saveloc does not have one
	if not saveloc.endswith('/'):
			saveloc += '/'


	# training output
	fig, axs = plt.subplots(nrows = outputdim*output_timesteps, squeeze=False)
	for i in range(outputdim):
		for j in range(output_timesteps):
			# plot predicted
			axs[i+j, 0].plot(preds[:, j, i], 'r--', label='Predicted'+outputdim_names[i])
			# plot target
			axs[i+j, 0].plot(target[:, j, i], 'g--', label='Actual'+outputdim_names[i])
			# Plot Properties
			axs[i+j, 0].set_title('Predicted vs Actual at time = t + {} for {}'.format(-1*lag+j, outputdim_names[i]))
			axs[i+j, 0].set_xlabel('Time points at {} minute(s) intervals'.format(timegap))
			axs[i+j, 0].set_ylabel('Actual Energy')
			axs[i+j, 0].grid(which='both',alpha=100)
			axs[i+j, 0].legend()
			axs[i+j, 0].minorticks_on()
	fig.savefig(saveloc+str(timegap)+'_LSTM_'+typeofplot+'prediction-{}.pdf'.format(Idx), bbox_inches='tight')
	plt.close(fig)

def detailedplot(timegap, xs, outputdim, output_timesteps, input_timesteps, pred, target, X_var,
saveloc, scaling: bool, Xscaler, yscaler, lag: int = -1, outputdim_names : list = [], typeofplot: str = 'train', 
				Idx: str = ''):
	
	if not outputdim_names:
		outputdim_names = ['Output']*outputdim
		
	_pred = np.empty_like(pred)
	_target = np.empty_like(target)
	_X_var = np.empty_like(X_var)

	plt.rcParams["figure.figsize"] = (15, 5*outputdim*output_timesteps)
	font = {'size':16}
	plt.rc('font',**font)
	plt.rc('legend',**{'fontsize':14})

	# Inerse scaling the data for each time step
	if scaling:
		for j in  range(output_timesteps):
			_pred[:,j,:] = yscaler.inverse_transform(pred[:,j,:])
			_target[:,j,:] = yscaler.inverse_transform(target[:,j,:])
		for j in  range(input_timesteps):
			_X_var[:,j,:] = Xscaler.inverse_transform(X_var[:,j,:])     

	# attach forward slash if saveloc does not have one
	if not saveloc.endswith('/'):
			saveloc += '/'


	# training output
	fig, axs = plt.subplots(nrows = outputdim*output_timesteps, squeeze=False)
	for i in range(outputdim):
		for j in range(output_timesteps):
			# plot predicted
			axs[i+j, 0].plot_date(xs, _pred[:, j, i], 'ro-', label='Predicted'+outputdim_names[i])
			# plot target
			axs[i+j, 0].plot_date(xs, _target[:, j, i], 'g*-', label='Actual'+outputdim_names[i])
			# plot other variables: temperature
			axs[i+j, 0].plot_date(xs, _X_var[:, -1, 0], 'b3-', label='Outside Air Temperature')
			# plot other variables: relative humidity
			axs[i+j, 0].plot_date(xs, _X_var[:, -1, 1], 'm4-', label='Outside Air Rel Humidity')
			# plot other variables: Supply Air Temperature
			axs[i+j, 0].plot_date(xs, _X_var[:, -1, 2], 'k^-', label='Supply Air Temperature')
			# plot other variables: flow value
			axs[i+j, 0].plot_date(xs, _X_var[:, -1, 4], 'c>-', label='Hot Water Flow Rate')
			# Plot Properties
			axs[i+j, 0].set_title('Predicted vs Actual at time = t + {} for {}'.format(-1*lag+j, outputdim_names[i]))
			axs[i+j, 0].set_xlabel('Time points at {} minute(s) intervals'.format(timegap))
			axs[i+j, 0].set_ylabel('Different Variables')
			axs[i+j, 0].grid(which='both',alpha=100)
			axs[i+j, 0].minorticks_on()
			axs[i+j, 0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
	fig.savefig(saveloc+str(timegap)+'min_LSTM_'+typeofplot+'_plot-{}.pdf'.format(Idx), bbox_inches='tight')
	plt.gcf().autofmt_xdate()
	plt.close(fig)

def single_bar_plot(bars: list, color, bar_label: str, saveloc: str, barwidth = 0.50, smoothcurve: bool = False,
 bar_annotate: bool = False, saveplot: bool = False, plot_name: str = 'BarPlot', xlabel: str = 'Xlabel', ylabel: str = 'ylabel',
 title: str = 'Title', xticktype: str = 'Bar', xticklist = None, plotwidth = None, plotheight = 15, fontsize = 16):

	if plotwidth is None:
		plt.rcParams["figure.figsize"] = (0.6*len(bars),plotheight)
	else:
		plt.rcParams["figure.figsize"] = (plotwidth,plotheight)
	font = {'size':fontsize, 'family': "Times New Roman"}
	plt.rc('font',**font)
	plt.rc('legend',**{'fontsize':fontsize})

	N = len(bars)
	ind = np.arange(N)

	plt.bar(ind,
	 bars, 
	 barwidth, 
	 color=color, 
	 label=bar_label)

	if xticklist is None:
		plt.xticks(ind, [xticktype + str(i) for i in range(1, N + 1)], rotation = 45)
	else:
		plt.xticks(ind, xticklist, rotation = 90)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.title(title)

	if smoothcurve:
		T = np.array([i for i in range(len(bars))])
		xnew = np.linspace(T.min(), T.max(), 300)
		spl = make_interp_spline(T, bars, k=3)  # type: BSpline
		power_smooth = spl(xnew)
		plt.plot(xnew, power_smooth, color='k', alpha=0.8)

	if bar_annotate:
		for i, v in enumerate(bars):
			plt.text(i - 0.30, bars[i], '{0:.1f}%'.format(np.abs(v)), color='g', fontweight='bold', fontsize=13, rotation= 0)

	if saveplot:
		# attach forward slash if saveloc does not have one
		if not saveloc.endswith('/'):
			saveloc += '/'
		plt.savefig(saveloc + 'Weekly Energy Predictions.png', bbox_inches='tight', dpi=300)
