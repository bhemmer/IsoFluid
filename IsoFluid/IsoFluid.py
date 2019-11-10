# modules
import numpy as np
import os
import datetime as dt
import re
from scipy.integrate import simps
from scipy.optimize import least_squares
from scipy.signal import convolve
from glob import glob


'''
Isotopy Evaluation

This module provides tools to analyze isotopy data from a Picarro Instrument.
The evaluation is carried out inside the :class:`~isotopy_evaluation.eval` class. The data is imported at the initialization, all calculations are implemented as methods on this class.

input / output variables which are used across all functions

time (numpy array): times of the measurements relative to some chosen reference time [s]
value (numpy array): arbitrary measurement data
water (numpy array): water concentration measured by picarro [ppmv]
isotope (numpy array): isotope ratio in Delta-notation [permil]
peak_start (list): list of times at which peaks start in the data set
peak_end (list): list of times at which peaks end in the data set

if you do not reference optional arguments of functions by name, be sure you reference all optional arguments in the right order
'''

# regular expression definitions
re_header = r"(\w+?)(?:$|\s+)"
re_float = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
re_data = r"(" + re_float + ")+?\s*(?:\;|\s*$)?"
re_commafloat = r"[-+]?[0-9]*\,?[0-9]+(?:[eE][-+]?[0-9]+)?"

class eval:
    """
    This class is used to evaluate isotopy data taken from a Picarro Instrument.

    It is created as part of a research project at the Institute for Environmental Physics at Heidelberg University.

    Important methods:

    - :func:`~isotopy_evaluation.eval.set`
    - :func:`~isotopy_evaluation.eval.update`
    - :func:`~isotopy_evaluation.eval.get`
    - :func:`~isotopy_evaluation.eval.write_data_to_file`
    """
    BG_CALC_MODE = ['const', 'fit']
    FIT_MODEL = ['lin', 'exp']
    VOLUME_MODE = ['area', 'height']

    def __init__(self, path_to_data, start_time, end_time, time_offset=dt.timedelta(hours=0.)):
        """On initialization of the class data from a specified folder is imported and the date and time of measured datapoints are converted to times in seconds.

        Args:
            path_to_data: Sting that specifies the path (relative or absolute) to the data from the Picarro Instrument. First directory inside the specified directory should be the year directory.
            start_time: Datetime object which specifies the date and time for which the evaluation should start.
            end_time: Datetime object which specifies the date and time for which the evaluation should end.
            time_offset: Datetime timedelta object which specifies the time offset between the time in the picarro device (and therefore in the saved data) and the real labtime.
        """
        # initialize empty data containers
        self.date = []
        self.t = np.array([])
        self.water = np.array([])
        self.water_rm = np.array([])
        self.H2 = np.array([])
        self.O18 = np.array([])

        # state variables
        self.calib = False

        # calculated values per peak
        self.peak_start = []
        self.peak_end = []
        self.bg_t = []
        self.bg_water = []
        self.bg_H2 = []
        self.bg_O18 = []
        self.bg_model_param = []
        self.bg_model_cost = []
        self.dH2 = []
        self.dO18 = []
        self.volume = []

        # initialize parameters to None
        # calibration
        self.H2_calib_param = None
        self.O18_calib_param = None
        self.volume_mode = None
        self.volume_calib_param = None
        # start peak detection
        self.start_slope = None
        self.start_slope_n = None
        self.ignore_intervals = None
        self.min_peak_separation = None
        # end peak detection
        self.end_slope = None
        self.end_slope_n = None
        self.end_slope_running_mean = None
        self.running_mean_n = None
        self.min_peak_size = None
        self.max_peak_size = None
        # background
        self.bg_calc_mode = None
        self.bg_model = None
        self.bg_fit_start_param = None
        self.bg_period = None
        self.bg_separation_before_peak = None
        self.bg_separation_after_peak = None

        # evaluation properties and paths
        self.name = None
        self.path_to_data = path_to_data  # input
        self.start_time = start_time
        self.end_time = end_time
        self.time_offset = time_offset
        self.tref = start_time
        self.path_to_results = None  # output
        self.verbose = 1

        self.load(path_to_data, start_time, end_time, time_offset=time_offset)
        return None

    def load(self, path_to_data, start_time, end_time, time_offset=dt.timedelta(hours=0.)):
        """Imports data from specified folder and converts the dates to time in seconds.
        """
        self.path_to_data = path_to_data
        self.start_time = start_time
        self.end_time = end_time
        self.time_offset = time_offset
        if self.verbose > 0: print('importing data ...')
        self.date, self.water, self.H2, self.O18 = import_picarro_data(path_to_data, start_time, end_time, time_offset=time_offset)
        self.calib = False
        if self.verbose > 0: print('convert date to time ...')
        self.t = time_conversion(self.date, self.tref)
        return None

    def set(self,
        name = None,
        path_to_data = None,  # evaluation properties
        start_time = None,
        end_time = None,
        time_offset = None,
        tref = None,
        path_to_results = None,
        H2_calib_param = None,  # calibration
        O18_calib_param = None,
        volume_mode = None,
        volume_calib_param = None,
        start_slope = None,
        start_slope_n = None,
        ignore_intervals = None,  # background
        min_peak_separation = None,
        end_slope = None,  # peak detection
        end_slope_n = None,
        end_slope_running_mean = None,
        running_mean_n = None,
        min_peak_size = None,
        max_peak_size = None,
        bg_calc_mode = None,
        bg_model = None,
        bg_fit_start_param = None,
        bg_period = None,
        bg_separation_before_peak = None,
        bg_separation_after_peak = None,
        verbose = None
        ):
        """ Sets the parameter of the evaluation.

        Evaluation **Properties and Paths**:

        Args:
            name: String which specifies the name of the Evaluation.
            path_to_data: String which specifies the path (relative or absolute) to the data from the Picarro Instrument. First directory inside the specified directory should be the year directory.
            start_time: Datetime object which specifies the date and time for which the evaluation should start.
            end_time: Datetime object which specifies the date and time for which the evaluation should end.
            time_offset: Datetime timedelta object which specifies the time offset between the time in the picarro device (and therefore in the saved data) and the real labtime.
            tref: Datetime object which specifies the date and time for which the time in seconds is 0.
            path_to_results: String which specifies the path to directory in whicht the textfile containing the evaluation parameters and resuls is written.
            verbose: Integer which specifies how much information should be given on whats going on. 0 = none, 1 = basic.

        Parameter for **Calibration**:

        Args:
            H2_calib_param: List of parameter for the linear calibration of the H2 values.
            O18_calib_param: List of parameter for the linear calibration of the O18 values.
            volume_mode: String which specifies the mode in which the volume for each peak is calculated. Either of 'area' or 'heigth'.
            volume_calib_param: List of parameter for the calibration of the watervolume associated with each peak.

        Parameter for **Peak Detection**:

        Args:
            start_slope: Float which specifies the value for slope above which a new peak is detected.
            start_slope_n: Float which specifies the distance of datapoints between which slope is calculated for the detection of the start of a peak.
            ignore_intervals: List of tuples (start, end) which specify intervals in which no peaks are detected
            min_peak_separation: Float which specifies the specifies minimal distance in seconds between start of two peaks.
            end_slope: Float which specifies the value for slope below which the end of a peak is detected.
            end_slope_n: Float which specifies the distance of datapoints between which slope is calculated for the detection of the end of a peak.
            end_slope_running_mean: True if running mean should be used for calculation of the slope for end of peak detection.
            running_mean_n: Integer which specifies how many neigbouring datapoints are used for averaging. If this is set the running mean is recalculated immediatly.
            min_peak_size: Float which specifies the minimal length of a peak in seconds.
            max_peak_size: Float which specifies the maximal length of a peak in seconds.


        Parameter for **Background Calculation**:

        Args:
            bg_calc_mode: String which specifies the mode in which the background is calculated. Either of 'const' or 'fit'. For 'fit' mode a bg_model has to be selected.
            bg_model: String which specifies the model which is fitted to the background before and after each peak. Either of 'lin' or 'exp'.
            bg_fit_start_param: List of parameter which are used as starting parameters for the background fit.
            bg_period: Float which specifies the length period before (and after) the peak for which the background is evaluated (or fitted).
            bg_separation_before_peak: Float which specifies the time in seconds between end of the interval used for background calculation (or fit) and start of peak.
            bg_separation_after_peak: Float which specifies the time in seconds between end of peak and start of intervals used for the fit of the background model.
        """
        # evaluation properties
        if name is not None:
            self.name = name
        if path_to_data is not None:
            self.path_to_data = path_to_data
        if start_time is not None:
            self.start_time = start_time
        if end_time is not None:
            self.end_time = end_time
        if time_offset is not None:
            self.time_offset = time_offset
        if tref is not None:  # load method already does this
            self.tref = tref
        if (path_to_data is not None or
                start_time is not None or
                end_time is not None):
            self.load(self.path_to_data, self.start_time, self.end_time, self.time_offset)
        elif tref is not None:  # load method already does this
            if self.verbose > 0: print('convert date to time ...')
            self.t = time_conversion(self.date, tref)
        if path_to_results is not None:
            self.path_to_results = path_to_results

        # parameters
        # calibration
        if O18_calib_param is not None:
            self.O18_calib_param = O18_calib_param
        if H2_calib_param is not None:
            self.H2_calib_param = H2_calib_param
        if volume_mode is not None:
            self.volume_mode = volume_mode
        if volume_calib_param is not None:
            self.volume_calib_param = volume_calib_param
        # peak detection
        if start_slope is not None:
            self.start_slope = start_slope
        if start_slope_n is not None:
            self.start_slope_n = start_slope_n
        if ignore_intervals is not None:
            self.ignore_intervals = ignore_intervals
        if min_peak_separation is not None:
            self.min_peak_separation = min_peak_separation
        if end_slope is not None:
            self.end_slope = end_slope
        if end_slope_n is not None:
            self.end_slope_n = end_slope_n
        if end_slope_running_mean is not None:
            self.end_slope_running_mean = end_slope_running_mean
        if running_mean_n is not None:
            self.running_mean_n = running_mean_n
            self.calc_running_mean()
        if min_peak_size is not None:
            self.min_peak_size = min_peak_size
        if max_peak_size is not None:
            self.max_peak_size = max_peak_size
        # background
        if bg_calc_mode is not None:
            self.bg_calc_mode = bg_calc_mode
        if bg_model is not None:
            self.bg_model = bg_model
        if bg_fit_start_param is not None:
            self.bg_fit_start_param = bg_fit_start_param
        if bg_period is not None:
            self.bg_period = bg_period
        if bg_separation_before_peak is not None:
            self.bg_separation_before_peak = bg_separation_before_peak
            if self.bg_separation_after_peak is None:
                self.bg_separation_after_peak = bg_separation_before_peak
        if bg_separation_after_peak is not None:
            self.bg_separation_after_peak = bg_separation_after_peak
        if verbose is not None:
            self.verbose = verbose
        return None

    def get(self, ref=None):
        """"Get Data or Results of the Evaluation.

        Args:
            ref: String which references to one of the following Variables.

                *Data* of Evaluation:

                - **date**: List which contains datetime objects for each measurement point.
                - **t**: Array which contains time in seconds for each measurement point.
                - **water**: Array which contains water measurements in ppmv.
                - **water_rm**: Array which contains water measurements in ppmv averaged over neighbouring datapoints.
                - **H2**: Array which contains delta 2H isotope measurements in permil.
                - **O18**: Array which contains delta 18O isotope measurements in permil.

                *Results* from Evaluation:

                - **peak_start**: List which contains time in seconds at which the peak starts.
                - **peak_end**: List which contains time in seconds at whicht the peak ends.
                - **bg_t**: List which contains for each peak an np.array of times in seconds, which are used the background calculation.
                - **bg_water**: List which contains value (*float*) water background for each peak.
                - **bg_H2**: List which contains value (*float*) H2 background for each peak.
                - **bg_O18**: List which contains value (*float*) O18  background for each peak.
                - **bg_model_param**: List which contains result parameter (as a *list*) from background fit for each peak.
                - **bg_model_cost**: List which contains final cost function value (as a *list*) from background fit for each peak.
                - **dH2**: List which contains the result value (*float*) of the dH2 isotopy for each peak.
                - **dO18**: List which contains  the result value (*float*) of the dO18 isotopy for each peak.
                - **volume**: List which contains the result value (*float*) of the volume for each peak.

        Returns:
            Variable which was referenced to by ref.
        """
        if ref in ['date', 'datetime', 'real_time']:
            return self.date
        if ref in ['t', 'time', 'duration']:
            return self.t
        elif ref in ['water', 'w', 'H20']:
            return self.water
        elif ref in ['water_rm', "w_rm", 'water running mean', 'running mean']:
            return self.water_rm
        elif ref in ['H2', 'H', 'Hydrogen']:
            return self.H2
        elif ref in ['O18', 'O', 'Oxygen']:
            return self.O18
        elif ref in ['peak_start', 'peak', 'start', 'start of peaks']:
            return self.peak_start
        elif ref in ['peak_end', 'end', 'end of peaks']:
            return self.peak_end
        elif ref in ['bg_t', 'bg_time', 'background_time', 'background time']:
            return self.bg_t
        elif ref in ['bg_water', 'bg_w', 'background_water', 'background water']:
            return self.bg_water
        elif ref in ['bg_H2', 'bg H2', 'bg_H', 'bg H', 'bg_D', 'bg D', 'background H2', 'background H', 'background D']:
            return self.bg_H2
        elif ref in ['bg_O18', 'bg O18', 'bg_O', 'bg O', 'background O18', 'background O']:
            return self.bg_O18
        elif ref in ['bg_model_param', 'bg_fit_param', 'background model parameter', 'fit param', 'bg param']:
            return self.bg_model_param
        elif ref in ['bg_model_cost', 'bg_fit_cost', 'background model cost', 'fit cost', 'bg cost']:
            return self.bg_model_cost

        elif ref in ['dH2', 'delta H2', 'delta D', 'delta H', 'result H2', 'result D', 'result H', 'H2 peak isotopy', 'D peak isotopy']:
            return self.dH2
        elif ref in ['dO18', 'delta O18', 'delta O', 'result O18', 'result O', 'O18 peak isotopy', 'O peak isotopy']:
            return self.dO18

        # TODO add get also parameters and properties
        elif ref in ['tref', 'reference_time', 'reference time', 'reference_date', 'reference date']:
            return self.tref
        else:
            print(ref, 'is not a valid reference')
            return None

    def calibrate(self, H2_calib_param=None, O18_calib_param=None):
        """Calibrate isotopy data from Picarro Instrument.

        Executes a linear calbration for H2 and O18 isotopy data with given parameters.

        Args:
            H2_calib_param: List of parameter for the linear calibration of the H2 values.
            O18_calib_param: List of parameter for the linear calibration of the O18 values.
        """
        if (H2_calib_param is not None or
            O18_calib_param is not None):
            self.H2_calib_param = H2_calib_param
            self.O18_calib_param = O18_calib_param
        elif self.calib == True:  # if no new parameter given and calibration already done do not redo calibration
            return self.O18, self.H2
        if self.verbose > 0: print('calibrating isotopy data ...')
        self.O18 = linear_calib(self.O18, self.O18_calib_param)
        self.H2 = linear_calib(self.H2, self.H2_calib_param)
        self.calib = True
        return None

    def calc_running_mean(self, running_mean_n=None):
        """Calculate running mean over water values

        Calculate running mean using scipy.signal.convole in mode 'same'. May lead to problems if values at the start or end of the data range are used, as values drop rapidly.

        Args:
            running_mean_n: Integer which specifies over how many water values should be averaged.
        """
        if running_mean_n is not None:
            self.running_mean_n = running_mean_n
        filter = np.ones(self.running_mean_n)/self.running_mean_n
        self.water_rm = convolve(self.water, filter, mode='same')
        return None


    def find_peak_start(self, start_slope=None, start_slope_n=None, min_peak_separation=None, ignore_intervals=None):
        '''Find start of peaks in water data.

        Start of peaks is found with a start slope criterion and stored as a list in the class attribute 'peak_start'.

        Args:
            start_slope: Float which specifies the value for slope above which a new peak is detected.
            start_slope_n: Float which specifies the distance of datapoints between which slope is calculated for the detection of the start of a peak.
            ignore_intervals: List of tuples (start, end) which specify intervals in which no peaks are detected
            min_peak_separation: Float which specifies the minimal distance in seconds between start of two peaks.

        Returns:
            List of times at which peaks start.
        '''
        if start_slope is not None:
            self.start_slope = start_slope
        if start_slope_n is not None:
            self.start_slope_n = start_slope_n
        if min_peak_separation is not None:
            self.min_peak_separation = min_peak_separation
        if ignore_intervals is not None:
            self.ignore_intervals = ignore_intervals

        peak_start = []
        if self.verbose > 0: print('calculating start of peaks ...')

        if self.start_slope is None:
            raise TypeError('please set start_slope')
        if self.min_peak_separation is None:
            raise TypeError('please set min_peak_separation')
        if self.start_slope_n is None:
            raise TypeError('please set start_slope_n')

        n = self.start_slope_n
        for i in range(self.water.shape[0]-n):
            if in_interval(self.t[i], self.ignore_intervals): continue
            dt = self.t[i+n] - self.t[i]
            dv = self.water[i+n] - self.water[i]
            slope = dv/dt
            if slope > self.start_slope:
                if not peak_start:
                    peak_start.append(self.t[i])  # first peak
                elif self.t[i] > peak_start[-1] + self.min_peak_separation:
                    peak_start.append(self.t[i])  # additional peaks
        self.peak_start = peak_start
        return self.peak_start

    def find_peak_end(self, end_slope=None, end_slope_n=None, end_slope_running_mean=None, running_mean_n=None, min_peak_size=None, max_peak_size=None):
        '''Find start of peaks in water data.

        End of peaks is found with a end slope criterion and stored as a list in the class attribute 'peak_end'.
        This method corrects for the additional slope due to the background model 'bg_model', if the underground is calculated in 'bg_calc_mode' 'fit'.
        If end_slope_running_mean is True, the running mean is calculated with scipy.signal.convolve (in mode 'same') and used for end peak detection.

        Args:
            end_slope: Float which specifies the value for slope below which the end of a peak is detected.
            end_slope_n: Float which specifies the distance of datapoints between which slope is calculated for the detection of the end of a peak.
            end_slope_running_mean: True if running mean should be used to average over neighbouring datapoints for slope calculation. Set to False if no running mean should be used.
            running_mean_n: Integer which specifies over how many water values are averaged for calculation of the end slope.
            min_peak_size: Float which specifies the minimal length of a peak in seconds.
            max_peak_size: Float which specifies the maximal length of a peak in seconds.

        Returns:
            List of times at which peaks end.
        '''
        if end_slope is not None:
            self.end_slope = end_slope
        if end_slope_n is not None:
            self.end_slope_n = end_slope_n
        if end_slope_running_mean is not None:
            self.end_slope_running_mean = end_slope_running_mean
        if running_mean_n is not None:
            self.running_mean_n = running_mean_n
            if not self.end_slope_running_mean:
                print('Warning: No running mean used despite running_mean_n being set. If running mean should be used please set end_slope_running_mean to true.')
        if min_peak_size is not None:
            self.min_peak_size = min_peak_size
        if max_peak_size is not None:
            self.max_peak_size = max_peak_size

        peak_end = []
        if self.verbose > 0: print('calculating end of peaks ...')

        if self.end_slope is None:
            raise TypeError('please set end_slope')
        if self.end_slope_n is None:
            raise TypeError('please set end_slope_n')
        if self.end_slope_running_mean and not self.running_mean_n:
            raise TypeError('please specifiy running_mean_n')
        if self.min_peak_size is None:
            raise TypeError('please set min_peak_size')
        if self.max_peak_size is None:
            raise TypeError('please set max_peak_size')

        for j in range(len(self.peak_start)):
            mask = ((self.t > self.peak_start[j] + self.min_peak_size)
                    & (self.t < self.peak_start[j] + self.max_peak_size))
            t = self.t[mask]
            if self.end_slope_running_mean:
                self.calc_running_mean()
                w = self.water_rm[mask]
            else:
                w = self.water[mask]

            if self.end_slope is None:
                print("please provide end_slope")
            end_slope = None
            n = self.end_slope_n
            for i in range(w.shape[0]-n):
                slope = (w[i+n] - w[i])/(t[i+n] - t[i])
                if i==0:  # in this case criterion is not time dependend, needs to be calculated only once
                    if self.bg_calc_mode=='fit' and self.bg_calc_mode=='lin' and self.bg_model_param is not None:
                        end_slope = self.end_slope - np.take(self.bg_model_param, 0, axis=1)  # subtract slope due to background
                    else:
                        end_slope = self.end_slope * np.ones(len(self.peak_start))
                if self.bg_calc_mode=='fit' and self.bg_calc_mode=='exp' and self.bg_model_param is not None:
                    A = np.take(self.bg_model_param, 0, axis=1)
                    B = np.take(self.bg_model_param, 1, axis=1)
                    end_slope = self.end_slope - (-A*B*np.exp(-B*t[i]))  # d/dt of exp model = -A*B*exp(-B*t)
                if slope > end_slope[j]:
                    peak_end.append(t[i])
                    break
            if len(peak_end) <= j:  # if no end was found, append max size
                peak_end.append(t[-1])

        self.peak_end = peak_end
        return self.peak_end

    def calc_background(self, bg_period=None, bg_separation_before_peak=None, bg_separation_after_peak=None, bg_calc_mode=None, bg_model=None, bg_fit_start_param=None):
        '''Calculates the background for water as well as H2 and O18 isotope data per peak.

        Background is calculated either
        - in mode 'const' from an interval before the the peak, specified by 'bg_period' and 'bg_separation_before_peak'
        - in mode 'fit' from a fit of an inteval before and one after the peak, using the function :func:`~isotopy_evaluation.fit_model`. A Model for the fit has to be specified in 'bg_model'
        The results are stored as a list in the class attribute 'bg_w', 'bg_H2' and 'bg_O18'.

        Args:
            bg_calc_mode: String which specifies the mode in which the background is calculated. Either of 'const' or 'fit'. For 'fit' mode a bg_model has to be selected.
            bg_model: String which specifies the model which is fitted to the background before and after each peak. Either of 'lin' or 'exp'.
            bg_fit_start_param: List of parameter which are used as starting parameters for the background fit.
            bg_period: Float which specifies the length period before (and after) the peak for which the background is evaluated (or fitted).
            bg_separation_before_peak: Float which specifies the time in seconds between end of the interval used for background calculation (or fit) and start of peak.
            bg_separation_after_peak: Float which specifies the time in seconds between end of peak and start of intervals used for the fit of the background model.
        '''
        if bg_period is not None:
            self.bg_period = bg_period
        if bg_separation_before_peak is not None:
            self.bg_separation_before_peak = bg_separation_before_peak
        if bg_calc_mode is not None:
            self.bg_calc_mode = bg_calc_mode
        if bg_model is not None:
            self.bg_model = bg_model
        if bg_fit_start_param is not None:
            self.bg_fit_start_param = bg_fit_start_param

        if self.bg_calc_mode not in self.BG_CALC_MODE:
            raise ValueError('please provide a valid bg_calc_mode. One of: ', self.BG_CALC_MODE)
        if self.bg_calc_mode == 'fit' and self.bg_model not in self.FIT_MODEL:
            raise ValueError('please provide one of the following bg_model: \'lin\', \'exp\'')
        if self.peak_start is None or not self.peak_start:
            raise ValueError('Please calculate peak_start first.')
        if self.peak_end is None or not self.peak_end:
            raise ValueError('Please calculate peak_end first.')

        bg_t = []
        bg_w = []
        bg_H2 = []
        bg_O18 = []
        bg_model_param = []
        bg_model_cost = []
        if self.verbose > 0: print('calculating background in mode \'' + self.bg_calc_mode + '\' ...')

        for i in range(len(self.peak_start)):
            t0 = self.peak_start[i]
            t1 = self.peak_end[i]
            mask = (self.t > t0 - self.bg_period - self.bg_separation_before_peak) & (self.t < t0 - self.bg_separation_before_peak)
            mask_after = (self.t > t1 + self.bg_separation_after_peak) & (self.t < t1 + self.bg_period + self.bg_separation_after_peak)
            # check that last peak is not in background
            if i > 0:  # only if not first peak
                if (t0 - self.bg_period - self.bg_separation_before_peak) < self.peak_end[i-1]:
                    mask = (self.t > self.peak_end[i-1] + self.bg_separation_after_peak) & (self.t < t0 - self.bg_separation_before_peak)  # add offset to end of last peak to be sure
                    print("background period lower than specified due to close previous peak")
            # check that next peak is not in background
            if i < len(self.peak_start)-1:  # not for last peak
                if (t1 + self.bg_period + self.bg_separation_after_peak) > self.peak_start[i+1]:
                    mask_after = (self.t > t1 + self.bg_separation_after_peak) & (self.t < self.peak_start[i+1] - self.bg_separation_before_peak)  # add offset before start of next peak to be sure
                    print("background period lower than specified due to close next peak")

            t = self.t[mask]
            t_after = self.t[mask_after]
            w = self.water[mask]
            w_after = self.water[mask_after]
            H2 = self.H2[mask]
            H2_after = self.H2[mask_after]
            O18 = self.O18[mask]
            O18_after = self.O18[mask_after]

            if self.bg_calc_mode=='const':
                # integrate over interval before peak to find mean
                # from Affolter et al. [2014]
                mean_water, mean_H2 = isotope_value(t, w, H2)
                mean_water, mean_O18 = isotope_value(t, w, O18)
                bg_t.append(t)
                bg_w.append(mean_water)
                bg_H2.append(mean_H2)
                bg_O18.append(mean_O18)
            elif self.bg_calc_mode=='fit':
                # calculate mean from integral over extrapolated model for water during peak and mean of isotope during interval before peak
                # background water is not constant in measurement data, isotope however is
                if self.bg_model not in self.FIT_MODEL:
                    print('please choose either mode \'const\' or \'fit\' \nfor mode \'fit\' please provide valid model:', self.FIT_MODEL)
                    return None
                if self.bg_fit_start_param and self.bg_model == 'lin' and len(self.bg_fit_start_param) != 2:
                    print('please provide 2 parameters for linear background model')
                    return None
                elif self.bg_fit_start_param and self.bg_model == 'exp' and len(self.bg_fit_start_param) != 3:
                    print('please provide 3 parameters for exponential background model')
                    return None

                # fit model to interval before and after peak
                t_both = np.append(t, t_after)
                w_both = np.append(w, w_after)
                fit_param, fit = fit_model(t_both, w_both, self.bg_model, start_param=self.bg_fit_start_param)

                if fit.success:
                    if self.verbose > 0: print("fit for peak", i, "successfull;", "cost/datapoint at solution: {:.2f}".format(fit.cost/t_both.shape[0]))
                else:
                    if self.verbose > 0: print(fit.message)

                # evaluate model during peak
                mask_peak = (self.t > t0) & (self.t < t1)
                t_peak = self.t[mask_peak]
                if self.bg_model == 'lin':
                    w_model = fit_param[0]*t_peak + fit_param[1]
                elif self.bg_model == 'exp':
                    w_model = fit_param[0]*np.exp(-fit_param[1]*t_peak) + fit_param[2]  # A*exp(-B*t) + C

                # calculate mean water and mean isotopy during peak
                mean_water, mean_H2 = isotope_value(t_peak, w_model, H2.mean()*np.ones(w_model.shape[0]))
                mean_water, mean_O18 = isotope_value(t_peak, w_model, O18.mean()*np.ones(w_model.shape[0]))

                bg_t.append(t)
                bg_w.append(mean_water)
                bg_H2.append(mean_H2)
                bg_O18.append(mean_O18)
                bg_model_param.append(fit_param)
                bg_model_cost.append(fit.cost)

        self.bg_t = bg_t
        self.bg_H2 = bg_H2
        self.bg_O18 = bg_O18
        self.bg_water = bg_w
        if self.bg_calc_mode == 'fit':
            self.bg_model_param = bg_model_param
            self.bg_model_cost = bg_model_cost
        return None

    def calc_isotopy(self):
        '''Calculates isotopy delta values per peak.

        The results are stored as a List in the class attribute 'dH2' and 'dO18'. This methods uses the function :func:`~isotopy_evaluation.isotope_value`.

        implemented after:
        S.Affolter, D. Fleitmann, and M. Leuenberger, 'New online method for water isotope analysis of speleothem fluid inclusions using laser absorption spectroscopy (WS-CRDS)', Clim. Past, 10, 1291-1304, 2014, doi:10.5194/cp-10-1291-2014

        Returns:
            Two lists of isotopy delta values for all peaks, for H2 and O18.
        '''

        dH2 = []
        dO18 = []
        if self.verbose > 0: print('calculating isotopy values ...')

        if self.peak_start is None or not self.peak_start:
            raise ValueError('Please calculate peak_start first.')

        for j in range(len(self.peak_start)):
            mask = ((self.t > self.peak_start[j])
                    & (self.t < self.peak_end[j]))
            t = self.t[mask]
            w = self.water[mask]
            H2 = self.H2[mask]
            O18 = self.O18[mask]
            w_b = self.bg_water[j]
            H2_b = self.bg_H2[j]
            O18_b = self.bg_O18[j]
            w_mix, H2_mix = isotope_value(t, w, H2)
            w_mix, O18_mix = isotope_value(t, w, O18)
            dH2.append((H2_mix*w_mix - H2_b*w_b) / (w_mix - w_b))  # equation 6 in Affolter [2014]
            dO18.append((O18_mix*w_mix - O18_b*w_b) / (w_mix - w_b))

        self.dH2 = dH2
        self.dO18 = dO18
        return self.dH2, self.dO18

    def calc_volume(self, volume_mode=None, volume_calib_param=None):
        '''Calculates the water volume in sample.

        The resulting volume values are stored as a list of values per peak in the class attribute 'volume'.

        Args:
            volume_mode: String which specifies based on which calibration the volume is calculated. Either 'area' or 'height'. In mode 'area' the area under peak in waterdata is used. Water volume has a linear relationship to the area under the peak. In mode 'height' the quadratic relationship of sample volume to the height of the peak in water data is used.
            volume_calib_param: List of parameter for calculation of the water volume of the sample. For linear relationship in mode 'area' parameters are specified as [slope, offset]. For quadratic relationship in mode 'height' parameters are specified as [a, b, offset].

        Returns:
            List of sample water volume value for all peaks.
        '''

        if volume_mode is not None:
            self.volume_mode = volume_mode
        if volume_calib_param is not None:
            self.volume_calib_param = volume_calib_param

        if (self.volume_mode == 'area') and (len(self.volume_calib_param) != 2):
            raise IndexError('please give 2 parameter for calibration in mode \'area\'')
        elif (self.volume_mode == 'height') and (len(self.volume_calib_param) != 3):
            raise IndexError('please give 3 parameter for calibration in mode \'height\'')

        volume = []
        if self.verbose > 0: print('calculating volume in mode \'' + self.volume_mode + '\' ...')

        for j in range(len(self.peak_start)):
            mask = ((self.t > self.peak_start[j])
                    & (self.t < self.peak_end[j]))
            t = self.t[mask]
            w = self.water[mask]
            w_b = self.bg_water[j]
            if self.volume_mode == 'area':
                mix_area = simps(w, t)
                bg_area = w_b * (t[-1] - t[0])
                signal_area = mix_area - bg_area
                volume.append(self.volume_calib_param[0]*signal_area +
                              self.volume_calib_param[1])
            elif self.volume_mode == 'height':
                signal_height = w.max() - w_b
                volume.append(self.volume_calib_param[0]*signal_height**2 +
                              self.volume_calib_param[1]*signal_height +
                              self.volume_calib_param[2])

        self.volume = volume
        return self.volume

    # TODO not finished yet, changes with background models
    def update(self):
        """Redoes the complete evaluation based on the parameters stored in the classes attributes.

        Peak end and background calculation are iterated twice as peak end may depend on the background model."""
        self.find_peak_start()
        self.find_peak_end()
        self.calc_background()
        # TODO iterate and measure change to find after which iteration results no longer change above given threshold
        self.find_peak_end()
        self.calc_background()
        self.calc_isotopy()
        self.calc_volume()
        return None

    def print_peak_properties(self):
        """Prints the characteristic values of peaks."""
        print("duration:")
        print("min peak duration:", self.min_peak_size, "\nmax peak duration:", self.max_peak_size)
        dur = []  # duration of peak
        for i in range(len(self.peak_start)):
            dur.append(self.peak_end[i] - self.peak_start[i])
        print("peak duration:", [round(d) for d in dur])

        print("\nseparation:")
        print("min peak separation:",self. min_peak_separation)
        sep = []  # time to next peak
        for i in range(len(self.peak_start)-1):
            sep.append(self.peak_start[i+1] - self.peak_end[i])
        print("peak separation: ", [round(s) for s in sep])

        print("\npeaks ")
        if not len(self.peak_start) == len(self.peak_end):
            print("peak_start and peak_end do not match")
        for i in range(len(self.peak_start)):
            if i == len(self.peak_start)-1: print("{:>2}  {:} - {:}  dur: {:=5.1f}".format(i, date(self.peak_start[i], self.tref).strftime("%d.%m. %H:%M"), date(self.peak_end[i], self.tref).strftime("%d.%m. %H:%M"), dur[i]))
            else: print("{:>2}  {:} - {:}  dur: {:=5.1f}  sep: {:=6.1f}".format(i, date(self.peak_start[i], self.tref).strftime("%d.%m. %H:%M"), date(self.peak_end[i], self.tref).strftime("%H:%M"), dur[i], sep[i]))
        return None

    def print_peak_results(self):
        """Prints the results of the evaluation for each peak."""
        if not len(self.peak_start) == len(self.peak_end):
            print("peak_start and peak_end do not match")
        for i in range(len(self.peak_start)):
            print("{:>2}  {:} - {:}   H2: {:=5.1f}   O18: {:=6.1f}  V: {:4.1f}".format(i, date(self.peak_start[i], self.tref).strftime("%d.%m. %H:%M"), date(self.peak_end[i], self.tref).strftime("%d.%m. %H:%M"), self.dH2[i], self.dO18[i], self.volume[i]))
        return None

    def write_data_to_file(self, name=None, path_to_results=None):
        """Writes data and used parameters to a .txt file.

        Args:
            name: String which specifies the name of the Evaluation.
            path_to_results: String which specifies the path to directory in whicht the textfile containing the evaluation parameters and resuls is written.
        """
        # check if exist & handle possibly with user input
        if name is not None:
            self.name = name
            print('name of evaluation set to: ' + self.name)
        if path_to_results is not None:
            self.path_to_results = path_to_results
        if self.name is None:
            print("please set name of evaluation")
            return None
        if self.path_to_results is None:
            print("please set path_to_results")
        date = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(self.tref.year, self.tref.month, self.tref.day, self.tref.hour, self.tref.minute, self.tref.second)
        filename = os.path.join(self.path_to_results, (date + "_" + self.name +'.txt'))
        if os.path.isfile(filename):
            print("This file already exists.")
            overwrite = input("Do you want to overwrite it? y/n: ")
            if overwrite == "y":
                pass
            else:
                return None

        if self.verbose > 0: print('Writing parameter and results to file ' + filename + ' ...')

        with open(filename, 'w') as f:
            # header
            f.write("Reference Time: {:02d}.{:02d}.{:04d} {:02d}:{:02d}:{:02d}\n".format(self.tref.day, self.tref.month, self.tref.year, self.tref.hour, self.tref.minute, self.tref.second))

            f.write("\n# Calibration \n")
            f.write("H2 calibration parameter: {}\n".format(self.H2_calib_param))
            f.write("O18 calibration parameter: {}\n".format(self.O18_calib_param))
            f.write("Volume evaluation from peak {}\n".format(self.volume_mode))
            f.write("Volume calibration parameters: {}\n".format(self.volume_calib_param))

            f.write("\n# Peak Detection\n")
            f.write("Threshold slope for start of peak: {}\n".format(self.start_slope))
            f.write("Distance of datapoints for slope calculation for start of peak: {}\n".format(self.start_slope_n))
            f.write("Threshold slope for end of peak: {}\n".format(self.end_slope))
            f.write("Distance of datapoints for slope end of peak detection: {}\n".format(self.end_slope_n))
            f.write("Running Mean used for end of peak detection: {}\n".format(self.end_slope_running_mean))
            if self.end_slope_running_mean:
                f.write("Number of water values averaged for end of peak detection: {}\n".format(self.running_mean_n))

            f.write("\n# Background Calculation\n")
            f.write("Background period: {}\n".format(self.bg_period))
            f.write("Separation to background before peak: {}\n".format(self.bg_separation_before_peak))
            f.write("Separation to background after peak: {}\n".format(self.bg_separation_after_peak))
            f.write("Water background calculated in mode: {}\n".format(self.bg_calc_mode))
            if self.bg_calc_mode == 'fit':
                f.write("Model used for water background fit: {}\n".format(self.bg_model))
                #f.write("Parameter from fit: {}\n".format(self.bg_model_param))

            # data
            f.write("\n")
            f.write("\n# Results\n")
            f.write("\n")
            f.write("{0:9} {1:9} {2:9} {3:9} {4:13}\n".format("start", "end", "dH2", "dO19", "sample volume"))
            f.write("{0:9} {1:9} {2:9} {3:9} {4:13}\n".format("[second]", "[second]", "[permil]", "[permil]", "[micro liter]"))
            for i in range(len(self.peak_start)):
                f.write("{0:9.2f} {1:9.2f} {2:+9.3f} {3:+9.3f} {4:13.3f}\n".format(self.peak_start[i], self.peak_end[i], self.dH2[i], self.dO18[i], self.volume[i]))
        return None

    # TODO
    def save(self):
        """Not implemented yet. Should store complete class to a file. (E.g. hdf5)"""
        return None

    # TODO init from file -> restore

# Import of Picarro files
def read_picarro_file(filename, time_offset=dt.timedelta(hours=0.), index=[20,21,22]):
    """Reads a single file of measurement data by a Picarro instrument

    Args:
        filename: String which specifies the filename of datafile. May also be a filename including the path to file, if file is not stored in current directory.
        time_offset: Datetime timedelta object which specifies the time offset between the time in the picarro device (and therefore in the saved data) and the real labtime.

    Returns:
        The header with description of read columns and four lists for time and the measured values for water, H2 and O18.
    """
    # index = [water, O18, H2]
    header=[]
    time=[]
    water=[]
    O18=[]
    H2=[]
    with open(filename,'r') as f:
        header.append(re.findall(re_header, f.readline()))  # extract header from first line
        for line in f:
            current = re.findall(re_data,line)
            time.append(dt.datetime(year = int(current[0]),
                                    month = -int(current[1]),  # negative to adjust for date formatting yyyy-mm-dd
                                    day = -int(current[2]),
                                    hour = int(current[3]),
                                    minute = int(current[4]))
                        + dt.timedelta(seconds=float(current[5]))  # add milliseconds as fraction of seconds
                        + time_offset)  # compensate for summertime or instrument time offset
            water.append(float(current[index[0]]))
            O18.append(float(current[index[1]]))
            H2.append(float(current[index[2]]))
    return header, time, water, H2, O18

def get_picarro_data(filelist=[], time_offset=dt.timedelta(hours=0.), index=[20,21,22]):
    """Read and combine data from list of files.

    Uses the 'read_picarro_file' function to import the individual files.
    Combines the data from individual files.

    Args:
        filelist: List of filenames (strings) or filenames including the path to files, if files are not stored in current directory.
        time_offset: Datetime timedelta object which specifies the time offset between the time in the picarro device (and therefore in the saved data) and the real labtime.

    Returns:
        A list of the times of measured datapoints as well as 3 numpy arrays of the values for water, H2 and O18.
    """
    # index = [water, O18, H2]
    time = []
    water = []
    O18 = []
    H2 = []
    if not filelist:
        filelist.extend(glob('*.dat'))  # import files in directory
    if not filelist:
        print("no files found to process")
        return time, np.array(water), np.array(O18), np.array(H2)
    filelist = sorted(filelist)
    for file in filelist:
        header_read, time_read, water_read, H2_read, O18_read = read_picarro_file(file, time_offset=time_offset, index=index)
        time.extend(time_read)
        water.extend(water_read)
        O18.extend(O18_read)
        H2.extend(H2_read)
    return time, np.array(water), np.array(H2), np.array(O18)

def import_picarro_data(dir_name, start, end, time_offset=dt.timedelta(hours=0.)):
    """Imports data from the Picarro instrument from the folder structure provided by the instrument.

    Args:
        dir_name: String which specifies the relative path to the directory the data is stored in. The directory should contain subdirectories corresponding to date (as in \'dir_name/yyyy/mm/dd/*.dat\').
        start: Datetime object which specifies the date and time from which on the data should be imported.
        end: Datetime object which specifies the date and time up to which the data should be imported.
        time_offset: Datetime timedelta object which specifies the time offset between the time in the picarro device (and therefore in the saved data) and the real labtime.

    Returns:
        A list of the times of measured datapoints as well as 3 numpy arrays of the values for water, H2 and O18.
    """

    files = []
    date = start
    while date <= end:
        files.extend(glob(os.path.join(dir_name,
                                       '{:04}'.format(date.year),
                                       '{:02}'.format(date.month),
                                       '{:02}'.format(date.day),
                                       '*.dat')))
        date = date + dt.timedelta(days=1.)

    # read all files
    t, water, H2, O18 = get_picarro_data(filelist=files, time_offset=time_offset)

    # return filtered for date in [start, end]
    ta = np.array(t)
    mask = (ta >= start) & (ta <= end)
    t = ta[mask].tolist()
    if len(t) == 0:
        print('no data in specified time interval')
        return None
    print('imported {0} datapoints from \n{1} to \n{2}'.format(len(t), t[0], t[-1]))
    return t, water[mask], H2[mask], O18[mask]

# calculations
def linear_calib(raw_data, calib_param):
    """Calibrates raw data by linear function.

    :param raw_data: List or np.array of raw data.
    :param list calib_param: Parameter of linear function used for calibration [slope, offset].
    """
    if not isinstance(calib_param, list):
        raise TypeError('Please provide calib_param as a list[slope, offset] for linear calibration')
    elif not calib_param or len(calib_param) != 2:
        raise ValueError('Please set calib_param as a list [slope, offset] for linear calibration')
    return calib_param[0]*raw_data + calib_param[1]

def fit_model(time, value, model, start_param=None):
    """Fits model with given starting parameters to values. Returns list of parameters.

    Uses the function scipy.optimize.least_squares.

    Args:
        time: Array with time of the data points in seconds.
        value: Array of values corresponding to times.
        model: String specifying the model used for the fit. Either 'lin' or   'exp' for linear (slope*value + offset) or exponential model (A*exp(-B*value) + offset).
        start_param: List of start parameters for given model. Eiter [slope, offset] for 'lin' model or [A, B, C] for 'exp' model.

    Returns:
        res.x: A list of fitting parameters corresponding to the model.
        res: Object returned by scipy.optimize.least_squares.
    """
    if model=='lin' and start_param is None:
        start_param = [1., 0.]
    elif model=='exp' and start_param is None:
        start_param = [1., 1., 0.]

    # fit model to intervals
    if model=='lin':
        fitfunc = lambda p, x: p[0]*x + p[1]
        errfunc = lambda p, x, y: fitfunc(p, x) - y
    elif model=='exp':
        fitfunc = lambda p, x: p[0]*np.exp(-p[1]*x) + p[2]
        errfunc = lambda p, x, y: fitfunc(p, x) - y

    res = least_squares(errfunc, start_param, args=(time, value))

    return res.x, res

def isotope_value(t, w, i):
    """Calculates the mean isotope values based on isotope values and corresponding water concentration.

    mean isotope = I Isotope(t)*water(t) dt / I water(t) dt

    mean water = I water(t) dt / I dt

    For the integration scipy.integrate.simps is used.

    Args:
        t: Numpy array with the time of the datapoints.
        w: Numpy array with the water value of the datapoints.
        i: Numpy array with the isotope of the datapoints.

    Returns:
        Two floats for the calculated water and isotope value.
    """
    # integrate with either
    # - trapz (trapezoidal rule, linear approx. error~O(h^2))
    # - simps (simpsons rule, quadratic approx. error~O(h^4))

    # calculation from Affolter et al. [2014]
    mean_isotope = simps(np.multiply(i, w), t) / simps(w, t) # I Isotope(t)*water(t) dt / I water(t) dt
    mean_water = simps(w, t) / (t[-1] - t[0])   # I water(t) dt / I dt
    return mean_water, mean_isotope

# Conversion: datetime <-> time in seconds
def date(seconds, tref):
    """ Convert elapsed time (in seconds) since reference time into datetime object.

    Args:
        seconds: Float, list or numpy array of the elapsed time (in seconds).
        tref: Datetime object which specifies the reference time of the evaluation.

    Returns:
        Datetime object(s) corresponding to the reference time plus the specified duration. List of datetime objects if multiple times were given.
    """
    if isinstance(seconds, (list, np.ndarray)):
        return [tref + dt.timedelta(seconds=s) for s in seconds]
    elif isinstance(seconds, (int, float)):
        return tref + dt.timedelta(seconds=seconds)
    else:
        raise TypeError("please provide time in seconds as either: int, float, list or numpy array")
        return None

def pdate(seconds, tref):
    """Print date and time of elapsed time since reference time.

    Args:
        seconds: Float, list or numpy array of the elapsed time (in seconds).
        tref: Datetime object which specifies the reference time of the evaluation.
    """
    if isinstance(seconds, list) or isinstance(seconds, np.ndarray):
        for s in seconds:
            print(seconds.index(s), str(s)+":", date(s, tref).strftime("%d.%m.%Y %H:%M:%S"))
    elif isinstance(seconds, float) or isinstance(seconds, int):
        print(str(seconds)+":",date(seconds, tref).strftime("%d.%m.%Y %H:%M:%S"))
    else:
        print("please provide time in seconds as either: int, float, list or numpy array")
    return None

def eval_time(time, tref):
    """Converts absolute (laboratory time) to time relative (in seconds) to reference time.

    Args:
        time: Absolute (laboratory time) as datetime object or as formatted string. Provide either of the following formats:

            - datetime timedelta objects
            - dd.mm.yyyy hh:mm:ss
            - dd.mm.yyyy hh:mm
            - hh:mm:ss
            - hh:mm

            tref: Datetime object which specifies the reference time of the evaluation.

    Returns:
        Float of the elapsed time since reference time.
    """

    if isinstance(time, dt.datetime): return (time - tref).total_seconds()
    elif isinstance(time, str):
        try:
            parsed_time = dt.datetime.strptime(time, "%d.%m.%Y %H:%M:%S")
        except:
            try:
                parsed_time = dt.datetime.strptime(time, "%d.%m.%Y %H:%M")
            except:
                try:
                    parsed_time = dt.datetime.strptime(time, "%d.%m. %H:%M:%S")
                    parsed_time = parsed_time.replace(year=tref.year)
                except:
                    try:
                        parsed_time = dt.datetime.strptime(time, "%d.%m. %H:%M")
                        parsed_time = parsed_time.replace(year=tref.year)
                    except:
                        try:
                            parsed_time = dt.datetime.strptime(time, "%H:%M:%S")
                            parsed_time = parsed_time.replace(year=tref.year, month=tref.month, day=tref.day)
                        except:
                            try:
                                parsed_time = dt.datetime.strptime(time, "%H:%M")
                                parsed_time = parsed_time.replace(year=tref.year, month=tref.month, day=tref.day)
                            except:
                                raise ValueError("could not parse time. Provide either of the following formats:\n- datetime objects\n- dd.mm.yyyy hh:mm:ss\n- dd.mm.yyyy hh:mm\n- dd.mm. hh:mm:ss\n- dd.mm. hh:mm\n- hh:mm:ss\n- hh:mm")
        return (parsed_time - tref).total_seconds()

def time_conversion(date_array, tref):
    """Converts a array of datetimes to an array of time in seconds (floats).

    Args:
        date_array: Numpy array of datetime objects of the measurement times.
        ref: Datetime object which specifies the date and time for which the time in seconds is 0.

    Returns:
        Numpy array of the elapsed times since reference time in seconds.
    """
    time = []
    for t in date_array:
        time.append((t-tref).total_seconds())
    return np.array(time)

# other helpers
def in_interval(t, list_intervals):
    """Checks if t is in one of the intervals.

    Args:
        t: Float which specifies the number which should be checked wether it is inside one of the intervals specified by 'list_intervals'.
        list_intervals: List of Intervals specified as tuples of start and end of the interval.

    Returns:
        True if the number was inside the interval, false if not or equal to lower or upper limit.
    """
    if not list_intervals:
        return False
    if  isinstance(list_intervals, list):
        for interval in list_intervals:
            if not isinstance(interval, tuple):
                raise TypeError('please specify interval(s) as list of tuple(s)')
            elif len(interval) != 2:
                raise IndexError('please specify interval(s) as list of tuple(s)')
    else:
        raise TypeError('please specify interval(s) as list of tuple(s)')

    for interval in list_intervals:
        if t > interval[0] and t < interval[1]:
            return True
    return False
