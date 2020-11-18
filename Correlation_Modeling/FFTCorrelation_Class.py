from FFT_Class import FFTProcessor, APFormFourierSeriesProcessor, FourierSeriesProcessor, LASSOFFTProcessor, \
    SCFormFourierSeriesProcessor, BayesianFFTProcessor
from TimeSeries_Class import merge_two_time_series_df, TimeSeries, WindowedTimeSeries
import pandas as pd
from numpy import ndarray
from Ploting.fast_plot_Func import *
from typing import Tuple, Iterable
from .utils import BivariateCorrelationAnalyser
from collections import namedtuple, OrderedDict
import re


class FFTCorrelationMeta(type):
    @staticmethod
    def _make_init(fields):
        code = f"def __init__(cls, *, {', '.join(fields)}, **kwargs):\n"
        for this_filed in fields:
            code += f'    cls.{this_filed} = {this_filed}\n'
        return code

    def __new__(mcs, name, bases, clsdict):
        if '__init__' not in clsdict:
            init_fields = ['correlation_func',
                           'n_fft',
                           'considered_frequency_unit']
            exec(mcs._make_init(init_fields), globals(), clsdict)
        clsobj = super().__new__(mcs, name, bases,
                                 clsdict)
        return clsobj


class FFTCorrelation(metaclass=FFTCorrelationMeta):
    __slots__ = (
        'time_series',
        'correlation_func',
        'n_fft',
        'considered_frequency_unit'
    )

    @property
    def sampling_period(self):
        return (self.time_series.index[1] - self.time_series.index[0]).seconds


class BivariateFFTCorrelation(FFTCorrelation):
    __slots__ = ('main_ifft',
                 'vice_ifft')

    def __init__(self, *,
                 _time_series: TimeSeries = None,
                 main_time_series_df: pd.DataFrame = None,
                 vice_time_series_df: pd.DataFrame = None,
                 main_considered_peaks_index: Union[tuple, list],
                 vice_considered_peaks_index: Union[tuple, list],
                 main_find_peaks_args: dict = None,
                 vice_find_peaks_args: dict = None,
                 **kwargs):
        """
        参数time_series和(main_time_series_df和vice_time_series_df)二选一。
        :param _time_series TimeSeries类或者其子类，默认第0列对应'main_time_series'，第1列对应'vice_time_series'

        :param main_time_series_df 主时间序列。比如，在load和temperature相关性建模中，
        temperature是主，load是副。这种关系主要影响corr_between_main_peaks_f_and_vice方法和
        corr_between_combined_main_peaks_f_and_vice方法的行为。
        temperature（主）会作为一个完整的时序的序列。但是load（副）会被fft分解

        :param vice_time_series_df 副时间序列

        :param main_considered_peaks_index main哪些分量会被考虑到（这是它们的索引）。注意，这与considered_frequency_unit
        有关，也与FFTProcessor类的find_peaks_of_fft_frequency方法有关

        :param vice_considered_peaks_index


        """
        super(BivariateFFTCorrelation, self).__init__(**kwargs)
        if (main_time_series_df is None) and (vice_time_series_df is None):
            self.time_series = _time_series  # type: TimeSeries
        else:
            self.time_series = merge_two_time_series_df(main_time_series_df,
                                                        vice_time_series_df)
        # 找peaks
        main_found_peaks, vice_found_peaks = self._find_principal_frequency_component(
            main_find_peaks_args=main_find_peaks_args,
            vice_find_peaks_args=vice_find_peaks_args
        )
        # 对考虑的分量分别单独进行ifft
        # x_ifft的key就是x_considered_peaks_index, value是一个tuple(hz频率，指定单位频率，幅值，角度，ifft的结果)
        main_considered_peaks_index = np.array(main_considered_peaks_index)
        main_considered_peaks_index = main_considered_peaks_index[main_considered_peaks_index <=
                                                                  np.max(main_found_peaks.index)]

        vice_considered_peaks_index = np.array(vice_considered_peaks_index)
        vice_considered_peaks_index = vice_considered_peaks_index[vice_considered_peaks_index <=
                                                                  np.max(vice_found_peaks.index)]
        self.main_ifft, self.vice_ifft = self._cal_ifft(
            main_found_peaks=main_found_peaks,
            vice_found_peaks=vice_found_peaks,
            main_considered_peaks_index=main_considered_peaks_index,
            vice_considered_peaks_index=vice_considered_peaks_index
        )

    def _do_fft(self) -> Tuple[FFTProcessor, FFTProcessor]:
        """
        分别进行fft
        :return Tuple[FFTProcessor, FFTProcessor]
        """
        main_time_series_fft = FFTProcessor(self.time_series.iloc[:, 0].values,
                                            sampling_period=self.sampling_period,
                                            name='main_time_series',
                                            n_fft=self.n_fft)  # type: FFTProcessor
        vice_time_series_fft = FFTProcessor(self.time_series.iloc[:, 1].values,
                                            sampling_period=self.sampling_period,
                                            name='vice_time_series',
                                            n_fft=self.n_fft)  # type: FFTProcessor
        return main_time_series_fft, vice_time_series_fft

    def _find_principal_frequency_component(self, *, main_find_peaks_args: dict = None,
                                            vice_find_peaks_args: dict = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        :return: tuple中带两个pd.DataFrame，每个都是FFTProcessor对象find_peaks_of_fft_frequency方法后的结果（reset_index()）
        """
        main_fft, vice_fft = self._do_fft()
        # 找peaks
        main_find_peaks_args = main_find_peaks_args or {}
        vice_find_peaks_args = vice_find_peaks_args or {}
        main_found_peaks = main_fft.find_peaks_of_fft_frequency(
            self.considered_frequency_unit,
            **main_find_peaks_args)
        vice_found_peaks = vice_fft.find_peaks_of_fft_frequency(
            self.considered_frequency_unit,
            **vice_find_peaks_args)
        return main_found_peaks[0].reset_index(), vice_found_peaks[0].reset_index()

    def _cal_ifft(self, *,
                  main_found_peaks: pd.DataFrame,
                  vice_found_peaks: pd.DataFrame,
                  main_considered_peaks_index: Iterable,
                  vice_considered_peaks_index: Iterable):
        OneIFFTResultsVal = namedtuple(
            'OneIFFTResultsVal',
            ('hz_f', 'considered_unit_f', 'magnitude', 'phase', 're_constructed_time_domain')
        )

        # key是int， 代表第几个peak。value是一个tuple(hz频率，指定单位频率，幅值，角度，ifft的结果)
        def one_ifft(considered_peaks_index, found_peaks: pd.DataFrame, ):
            """
            :param considered_peaks_index 考虑第几个peaks
            :param found_peaks peaks self._find_principal_frequency_component方法的结果的元素
            """

            temp = {key: None for key in considered_peaks_index}
            for key in temp:
                hz_f = found_peaks['frequency'][key]
                considered_unit_f = found_peaks[self.considered_frequency_unit][key]
                magnitude = found_peaks['magnitude'][key]
                phase = found_peaks['phase angle (rad)'][key]

                re_constructed_time_domain = APFormFourierSeriesProcessor(frequency=np.array([hz_f]),
                                                                          magnitude=np.array([magnitude]),
                                                                          phase=np.array([phase]))
                re_constructed_time_domain = re_constructed_time_domain(self.time_series.index, False)
                temp[key] = OneIFFTResultsVal(
                    hz_f=hz_f,
                    considered_unit_f=considered_unit_f,
                    magnitude=magnitude,
                    phase=phase,
                    re_constructed_time_domain=re_constructed_time_domain
                )
            return temp

        main_ifft = one_ifft(main_considered_peaks_index, main_found_peaks)
        vice_ifft = one_ifft(vice_considered_peaks_index, vice_found_peaks)
        return main_ifft, vice_ifft

    @property
    def main_time_series(self) -> TimeSeries:
        return TimeSeries(self.time_series.iloc[:, 0])

    @property
    def vice_time_series(self) -> TimeSeries:
        return TimeSeries(self.time_series.iloc[:, 1])

    def corr_between_pairwise_peak_f_ifft(self) -> dict:
        """
        :return 一个dict。key代表考虑的相关性系数（符合CorrelationFuncMapper类的定义），
        value是一个pd.DataFrame，column的名字代表main序列中选定的第几个fft peak，
        row的名字代表vice序列中选定的第几个fft peak
        """
        pairwise_correlation = {key: pd.DataFrame(columns=list(self.main_ifft.keys()),
                                                  index=list(self.vice_ifft.keys())) for key in self.correlation_func}
        for key in pairwise_correlation:
            for main_ifft_key in self.main_ifft.keys():
                for vice_ifft_key in self.vice_ifft.keys():
                    corr = BivariateCorrelationAnalyser(self.main_ifft[main_ifft_key][-1],
                                                        self.vice_ifft[vice_ifft_key][-1])
                    one_result = corr(key)[0]
                    # 将一次pairwise的计算写入pairwise_correlation中key对应的value（即：一个pd.DataFrame）
                    pairwise_correlation[key].loc[vice_ifft_key, main_ifft_key] = one_result
        return pairwise_correlation

    def corr_between_main_and_one_selected_vice_peak_f_ifft(self) -> dict:
        """
        这个函数和上面的corr_between_pairwise_peaks_f相似。但是只对vice做了fft，然后逆变换选定的*一个*peak，用得到的
        reconstructed signal去和main的原信号计算相关性
        :return 一个dict。key代表考虑的相关性系数（符合CorrelationFuncMapper类的定义），
        value是一个pd.DataFrame，column的名字默认是'main time series'，
        row的名字代表vice序列中选定的第几个fft peak
        """
        main_peaks_f_and_vice_correlation = {
            key: pd.DataFrame(columns=('main time series',),
                              index=list(self.vice_ifft.keys())) for key in self.correlation_func}
        for key in main_peaks_f_and_vice_correlation:
            for vice_ifft_key in self.vice_ifft.keys():
                corr = BivariateCorrelationAnalyser(self.main_time_series.values,
                                                    self.vice_ifft[vice_ifft_key][-1])
                one_result = corr(key)[0]
                main_peaks_f_and_vice_correlation[key].loc[vice_ifft_key] = one_result
        return main_peaks_f_and_vice_correlation

    def corr_between_main_and_combined_selected_vice_peaks_f_ifft(
            self, *,
            vice_extra_hz_f: Iterable = None,
            do_lasso_fitting_args: dict = None,
            do_bayesian_fitting_args: dict = None) -> Tuple[SCFormFourierSeriesProcessor, dict]:
        """
        利用lasso去fit选择的frequencies form的Fourier expansion

        :param vice_extra_hz_f 给vice加入额外的频率用于分解

        :param do_lasso_fitting_args LASSOFFTProcessor对象的do_lasso_fitting函数的args

        :param do_bayesian_fitting_args BayesianFFTProcessor对象do_bayesian_fitting函数的args

        :return 一个dict。key代表考虑的相关性系数（符合CorrelationFuncMapper类的定义），
        value是tuple，里面的元素是排序好的namedtuple -> 属性frequency_combination是一个tuple，属性corr_value是相关性值
        """
        vice_extra_hz_f = vice_extra_hz_f or np.array([])

        # 注意fourier_full_results包含了base量，做combination的时候用不到
        frequency = np.concatenate((vice_extra_hz_f, [x[0] for x in self.vice_ifft.values()]))

        # 如果没给lasso，就用bayesian
        if do_bayesian_fitting_args is not None:
            # 设置bayesian默认参数
            do_bayesian_fitting_args = do_bayesian_fitting_args or {}
            bayesian_fitting = BayesianFFTProcessor(
                frequency=frequency,
                target=self.vice_time_series
            ).do_bayesian_fitting(**do_bayesian_fitting_args)
            sc_form_fourier_series_processor = bayesian_fitting[-1]
        # 优先用lasso去fit (#强制利用use_lasso_fft_to_re_estimate)
        else:
            # 设置lasso默认参数
            do_lasso_fitting_args = do_lasso_fitting_args or {'alpha': 0.0001,
                                                              'max_iter': 1_000_000,
                                                              'tol': 1e-8,
                                                              'random_state': 0}
            lasso_fitting = LASSOFFTProcessor(
                frequency=frequency,
                target=self.vice_time_series
            ).do_lasso_fitting(**do_lasso_fitting_args)
            sc_form_fourier_series_processor = lasso_fitting[-1]  # type: SCFormFourierSeriesProcessor

        fourier_full_results = sc_form_fourier_series_processor(self.vice_time_series.index,
                                                                return_raw=True)[-1]
        # combination
        combination = sc_form_fourier_series_processor.combination_of_frequency_selector(
            remove_base=True,
            call__raw_results=fourier_full_results
        )
        # 计算相关性系数并且sort
        PartlyCallCorr = namedtuple("PartlyCallCorr",
                                    ("frequency_combination", "partly_combination_reconstructed", "corr_value"))
        final_correlation_results = {key: None for key in self.correlation_func}
        for key in final_correlation_results:
            this_corr_func_results = []
            for this_combination in combination:
                corr = BivariateCorrelationAnalyser(self.main_time_series.values.flatten(),
                                                    this_combination.partly_combination_reconstructed)
                this_corr_func_results.append(
                    PartlyCallCorr(
                        frequency_combination=this_combination.frequency_combination,
                        partly_combination_reconstructed=this_combination.partly_combination_reconstructed,
                        corr_value=corr(key)[0]
                    )
                )
            # 排序
            this_corr_func_results = sorted(this_corr_func_results,
                                            key=lambda x: x.corr_value, reverse=True)
            final_correlation_results[key] = tuple(this_corr_func_results)

        return sc_form_fourier_series_processor, final_correlation_results
