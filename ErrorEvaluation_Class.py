from Data_Preprocessing import float_eps
import warnings
from Ploting.fast_plot_Func import *
from typing import Callable


class ErrorEvaluation:
    def __init__(self, *, target: Sequence, model_output: Sequence, **kwargs):
        assert target.__len__() == model_output.__len__()

        self.target = target
        self.model_output = model_output


class DeterministicError(ErrorEvaluation):
    def cal_error(self, error_name: str) -> float:
        if error_name == 'mean_absolute_error':
            error = self.cal_mean_absolute_error()
        elif error_name == 'max_absolute_error':
            error = self.cal_max_absolute_error()
        elif error_name == 'weighted_mean_absolute_error':
            error = self.cal_weighted_mean_absolute_error()
        elif error_name == 'root_mean_square_error':
            error = self.cal_root_mean_square_error()
        elif error_name == 'weighted_root_mean_square_error':
            error = self.cal_weighted_root_mean_square_error()
        else:
            raise Exception('Fatal error: cannot calculate assigned error type')
        return error

    def cal_max_absolute_error(self) -> float:
        error = np.nanmax(np.abs(self.target - self.model_output))
        return float(error)

    def cal_mean_absolute_error(self) -> float:
        error = np.nanmean(np.abs(self.target - self.model_output))
        return float(error)

    def cal_weighted_mean_absolute_error(self) -> float:
        pass

    def cal_root_mean_square_error(self) -> float:
        error = np.sqrt(np.nanmean((self.target - self.model_output) ** 2))
        return float(error)

    def cal_weighted_root_mean_square_error(self) -> float:
        pass

    def cal_mean_absolute_percentage_error(self) -> float:
        return float(np.mean(np.abs((self.target - self.model_output) / self.target))) * 100


class EnergyBasedError(ErrorEvaluation):
    __slots__ = ('time_step',)

    def __init__(self, *, target: ndarray, model_output: ndarray, time_step: float):
        """

        :param time_step指相邻记录的时间间隔，单位是小时
        """
        super().__init__(target=target, model_output=model_output)
        self.time_step = time_step
        if (sum(np.isnan(target)) > 0) or (sum(np.isnan(model_output)) > 0):
            warnings.warn("At least one 'target' or 'model_output' is nan")

    def do_calculation(self):
        over_estimate_mask = self.model_output > self.target

        over_estimate = np.nansum(
            self.model_output[over_estimate_mask] - self.target[over_estimate_mask]) * self.time_step
        under_estimate = np.nansum(
            self.target[~over_estimate_mask] - self.model_output[~over_estimate_mask]) * self.time_step
        over_minus_under_estimate = over_estimate - under_estimate
        over_plus_under_estimate = over_estimate + under_estimate

        target_total_when_over = np.nansum(self.target[over_estimate_mask]) * self.time_step
        target_total_when_under = np.nansum(self.target[~over_estimate_mask]) * self.time_step
        target_total = np.nansum(self.target) * self.time_step

        over_estimate_in_pct = over_estimate / target_total_when_over * 100
        under_estimate_in_pct = under_estimate / target_total_when_under * 100
        model_output_total_dividing_target_total = over_minus_under_estimate / target_total * 100
        model_output_total_plus_dividing_target_total = over_plus_under_estimate / target_total * 100
        return {'over_estimate': over_estimate,
                'over_estimate_in_pct': over_estimate_in_pct,
                'target_total_when_over': target_total_when_over,
                #
                'under_estimate': under_estimate,
                'under_estimate_in_pct': under_estimate_in_pct,
                'target_total_when_under': target_total_when_under,
                #
                'model_output_total': over_minus_under_estimate,
                'model_output_total_dividing_target_total': model_output_total_dividing_target_total,
                'target_total': target_total,
                #
                'model_output_total_plus': over_plus_under_estimate,
                'model_output_total_plus_dividing_target_total': model_output_total_plus_dividing_target_total
                }


class ProbabilisticError(ErrorEvaluation):
    monte_carlo_sample_size: int = 100_000
    np.random.seed(0)

    def cal_continuous_ranked_probability_score(self, integral_boundary: Sequence[Union[int, float]]) -> ndarray:
        # ref: https://www.lokad.com/continuous-ranked-probability-score#Numerical_evaluation_4
        # The elements of model_output should be Callable instance, which is the predicted CDF (given x, should
        # return its corresponding CDF value)
        assert isinstance(self.model_output[0], Callable)

        def cal_crps_for_single_sample(actual_val, predicted_cdf):
            nonlocal integral_boundary

            part_1 = np.mean(predicted_cdf(np.random.uniform(integral_boundary[0],
                                                             actual_val,
                                                             self.monte_carlo_sample_size)) ** 2)
            part_2 = np.mean((predicted_cdf(np.random.uniform(actual_val,
                                                              integral_boundary[1],
                                                              self.monte_carlo_sample_size)) - 1) ** 2)
            return part_1 + part_2

        crps_ans = np.full(self.target.__len__(), np.nan)
        for i in range(len(crps_ans)):
            if self.model_output[i] is not None:
                now_crps = cal_crps_for_single_sample(self.target[i], self.model_output[i])
                crps_ans[i] = now_crps

        return crps_ans

    def cal_pinball_loss(self, quantiles_to_assess: ndarray = None) -> ndarray:
        # ref: https://www.lokad.com/pinball-loss-function-definition
        quantiles = quantiles_to_assess or np.arange(0.001, 1., 0.001)
        # The elements of model_output should be Callable instance, which is the predicted CDF (given x, should
        # return its corresponding CDF value)
        assert isinstance(self.model_output[0], Callable)

        def cal_pinball_loss_for_single_sample(actual_val, predicted_cdf):
            nonlocal quantiles

            one_ans = np.full(quantiles.__len__(), np.nan)

            forecasted = predicted_cdf(quantiles)

            mask_act_gte = actual_val >= forecasted
            one_ans[mask_act_gte] = (actual_val - forecasted[mask_act_gte]) * quantiles[mask_act_gte]
            one_ans[~mask_act_gte] = (forecasted[~mask_act_gte] - actual_val) * (1 - quantiles[~mask_act_gte])

            return np.mean(one_ans)

        pinball_loss_ans = np.full(self.target.__len__(), np.nan)
        for i in range(len(pinball_loss_ans)):
            if self.model_output[i] is not None:
                now_pinball_loss = cal_pinball_loss_for_single_sample(self.target[i], self.model_output[i])
                pinball_loss_ans[i] = now_pinball_loss

        return pinball_loss_ans

    def cal_winker_score(self, alpha_val: Union[int, float]) -> ndarray:
        # ref: https://otexts.com/fpp3/distaccuracy.html
        # The elements of model_output should be Callable instance, which is the predicted CDF (given x, should
        # return its corresponding CDF value)
        assert isinstance(self.model_output[0], Callable)

        def cal_winker_score_for_single_sample(actual_val, predicted_cdf):
            nonlocal alpha_val
            cdf_lower_alpha, cdf_upper_alpha = predicted_cdf([0.5 * alpha_val, 1. - 0.5 * alpha_val])
            assert cdf_upper_alpha >= cdf_lower_alpha
            interval_length = cdf_upper_alpha - cdf_lower_alpha

            if actual_val < cdf_lower_alpha:
                return interval_length + 2 / alpha_val * (cdf_lower_alpha - actual_val)
            elif actual_val <= cdf_upper_alpha:
                return interval_length
            else:
                return interval_length + 2 / alpha_val * (actual_val - cdf_upper_alpha)

        winker_score_ans = np.full(self.target.__len__(), np.nan)
        for i in range(len(winker_score_ans)):
            if self.model_output[i] is not None:
                now_winker_score = cal_winker_score_for_single_sample(self.target[i], self.model_output[i])
                winker_score_ans[i] = now_winker_score

        return winker_score_ans


class ProbabilisticErrorIETPaperMethod(ProbabilisticError):

    def do_calculation(self):
        def cal_one_epsilon(ref, model):
            return {'rmse': DeterministicError(target=ref, model_output=model).cal_root_mean_square_error(),
                    'mae': DeterministicError(target=ref, model_output=model).cal_mean_absolute_error()}

        epsilon_rmse = np.mean(
            [cal_one_epsilon(self.target[:, 0], self.model_output[:, 0])['rmse'],
             cal_one_epsilon(self.target[:, 1], self.model_output[:, 1])['rmse']]
        )

        epsilon_mae = np.mean(
            [cal_one_epsilon(self.target[:, 0], self.model_output[:, 0])['mae'],
             cal_one_epsilon(self.target[:, 1], self.model_output[:, 1])['mae']]
        )

        # 移除有可能出现的inf
        delta_u_all = ((self.model_output[:, 1] - self.model_output[:, 0]) / (self.target[:, 1] - self.target[:, 0]))
        delta_u_all[np.isinf(delta_u_all)] = np.nan
        delta_u = np.nanmean(delta_u_all - 1)

        return {'epsilon_rmse': epsilon_rmse,
                'epsilon_mae': epsilon_mae,
                'delta_u': delta_u}
