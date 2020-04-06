import datetime
import time
import numpy as np
from typing import Tuple, Iterable
from dateutil import tz as dateutil_tz


def np_datetime64_to_datetime(date_and_time: np.datetime64, tz=None) -> datetime.datetime:
    timestamp = ((date_and_time - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    utc_datetime = datetime.datetime.utcfromtimestamp(timestamp).replace(tzinfo=dateutil_tz.gettz('UTC'))
    return utc_datetime.astimezone(tz=tz)


def datetime64_ndarray_to_datetime_tuple(date_and_time: Iterable[np.datetime64], tz=None) -> Tuple[datetime.datetime]:
    return tuple([np_datetime64_to_datetime(this_date_and_time, tz) for this_date_and_time in date_and_time])


def find_nearest_datetime_idx_in_datetime_iterable(datetime_iterable: Iterable[datetime.datetime],
                                                   datetime_to_find: datetime.datetime) -> int:
    datetime_to_find = time.mktime(datetime_to_find.timetuple())
    date_time_delta = np.array([(time.mktime(x.timetuple()) - datetime_to_find) for x in datetime_iterable])
    return int(np.argmin(np.abs(date_time_delta)))


def datetime_to_mktime(datetime_: datetime.datetime):
    return time.mktime(datetime_.timetuple())
