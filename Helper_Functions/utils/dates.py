from datetime import datetime, timedelta
from collections import OrderedDict

date_format = '%Y-%m-%d'
date_format_full = '%Y-%m-%d %H:%M:%S'

def dt_str(date):
    """Convert Python datetime.datetime object to string YYYY-MM-DD."""
    return date.strftime(date_format)


yesterday_dt = dt_str(datetime.now() - timedelta(days=1))


def shift_date(date, add_sub, shift_by):
    if add_sub:
        return (datetime.strptime(date, date_format) + timedelta(shift_by)).strftime(date_format)
    else:
        return (datetime.strptime(date, date_format) - timedelta(shift_by)).strftime(date_format)


def avg_date(date_metric, date=None):
    if date_metric == 'WEEKLY AVG':
        date_obj = {
        'dt_start': shift_date(date if date else yesterday_dt, False, 7),
        'dt_end': date if date else yesterday_dt
        }
        return date_obj
    elif date_metric == 'WEEK AGO':
        date_obj = {
            'dt_start': shift_date(date if date else yesterday_dt, False, 7),
            'dt_end': shift_date(date if date else yesterday_dt, False, 7)
        }
        return date_obj
    elif date_metric == 'TWENTY EIGHT':
        date_obj = {
            'dt_start': shift_date(date if date else yesterday_dt, False, 28),
            'dt_end': shift_date(date if date else yesterday_dt, False, 28)
        }
        return date_obj
    else:
        date_obj = {
            'dt_start': shift_date(date if date else yesterday_dt, False, 28),
            'dt_end': date if date else yesterday_dt
        }
        return date_obj

def adj_date(dates):
    new_dates = OrderedDict()
    order = range(len(dates))
    for period in order:
        new_dates[str(period)] = {}
        new_dates[str(period)]['dt_start'] = shift_date(dates[period]['dt_start'], False, 7)
        new_dates[str(period)]['dt_end'] = shift_date(dates[period]['dt_end'], False, 7)
    return new_dates.values()
