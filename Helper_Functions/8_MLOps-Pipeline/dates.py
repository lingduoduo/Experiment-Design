from datetime import datetime, timedelta
from collections import OrderedDict

date_format = '%Y-%m-%d'
date_format_full = '%Y-%m-%d %H:%M:%S'


def correct_week_year(week, month, year):
    if (week > 50) and month == 1:
        year -= 1
    return year


def dt_str(date):
    """Convert Python datetime.datetime object to string YYYY-MM-DD."""
    return date.strftime(date_format)


def shift_date(date, add_sub, shift_by):
    if add_sub:
        return (datetime.strptime(date, date_format) + timedelta(shift_by)).strftime(date_format)
    else:
        return (datetime.strptime(date, date_format) - timedelta(shift_by)).strftime(date_format)
