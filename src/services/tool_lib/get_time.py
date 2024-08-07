def get_time():
	from datetime import datetime
	# return datetime.now().strftime('Year: %Y\nMonth: %m\nDay: %d\nHour: %H \nMinute: %M\nSecond: %S')
	return datetime.now().strftime('%Y-%m-%d %H:%M')

