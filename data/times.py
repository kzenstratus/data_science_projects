import sqlite3
import datetime
import pandas as pd
conn = sqlite3.connect('database.sqlite')
c = conn.cursor()

a = c.execute("SELECT ExtractedDateSent From Emails")
l = a.fetchall()

def to_date(str_date):
	datetime.datetime.strptime(str_date, "%d%m%Y").date()

print type(l[0][0])