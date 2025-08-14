# קובץ: save_data.py
import pandas as pd
import sqlite3

df = pd.read_excel(r"C:\Users\PC\Desktop\פרויקט גמר\פרויקט\processed_data_after_modeling.xlsx")
conn = sqlite3.connect("employees.db")
df.to_sql("employees", conn, if_exists="replace", index=False)
conn.close()



