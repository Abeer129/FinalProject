import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import os
import webbrowser
import numpy as np
import json
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import f_oneway
import plotly.io as pio
import plotly.figure_factory as ff


# --- טען נתונים מהמסד
def load_data_from_db(path="employees.db"):
    conn = sqlite3.connect(path)
    df = pd.read_sql_query("SELECT * FROM employees", conn)
    conn.close()
    return df

# --- קידוד עמודות טקסט
def encode_categorical(df):
    label_encoders = {}
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

# --- אימון מודל, הערכה וחיזוי עזיבה
def train_and_predict(df, label_encoders):
    meta_cols = ['EmpID', 'FirstName', 'LastName']
    X = df.drop(columns=['Is_Exited'] + meta_cols)
    y = df['Is_Exited']

    # חלוקה לאימון ובדיקה
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # הדפסת תוצאות הערכה
    print("=== Accuracy:", accuracy_score(y_test, y_pred))
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # חיזוי עזיבה לעובדים פעילים בלבד
    df_active = df[df['Is_Exited'] == 0]
    X_predict = df_active.drop(columns=['Is_Exited'] + meta_cols)
    exit_probabilities = model.predict_proba(X_predict)[:, 1]

    df_result = df_active[meta_cols].copy()
    df_result['Predicted_Exit_Probability'] = exit_probabilities

    # סיווג רמת סיכון
    def classify_risk(prob):
        prob = round(prob, 2)
        if prob >= 0.75:
            return 'High'
        elif prob >= 0.55:
            return 'Medium'
        else:
            return 'Low'

    df_result['Risk_Level'] = df_result['Predicted_Exit_Probability'].apply(classify_risk)

    # שחזור שמות
    df_result['FirstName'] = label_encoders['FirstName'].inverse_transform(df_result['FirstName'])
    df_result['LastName'] = label_encoders['LastName'].inverse_transform(df_result['LastName'])

    return df_result.sort_values(by='Predicted_Exit_Probability', ascending=False)


# -- פונקציה 4: גרף עוגה
def create_pie_chart(df_result):
    risk_counts = df_result['Risk_Level'].value_counts().reindex(['High', 'Medium', 'Low'])
    fig = px.pie(
        names=risk_counts.index,
        values=risk_counts.values,
        color=risk_counts.index,
        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'lightgreen'}
    )

    # העברת המקרא לשמאל המסך
    fig.update_layout(
        legend=dict(
            x=-0.1,           # שמאלה לגמרי
            xanchor="left",
            y=0.5,
            yanchor="middle"
        )
    )

    path = "risk_pie_chart.html"
    fig.write_html(path)
    return open(path, encoding="utf-8").read()

###################################################################################
################################################################################
#rating_risk_chart/1
# -- פונקציה 5: גרף לפי דירוג עובד נוכחי
def create_rating_risk_chart(df_result, df_original, label_encoders):
    if 'Current Employee Rating' not in df_original.columns:
        print("העמודה 'Current Employee Rating' לא נמצאה במסד הנתונים.")
        return ""

    df_joined = df_result.merge(
        df_original[['EmpID', 'Current Employee Rating']],
        on='EmpID',
        how='left'
    )

    if 'Current Employee Rating' in label_encoders:
        le = label_encoders['Current Employee Rating']
        mapping = dict(zip(range(len(le.classes_)), le.classes_))
        df_joined['Current Employee Rating'] = df_joined['Current Employee Rating'].map(lambda x: mapping.get(x, str(x)))

    grouped = df_joined.groupby(['Current Employee Rating', 'Risk_Level']).size().reset_index(name='Count')

    color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'lightgreen'}
    fig = px.bar(
        grouped,
        x='Current Employee Rating',
        y='Count',
        color='Risk_Level',
        barmode='group',
        text='Count',  # ✅ הוספת טקסט לעמודות
        labels={
            'Current Employee Rating': 'Current Employee Rating',
            'Count': 'Number of Employees',
            'Risk_Level': 'Risk Level'
        },
        color_discrete_map=color_map
    )

    # ✅ הצגת הטקסט מעל העמודות
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', height=700)

    # --- יצירת קובץ Excel
    high_risk = df_joined[df_joined['Risk_Level'] == 'High']
    high_risk_export = high_risk[['EmpID', 'FirstName', 'LastName', 'Current Employee Rating', 'Predicted_Exit_Probability']]
    excel_path = "high_risk_by_rating.xlsx"
    high_risk_export.to_excel(excel_path, index=False)

    # --- יצירת קובץ HTML + כפתור
    output_file = "risk_by_rating.html"
    fig.write_html("temp_graph.html")

    with open("temp_graph.html", encoding="utf-8") as f:
        graph_html = f.read()

    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Risk by Rating</title>
    </head>
    <body>
        <h1>Risk by Current Employee Rating</h1>

        <!-- כפתור להורדת קובץ Excel -->
        <a href="{excel_path}" download style="display: inline-block; margin-bottom: 20px; background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
             Download High Risk Excel
        </a>

        <!-- גרף -->
        {graph_html}
    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_html)

    return full_html

########################
###risk_by_department_chart/2
def create_risk_by_department_chart(df_results, df_original, label_encoders):
    # איחוד נתונים
    df_joined = df_results.merge(df_original[['EmpID', 'DepartmentType']], on='EmpID', how='left')
    df_joined['DepartmentType'] = df_joined['DepartmentType'].astype(str).str.strip()

    # שחזור קידודים אם היה
    if 'DepartmentType' in label_encoders:
        le = label_encoders['DepartmentType']
        mapping = dict(zip(range(len(le.classes_)), le.classes_))
        df_joined['DepartmentType'] = df_joined['DepartmentType'].map(lambda x: mapping.get(x, str(x)))

    # קיבוץ לגרף
    grouped = df_joined.groupby(['DepartmentType', 'Risk_Level']).size().reset_index(name='Count')

    # צבעים
    color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'lightgreen'}
    fig = px.bar(
        grouped,
        x='DepartmentType',
        y='Count',
        color='Risk_Level',
        barmode='group',
        text='Count',  # הוספת טקסט על העמודות
        labels={'DepartmentType': 'Department', 'Count': 'Number of Employees', 'Risk_Level': 'Risk Level'},
        color_discrete_map=color_map
    )

    fig.update_layout(height=700)  # במקום ברירת מחדל

    # הצגת הטקסט אוטומטית
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', height=700)

    # יצירת קובץ Excel לסיכון גבוה
    high_risk = df_joined[df_joined['Risk_Level'] == 'High']
    high_risk_export = high_risk[['EmpID', 'FirstName', 'LastName', 'DepartmentType', 'Predicted_Exit_Probability']]
    excel_path = "high_risk_by_department.xlsx"
    high_risk_export.to_excel(excel_path, index=False)

    # שמירה זמנית של הגרף
    fig.write_html("temp_dept_graph.html")

    with open("temp_dept_graph.html", encoding="utf-8") as f:
        graph_html = f.read()

    # יצירת דף HTML עם כפתור
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Risk by Department</title>
    </head>
    <body>
        <h1>Risk by Department</h1>

        <!-- כפתור להורדת קובץ Excel -->
        <a href="{excel_path}" download style="display: inline-block; margin-bottom: 20px; background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
             Download High Risk Excel
        </a>

        <!-- גרף -->
        {graph_html}
    </body>
    </html>
    """

    # שמירה לקובץ סופי
    output_file = "risk_by_dept_and_level.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_html)

    return full_html

########################
##Performance Score/3

def create_performance_risk_chart(df_result, df_original, label_encoders):
    df_joined = df_result.merge(
        df_original[['EmpID', 'Performance Score']],
        on='EmpID',
        how='left'
    )

    df_joined['Performance Score'] = df_joined['Performance Score'].astype(str).str.strip()

    if 'Performance Score' in label_encoders:
        le = label_encoders['Performance Score']
        mapping = dict(zip(range(len(le.classes_)), le.classes_))
        df_joined['Performance Score'] = df_joined['Performance Score'].map(lambda x: mapping.get(x, str(x)))

    grouped = df_joined.groupby(['Performance Score', 'Risk_Level']).size().reset_index(name='Count')

    color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'lightgreen'}
    risk_levels = ['High', 'Medium', 'Low']

    fig = go.Figure()
    fig.update_layout(height=700)  # במקום ברירת מחדל


    for risk in risk_levels:
        sub_df = grouped[grouped['Risk_Level'] == risk]
        fig.add_trace(go.Bar(
            x=sub_df['Performance Score'],
            y=sub_df['Count'],
            name=risk,
            marker_color=color_map[risk],
            text=sub_df['Count'],
            textposition='outside'  # או 'auto'
        ))

    fig.update_layout(
        barmode='group',
        xaxis_title="Performance Score",
        yaxis_title="Number of Employees",
        legend_title="Risk Level"

    )

    # --- ייצוא לקובץ Excel לעובדים בסיכון גבוה
    high_risk = df_joined[df_joined['Risk_Level'] == 'High']
    high_risk_export = high_risk[['EmpID', 'FirstName', 'LastName', 'Performance Score', 'Predicted_Exit_Probability']]
    excel_path = "high_risk_by_performance.xlsx"
    high_risk_export.to_excel(excel_path, index=False)

    # --- שמירה כ-HTML כולל כפתור
    fig.write_html("temp_perf_graph.html")
    with open("temp_perf_graph.html", encoding="utf-8") as f:
        graph_html = f.read()

    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Risk by Performance Score</title>
    </head>
    <body>
        <h1>Risk by Performance Score</h1>

        <a href="{excel_path}" download style="display: inline-block; margin-bottom: 20px; background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
             Download High Risk Excel
        </a>

        {graph_html}
    </body>
    </html>
    """

    output_file = "risk_by_performance_score.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_html)

    return full_html

###############
##marital_risk_chart/4
def create_marital_risk_chart(df_result, df_original, label_encoders):
    # צירוף עמודת MaritalDesc
    df_joined = df_result.merge(df_original[['EmpID', 'MaritalDesc']], on='EmpID', how='left')

    # ניקוי רווחים מיותרים
    df_joined['MaritalDesc'] = df_joined['MaritalDesc'].astype(str).str.strip()

    # שחזור שמות מקוריים אם קודדו
    if 'MaritalDesc' in label_encoders:
        le = label_encoders['MaritalDesc']
        mapping = dict(zip(range(len(le.classes_)), le.classes_))
        df_joined['MaritalDesc'] = df_joined['MaritalDesc'].map(lambda x: mapping.get(x, str(x)))

    # קיבוץ נתונים
    grouped = df_joined.groupby(['MaritalDesc', 'Risk_Level']).size().reset_index(name='Count')

    # יצירת גרף היסטוגרמה עם מספרים
    fig = px.bar(
        grouped,
        x='MaritalDesc',
        y='Count',
        color='Risk_Level',
        barmode='group',
        text='Count',  # ✅ הצגת מספרים
        labels={'MaritalDesc': 'Marital Status', 'Count': 'Number of Employees', 'Risk_Level': 'Risk Level'},
        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'lightgreen'}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', height=700)

    # --- יצירת קובץ Excel לעובדים עם סיכון גבוה
    high_risk = df_joined[df_joined['Risk_Level'] == 'High']
    excel_path = "high_risk_by_marital.xlsx"
    if not high_risk.empty:
        export_cols = ['EmpID', 'FirstName', 'LastName', 'MaritalDesc', 'Predicted_Exit_Probability']
        available_cols = [col for col in export_cols if col in high_risk.columns]
        high_risk[available_cols].to_excel(excel_path, index=False)
    else:
        with open(excel_path, 'w') as f:
            f.write("No high risk employees found for marital status.")

    # --- שמירה זמנית של הגרף
    fig.write_html("temp_marital_graph.html")
    with open("temp_marital_graph.html", encoding="utf-8") as f:
        graph_html = f.read()

    # --- יצירת דף HTML עם כפתור להורדת Excel
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Risk by Marital Status</title>
    </head>
    <body>
        <h1>Risk by Marital Status</h1>

        <!-- כפתור להורדת קובץ Excel -->
        <a href="{excel_path}" download style="display: inline-block; margin-bottom: 20px; background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
             Download High Risk Excel
        </a>

        <!-- גרף -->
        {graph_html}
    </body>
    </html>
    """

    # --- שמירה לקובץ סופי
    output_file = "risk_by_marital.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_html)

    return full_html


###############from plotly.subplots import make_subplots
#######gendercode/5

def create_gendercode_pie_chart(df_result, df_original, label_encoders):
    # מיזוג GenderCode עם התוצאות
    df_joined = df_result.merge(
        df_original[['EmpID', 'GenderCode']],
        on='EmpID',
        how='left'
    )

    # ניקוי רווחים
    df_joined['GenderCode'] = df_joined['GenderCode'].astype(str).str.strip()

    # שחזור ערכים מקוריים אם עברו קידוד
    if 'GenderCode' in label_encoders:
        le = label_encoders['GenderCode']
        mapping = dict(zip(range(len(le.classes_)), le.classes_))
        df_joined['GenderCode'] = df_joined['GenderCode'].map(lambda x: mapping.get(x, str(x)))

    # רשימת קטגוריות
    categories = sorted(df_joined['GenderCode'].dropna().unique())
    n = len(categories)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    # הגדרת סאבפלטים
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{'type': 'domain'} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=categories
    )

    # יצירת גרף עוגה לכל קטגוריה
    for i, category in enumerate(categories):
        row = i // cols + 1
        col = i % cols + 1
        sub_df = df_joined[df_joined['GenderCode'] == category]
        counts = sub_df['Risk_Level'].value_counts().reindex(['High', 'Medium', 'Low'], fill_value=0)

        fig.add_trace(
            go.Pie(
               
                labels=counts.index,
                values=counts.values,
                name=str(category),
                marker=dict(colors=['red', 'orange', 'lightgreen']),
                textinfo='percent'  # <--- כאן נשאר רק אחוזים (בלי שמות הקטגוריות)
            ),
            row=row,
            col=col
        )

    # עיצוב גרף
    fig.update_layout(
        showlegend=True,
        height=rows * 400
    )

    # שמירת גרף זמני
    fig.write_html("temp_gender_graph.html")
    with open("temp_gender_graph.html", encoding="utf-8") as f:
        graph_html = f.read()

    # יצירת Excel של עובדים בסיכון גבוה
    high_risk = df_joined[df_joined['Risk_Level'] == 'High']
    excel_path = "high_risk_by_gender.xlsx"
    export_cols = ['EmpID', 'FirstName', 'LastName', 'GenderCode', 'Predicted_Exit_Probability']
    high_risk[export_cols].to_excel(excel_path, index=False)

    # דף HTML סופי
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Risk by Gender</title>
    </head>
    <body>
        <h1>Risk by Gender</h1>

        <!-- כפתור להורדת קובץ Excel -->
        <a href="{excel_path}" download style="display: inline-block; margin-bottom: 20px;
           background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
             Download High Risk Excel
        </a>

        <!-- גרף -->
        {graph_html}
    </body>
    </html>
    """

    # שמירה לקובץ סופי
    output_file = "risk_by_gendercode.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_html)

    return full_html


###########################
###division_risk_chart/6
def create_division_risk_chart(df_result, df_original, label_encoders):
    # מיזוג נתוני חטיבה
    df_joined = df_result.merge(df_original[['EmpID', 'Division']], on='EmpID', how='left')
    df_joined['Division'] = df_joined['Division'].astype(str).str.strip()

    # שחזור שמות מקוריים אם קודדו
    if 'Division' in label_encoders:
        le = label_encoders['Division']
        mapping = dict(zip(range(len(le.classes_)), le.classes_))
        df_joined['Division'] = df_joined['Division'].map(lambda x: mapping.get(x, str(x)))

    # קיבוץ נתונים
    grouped = df_joined.groupby(['Division', 'Risk_Level']).size().reset_index(name='Count')

    # גרף עם מספרים
    color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'lightgreen'}
    fig = px.bar(
        grouped,
        x='Division',
        y='Count',
        color='Risk_Level',
        barmode='group',
        labels={'Division': 'Division', 'Count': 'Number of Employees', 'Risk_Level': 'Risk Level'},
        color_discrete_map=color_map,
        text='Count'  # הוספת מספרים על העמודות
    )

    fig.update_layout(height=700)  # במקום ברירת מחדל
    fig.update_traces(textposition='outside')  # מיקום המספרים

    # יצירת קובץ Excel לעובדים בסיכון גבוה
    if 'Predicted_Exit_Probability' in df_joined.columns:
        high_risk = df_joined[df_joined['Risk_Level'] == 'High']
        high_risk_export = high_risk[['EmpID', 'FirstName', 'LastName', 'Division', 'Predicted_Exit_Probability']]
        excel_path = "high_risk_by_division.xlsx"
        high_risk_export.to_excel(excel_path, index=False)
    else:
        excel_path = None

    # שמירה זמנית של הגרף
    fig.write_html("temp_division_graph.html")
    with open("temp_division_graph.html", encoding="utf-8") as f:
        graph_html = f.read()

    # יצירת דף HTML עם כפתור
    excel_button_html = f"""
    <a href="{excel_path}" download style="display: inline-block; margin-bottom: 20px; background-color: #4CAF50;
    color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
        Download High Risk Excel
    </a>
    """ if excel_path else ""

    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Risk by Division</title>
    </head>
    <body>
        <h1>Risk by Division</h1>
        {excel_button_html}
        {graph_html}
    </body>
    </html>
    """

    output_file = "risk_by_division.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_html)

    return full_html
##########################
####tenure_risk_chart/7
def create_tenure_risk_chart(df_result, df_original):
    # טווחים לותק
    bins = [0, 2, 4, 6, 8, float('inf')]
    labels = ['0-2', '2-4', '4-6', '6-8', '+8']

    # צירוף ותק
    df_joined = df_result.merge(df_original[['EmpID', 'Tenure']], on='EmpID', how='left')

    # חישוב טווחי ותק
    df_joined['TenureRange'] = pd.cut(df_joined['Tenure'], bins=bins, labels=labels, right=False)

    # קיבוץ לפי טווח ותק ורמת סיכון
    grouped = df_joined.groupby(['TenureRange', 'Risk_Level']).size().reset_index(name='Count')

    # גרף עם מספרים
    fig = px.bar(
        grouped,
        x='TenureRange',
        y='Count',
        color='Risk_Level',
        barmode='group',
        labels={'TenureRange': 'Tenure Range (Years)', 'Count': 'Number of Employees', 'Risk_Level': 'Risk Level'},
        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'lightgreen'},
        text='Count'
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(height=700)  # במקום ברירת מחדל


    # יצירת קובץ אקסל לעובדים בסיכון גבוה
    if 'Predicted_Exit_Probability' in df_joined.columns:
        high_risk = df_joined[df_joined['Risk_Level'] == 'High']
        excel_path = "high_risk_by_tenure.xlsx"
        high_risk_export = high_risk[['EmpID', 'FirstName', 'LastName', 'Tenure', 'Predicted_Exit_Probability']]
        high_risk_export.to_excel(excel_path, index=False)
    else:
        excel_path = None

    # שמירה זמנית
    fig.write_html("temp_tenure_graph.html")
    with open("temp_tenure_graph.html", encoding="utf-8") as f:
        graph_html = f.read()

    # יצירת כפתור להורדת אקסל
    excel_button_html = f"""
    <a href="{excel_path}" download style="display: inline-block; margin-bottom: 20px; background-color: #4CAF50;
    color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
        Download High Risk Excel
    </a>
    """ if excel_path else ""

    # דף HTML מלא
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Risk by Tenure</title>
    </head>
    <body>
        <h1>Risk by Tenure</h1>
        {excel_button_html}
        {graph_html}
    </body>
    </html>
    """

    output_file = "tenure_risk_chart.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_html)

    return full_html
#########################
####ge_risk_chart/8

def create_age_risk_chart(df_result, df_original):
    # איחוד עמודת גיל
    df_joined = df_result.merge(df_original[['EmpID', 'Age']], on='EmpID', how='left')

    # יצירת טווחי גיל
    bins = [18, 25, 35, 45, 55, 65, float('inf')]
    labels = ['18-25', '25-35', '35-45', '45-55', '55-65', '+65']
    df_joined['AgeRange'] = pd.cut(df_joined['Age'], bins=bins, labels=labels, right=False)

    # קיבוץ לפי טווח גיל ורמת סיכון
    grouped = df_joined.groupby(['AgeRange', 'Risk_Level']).size().reset_index(name='Count')

    # גרף עם מספרים
    fig = px.bar(
        grouped,
        x='AgeRange',
        y='Count',
        color='Risk_Level',
        barmode='group',
        labels={'AgeRange': 'Age Range', 'Count': 'Number of Employees', 'Risk_Level': 'Risk Level'},
        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'lightgreen'},
        text='Count'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(height=700)  # במקום ברירת מחדל


    # יצירת קובץ Excel לעובדים בסיכון גבוה
    if 'Predicted_Exit_Probability' in df_joined.columns:
        high_risk = df_joined[df_joined['Risk_Level'] == 'High']
        excel_path = "high_risk_by_age.xlsx"
        high_risk_export = high_risk[['EmpID', 'FirstName', 'LastName', 'Age', 'Predicted_Exit_Probability']]
        high_risk_export.to_excel(excel_path, index=False)
    else:
        excel_path = None

    # שמירה זמנית של הגרף
    fig.write_html("temp_age_graph.html")
    with open("temp_age_graph.html", encoding="utf-8") as f:
        graph_html = f.read()

    # יצירת כפתור להורדת אקסל
    excel_button_html = f"""
    <a href="{excel_path}" download style="display: inline-block; margin-bottom: 20px; background-color: #4CAF50;
    color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
        Download High Risk Excel
    </a>
    """ if excel_path else ""

    # דף HTML מלא
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Risk by Age Range</title>
    </head>
    <body>
        <h1>Risk by Age Range</h1>
        {excel_button_html}
        {graph_html}
    </body>
    </html>
    """

    output_file = "risk_by_age_range.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_html)

    return full_html


#########################################################################3
# -- פונקציה 6: עיצוב צבעוני
def create_styled_table(df_result, df_original):
    rows_html = ""
    for idx, (_, row) in enumerate(df_result.iterrows(), 1):
        emp_id = row['EmpID']
        first = row['FirstName']
        last = row['LastName']
        prob = row['Predicted_Exit_Probability']
        risk = row['Risk_Level']

        full_info = df_original[df_original['EmpID'] == emp_id].to_dict('records')[0]

        bg_color = 'lightgreen' if risk == 'Low' else 'orange' if risk == 'Medium' else 'red'
        text_color = 'black' if risk != 'High' else 'white'

        rows_html += f"""
        <tr style="background-color:{bg_color}; color:{text_color};">
            <td>{risk}</td>
            <td>{prob:.2f}</td>
            <td>{last}</td>
            <td>{first}</td>
            <td><a href="#" onclick='showDetails("{emp_id}")'>{emp_id}</a></td>
            <td>{idx}</td>
        </tr>
        """

    table_html = f"""
    <table>
        <thead>
            <tr>
                <th>Risk Level</th>
                <th>Exit Probability</th>
                <th>Last Name</th>
                <th>First Name</th>
                <th>EmpID</th>
                <th> </th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """
    df_result.to_excel("high_risk_table.xlsx", index=False)

    return table_html


######################################################################################

def create_correlation_heatmap(df_encoded, output_path="correlation_heatmap.html"):
    # עמודות מזהות שלא רלוונטיות
    drop_cols = ['EmpID', 'FirstName', 'LastName']
    df_filtered = df_encoded.drop(columns=[col for col in drop_cols if col in df_encoded.columns])

    # קורולציה רק בין עמודות נומריות
    numeric_df = df_filtered.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numeric_df.corr().round(2)

    # הפיכת המטריצה למערך תצוגה
    z = corr_matrix.values
    x = list(corr_matrix.columns)
    y = list(corr_matrix.index)

    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='RdBu', showscale=True, zmin=-1, zmax=1)
    fig.update_layout(
        title="Correlation Heatmap of Columns",
        width=900,
        height=500
    )

    # שמירה
    pio.write_html(fig, file=output_path, auto_open=False)
    with open(output_path, encoding="utf-8") as f:
        return f.read()



#########################################################################################

from scipy.stats import chi2_contingency
def create_chi_square_heatmap_html(df_encoded, output_html="chi_square_heatmap.html"):
    results = []

    for col in df_encoded.columns:
        if col == 'Is_Exited':
            continue
        if 1 < df_encoded[col].nunique() < 20:
            try:
                contingency = pd.crosstab(df_encoded[col], df_encoded['Is_Exited'])
                chi2, p, dof, expected = chi2_contingency(contingency)
                results.append({
                    'Feature': col,
                    'Chi²': round(chi2, 2),
                    'p-value': round(p, 5)
                })
            except:
                continue

    df_results = pd.DataFrame(results).sort_values(by='Chi²', ascending=False)

    # ערכי z ו־annotations בפורמט תואם ל־heatmap
    z = [[val] for val in df_results['Chi²']]
    annotations = [[f"{val}<br>p={p}"] for val, p in zip(df_results['Chi²'], df_results['p-value'])]

   

    fig = ff.create_annotated_heatmap(
        z=z,
        x=["Chi² (p-value)"],
        y=df_results['Feature'].tolist(),
        annotation_text=annotations,
        colorscale='Blues',
        showscale=True,
        zmin=0
    )

    fig.update_layout(
        title="Chi-Square Test Heatmap by Features",
        width=800,
        height=500 + len(z) * 25
    )

    pio.write_html(fig, file=output_html, auto_open=False)
    return output_html

############################################################################################

def anova_test_plot_html(df, category_col='DepartmentType', value_col='Age', output_html='anova_test_plot.html'):
    if category_col not in df.columns or value_col not in df.columns:
        return f"Missing required columns: {category_col}, {value_col}"

    # סינון רק לעמודות הרצויות
    df_filtered = df[[category_col, value_col]].dropna()

    # קיבוץ הערכים לפי קטגוריה
    grouped_values = [group[value_col].values for name, group in df_filtered.groupby(category_col)]

    # הרצת מבחן ANOVA
    anova_result = f_oneway(*grouped_values)

    # יצירת BoxPlot עם Plotly
    fig = go.Figure()
    for name, group in df_filtered.groupby(category_col):
        fig.add_trace(go.Box(y=group[value_col], name=str(name)))

    fig.update_layout(
        title=f"ANOVA Test by {category_col} (F={anova_result.statistic:.2f}, p={anova_result.pvalue:.5f})",
        yaxis_title=value_col,
        width=900,
        height=500
    )

    # שמירה לקובץ HTML
    pio.write_html(fig, file=output_html, auto_open=False)
    return os.path.abspath(output_html)

##############################################################################

def anova_tenure_vs_performance_html(df, category_col='Performance Score', value_col='Tenure', output_html="anova_tenure_vs_performance.html"):
    if category_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Missing required columns: {category_col} or {value_col}")

    # סינון נתונים חסרים
    df_filtered = df[[category_col, value_col]].dropna()
    grouped_values = [group[value_col].values for _, group in df_filtered.groupby(category_col)]

    # הרצת ANOVA
    anova_result = f_oneway(*grouped_values)
    title = f"ANOVA Test by {category_col} (F={anova_result.statistic:.2f}, p={anova_result.pvalue:.5f})"

    # יצירת Box Plot
    fig = go.Figure()
    for name, group in df_filtered.groupby(category_col):
        fig.add_trace(go.Box(
            y=group[value_col],
            name=str(name),
            boxmean='sd'
        ))

    fig.update_layout(
        title=title,
        xaxis_title=category_col,
        yaxis_title=value_col,
        width=900,
        height=500
    )

    # שמירה ל־HTML
    pio.write_html(fig, file=output_html, auto_open=False)
    return os.path.abspath(output_html)

#################################################################################

# פונקציה בין Age ל־Performance Score
def anova_age_vs_performance_html(df, category_col='Performance Score', value_col='Age', output_html="anova_age_vs_performance.html"):
    if category_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Missing required columns: {category_col} or {value_col}")

    df_filtered = df[[category_col, value_col]].dropna()
    grouped_values = [group[value_col].values for _, group in df_filtered.groupby(category_col)]

    # מבחן ANOVA
    anova_result = f_oneway(*grouped_values)
    title = f"ANOVA Test by {category_col} (F={anova_result.statistic:.2f}, p={anova_result.pvalue:.5f})"

    fig = go.Figure()
    for name, group in df_filtered.groupby(category_col):
        fig.add_trace(go.Box(
            y=group[value_col],
            name=str(name),
            boxmean='sd'
        ))

    fig.update_layout(
        title=title,
        xaxis_title=category_col,
        yaxis_title=value_col,
        width=900,
        height=500
    )

    # שמירה לקובץ HTML
    pio.write_html(fig, file=output_html, auto_open=False)
    return os.path.abspath(output_html)

##################################################################################

def anova_tenure_vs_employee_type_html(df, category_col='EmployeeType', value_col='Tenure', output_html="anova_tenure_vs_employee_type.html"):
    if category_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Missing required columns: {category_col} or {value_col}")

    # הסרת ערכים חסרים
    df_filtered = df[[category_col, value_col]].dropna()
    grouped_values = [group[value_col].values for _, group in df_filtered.groupby(category_col)]

    # הרצת ANOVA
    anova_result = f_oneway(*grouped_values)
    title = f"ANOVA Test by {category_col} (F={anova_result.statistic:.2f}, p={anova_result.pvalue:.5f})"

    # יצירת Box Plot
    fig = go.Figure()
    for name, group in df_filtered.groupby(category_col):
        fig.add_trace(go.Box(
            y=group[value_col],
            name=str(name),
            boxmean='sd'
        ))

    fig.update_layout(
        title=title,
        xaxis_title=category_col,
        yaxis_title=value_col,
        width=900,
        height=500
    )

    # שמירה לקובץ HTML
    pio.write_html(fig, file=output_html, auto_open=False)
    return os.path.abspath(output_html)
##################################################################################

def anova_tenure_vs_jobfunction_html(df, category_col='JobFunctionDescription', value_col='Tenure', output_html="anova_tenure_vs_jobfunction.html"):
    if category_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Missing required columns: {category_col} or {value_col}")

    # הסרת ערכים חסרים
    df_filtered = df[[category_col, value_col]].dropna()
    grouped_values = [group[value_col].values for _, group in df_filtered.groupby(category_col)]

    # הרצת מבחן ANOVA
    anova_result = f_oneway(*grouped_values)
    title = f"ANOVA Test by {category_col} (F={anova_result.statistic:.2f}, p={anova_result.pvalue:.5f})"

    # יצירת גרף Box Plot
    fig = go.Figure()
    for name, group in df_filtered.groupby(category_col):
        fig.add_trace(go.Box(
            y=group[value_col],
            name=str(name),
            boxmean='sd'
        ))

    fig.update_layout(
        title=title,
        xaxis_title=category_col,
        yaxis_title=value_col,
        width=1100,
        height=500
    )

    # שמירה ל־HTML
    pio.write_html(fig, file=output_html, auto_open=False)
    return os.path.abspath(output_html)

####################################################################################


def anova_tenure_vs_payzone_html(df, category_col='PayZone', value_col='Tenure', output_html="anova_tenure_vs_payzone.html"):
    if category_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Missing required columns: {category_col} or {value_col}")

    # סינון נתונים חסרים
    df_filtered = df[[category_col, value_col]].dropna()
    grouped_values = [group[value_col].values for _, group in df_filtered.groupby(category_col)]

    # הרצת מבחן ANOVA
    anova_result = f_oneway(*grouped_values)
    title = f"ANOVA Test by {category_col} (F={anova_result.statistic:.2f}, p={anova_result.pvalue:.5f})"

    # יצירת גרף Box Plot
    fig = go.Figure()
    for name, group in df_filtered.groupby(category_col):
        fig.add_trace(go.Box(
            y=group[value_col],
            name=str(name),
            boxmean='sd'
        ))

    fig.update_layout(
        title=title,
        xaxis_title=category_col,
        yaxis_title=value_col,
        width=800,
        height=500
    )

    # שמירה לקובץ HTML
    pio.write_html(fig, file=output_html, auto_open=False)
    return os.path.abspath(output_html)

####################################################################################3
def build_employee_json(df_original):
    employee_details = {}
    for _, row in df_original.iterrows():
        emp_id = row["EmpID"]
        details = "<div style='direction: ltr; text-align: left;'>"

        for col, val in row.items():
            if col == "Is_Exited":
                continue  # דלג על התכונה הזו

            if col in ["Age", "Tenure"]:
                val = round(val, 2)  # עיגול לגיל ו־Tenure

            details += f"<strong>{col}:</strong> {val}<br>"

        details += "</div>"
        employee_details[emp_id] = details

    return json.dumps(employee_details, ensure_ascii=False)


# -- פונקציה 7: בניית HTML מתבנית
def build_html_report(template_path, risk_chart, table_html, employee_json):
    with open(template_path, encoding="utf-8") as f:
        template = f.read()
    return template \
        .replace("{{ risk_pie_chart }}", risk_chart) \
        .replace("{{ styled_table }}", table_html) \
        .replace("{{ employee_data_json }}", employee_json)


# -- פונקציה 8: שמירה ופתיחה
def save_and_open_report(html_content, filename="final_risk_report.html"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    #webbrowser.open(filename)



# -- פונקציית # -- פונקציית MAIN
def main():
    df = load_data_from_db()
    df_original = df.copy()
    employee_json = build_employee_json(df_original)

    df_encoded, encoders = encode_categorical(df)
    df_results = train_and_predict(df, encoders)

   
    pie_html = create_pie_chart(df_results)
    table_html = create_styled_table(df_results, df_original)
   
     # גרף קורולציה - חדש
    correlation_html = create_correlation_heatmap(df_encoded)
   
    # הוספת הגרף של דירוג העובדים
    rating_chart_html = create_rating_risk_chart(df_results, df_original, encoders)
    dept_chart_html = create_risk_by_department_chart(df_results, df_original, encoders)
    performance_chart_html = create_performance_risk_chart(df_results, df_original, encoders)
    marital_chart_html = create_marital_risk_chart(df_results, df_original, encoders)
    gendercode_html = create_gendercode_pie_chart(df_results, df_original, encoders)
    division_chart_html = create_division_risk_chart(df_results, df_original, encoders)
    tenure_chart_html = create_tenure_risk_chart(df_results, df_original)
    age_chart_html = create_age_risk_chart(df_results, df_original)
   



    # 👇 אם אתה רוצה לשלב אותו גם בדו"ח HTML – תעדכן גם את template.html בהתאם:
    final_html = build_html_report("template.html", pie_html, table_html, employee_json)
    save_and_open_report(final_html)
   
   
    df = load_data_from_db()
    df_encoded, _ = encode_categorical(df)
    chi_path = create_chi_square_heatmap_html(df_encoded)

    df = load_data_from_db()
    anova_html = anova_test_plot_html(df, category_col='DepartmentType', value_col='Age')
   
    df = load_data_from_db()
    anova_path = anova_tenure_vs_performance_html(df)

    anova_age_vs_performance_html(df)
    html_path = anova_age_vs_performance_html(df)

    html_path = anova_tenure_vs_employee_type_html(df)

    html_path = anova_tenure_vs_jobfunction_html(df)

    html_path = anova_tenure_vs_payzone_html(df)


   
# -- הרצה
if __name__ == "__main__":
    webbrowser.open("home.html")
    main()

  
