import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\osama\Desktop\popul.any\population_data.csv")   
print(df.head())
print(df.info())


df=df.dropna()  # Remove rows with missing values

df=df.drop_duplicates()  # Remove duplicate rows


df.sort_values('Population 2025', ascending=False).head(10)

print(df.head(10))



df['Yearly Change'] = df['Yearly Change'].astype(str)               # نحوله إلى نصوص
df['Yearly Change'] = df['Yearly Change'].str.replace('−', '-', regex=False)  # نحول الناقص الغريبة إلى العادية
df['Yearly Change'] = df['Yearly Change'].str.replace('%', '', regex=False)   # نحذف علامة %
df['Yearly Change'] = df['Yearly Change'].astype(float)             # نحول إلى أرقام

for col in df.columns:
    if df[col].astype(str).str.contains('%').any():
        df[col] = df[col].astype(str).str.replace('−', '-', regex=False)
        df[col] = df[col].str.replace('%', '', regex=False)
        df[col] = df[col].astype(float)






mean_age=df["Median Age"].mean()
median_age=df["Median Age"].median()
print("Mean,median .Median Age:", mean_age, median_age)

mean_pop=df["Population 2025"].mean()
print("Mean Population 2025:", mean_pop)

df.nlargest(5, 'Fert. Rate')[['Country (or dependency)', 'Fert. Rate']]
print("Top 5 countries with highest Fertility Rate and their Population Density:")
plt.subplots(figsize=(10,6))
top5_fertility = df.nlargest(5, "Fert. Rate")[["Country (or dependency)", "Fert. Rate"]]
plt.bar(top5_fertility["Country (or dependency)"], top5_fertility["Fert. Rate"], color='skyblue')
plt.xlabel("Country (or dependency)")
plt.ylabel("Fertility Rate")
plt.title("Top 5 Countries with Highest Fertility Rate")
plt.show()

df.nlargest(5,"Density (P/Km²)")[["Country (or dependency)","Density (P/Km²)"]]
print("Top 5 countries with highest Population Density:")
plt.subplots(figsize=(10,6))    
top5_density = df.nlargest(5, "Density (P/Km²)")[["Country (or dependency)", "Density (P/Km²)"]]
plt.bar(top5_density["Country (or dependency)"], top5_density["Density (P/Km²)"], color='salmon')
plt.xlabel("Country (or dependency)")
plt.ylabel("Population Density (P/Km²)")
plt.title("Top 5 Countries with Highest Population Density")
plt.show()

df.nlargest(10,"Population 2025")[["Country (or dependency)","Population 2025"]]
print("Top 10 countries with highest Population 2025:")
plt.subplots(figsize=(10,6))
top10_population = df.nlargest(10, "Population 2025")[["Country (or dependency)", "Population 2025"]]
plt.bar(top10_population["Country (or dependency)"], top10_population["Population 2025"], color='lightgreen')
plt.xlabel("Country (or dependency)")
plt.ylabel("Population 2025")
plt.title("Top 10 Countries with Highest Population 2025")
plt.show()


df_urban_pop = df[['Country (or dependency)', 'Urban Pop %']].sort_values('Urban Pop %', ascending=False).head(10)
print("Top 10 countries with highest Urban Population Percentage:")
plt.subplots(figsize=(10,6))
top10_urban_pop = df_urban_pop[["Country (or dependency)", "Urban Pop %"]]
plt.bar(top10_urban_pop["Country (or dependency)"], top10_urban_pop["Urban Pop %"], color='orange')
plt.xlabel("Country (or dependency)")
plt.ylabel("Urban Population Percentage")
plt.title("Top 10 Countries with Highest Urban Population Percentage")
plt.show()


sns.scatterplot(x="Median Age", y="Fert. Rate", data=df, color='purple')
plt.title("Median Age vs Fertility Rate")
plt.show()

df.columns = df.columns.str.strip()

df['Yearly Change'] = df['Yearly Change'].astype(str).str.strip()            # إزالة الفراغات
df['Yearly Change'] = df['Yearly Change'].str.replace('−', '-', regex=False) # تصحيح الناقص
df['Yearly Change %'] = df['Yearly Change'].str.replace('%', '').astype(float)

df.sort_values("Yearly Change %",ascending=False)[["Country (or dependency)","Yearly Change %"]].head(10)
print("Top 10 countries with highest Yearly Change Percentage:")
plt.subplots(figsize=(10,6))
top10_yearly_change = df.sort_values("Yearly Change %", ascending=False).head(10)[["Country (or dependency)", "Yearly Change %"]]
plt.bar(top10_yearly_change["Country (or dependency)"], top10_yearly_change["Yearly Change %"], color='teal')
plt.xlabel("Country (or dependency)")
plt.ylabel("Yearly Change Percentage")
plt.title("Top 10 Countries with Highest Yearly Change Percentage")
plt.show()


top_growth=df.sort_values("Yearly Change %",ascending=False).head(10)
sns.barplot(x="Yearly Change %", y="Country (or dependency)", data=top_growth, palette="viridis")
plt.title("Top 10 Countries with Highest Yearly Change Percentage")
plt.show()


corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix") 
plt.show()


df.groupby("Country (or dependency)")["Fert. Rate"].mean()
print("Average Fertility Rate by Country:")
avg_fertility_by_country = df.groupby("Country (or dependency)")["Fert. Rate"].mean()
plt.subplots(figsize=(10,10))
avg_fertility_by_country.plot(kind='bar', color='mediumpurple')
plt.ylabel("Average Fertility Rate")
plt.title("Average Fertility Rate by Country")
plt.show()


sns.pairplot(df[["Median Age", "Fert. Rate", "Density (P/Km²)", "Urban Pop %"]])
plt.show()


sns.barplot(
    x="Yearly Change %",
    y="Country (or dependency)",
    data=top_growth,
    hue="Country (or dependency)",
    palette="viridis",
    legend=False
)
plt.title("Top 10 Countries with Highest Yearly Change Percentage")
plt.show()  




# تنظيف العمود Yearly Change

from sklearn.linear_model import LinearRegression

# تنظيف الأعمدة
df['Yearly Change %'] = df['Yearly Change %'].astype(float)
df['Urban Pop %'] = df['Urban Pop %'].astype(float)

# اختيار المتغيرات
x = df[['Fert. Rate', 'Median Age', 'Urban Pop %', 'Yearly Change %']]
y = df['Population 2025']

# تدريب النموذج
model = LinearRegression()
model.fit(x, y)

print("Model trained successfully!")

plt.figure(figsize=(10,6))
sns.regplot(x='Fert. Rate', y='Population 2025', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Fertility Rate vs Population 2025')
plt.show()




sns.lmplot(x='Median Age', y='Population 2025', data=df, aspect=1.5, scatter_kws={'alpha':0.5}, line_kws={'color':'green'})
plt.title('Median Age vs Population 2025')
plt.show()

df['Predicted Pop'] = model.predict(x)
plt.figure(figsize=(10,6))
plt.scatter(df['Population 2025'], df['Predicted Pop'], alpha=0.5)
plt.plot([df['Population 2025'].min(), df['Population 2025'].max()],
         [df['Population 2025'].min(), df['Population 2025'].max()],
         'r--')  # خط مثالي للتوقع
plt.xlabel("Actual Population 2025")
plt.ylabel("Predicted Population 2025")
plt.title("Actual vs Predicted Population")
plt.show()


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# افترض أن df جاهزة و y = df['Population 2025']
y = df['Population 2025'].astype(float)

# الأعمدة الرقمية فقط (عدا الهدف)
x_numerical = df.select_dtypes(exclude=['object']).drop(columns=['Population 2025'])

# الأعمدة النصية
x_categorical = df.select_dtypes(include=['object'])

# تحويل الأعمدة النصية إلى أرقام
label_encoder = LabelEncoder()
x_categorical_encoded = x_categorical.apply(label_encoder.fit_transform)

# دمج الأعمدة الرقمية مع النصية المحوّلة
X = pd.concat([x_numerical, x_categorical_encoded], axis=1)

# إنشاء نموذج Random Forest
regressor = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True)

# تدريب النموذج
regressor.fit(X, y)

# طباعة OOB Score
print("OOB Score:", regressor.oob_score_)
# التنبؤ باستخدام النموذج
y_pred = regressor.predict(X)
df['RF Predicted Pop'] = y_pred

plt.figure(figsize=(10,6))
plt.scatter(df['Population 2025'], df['RF Predicted Pop'], alpha=0)
plt.plot([df['Population 2025'].min(), df['Population 2025'].max()],
         [df['Population 2025'].min(), df['Population 2025'].max()],
         'r--')  # خط مثالي للتوقع  
plt.xlabel("Actual Population 2025")
plt.ylabel("RF Predicted Population 2025")
plt.title("Actual vs RF Predicted Population")
plt.show()

# نفترض أن لدينا تغييرات متوقعة لكل متغير
df_2026 = df.copy()

# تحديث بعض الأعمدة لتوقع 2026
# مثال: زيادة بسيطة في Fertility Rate، Urban Pop % و Yearly Change %
df_2026['Fert. Rate'] = df_2026['Fert. Rate'] * 1.01       # زيادة 1%
df_2026['Median Age'] = df_2026['Median Age'] + 1           # ارتفاع العمر الوسيط سنة
df_2026['Urban Pop %'] = df_2026['Urban Pop %'] * 1.02     # زيادة 2%
df_2026['Yearly Change %'] = df_2026['Yearly Change %'] * 1.01

from sklearn.preprocessing import LabelEncoder

# اختيار الأعمدة النصية (مثل أسماء الدول)
x_categorical_2026 = df_2026.select_dtypes(include=['object'])
label_encoder = LabelEncoder()
x_categorical_encoded_2026 = x_categorical_2026.apply(label_encoder.fit_transform)

# دمج الأعمدة الرقمية والنصية المشفرة
x_numerical_2026 = df_2026.select_dtypes(exclude=['object']).drop(columns=['Population 2025'])
X_2026 = pd.concat([x_numerical_2026, x_categorical_encoded_2026], axis=1)

df['RF Predicted Pop'] = regressor.predict(X)  # التنبؤ على بيانات 2025


# ترتيب الأعمدة مثل بيانات التدريب
X_2026_fixed = X_2026[X.columns]  # X هو البيانات التي دربت عليها النموذج مسبقًا

# التنبؤ
df_2026['Population 2026 Predicted'] = regressor.predict(X_2026_fixed)


# الرسم البياني
plt.figure(figsize=(10,6))
plt.scatter(df['Population 2025'], df['RF Predicted Pop'], alpha=0.5, label='2025 Actual vs Predicted')
plt.scatter(df_2026['Population 2025'], df_2026['Population 2026 Predicted'], alpha=0.5, label='2026 Predicted')
plt.plot([df['Population 2025'].min(), df['Population 2025'].max()],
         [df['Population 2025'].min(), df['Population 2025'].max()],
         'r--', label='Ideal Line')
plt.xlabel("Actual Population")
plt.ylabel("Predicted Population")
plt.title("Actual vs Predicted Population")
plt.legend()
plt.show()


from sklearn.preprocessing import LabelEncoder

# 1. الأعمدة النصية
x_categorical_2026 = df_2026.select_dtypes(include=['object'])
label_encoder = LabelEncoder()
x_categorical_encoded_2026 = x_categorical_2026.apply(label_encoder.fit_transform)

# 2. الأعمدة الرقمية (مثل Fert. Rate, Median Age...)
x_numerical_2026 = df_2026.select_dtypes(exclude=['object']).drop(columns=['Population 2025'])

# 3. دمج الأعمدة الرقمية والنصية المشفرة
X_2026 = pd.concat([x_numerical_2026, x_categorical_encoded_2026], axis=1)

# 4. ترتيب الأعمدة مثل بيانات التدريب
X_2026_fixed = X_2026[X.columns]

# 5. التنبؤ بعدد السكان
df_2026['Population 2026 Predicted'] = regressor.predict(X_2026_fixed)

# 6. عرض النتائج
print(df_2026[['Country (or dependency)', 'Population 2026 Predicted']])

import matplotlib.pyplot as plt
import seaborn as sns

# 1. ترتيب الدول حسب التنبؤ بعدد السكان لعام 2026
top10_2026 = df_2026.sort_values('Population 2026 Predicted', ascending=False).head(10)

# 2. رسم شريطي للدول العشر الأعلى
plt.figure(figsize=(12,6))
sns.barplot(
    x='Country (or dependency)', 
    y='Population 2026 Predicted', 
    data=top10_2026, 
    palette='viridis'
)

# 3. إعدادات الرسم
plt.xticks(rotation=45)  # تدوير أسماء الدول لتظهر واضحة
plt.xlabel("Country")
plt.ylabel("Predicted Population 2026")
plt.title("Top 10 Most Populated Countries in 2026 (Predicted)")
plt.tight_layout()
plt.show()
