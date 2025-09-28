# Import libraries (pandas for data, matplotlib for plots in insights)
import pandas as pd
import matplotlib.pyplot as plt


# Step 1: Data Exploration 
url = "https://www.dropbox.com/scl/fi/d5v08knbo3gha4xn2a44h/student_wellbeing_dataset.csv?rlkey=72iirijqpk6g2290myzng7pvi&e=1&st=kwwvfj6i&dl=1"
df = pd.read_csv(url) 

print("=== Data Exploration ===")
print("Dataset Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info (data types and non-null counts):")
print(df.info())
print("\nDataset Description (stats for numerical columns):")
print(df.describe())

print("\nMissing Values (count per column):")
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0])

duplicates = df.duplicated().sum()
print(f"\nNumber of Duplicate Rows: {duplicates}")

print("\nUnique values in categorical columns:")
print("Extracurricular:", df['Extracurricular'].unique())
print("Stress_Level:", df['Stress_Level'].unique())


# Step 2: Data Preprocessing 
print("\n=== Data Preprocessing ===")
initial_shape = df.shape
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape} (removed {initial_shape[0] - df.shape[0]} rows)")

# Handle missing values ONLY for numerical columns (fill with median)
numerical_cols = ['Hours_Study', 'Sleep_Hours', 'Screen_Time', 'Attendance', 'CGPA']
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled missing values in '{col}' with median: {median_val}")

print("\nMissing Values after Preprocessing:")
print(df.isnull().sum().sum())

# Create a temporary copy for analysis (encode categories only here for math operations)
df_analysis = df.copy()
df_analysis['Extracurricular'] = df_analysis['Extracurricular'].map({'Yes': 1, 'No': 0})
df_analysis['Stress_Level'] = df_analysis['Stress_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
# Drop Student_ID from analysis copy (not needed for stats)
df_analysis = df_analysis.drop('Student_ID', axis=1)


# Step 3: Exploratory Data Analysis (EDA) - Use encoded df_analysis (text-based, updated cols)
print("\n=== Exploratory Data Analysis (EDA) ===")

corr_cols = ['Hours_Study', 'Sleep_Hours', 'Screen_Time', 'Attendance', 'CGPA']
correlations = df_analysis[corr_cols].corr()['CGPA'].sort_values(ascending=False)
print("\nCorrelations with CGPA (1=strong positive, -1=strong negative):")
print(correlations)

stress_analysis = df_analysis.groupby('Stress_Level')['CGPA'].agg(['mean', 'count', 'std']).round(2)
print("\nCGPA by Stress_Level (0:Low, 1:Medium, 2:High):")
print(stress_analysis)

extra_analysis = df_analysis.groupby('Extracurricular')['CGPA'].agg(['mean', 'count', 'std']).round(2)
print("\nCGPA by Extracurricular (0:No, 1:Yes):")
print(extra_analysis)

key_stats = df_analysis[['Hours_Study', 'Sleep_Hours', 'Screen_Time', 'CGPA']].describe().round(2)
print("\nOverall Stats for Key Features:")
print(key_stats)


# Step 4: Insights (Use df_analysis for computations, visuals updated with new col names)
print("\n=== Insights (Based on Full Dataset Statistics, with Visualizations) ===")

# Insight 1: Hours_Study and CGPA (scatter plot)
study_corr = df_analysis['Hours_Study'].corr(df_analysis['CGPA'])
print(f"1. Hours_Study strongly correlate with CGPA (correlation: {study_corr:.3f}). Higher study time links to better grades.")
plt.figure(figsize=(8, 5))
plt.scatter(df_analysis['Hours_Study'], df_analysis['CGPA'], alpha=0.5)
plt.xlabel('Hours_Study')
plt.ylabel('CGPA')
plt.title(f'Hours_Study vs CGPA (Corr: {study_corr:.3f})')
plt.grid(True)
plt.savefig('insight1_study_corr.png')
plt.show()

# Insight 2: Sleep_Hours and CGPA (bar plot for bins)
df_analysis['Sleep_Bin'] = pd.cut(df_analysis['Sleep_Hours'], bins=[0, 6, 8, 24], labels=['<6h', '6-8h', '>8h'])
sleep_insight = df_analysis.groupby('Sleep_Bin', observed=False)['CGPA'].mean().round(2)
print("2. Average CGPA by Sleep Bins:")
print(sleep_insight)
print("   - Optimal sleep (6-8h) yields the highest CGPA.")
plt.figure(figsize=(8, 5))
sleep_insight.plot(kind='bar')
plt.title('Average CGPA by Sleep Bins')
plt.xlabel('Sleep_Hours')
plt.ylabel('Average CGPA')
plt.savefig('insight2_sleep_bins.png')
plt.show()

# Insight 3: Screen_Time impact (bar plot)
median_screen = df_analysis['Screen_Time'].median()
high_screen_cgpa = df_analysis[df_analysis['Screen_Time'] > median_screen]['CGPA'].mean().round(2)
low_screen_cgpa = df_analysis[df_analysis['Screen_Time'] <= median_screen]['CGPA'].mean().round(2)
print(f"3. High screen time (>median {median_screen:.1f}h) CGPA: {high_screen_cgpa} vs Low: {low_screen_cgpa}. More screen time lowers performance.")
plt.figure(figsize=(8, 5))
screen_groups = ['Low Screen Time', 'High Screen Time']
screen_cgpa = [low_screen_cgpa, high_screen_cgpa]
plt.bar(screen_groups, screen_cgpa)
plt.title('CGPA by Screen_Time Level')
plt.ylabel('Average CGPA')
plt.savefig('insight3_screen_time.png')
plt.show()

# Insight 4: Attendance and CGPA (scatter plot)
attend_corr = df_analysis['Attendance'].corr(df_analysis['CGPA'])
print(f"4. Attendance has a strong positive correlation with CGPA ({attend_corr:.3f}). Regular attendance boosts grades.")
plt.figure(figsize=(8, 5))
plt.scatter(df_analysis['Attendance'], df_analysis['CGPA'], alpha=0.5)
plt.xlabel('Attendance (%)')
plt.ylabel('CGPA')
plt.title(f'Attendance vs CGPA (Corr: {attend_corr:.3f})')
plt.grid(True)
plt.savefig('insight4_attendance_corr.png')
plt.show()

# Insight 5: Stress_Level effect (bar plot)
print(f"5. Low stress students average {stress_analysis.loc[0, 'mean']:.2f} CGPA vs High stress: {stress_analysis.loc[2, 'mean']:.2f}. Managing stress improves outcomes.")
plt.figure(figsize=(8, 5))
stress_levels = ['Low (0)', 'Medium (1)', 'High (2)']
stress_means = [stress_analysis.loc[i, 'mean'] for i in range(3)]
plt.bar(stress_levels, stress_means)
plt.title('Average CGPA by Stress_Level')
plt.ylabel('Average CGPA')
plt.savefig('insight5_stress_level.png')
plt.show()

# Insight 6: Extracurricular benefit (bar plot)
extra_diff = extra_analysis.loc[1, 'mean'] - extra_analysis.loc[0, 'mean']
print(f"6. Students with extracurriculars have {extra_diff:.2f} higher average CGPA than those without.")
plt.figure(figsize=(8, 5))
extra_groups = ['No (0)', 'Yes (1)']
extra_means = [extra_analysis.loc[i, 'mean'] for i in range(2)]
plt.bar(extra_groups, extra_means)
plt.title('Average CGPA by Extracurricular')
plt.ylabel('Average CGPA')
plt.savefig('insight6_extracurricular.png')
plt.show()


# Export cleaned dataset (Updated: With new column names, Student_ID, original categories)
print("\nCleaned dataset (full rows) exported to 'cleaned_student_data.csv' (in the same folder as this script)")
print("It includes Student_ID, new column names, and original categorical labels (missings in categoricals remain as NaN).")
df.to_csv('cleaned_student_data.csv', index=False)
print(f"Final dataset shape: {df.shape}")
print("\nFirst few rows of exported data (for verification):")
print(df.head())
