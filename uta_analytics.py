import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from matplotlib.patches import Patch

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Load the dataset

# Load the dataset
df = pd.read_csv('/Users/arnob/Dropbox/UTA_Analytics/Synthetic_Dataset_2023.csv')


##############################################################
#                        Pie Chart                           #
##############################################################
# Count retention vs no retention
retention_counts = df['OneYearRetention'].value_counts().sort_index()
labels = ['No Retention (0)', 'Retention (1)']
sizes = retention_counts.values
colors = [(0, 0, 1, 0.4), (1, 0, 0, 0.4)]  # blue and red with alpha = 0.4

# Plot fancy disk-style pie chart
plt.figure(figsize=(5, 5))
wedges, texts, autotexts = plt.pie(
    sizes,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    wedgeprops=dict(width=0.5, edgecolor='white')  # donut style
)

plt.setp(autotexts, size=12, weight="bold", color="black")
plt.setp(texts, size=10)

# Smaller, precise legend
legend_elements = [
    Patch(facecolor=colors[0], edgecolor='black', label=f'{labels[0]}: {sizes[0]:,}'),
    Patch(facecolor=colors[1], edgecolor='black', label=f'{labels[1]}: {sizes[1]:,}')
]

plt.legend(
    handles=legend_elements,
    title="Retention Status",
    loc="upper right",
    fontsize=10,
    title_fontsize=11,
    handlelength=1.2,
    handletextpad=0.5,
    borderpad=0.5,
    labelspacing=0.4
)

plt.title("One-Year Retention Status of Students", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()



##############################################################
#                Barplot for College:                        #
##############################################################

# Create a grouped dataframe: count of retention by college
grouped = df.groupby(['FirstTermEnrolledCollege', 'OneYearRetention']).size().unstack(fill_value=0)
grouped.columns = ['No Retention (0)', 'Retention (1)']

# Set custom colors with alpha=0.4
colors = [(0, 0, 1, 0.4), (1, 0, 0, 0.4)]  # Blue and red with transparency

# Plotting
ax = grouped.plot(kind='bar', figsize=(12, 6), width=0.7, color=colors, edgecolor='black')

# Add percentage of total students on top of each bar (not per group)
total_students = df.shape[0]
for idx, (index, row) in enumerate(grouped.iterrows()):
    total_group = row.sum()
    for i, val in enumerate(row):
        percentage = (val / total_students) * 100
        ax.text(
            idx + (i - 0.5) * 0.2,  # slight horizontal shift for each bar in the group
            val + 1,                # just above the bar
            f'{percentage:.1f}%',
            ha='center', va='bottom', fontsize=10, color='black'
        )

    ratio_1 = int(round(100 * row['No Retention (0)'] / total_group))
    ratio_2 = 100 - ratio_1
    ratio_text = f"{ratio_1}:{ratio_2}"
    ax.text(
        idx, -5, ratio_text,
        ha='center', va='top', fontsize=9, color='black'
    )

# Create matching legend
legend_elements = [
    Patch(facecolor=colors[0], edgecolor='black', label='No Retention (0)'),
    Patch(facecolor=colors[1], edgecolor='black', label='Retention (1)')
]
plt.legend(handles=legend_elements, title='Retention Status')

# Styling
plt.title('One-Year Retention by First Term Enrolled College', fontsize=16, weight='bold')
plt.xlabel('First Term Enrolled College', fontsize=12)
plt.ylabel('Number of Students', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()





import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and clean data
df = pd.read_excel("Synthetic_Dataset_2023.xlsx")
df.columns = df.columns.str.strip()

# Reduce dataset size to avoid 25MB limit
df_small = df.sample(n=3000, random_state=42)

# Separate target and features
X = df_small.drop(columns="OneYearRetention")
y = df_small["OneYearRetention"]

# Encode categorical features
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model to pickle
with open("model_rf.pkl", "wb") as f:
    pickle.dump(model, f)


##############################################################
#                Barplot for Gender:                        #
##############################################################
# Create a grouped dataframe: count of retention by gender
grouped = df.groupby(['Gender', 'OneYearRetention']).size().unstack(fill_value=0)
grouped.columns = ['No Retention (0)', 'Retention (1)']

# Set custom colors with alpha=0.4
colors = [(0, 0, 1, 0.4), (1, 0, 0, 0.4)]  # Blue and red with transparency

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
grouped.plot(kind='bar', ax=ax, width=0.7, color=colors, edgecolor='black')

# Add percentage of total students on top of each bar (not per group)
total_students = df.shape[0]
for idx, (gender, row) in enumerate(grouped.iterrows()):
    total_group = row.sum()
    for i, val in enumerate(row):
        percentage = (val / total_students) * 100
        ax.text(
            idx + (i - 0.5) * 0.2,
            val + 1,
            f'{percentage:.1f}%',
            ha='center', va='bottom', fontsize=10, color='black'
        )

    # Add group ratio like 14:86 under each gender
    ratio_1 = int(round(100 * row['No Retention (0)'] / total_group))
    ratio_2 = 100 - ratio_1
    ratio_text = f"{ratio_1}:{ratio_2}"
    ax.text(
        idx, -5, ratio_text,
        ha='center', va='top', fontsize=9, color='black'
    )

# Create matching legend
legend_elements = [
    Patch(facecolor=colors[0], edgecolor='black', label='No Retention (0)'),
    Patch(facecolor=colors[1], edgecolor='black', label='Retention (1)')
]
plt.legend(handles=legend_elements, title='Retention Status')

# Styling
plt.title('One-Year Retention by Gender', fontsize=16, weight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Number of Students', fontsize=12)
plt.xticks(rotation=0, ha='center')
plt.tight_layout()

plt.show()




##############################################################
#                  Features Selection                        #
##############################################################


df.columns = df.columns.str.strip()

# Separate features and target
X = df.drop(columns='OneYearRetention')
y = df['OneYearRetention']

# Encode categorical variables
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Fit Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Extract feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot vertical bars
plt.figure(figsize=(8, 4))
barplot = sns.barplot(x='Feature', y='Importance', data=importance_df, color='red', alpha=0.4)

# Annotate each bar with the importance score on top
for p in barplot.patches:
    height = p.get_height()
    plt.text(p.get_x() + p.get_width()/2, height + 0.002, f'{height:.3f}',
             ha='center', va='bottom', fontsize=10)

# Styling
plt.title('The relative importance of each variable', fontsize=16, weight='bold')
plt.ylabel('Relative Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()





# Preprocess
df.columns = df.columns.str.strip()
X = df.drop(columns='OneYearRetention')
y = df['OneYearRetention']

# Encode categorical features
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels = ['Not Retained', 'Retained']

for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot base heatmap with greyscale color
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', cbar=False,
                xticklabels=labels, yticklabels=labels, ax=axes[idx],
                linewidths=1, linecolor='black',
                annot_kws={"size": 14, "weight": "bold", "color": "black"})

    # Overlay ash grey on diagonal cells
    for (i, j), val in np.ndenumerate(cm):
        if i == j:
            rect = plt.Rectangle((j, i), 1, 1, fill=True, color='#d3d3d3', zorder=2, lw=0)
            axes[idx].add_patch(rect)
            # Re-draw annotation text on top
            axes[idx].text(j + 0.5, i + 0.5, str(val),
                           ha='center', va='center', fontsize=14, fontweight='bold', color='black')

    axes[idx].set_title(f"{name} Confusion Matrix", fontsize=13, weight='bold')
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

plt.tight_layout()
plt.show()

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and clean data
df = pd.read_csv("Model.csv")
df.columns = df.columns.str.strip()

# Reduce dataset size to avoid 25MB limit
df_small = df.sample(n=3000, random_state=42)

# Separate target and features
X = df_small.drop(columns="OneYearRetention")
y = df_small["OneYearRetention"]

# Encode categorical features
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model to pickle
with open("model_rf.pkl", "wb") as f:
    pickle.dump(model, f)
