import matplotlib.pyplot as plt
import seaborn as sns


# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Display the first few rows
print(titanic.head())

# Summary of dataset
print(titanic.info())


### 2. Univariate Analysis (Single variable visualization)

# a.  Distribution of Numeric Data
# Visualize the age distribution of passengers.

plt.figure(figsize=(8,5))
sns.histplot(titanic['age'].dropna(), bins=30, kde=True, color='blue')
plt.title('Age Distribution of Titanic Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# b. Count Plot for Categorical Data
# Count Plot for passenger class.

sns.countplot(data=titanic, x='class', palette='viridis')
plt.title('Passenger Class Distribution')
plt.show()

# c. Pie Chart
# Survival rate representation.

titanic['survived'].value_counts().plot(kind='pie',
        labels=['Did Not Survive', 'Survive'], autopct='%1.1f%%',  colors=['Orange','green'])
plt.title('Survival Rate')
plt.ylabel('')
plt.show()

### 3. Bivariate Analysis (Relationships Between Two Variables)

# a. Boxplot
# Compare age distribution across passenger Classes.

plt.figure(figsize=(8, 5))
sns.boxplot(data=titanic, x='class', y='age', palette='coolwarm')
plt.title('Age Distribution Across Passenger Classes')
plt.show()

# Barplot
# Survival rate by passenger class.

sns.barplot(data=titanic, x='class', y='survived', ci=None, palette='viridis')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Scatterplot
# Fare vs. Age

plt.figure(figsize=(8, 5))
sns.scatterplot(data=titanic, x='age', y='fare', hue='sex', style='class', palette='deep')
plt.title('Fare vs. Age')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

### 4. Multivariate Analysis
#a Heatmap
# Correlation matrix for numeric variables.
# Heatmap missing values: Fill missing numeric values with the Mean
numeric_data = titanic.select_dtypes(include='number').copy()
numeric_data = numeric_data.fillna(numeric_data.mean()) #imputation techniques
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Handle missing values in selected columns for the pairplots
pairplots_data = titanic[['age','fare','pclass','survived']].dropna()

# b. Pairplots
# Visualization multiple relationships.
sns.pairplot(titanic[['age','fare', 'pclass', 'survived']], hue='survived', palette='husl')
plt.suptitle('Pairplots of Titanic Dataset', y=1.02)
plt.show()

### 5. Advanced Visualizations

# a. Violin Plot
# Distribution of age by survival status.

sns.violinplot(data=titanic, x='survived', y='age', split=True, hue='sex', palette='muted')
plt.title('Age Distribution by Survival Status and Gender')
plt.show()

# b. Swarmplot
# Survival by class and gender.

sns.swarmplot(data=titanic, x='class', y='age', hue='survived', palette='Set1', dodge=True)
plt.title('Age Distribution by Class and Survival')
plt.show()

# c. FaceGrid
# Survival rates across age for each class.

g = sns.FacetGrid(titanic, col='class', hue='survived', aspect=1.2, palette='Set2')
g.map(sns.kdeplot, 'age', fill=True)
g.add_legend()
plt.show()

# Advanced plot to uncover complex relationships.
# FaceGrid for breaking data into smaller groups.

### 6. Categorical and Time-Based Analysis
# a) Countplot with Gender and Survival
# Survival comparison by gender.

sns.countplot(data=titanic, x='sex', hue='survived', palette='pastel')
plt.title('Survival by Gender')
plt.show()

# b. Lineplot (If Time Data Available)
# simulating passenger age trends.

sns.lineplot(data=titanic.sort_values(by='age'), x='age', y='fare', hue='class', palette='coolwarm')
plt.title('Fare vs Trends by Class')
plt.show()

### 7. Customizing Visuals for Better Presentation
# Add annotations and themes for polished visuals.

sns.set_theme(style='whitegrid')
plt.figure(figsize=(8, 5))
sns.barplot(data=titanic, x='embark_town', y='fare', ci=None, hue='class', palette='cool')
plt.title('Fare Paid by Embarkation Town and Class')
plt.xlabel('Embarkation Town')
plt.ylabel('Fare')
plt.legend(title='Class')
plt.show()