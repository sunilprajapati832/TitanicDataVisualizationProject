# Titanic Data Visualization Project

This project explores the Titanic dataset through a wide range of data visualization techniques using Python.
It covers:
1. **Univariate Analysis** — Distribution of individual variables  
2. **Bivariate Analysis** — Relationships between two variables  
3. **Multivariate Analysis** — Correlation between multiple variables  
4. **Advanced Visualizations** — Violin plots, swarm plots and FacetGrids  
5. **Categorical & Time-Based Analysis** — Class-wise and gender-based insights

## Objective
The goal of this project is to perform **Exploratory Data Analysis (EDA)** on the Titanic dataset to discover key insights about **passenger survival patterns**, demographic trends and relationships between various features such as age, class, fare and gender.  

## Dataset

| Feature | Description |
|----------|-------------|
| `survived` | Survival (0 = No, 1 = Yes) |
| `pclass` | Passenger class (1 = Upper, 2 = Middle, 3 = Lower) |
| `sex` | Gender |
| `age` | Age of passenger |
| `sibsp` | Number of siblings/spouses aboard |
| `parch` | Number of parents/children aboard |
| `fare` | Passenger fare |
| `embarked` | Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton) |
| `class` | Passenger class (categorical) |
| `who`, `adult_male`, `deck`, `embark_town`, `alive`, `alone` | Additional derived columns |

📌 You can find the dataset on Kaggle Titanic - Machine Learning from Disaster

## Tools & Libraries Used
Python (PyCharm IDE)

# matplotlib.pyplot – For base-level plotting and customization.

🐧 seaborn – For high-level, attractive statistical visualizations.

# Visualizations Performed

1️⃣ Univariate Analysis
Histogram – Age distribution

Countplot – Passenger class

Pie Chart – Survival rate

2️⃣ Bivariate Analysis
Boxplot – Age distribution across classes

Barplot – Survival rate by class

Scatterplot – Fare vs. Age

3️⃣ Multivariate Analysis
Heatmap – Correlation matrix (after handling missing values)

Pairplots – Multiple variable relationships

4️⃣ Advanced Visualizations
Violin Plot – Age distribution by survival status

Swarmplot – Survival based on class and gender

FacetGrid – Age and class breakdown with survival overlay

5️⃣ Categorical & Time-Based Insights
Countplot – Gender vs survival

Lineplot – Simulated trend analysis with age

6️⃣ Customization & Styling
Applied themes, annotations, and visual polish to make graphs more presentable and insightful.

📂 Project Structure
├── titanic_visualization.ipynb   # Main project notebook
├── titanic.csv                   # Dataset file
├── README.md                     # Project overview
└── /images (Figures)             # Folder containing exported plots

## Connect with Me 🤝
If you found this project interesting, let’s connect!  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow%20Me-blue?logo=linkedin&style=for-the-badge)](https://www.linkedin.com/in/sunil-prajapati832) 
