# Gender, Experience, and Compensation in Modern Workplaces: A Machine Learning Analysis of Salary Survey Data Using Machine Learning and Time Series Forecasting

### Project OverView

This research aims to understand how various demographic and professional variables impact compensation in modern workplaces. The primary goal is to promote fairness, equity, and evidence-based decisions within the workplace by studying these relationships. The project utilizes Machine Learning (ML) and time-series (t-series) forecasting for its analysis.

### Data Sources

The research utilizes the **AskAManager.org salary survey dataset**. This dataset offers a unique opportunity to study compensation dynamics as it contains real-world, real-people salary data. The dataset is comprehensive, including both structured attributes such as age, gender, education, and salary, and unstructured attributes like free-form job titles, compensation context, and additional earnings. It covers a wide diversity of employee experience across multiple industries, countries, and demographic groups. Furthermore, the dataset is dynamic, allowing for the application of time series forecasting to simulate future compensation trends.

### Tools 
  - Excel (Data Cleaning)
  
  - Google Colab (Data Analysis and Visualisation)
    
  ### Data Cleaning and Preparation
  To ensure accuracy and avoid discrepancies, a rigorous data cleaning and preprocessing phase is undertaken, particularly for the structured and unstructured attributes within the dataset. This phase includes:
  
    - Text standardization of job titles using Natural Language Processing (NLP) tools like spaCy and TF-IDF.
    
    - Currency normalization based on current and historical exchange rates.
    
    - Outlier detection and treatment using statistical methods such as z-scores and Interquartile Range methods.
    
    - Imputation of missing values using k-Nearest Neighbors or regression-based approaches.

  ### Exploratory Data Analysis

  The project's research objectives guide the exploratory data analysis and subsequent modeling, addressing core questions about compensation dynamics:
  
 - "How does gender, race, and age influence base salary and additional compensation when controlling experience, education, and job type?"
  
- "Which features mostly predict salary outcomes across different sectors and roles?"
  
-  "What predictive frameworks most effectively estimate salary using demographics and professional attributes?"
  
- "What is the distribution of annual salary by gender and education level?"
  
- "How is additional compensation distributed across experience levels, segmented by gender?"

### Data Analysis
  - How does gender, race, and age influence base salary and additional compensation when controlling experience, education, and job type?
```python
# Visualization 1: Compensation by Gender
plt.figure(figsize=(12, 6))
sns.boxplot(x='Gender', y='Total_Compensation', data=df)
plt.title('Total Compensation Distribution by Gender')
plt.ylabel('Total Compensation (USD)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
```python
# Visualization 2: Compensation by Age and Gender
plt.figure(figsize=(12, 6))
sns.barplot(x='Age_Midpoint', y='Total_Compensation', hue='Gender', data=df, ci=None)
plt.title('Average Compensation by Age and Gender')
plt.ylabel('Average Total Compensation (USD)')
plt.xlabel('Age Midpoint')
plt.tight_layout()
plt.show()
```
```python
# Visualization 3: Experience vs Compensation by Gender
plt.figure(figsize=(12, 6))
sns.scatterplot(
    x='Total_Experience', y='Total_Compensation', hue='Gender',
    size='Age_Midpoint', sizes=(40, 200), alpha=0.7, data=df
)
plt.title('Experience vs Compensation by Gender and Age')
plt.ylabel('Total Compensation (USD)')
plt.xlabel('Total Experience (Years)')
plt.tight_layout()
plt.show()
```

  - What predictive frameworks most effectively estimate salary using demographic and professional attributes?
 ```Python
sample_df = shuffle(df).sample(n=500, random_state=42)
features = ['Age_Range', 'Industry', 'Job_Title', 'Country', 'Education_Level', 'Gender', 'Total_Experience', 'Field_Experience']
target = 'Total_Compensation'

X = sample_df[features].fillna('Missing')
y = sample_df[target].fillna(sample_df[target].median())

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Evaluate models
results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error')
    results[name] = -scores.mean()

# print RMSE for each model
for model_name, rmse in results.items():
    print(f"{model_name}: RMSE = {rmse:.2f}")
```

- Which features most predict salary outcomes across different sectors and roles
```python
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit model
pipeline.fit(X, y)

# Extract feature names
encoded_features = pipeline.named_steps['preprocessor'].transformers_[0][2] + \
    pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols).tolist()

# feature importances
importances = pipeline.named_steps['regressor'].feature_importances_

feat_importance = pd.DataFrame({
    'Feature': encoded_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot top features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feat_importance.head(15))
plt.title('Top Features Predicting Salary')
plt.tight_layout()
plt.show()
```

### Results/ Findings
  
