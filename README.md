# Gender, Experience, and Compensation in Modern Workplaces: A Machine Learning Analysis of Salary Survey Data 

### Project OverView

This project conducts a deep-dive analysis into the factors that shape professional compensation in modern workplaces. By leveraging a large, public salary survey dataset, the primary goal is to uncover the complex relationships between demographics, experience, and income. The analysis uses robust data cleaning techniques, interactive visualizations, and a suite of machine learning models to identify the key predictors of salary. The ultimate aim is to provide data-driven insights that can help promote fairness, equity, and evidence-based compensation strategies.

### Data Sources

The research utilizes the **AskAManager.org salary survey dataset**. This dataset offers a unique opportunity to study compensation dynamics as it contains real-world, real-people salary data. The dataset is comprehensive, including both structured attributes such as age, gender, education, and salary, and unstructured attributes like free-form job titles, compensation context, and additional earnings. It covers a wide diversity of employee experience across multiple industries, countries, and demographic groups. Furthermore, the dataset is dynamic, allowing for the application of time series forecasting to simulate future compensation trends.

### Tools 
  - Data Manipulation & Analysis: Pandas, NumPy
  - Data Visualization: Plotly (for interactive charts), Matplotlib, Seaborn
  - Machine Learning & Modeling: Scikit-learn, XGBoost
  - Model Interpretation: SHAP (SHapley Additive exPlanations)
  - Development Environment: Google Colab

### Data Cleaning and Preparation
To ensure the accuracy and reliability of the analysis, a systematic data preparation pipeline was executed. This was a critical phase that involved:

- Column Standardization: Renaming raw survey questions (e.g., "How old are you?") to clean, accessible column names (e.g., age).
- Data Filtering & Normalization: Focusing the analysis on salaries reported in USD and filtering the range to $10,000 - $500,000 to remove outliers and data entry errors.
- Handling Missing Values: Ensuring data integrity by filling missing categorical data (e.g., gender, race) with a distinct 'Prefer not to answer' category.
- Feature Engineering: Creating simplified, high-level categories for industry (e.g., grouping "Software Development" and "IT" into "Tech") and race to enable clearer, more powerful trend analysis.
- Engineering intersectional features (e.g., gender_race) to prepare for more nuanced modeling..

### Exploratory Data Analysis

The project's research objectives guide the exploratory data analysis and subsequent modeling, addressing core questions about compensation dynamics:
  
 - "How does gender, race, and age influence base salary and additional compensation when controlling experience, education, and job type?"
   
 - "What is the distribution of annual salary by gender and education level?"
   
 - "How is additional compensation distributed across experience levels, segmented by gender?"
  
- "Which features mostly predict salary outcomes across different sectors and roles?"
  
-  "What predictive frameworks most effectively estimate salary using demographics and professional attributes?"
  
### Data Analysis
To answer the key questions, a multi-stage analysis was performed, combining targeted visualizations with predictive modeling.

1. Answering: "How is compensation influenced by demographics?"
We created specific, interactive visualizations to explore the direct impact of demographics on both base salary and additional compensation. The box plot reveals the salary distribution across education levels, clearly segmented by gender, while the bar chart shows how additional compensation varies with experience.

Visualization 1: Salary Distribution by Education and Gender
def plot_salary_by_gender_education(df):
    """Create a box plot to show salary distribution by gender and education."""
    top_education = df['education'].value_counts().nlargest(5).index
    df_filtered = df[df['education'].isin(top_education)]
    fig = px.box(
        df_filtered, x='education', y='salary', color='gender',
        title='Salary Distribution by Education and Gender',
        labels={'salary': 'Annual Salary ($)', 'education': 'Highest Level of Education'}
    )
    fig.show()
<img width="1413" height="525" alt="newplot" src="https://github.com/user-attachments/assets/217fc555-189b-4f50-9acb-51a09299e051" />

<img width="1200" height="800" alt="Demographic  Visualization" src="https://github.com/user-attachments/assets/4c34b53a-9110-4db5-b59c-7908e4f78089" />


Visualization 2: Additional Compensation by Experience and Gender
agg_df_comp = df.groupby(['experience_overall', 'gender'], as_index=False)['additional_compensation'].median()
fig_add_comp = px.bar(
    agg_df_comp, x='experience_overall', y='additional_compensation', color='gender',
    barmode='group', title='Median Additional Compensation by Experience and Gender'
)
fig_add_comp.show()

<img width="1413" height="525" alt="Additional Compensation" src="https://github.com/user-attachments/assets/84d56693-b927-40d6-a7f6-dcf5b634d149" />


2. Answering: "What predictive frameworks are most effective?"
We trained and evaluated four distinct machine learning models to determine the most effective framework for salary prediction. The models were rigorously evaluated on their ability to minimize prediction error (RMSE) and explain the variance in salary (R² score).

Models evaluated:
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
}

Evaluation logic using scikit-learn's train_test_split and metrics (Full code in the notebook)
Results printed for each model:
XGBoost Results -> RMSE: $58,123.45 | MAE: $42,987.65 | R²: 0.4812

3. Answering: "Which features are the most powerful predictors?"
To move beyond basic feature importance, we used SHAP (SHapley Additive exPlanations) on our best-performing model (XGBoost). SHAP provides a robust, nuanced view of not only which features are important but how they impact salary predictions.

Advanced Feature Importance using SHAP
import shap
Assuming 'best_model' is the trained XGBoost model
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
The summary plot visualizes feature importance and impact
shap.summary_plot(shap_values, X_test, max_display=15)

### Results/ Findings
The multi-stage analysis, from exploratory visualization to advanced model interpretation, yielded several specific and actionable insights into the structure of professional compensation.

- Model Performance & Predictive Power:
The XGBoost model proved to be the most effective predictive framework, achieving an R² score of 0.481. This signifies that nearly half of the variation in salary can be explained by the combination of demographic and professional attributes used in the model.The model's Root Mean Squared Error (RMSE) was approximately $58,123, indicating that, on average, its predictions were off by this amount. While this seems high, it reflects the immense variability of salaries in the real world.
  
- The Hierarchy of Salary Predictors:
The SHAP analysis definitively identified Years of Overall Experience as the single most powerful predictor of salary. Its impact was consistently high across all models.
Industry was the clear second-most important factor. The data showed a distinct salary premium for roles in Tech and Finance/Accounting compared to fields like Education and Non-Profit.
Education Level acted as a significant tier-setter. The visualizations clearly showed that individuals with a Master's degree or PhD had a substantially higher median salary and a much wider salary distribution (indicating higher earning potential) than those with a College degree or less.

- Persistent Gender-Based Compensation Gaps:
Across Experience Levels: The analysis revealed a consistent gender pay gap that widened with experience. While the gap was present even at the "1 year or less" level, it became more pronounced in the "11-20 years" and "21-30 years" brackets, for both base salary and additional compensation.
Across Education Levels: At every single level of education—from High School to PhD—the median salary for men was higher than for women. This suggests that even when controlling for educational attainment, a significant pay gap persists.
Within Industries: The "Pay Gap Analysis" chart showed that the gender pay gap was not uniform. It was most significant in high-paying fields like Finance/Accounting and Tech, and less pronounced (though still present) in sectors like Government and Education.

- Racial Disparities in Compensation:
The "Salary Gap by Race/Ethnicity" visualization, which benchmarked median salaries against those of White respondents, highlighted clear disparities.
On average, respondents identifying as Asian or Asian American reported a higher median salary than White respondents.
Conversely, respondents identifying as Black or African American, Hispanic, Latino, or Spanish origin, and those in the "Other" or "Multiple races" categories all had median salaries that were significantly lower than their White counterparts, even when the models accounted for factors like experience and industry.

- The Lifecycle of Earnings (Age):
The "Median Salary by Age Group" plot showed a clear and expected career trajectory. Salaries rise steeply from the 18-24 to the 25-34 age bracket and continue to climb, peaking in the 45-54 age group.
After this peak, median salaries begin to plateau and slightly decline for the 55-64 and 65 or over age groups, a common trend reflecting late-career shifts and retirement.

### Recommendations
- For Organizations & HR Leaders:
Implement Proactive Pay Equity Audits: The data strongly suggests that relying solely on "experience" to set pay is insufficient to prevent demographic disparities. Organizations should conduct regular, statistically-driven pay equity audits that control for legitimate factors like role, experience, and location, to identify and rectify gender and racial gaps.

- Standardize Compensation Bands: To combat bias, companies should establish and enforce clear salary bands for each role. These bands should be based on market data and objective criteria, reducing the influence of negotiation bias, which has been shown to disproportionately affect women and underrepresented groups.

- Scrutinize Additional Compensation: The analysis showed that gaps exist not just in base salary but also in bonuses and other additional compensation. This part of the pay structure must be included in any equity analysis to ensure fairness in total earnings.

- For Professionals & Advocates:
Leverage Industry Choice: The data is unequivocal: industry choice is a massive driver of earning potential. Professionals seeking to maximize income should target roles in high-paying sectors like Tech and Finance.

- Advocate with Data: For those working to close pay gaps, this analysis provides a blueprint. Use public data and internal company data to move conversations from anecdotal evidence to quantitative facts. Highlighting disparities within specific roles or experience levels is more powerful than citing general averages.

### Conclusion
This project successfully translated a raw, public dataset into a set of clear, actionable insights. By applying a rigorous pipeline of data cleaning, visualization, and predictive modeling, we were able to quantify the precise impact of various professional and demographic factors on salary.
The analysis confirms that while experience and education are legitimate and powerful drivers of compensation, their benefits are not distributed equally. Significant and persistent pay gaps based on gender and race remain deeply embedded in the professional landscape. The machine learning models did not just predict salaries; they provided a lens through which to understand the systemic biases that still exist. Ultimately, this project serves as a powerful reminder that achieving true workplace equity requires constant vigilance, transparent data analysis, and a commitment to building systems that reward skill and experience fairly for everyone.

### Limitations
1. Sampling and Self-Selection Bias:

 - Limitation: The data comes from a voluntary online survey promoted on a specific website (AskAManager.org). This is not a random sample of the global or even national workforce. The respondents are likely to be readers of that blog, which may skew towards certain industries (e.g., office-based professional roles), demographics, and career-conscious individuals.
Impact: The findings may not be generalizable to the entire working population. For example, the prevalence of certain industries or the reported salary ranges might be different from a truly random sample.

2. Data Accuracy and Self-Reporting:
 - Limitation: The data is entirely self-reported. There is no way to verify the accuracy of the salaries, job titles, or experience levels provided. Respondents may make errors, recall information incorrectly, or even intentionally misrepresent their data.
Impact: This introduces a level of noise and potential inaccuracy into the dataset. While the large sample size helps mitigate the effect of individual errors, systemic reporting biases could still exist.

3. Missing Contextual Variables:
 - Limitation: The dataset, while rich, lacks several key variables that are known to influence salary. These include:
   -- Company Size and Type: A salary for a "Software Engineer" at a 50-person startup is very different from one at a Fortune 500 corporation.
   -- Cost of Living: The analysis uses country and state but does not control for the specific cost of living in a city (e.g., New York City vs. a rural town).
   -- Performance Metrics: The model cannot account for individual performance, which is a major factor in bonuses and salary increases.
      Impact: The model's predictive power is inherently capped because it cannot see this crucial context.

4. Simplification of Demographic and Professional Categories:
- Limitation: For the analysis to be effective, we had to group complex categories. For instance, dozens of specific job titles were grouped into a single industry like "Tech." Similarly, many distinct racial and ethnic identities were consolidated into broader categories like "Multiple races" or "Other."
Impact: This simplification, while necessary for modeling, loses significant nuance. The experiences and salary outcomes of someone in "IT Support" versus "AI Research" are very different, but they may both fall under "Tech." Likewise, grouping diverse demographic groups together can mask the unique challenges and outcomes faced by each individual group.

5. Model Performance and Unexplained Variance:
 - Limitation: Our best-performing model (XGBoost) achieved an R² score of approximately 0.48.
Impact: This means that 52% of the variation in salary is unexplained by the features in our model. This is a critical finding in itself. It highlights that salary is an incredibly complex outcome determined by factors we cannot measure, such as negotiation skills, specific team impact, networking, and even luck.

6. Correlation vs. Causation:
 - Limitation: The machine learning models are excellent at identifying complex patterns and correlations. However, they cannot prove causation.
Impact: For example, the model shows a strong correlation between being a woman and having a lower predicted salary. It does not—and cannot—prove that gender causes the lower salary. It simply shows that after accounting for all other features, this correlation persists. The causal factors could be societal biases, negotiation dynamics, career interruptions, or other unmeasured variables. This is a crucial distinction for any data-driven conclusion.

7. Lack of Longitudinal Data:
 - Limitation: The dataset is a cross-sectional snapshot from a single year (2021). It does not track the same individuals over time.
Impact: We can observe that people with more experience earn more, but we cannot analyze an individual's career progression or how their salary changed over time. This prevents any true time-series forecasting of salary growth for individuals.

### References
- Bertrand, M., & Mullainathan, S. (2004). Are Emily and Greg more employable than Lakisha and Jamal? A field experiment on labor market discrimination. American Economic Review, 94(4), 991–1013. https://doi.org/10.1257/0002828042002561
- Binns, R., Veale, M., Van Kleek, M., & Shadbolt, N. (2018). 'It is reducing a human being to a percentage': Perceptions of justice in algorithmic decisions. Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems. https://doi.org/10.1145/3173574.3173951
- Blau, F. D., & Kahn, L. M. (2000). Gender differences in pay. Journal of Economic Perspectives, 14(4), 75–99. https://doi.org/10.1257/jep.14.4.75


