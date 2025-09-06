import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
df = pd.read_csv('data/data_pre.csv')

print("Data shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData info:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Data preprocessing - remove completely empty or constant columns
df_cleaned = df.dropna(axis=1, how='all')
constant_columns = [col for col in df_cleaned.columns if df_cleaned[col].nunique() <= 1]
df_cleaned = df_cleaned.drop(columns=constant_columns)

print(f"\nRemoved {len(constant_columns)} constant columns: {constant_columns}")

# Select numeric columns for analysis
numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
df_numeric = df_cleaned[numeric_columns]

print(f"\nNumeric columns for analysis ({len(numeric_columns)}):")
print(numeric_columns.tolist())

# Calculate correlation matrix
correlation_matrix = df_numeric.corr()

# Create correlation matrix heatmap
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 6})  # Smaller font for annotations
plt.title('Correlation Matrix (Pearson Coefficients)', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Find strong correlations (|r| > 0.7)
strong_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            strong_correlations.append((
                correlation_matrix.columns[i], 
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

# Sort by correlation strength
strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

print("\nStrong correlations (|r| > 0.7):")
for corr in strong_correlations:
    print(f"{corr[0]} with {corr[1]}: {corr[2]:.4f}")


# Continue from your code...

# Continue from your code...

# Filter out rows where Sex is 0
df_filtered = df_cleaned[df_cleaned['Sex'] != 0].copy()

print(f"Original dataset shape: {df_cleaned.shape}")
print(f"Filtered dataset shape (Sex != 0): {df_filtered.shape}")

# Calculate correlation with YP (excluding PN and Sex) on filtered data
correlation_matrix_filtered = df_filtered.select_dtypes(include=[np.number]).corr()
yp_correlations = correlation_matrix_filtered['YP'].drop(['PN', 'Sex'], errors='ignore').sort_values(key=abs, ascending=False)

# Select top 10 most correlated features (excluding YP itself)
top_10_features = yp_correlations.drop('YP').head(10)
print("Top 10 features most correlated with YP (after filtering Sex=0):")
print(top_10_features)

# Prepare features and target variable
X = df_filtered[top_10_features.index]
y = df_filtered['YP']

# Check and handle missing values
print(f"\nMissing values in features: {X.isnull().sum().sum()}")
print(f"Missing values in target: {y.isnull().sum()}")

# If there are missing values, handle them
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    from sklearn.impute import SimpleImputer
    
    # Impute features with mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)
    
    # Impute target variable with mean
    if y.isnull().sum() > 0:
        y = y.fillna(y.mean())

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create and train linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Coefficient of Determination (R²): {r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (YP) - Sex != 0')
plt.savefig('actual_vs_predicted_filtered.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot - Sex != 0')
plt.savefig('residual_plot_filtered.png', dpi=300, bbox_inches='tight')
plt.show()

# Output model coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nModel Coefficients:")
print(coefficients)

# Check model assumptions - normally distributed residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Residual Distribution - Sex != 0')
plt.savefig('residual_distribution_filtered.png', dpi=300, bbox_inches='tight')
plt.show()

# Perform cross-validation for more robust performance evaluation
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"\nCross-Validation R² Scores: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")

# Save model for future predictions
import joblib
joblib.dump(model, 'yp_regression_model_filtered.pkl')
print("\nModel saved as 'yp_regression_model_filtered.pkl'")

# Optional: Check for multicollinearity using VIF
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    
    X_with_const = add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
    
    print("\nVariance Inflation Factors (VIF):")
    print(vif_data)
except ImportError:
    print("\nstatsmodels not installed, skipping VIF calculation")

# Continue from previous code...

# Continue from previous code...

# Import MLPRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Prepare the data (using the same filtered dataset and features as before)
X = df_filtered[top_10_features.index]
y = df_filtered['YP']

# 检查并处理NaN值
print("检查数据中的NaN值...")
print(f"X中的NaN值数量: {X.isnull().sum().sum()}")
print(f"y中的NaN值数量: {y.isnull().sum()}")

# 如果有NaN值，进行处理
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    print("处理NaN值...")
    # 对特征使用均值填充
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)
    
    # 对目标变量使用均值填充
    if y.isnull().sum() > 0:
        y = y.fillna(y.mean())

# 确认没有NaN值
print(f"处理后X中的NaN值数量: {X.isnull().sum().sum()}")
print(f"处理后y中的NaN值数量: {y.isnull().sum()}")

# 确保数据中没有无穷大的值
X = X.replace([np.inf, -np.inf], np.nan)
if X.isnull().sum().sum() > 0:
    X = X.fillna(X.mean())

print(f"最终X中的NaN值数量: {X.isnull().sum().sum()}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 创建一个简化的MLP模型，先不使用网格搜索
print("创建并训练MLP模型...")
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        alpha=0.001,
        learning_rate_init=0.01,
        random_state=42,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1
    ))
])

# 训练模型
mlp_pipeline.fit(X_train, y_train)

# 获取训练历史
mlp_model = mlp_pipeline.named_steps['mlp']
print(f"训练迭代次数: {mlp_model.n_iter_}")
print(f"最终损失: {mlp_model.loss_:.4f}")

# Make predictions with the MLP model
y_pred_mlp = mlp_pipeline.predict(X_test)

# Evaluate MLP model
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

print("\nMLP Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse_mlp:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_mlp:.4f}")
print(f"Mean Absolute Error (MAE): {mae_mlp:.4f}")
print(f"Coefficient of Determination (R²): {r2_mlp:.4f}")

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(mlp_model.loss_curve_)
plt.title('MLP Training Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('mlp_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot actual vs predicted values for MLP
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_mlp, alpha=0.6, label='MLP Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (MLP) - Sex != 0')
plt.legend()
plt.savefig('actual_vs_predicted_mlp.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot residuals for MLP
residuals_mlp = y_test - y_pred_mlp
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_mlp, residuals_mlp, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot (MLP) - Sex != 0')
plt.savefig('residual_plot_mlp.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot residual distribution for MLP
plt.figure(figsize=(10, 6))
sns.histplot(residuals_mlp, kde=True)
plt.xlabel('Residuals')
plt.title('Residual Distribution (MLP) - Sex != 0')
plt.savefig('residual_distribution_mlp.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the MLP model
joblib.dump(mlp_pipeline, 'yp_mlp_model.pkl')
print("\nMLP model saved as 'yp_mlp_model.pkl'")

# 如果模型表现良好，可以尝试进行超参数调优
if r2_mlp > 0.5:  # 如果R²大于0.5，说明模型有一定预测能力
    print("模型表现良好，尝试进行超参数调优...")
    
    # 定义参数网格
    param_grid = {
        'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__alpha': [0.0001, 0.001, 0.01],
        'mlp__learning_rate_init': [0.001, 0.01]
    }
    
    # 创建基础管道
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(random_state=42, max_iter=1000, early_stopping=True))
    ])
    
    # 执行网格搜索
    mlp_grid = GridSearchCV(base_pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
    mlp_grid.fit(X_train, y_train)
    
    # 获取最佳模型
    best_mlp = mlp_grid.best_estimator_
    
    # 打印最佳参数
    print(f"Best parameters: {mlp_grid.best_params_}")
    print(f"Best cross-validation R²: {mlp_grid.best_score_:.4f}")
    
    # 使用最佳模型进行预测
    y_pred_best_mlp = best_mlp.predict(X_test)
    r2_best_mlp = r2_score(y_test, y_pred_best_mlp)
    
    print(f"Test R² with best model: {r2_best_mlp:.4f}")
    
    # 保存最佳模型
    joblib.dump(best_mlp, 'yp_mlp_best_model.pkl')
    print("Best MLP model saved as 'yp_mlp_best_model.pkl'")
    
    # 比较不同模型的性能
    print(f"\nModel Comparison:")
    print(f"Basic MLP R²: {r2_mlp:.4f}")
    print(f"Tuned MLP R²: {r2_best_mlp:.4f}")
else:
    print("模型表现不佳，可能需要重新考虑特征选择或数据预处理。")