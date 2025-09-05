import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 1. 加载数据
data_path = "data/data.csv"
df = pd.read_csv(data_path)

# 检查数据
print("数据形状:", df.shape)
print("数据前几行:")
print(df.head())
print("\n数据描述:")
print(df.describe())

# 2. 准备数据
X = df.iloc[:, :-1].values  # 所有列除了最后一列作为特征
y = df.iloc[:, -1].values   # 最后一列作为目标

# 3. 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# 4. 划分训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")
print(f"测试集大小: {len(X_test)}")

# 5. 数据增强函数
def augment_data(X, y, noise_level=0.05, num_augmented=0):
    """
    通过添加噪声和生成合成样本增强数据
    num_augmented: 要生成的额外样本数量 (0表示不增强)
    """
    if num_augmented <= 0:
        return X, y
    
    X_augmented = []
    y_augmented = []
    
    # 复制原始数据
    X_augmented.extend(X)
    y_augmented.extend(y)
    
    # 添加噪声增强
    for _ in range(num_augmented):
        # 随机选择样本
        idx = np.random.randint(0, len(X))
        sample_X = X[idx].copy()
        sample_y = y[idx]
        
        # 添加高斯噪声
        noise_X = np.random.normal(0, noise_level, size=sample_X.shape)
        augmented_X = sample_X + noise_X
        
        # 添加目标变量的轻微噪声
        noise_y = np.random.normal(0, noise_level/2)
        augmented_y = sample_y + noise_y
        
        X_augmented.append(augmented_X)
        y_augmented.append(augmented_y)
    
    return np.array(X_augmented), np.array(y_augmented)

# 应用数据增强 (将训练集扩大一倍)
X_train_aug, y_train_aug = augment_data(
    X_train, y_train, 
    noise_level=0.05, 
    num_augmented=len(X_train)
)

print(f"增强后训练集大小: {len(X_train_aug)}")

# 6. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_aug)
y_train_tensor = torch.FloatTensor(y_train_aug).view(-1, 1)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 7. 定义神经网络模型
class RegressionNN(nn.Module):
    def __init__(self, input_size):
        super(RegressionNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# 8. 初始化模型、损失函数和优化器
input_size = X_train.shape[1]
model = RegressionNN(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20, verbose=True
)

# 9. 训练模型（带早停和验证）
num_epochs = 1000
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 200
patience_counter = 0

print("开始训练...")
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    epoch_train_loss = 0
    
    for batch_X, batch_y in train_loader:
        # 前向传播
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
    
    # 计算平均训练损失
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # 验证模式
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            val_predictions = model(batch_X)
            val_loss = criterion(val_predictions, batch_y)
            epoch_val_loss += val_loss.item()
    
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # 学习率调度
    scheduler.step(avg_val_loss)
    
    # 早停机制
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    # 每50个epoch打印一次进度
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # 检查早停条件
    if patience_counter >= patience:
        print(f"早停于第 {epoch+1} 个epoch")
        break

print("训练完成!")

# 10. 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 11. 评估模型
model.eval()
with torch.no_grad():
    # 在测试集上进行预测
    test_predictions = []
    test_targets = []
    
    for batch_X, batch_y in test_loader:
        batch_pred = model(batch_X)
        test_predictions.extend(batch_pred.numpy())
        test_targets.extend(batch_y.numpy())
    
    # 转换为numpy数组
    y_pred_scaled = np.array(test_predictions)
    y_test_scaled = np.array(test_targets)
    
    # 反标准化
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test_scaled)
    
    # 计算性能指标
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred)
    mae = np.mean(np.abs(y_test_original - y_pred))
    
    # 计算相关系数
    correlation = np.corrcoef(y_test_original.flatten(), y_pred.flatten())[0, 1]
    
    print(f"\n模型性能:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"相关系数: {correlation:.6f}")

# 12. 绘制训练和验证损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()

# 13. 绘制预测值与真实值对比
plt.figure(figsize=(10, 5))
plt.scatter(y_test_original, y_pred, alpha=0.7)
max_val = max(y_test_original.max(), y_pred.max())
min_val = min(y_test_original.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
plt.grid(True)
plt.savefig('predictions_vs_true.png')
plt.show()

# 14. 绘制残差图
residuals = y_test_original - y_pred
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.savefig('residual_plot.png')
plt.show()

# 15. 绘制残差分布
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.grid(True)
plt.savefig('residual_distribution.png')
plt.show()

# 16. 示例预测函数
def predict(W, BMI):
    # 准备输入数据
    input_data = np.array([[W, BMI]])
    input_scaled = scaler_X.transform(input_data)
    input_tensor = torch.FloatTensor(input_scaled)
    
    # 预测
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(input_tensor).numpy()
        prediction = scaler_y.inverse_transform(prediction_scaled)
    
    return prediction[0][0]

# 示例预测
example_W = 15.0
example_BMI = 35.0
prediction = predict(example_W, example_BMI)
print(f"\n示例预测: W={example_W}, BMI={example_BMI} -> Y={prediction:.6f}")

# 17. 保存模型和标准化器
torch.save(model.state_dict(), 'final_model.pth')
import joblib
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("模型和标准化器已保存")

# 18. 特征重要性分析（使用排列重要性）
from sklearn.inspection import permutation_importance

# 创建一个简单的包装器类，使PyTorch模型与sklearn兼容
class TorchModelWrapper:
    def __init__(self, model, scaler_X, scaler_y):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).numpy()
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred.flatten()

# 计算排列重要性
print("\n计算特征重要性...")
wrapper = TorchModelWrapper(model, scaler_X, scaler_y)
X_test_original = scaler_X.inverse_transform(X_test)

# 使用较小的样本子集以加快计算速度
sample_idx = np.random.choice(len(X_test_original), min(100, len(X_test_original)), replace=False)
X_sample = X_test_original[sample_idx]
y_sample = y_test_original[sample_idx].flatten()

result = permutation_importance(
    wrapper, X_sample, y_sample, 
    n_repeats=10, random_state=42, scoring='r2'
)

# 显示特征重要性
feature_names = df.columns[:-1]
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)

print("\n特征重要性 (排列重要性):")
print(importance_df)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'], xerr=importance_df['std'])
plt.xlabel('重要性')
plt.title('特征重要性 (排列重要性)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()