import pandas as pd
import numpy as np
import matplotlib
# 使用非交互式后端，避免显示问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")  # 忽略字体警告

# 设置英文默认字体
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
sns.set_style("whitegrid")  # 设置seaborn样式

# 中文列名到英文列名的完整映射（包含所有列）
COLUMN_MAPPING = {
    '序号': 'serial_number',
    '孕妇代码': 'pregnant_woman_id',
    '年龄': 'age',
    '身高': 'height',
    '体重': 'weight',
    '末次月经': 'last_menstrual_period',
    'IVF妊娠': 'ivf_pregnancy',
    '检测日期': 'detection_date',
    '检测抽血次数': 'number_of_blood_tests',
    '检测孕周': 'gestational_week',
    '孕妇BMI': 'maternal_bmi',
    '原始读段数': 'raw_reads',
    '在参考基因组上比对的比例': 'alignment_rate_to_reference_genome',
    '重复读段的比例': 'duplicate_reads_rate',
    '唯一比对的读段数  ': 'uniquely_aligned_reads',
    'GC含量': 'gc_content',
    '13号染色体的Z值': 'chromosome_13_z_score',
    '18号染色体的Z值': 'chromosome_18_z_score',
    '21号染色体的Z值': 'chromosome_21_z_score',
    'X染色体的Z值': 'chromosome_x_z_score',
    'Y染色体的Z值': 'chromosome_y_z_score',
    'Y染色体浓度': 'y_chromosome_concentration',
    'X染色体浓度': 'x_chromosome_concentration',
    '13号染色体的GC含量': 'chromosome_13_gc_content',
    '18号染色体的GC含量': 'chromosome_18_gc_content',
    '21号染色体的GC含量': 'chromosome_21_gc_content',
    '被过滤掉读段数的比例': 'filtered_reads_rate',
    '染色体的非整倍体': 'chromosomal_aneuploidy',
    '怀孕次数': 'number_of_pregnancies',
    '生产次数': 'number_of_deliveries',
    '胎儿是否健康': 'fetal_health_status'
}

def preprocess_and_analyze():
    # 定义文件路径
    excel_path = os.path.join("data", "ori.xlsx")
    csv_path = os.path.join("data", "processed_data_english.csv")
    
    # 创建data目录（如果不存在）
    if not os.path.exists("data"):
        os.makedirs("data")
    
    try:
        # 读取Excel文件
        print(f"Reading Excel file: {excel_path}")
        df = pd.read_excel(excel_path)
        
        # 将列名改为英文
        # 只映射数据中存在的列，避免KeyError
        existing_columns = df.columns.intersection(COLUMN_MAPPING.keys())
        column_mapping_subset = {col: COLUMN_MAPPING[col] for col in existing_columns}
        df = df.rename(columns=column_mapping_subset)
        
        # 数据预处理
        print("Starting data preprocessing...")
        
        # 转换日期格式
        if 'last_menstrual_period' in df.columns:
            df['last_menstrual_period'] = pd.to_datetime(df['last_menstrual_period'], errors='coerce')
        
        if 'detection_date' in df.columns:
            # 处理数字格式的日期（如20230429）
            if pd.api.types.is_integer_dtype(df['detection_date']):
                df['detection_date'] = pd.to_datetime(df['detection_date'].astype(str), format='%Y%m%d', errors='coerce')
            else:
                df['detection_date'] = pd.to_datetime(df['detection_date'], errors='coerce')
        
        # 处理孕周列，提取数值
        if 'gestational_week' in df.columns:
            df['gestational_week_numeric'] = df['gestational_week'].str.extract('(\d+\.?\d*)').astype(float)
        
        # 转换分类变量为英文
        if 'ivf_pregnancy' in df.columns:
            df['ivf_pregnancy'] = df['ivf_pregnancy'].replace({'自然受孕': 'natural_conception', 'IVF': 'ivf'})
        
        if 'fetal_health_status' in df.columns:
            df['fetal_health_status'] = df['fetal_health_status'].replace({'是': 'healthy', '否': 'unhealthy'})
        
        # 新增：计算染色体浓度的最早达标时间
        if all(col in df.columns for col in ['gestational_week_numeric', 'y_chromosome_concentration', 'x_chromosome_concentration']):
            print("Calculating earliest qualified week for chromosome concentration...")
            
            # 创建一个函数来判断单次检测是否达标
            def is_qualified(row):
                # 检查孕周是否在有效范围内
                if pd.isna(row['gestational_week_numeric']) or row['gestational_week_numeric'] < 10 or row['gestational_week_numeric'] > 25:
                    return False
                
                # 检查Y染色体浓度是否达标（男胎）- 使用0.04而不是4%
                if not pd.isna(row['y_chromosome_concentration']) and row['y_chromosome_concentration'] >= 0.04:
                    return True
                
                # 检查X染色体浓度是否正常（女胎）- 这里假设X染色体浓度没有异常的标准是浓度不为0且在正常范围内
                # 注意：实际应用中可能需要更精确的标准
                if not pd.isna(row['x_chromosome_concentration']) and row['x_chromosome_concentration'] > 0:
                    return True
                
                return False
            
            # 应用函数判断每次检测是否达标
            df['is_qualified'] = df.apply(is_qualified, axis=1)
            
            # 按孕妇分组，找到每个孕妇最早达标的孕周
            earliest_qualified_week = df[df['is_qualified']].groupby('pregnant_woman_id')['gestational_week_numeric'].min().reset_index()
            earliest_qualified_week = earliest_qualified_week.rename(
                columns={'gestational_week_numeric': 'earliest_qualified_week'}
            )
            
            # 将最早达标时间合并回原始数据
            df = df.merge(earliest_qualified_week, on='pregnant_woman_id', how='left')
            
            # 对于没有达标记录的孕妇，将最早达标时间设为NaN
            df.loc[df['earliest_qualified_week'].isna(), 'earliest_qualified_week'] = np.nan
            
            print("Earliest qualified week calculation completed.")
        else:
            print("Warning: Required columns for chromosome concentration analysis not found.")
        
        # 处理缺失值
        print(f"Missing values in dataset:\n{df.isnull().sum()}")
        
        # 保存为CSV（英文列名）
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Preprocessed data saved to: {csv_path}")
        
        # 相关性分析 - 选择数值型列
        # 仅使用is_qualified为True的数据
        qualified_df = df[df['is_qualified'] == True] if 'is_qualified' in df.columns else df
        numeric_cols = qualified_df.select_dtypes(include=['float64', 'int64']).columns
        
        # 确保Y染色体浓度在数值列中
        target_col = 'y_chromosome_concentration'
        if target_col not in numeric_cols:
            print(f"Warning: Could not find '{target_col}' column, cannot perform correlation analysis")
            return
        
        # 计算相关性
        print(f"Calculating correlations with {target_col} using only qualified data...")
        correlations = qualified_df[numeric_cols].corr()[target_col].sort_values(ascending=False)
        
        # 显示相关性结果
        print(f"\nCorrelations with {target_col} (from highest to lowest, qualified data only):")
        print(correlations)
        
        # 创建第一个图：相关性热力图（仅使用达标数据）
        plt.figure(figsize=(12, 10))
        corr_subset = qualified_df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_subset, dtype=bool))
        sns.heatmap(corr_subset, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                   vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
        plt.title('Correlation Heatmap of All Metrics (Qualified Data Only)', fontsize=16)
        plt.tight_layout()
        
        # 保存第一个图表
        plt.savefig(os.path.join("data", "correlation_heatmap_qualified.png"), dpi=300, bbox_inches='tight')
        print("Correlation heatmap (qualified data only) saved to: data/correlation_heatmap_qualified.png")
        plt.close()  # 关闭图形，避免内存泄漏
        
        # 创建第二个图：相关性条形图（仅使用达标数据）
        plt.figure(figsize=(12, 6))
        # 确保只使用英文列名
        correlations_english = correlations.copy()
        correlations_english.index = [COLUMN_MAPPING.get(col, col) for col in correlations_english.index]
        
        # 过滤掉目标列本身
        correlations_to_plot = correlations_english.drop(target_col, errors='ignore')
        correlations_to_plot.plot(kind='bar')
        plt.title(f'Correlation with {target_col} (Qualified Data Only)', fontsize=16)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存第二个图表
        plt.savefig(os.path.join("data", "y_chromosome_correlations_qualified.png"), dpi=300, bbox_inches='tight')
        print("Correlation bar chart (qualified data only) saved to: data/y_chromosome_correlations_qualified.png")
        plt.close()  # 关闭图形，避免内存泄漏
        
        # 输出强相关性的指标（绝对值大于0.5）
        strong_correlations = correlations[abs(correlations) > 0.1]
        if len(strong_correlations) > 1:  # 排除自身
            print(f"\nMetrics with strong correlation to {target_col} (|r| > 0.5, qualified data only):")
            print(strong_correlations.drop(target_col))
        else:
            print(f"\nNo strong correlations found with {target_col} (|r| > 0.5, qualified data only)")
            
    except FileNotFoundError:
        print(f"Error: File not found - {excel_path}")
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    preprocess_and_analyze()