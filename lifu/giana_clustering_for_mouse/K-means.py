import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 加载数据
file_path = '/Users/lifushen/Desktop/giana_for_mouse/output_encoded_ab.csv'  # 确保路径正确
data = pd.read_csv(file_path)

# 解析数字串为数值数组，添加错误处理
def parse_floats(text):
    try:
        return np.fromstring(text, sep=' ')
    except ValueError:
        print(f"Warning: Could not convert the following data to floats: {text}")
        return np.array([])  # 返回一个空数组，如果需要可以修改为合适的默认值

data['encoded_array'] = data['encoded_CDR3a+encoded_CDR3b'].apply(parse_floats)

# 过滤掉任何因解析错误而产生空数组的行
data = data[data['encoded_array'].map(len) > 0]

# 构建一个新的 DataFrame 用于聚类，其中每个元素是一个特征向量
feature_data = np.vstack(data['encoded_array'].values)

# 应用 k-means 聚类，显式设置 n_init 参数
kmeans = KMeans(n_clusters=50, random_state=0, n_init=10).fit(feature_data)

# 将聚类结果添加到原始 DataFrame
data['cluster_id'] = kmeans.labels_

# 计算聚类指标
silhouette_avg = silhouette_score(feature_data, kmeans.labels_)
calinski_harabasz = calinski_harabasz_score(feature_data, kmeans.labels_)
davies_bouldin = davies_bouldin_score(feature_data, kmeans.labels_)

print('Silhouette Score:', silhouette_avg)
print('Calinski-Harabasz Score:', calinski_harabasz)
print('Davies-Bouldin Score:', davies_bouldin)

# 保存修改后的 DataFrame 到新的 CSV 文件
output_file_path = '/Users/lifushen/Desktop/giana_for_mouse/output_file.csv'  # 输出文件路径
data.to_csv(output_file_path, index=False, columns=['encoded_CDR3a+encoded_CDR3b', 'antigen.epitope', 'cluster_id'])

print('聚类完成，结果已保存至:', output_file_path)
