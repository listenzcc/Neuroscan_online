import numpy as np
import scipy.io as sio


# -----------------------
# 训练部分函数定义
# -----------------------

# Step 1: Load training data from a .mat file
def load_data(file_path):
    # data = sio.loadmat(file_path)
    data = np.load(file_path)
    X = data['data']  # Shape: (N, 9), where N is the number of samples
    y = data['labels']  # Shape: (N, ), where N is the number of samples
    return X, y


# Step 2: Train the weight matrix W using the Least Squares method (Linear Regression)
# def train_weight_matrix(X, y):
#     # Add a column of ones to X for the bias term (intercept)
#     X_bias = np.c_[np.ones(X.shape[0]), X]
#     # Calculate the weight matrix W using the Normal Equation: W = (X'X)^-1 * X'y
#     W = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
#     return W

def train_weight_matrix(X, y):
    # 添加偏置项（截距）
    X_bias = np.c_[np.ones(X.shape[0]), X]  # Shape: (N, 10)

    # 使用伪逆求解 W，避免 X'X 不可逆的问题
    W = np.linalg.pinv(X_bias) @ y          # W Shape: (10, )
    return W

# Step 3: Save the trained weight matrix W to a .mat file
def save_weight_matrix(W, output_file):
    sio.savemat(output_file, {'W': W})



# -----------------------
# 基础归属函数定义
# -----------------------

def triangle(x, a, b, c):
    """三角形隶属函数"""
    if x == b:
        return 1.0
    elif x <= a or x >= c:
        return 0.0
    elif x < b:
        return (x - a) / (b - a) if b - a != 0 else 0.0
    else:
        return (c - x) / (c - b) if c - b != 0 else 0.0


def trapezoid(x, a, b, c, d):
    """梯形隶属函数"""
    if b <= x <= c:
        return 1.0
    elif x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a) if b - a != 0 else 0.0
    elif c < x < d:
        return (d - x) / (d - c) if d - c != 0 else 0.0
    else:
        return 0.0


# -----------------------
# 为各节点定义隶属函数
# -----------------------

def get_triangular_memberships(value):
    """
    对于注意力、警觉度、人员绩效预测等使用
    隶属函数：低、 中、 高
    """
    memberships = {}
    memberships['low'] = triangle(value, 0.0, 0.0, 0.5)
    memberships['medium'] = triangle(value, 0.25, 0.5, 0.75)
    memberships['high'] = triangle(value, 0.5, 1.0, 1.0)
    return memberships


def get_cog_memberships(value):
    """
    对于注意力、警觉度、人员绩效预测等使用
    隶属函数：低、 中、 高
    """
    memberships = {}
    memberships['low'] = triangle(value, 0.0, 0.0, 0.4)
    memberships['medium'] = triangle(value, 0.2, 0.5, 0.8)
    memberships['high'] = triangle(value, 0.6, 1.0, 1.0)
    return memberships

def get_diff_conf_memberships(value):
    """
    对于注意力、警觉度、人员绩效预测等使用
    隶属函数：低、 中、 高
    """
    memberships = {}
    memberships['low'] = triangle(value, 0.0, 0.0, 0.4)
    memberships['medium'] = triangle(value, 0.3, 0.5, 0.7)
    memberships['high'] = triangle(value, 0.7, 1.0, 1.0)
    return memberships


def get_environment_memberships(value):
    """
    对于环境因素，采用梯形函数：
      - poor: trapezoid(x, 0, 0, 0.25, 0.5)
      - average: trapezoid(x, 0.25, 0.5, 0.5, 0.75)
      - good: trapezoid(x, 0.5, 0.75, 1.0, 1.0)
    """
    memberships = {}
    memberships['poor'] = trapezoid(value, 0.0, 0.0, 0.25, 0.5)
    memberships['average'] = trapezoid(value, 0.25, 0.5, 0.5, 0.75)
    memberships['good'] = trapezoid(value, 0.5, 0.75, 1.0, 1.0)
    return memberships


def get_urgency_memberships(value):
    """
    对于任务紧急程度，同样采用梯形函数，但返回低、中、高
    """
    memberships = {}
    memberships['low'] = trapezoid(value, 0.0, 0.0, 0.25, 0.5)
    memberships['medium'] = trapezoid(value, 0.25, 0.5, 0.5, 0.75)
    memberships['high'] = trapezoid(value, 0.5, 0.75, 1.0, 1.0)
    return memberships


# -----------------------
# 模糊规则定义
# -----------------------

def fuzzy_rule_performance(att, alert, cog, ext, diff):
    """
    规则 A：人员绩效评价（节点7）
        规则 A1：如果 注意力为 高 且 警觉度为 高 且 认知负荷为 低 且 输入绩效预测值为 中或高 且 任务难度感知为 中等或低 那么 人员绩效评价为 高。(预测的是高、同时状态中至少有两项是好的；或者预测的是中，同时 难度感知是低)
        规则 A2：如果 注意力和警觉度为 低 或 警觉度和认知负荷为 低和高 或 注意力和认知负荷为 低和高 那么 人员绩效评价为 低。
        规则 A3：如果 注意力和警觉度均为 中等 且 认知负荷为 中等 且 输入绩效预测值为 中等或低 且 任务难度感知为 中等或低  那么 人员绩效评价为 中。
    """
    # ruleA1 = min(att['high'], alert['high'], cog['low'], max(ext['high'], ext['medium']), max(diff['low'], diff['medium']))
    ruleA1 = max(min(ext['high'], max(min(att['high'], alert['high']), min(alert['high'], cog['low']), min(cog['low'], att['high']))), min(ext['medium'], diff['low']))
    ruleA2 = max(min(att['low'], cog['high']), min(att['low'], alert['low']), min(alert['low'], cog['high']))
    # ruleA3 = min(att['medium'], alert['medium'], cog['medium'], max(att['low'], alert['low'], cog['low']), max(ext['medium'], ext['low']), max(diff['low'], diff['medium']))
    ruleA3 = min(max(min(att['medium'], alert['medium']), min(alert['medium'], cog['medium']), min(att['medium'], cog['medium'])), max(att['low'], alert['low'], cog['low']), max(ext['medium'], ext['low']), max(diff['low'], diff['medium']))
    membership = {'low': ruleA2, 'medium': ruleA3, 'high': ruleA1}
    return membership


def fuzzy_rule_difficulty(cog, ext, env, urg):
    """
    规则 B：任务难度感知（节点8）
        规则 B1：如果 认知负荷为 高或中 且 输入绩效预测值为 中或低 环境因素为 差 且 任务紧急度为 中等或高 那么 任务难度感知为 高。
        规则 B2：如果 认知负荷为 低或中 且 输入绩效预测值为 中等或高 且 环境因素为 好 那么 任务难度感知为 低。
        规则 B3：如果 认知负荷为 中 且 环境因素为 中 且 任务紧急度为 中等或高 那么 任务难度感知为 中。
    """
    ruleB1 = min(max(cog['high'], cog['medium']), max(ext['low'], ext['medium']), env['poor'])
    ruleB2 = min(max(cog['low'], cog['medium']), max(ext['high'], ext['medium']), env['good'], urg['low'])
    ruleB3 = min(cog['medium'], env['average'], max(urg['high'], urg['medium']))
    membership = {'low': ruleB2, 'medium': ruleB3, 'high': ruleB1}
    return membership


def fuzzy_rule_confidence(perf, diff):
    """
    规则 C：决策信心（节点9）
        规则 C1：如果 人员绩效评价为 高 且 任务难度感知为 低 那么 决策信心为 高。
        规则 C2：如果 人员绩效评价为 低 或 任务难度感知为 高 那么 决策信心为 低。
        规则 C3：如果 人员绩效评价为 中 且 任务难度感知为 中 那么 决策信心为 中。
    """
    ruleC1 = min(perf['high'], diff['low'])
    ruleC2 = max(perf['low'], diff['high'])
    ruleC3 = min(perf['medium'], diff['medium'])
    membership = {'low': ruleC2, 'medium': ruleC3, 'high': ruleC1}
    return membership


def fuzzy_rule_scanning(urg, conf, perf, diff):
    """
    规则 D：机器扫描速率（节点10），输出只有低和高两类
        规则 D1：如果 任务紧急程度为 高 或 （任务难度感知为 低或中 且 决策信心为 中或高 且 绩效评价为 中或高） 那么 无人机飞行速率为 高速。
        规则 D2：如果 任务紧急程度为低 且 任务难度感知为 高 且 决策信心为 低 且 绩效评价为 低 那么 无人机飞行速率为 低速。
        规则 D3：如果 （任务难度感知为 中或低 且 绩效评价为 中或高） 那么 无人机飞行速率倾向于 高速；否则 倾向于 低速。
    """
    ruleD1 = max(urg['high'], min(max(conf['high'], conf['medium']), max(diff['low'], diff['medium']), max(perf['high'], perf['medium'])))
    ruleD2 = min(urg['low'], conf['low'], diff['high'], perf['low'])
    ruleD3 = min(max(perf['medium'], perf['high']), max(diff['medium'], diff['low']))
    membership = {'low': ruleD2, 'high': max(ruleD1, ruleD3)}
    return membership


def defuzzify_scanning(membership):
    """
    对扫描速率输出进行去模糊化。假设：
      - "low" 对应 0
      - "high" 对应 1
    使用加权平均法：
      output = (low*0 + high*1) / (low + high)
    """
    total = membership['low'] + membership['high']
    if total == 0:
        return 0  # 或设定一个默认值
    return membership['high'] / total


def defuzzify_centroid(x_values, membership_values):
    """
    x_values: np.array, shape = (n_samples,)  # 可能的输出值，比如[0, 0.1, 0.2, ..., 1.0]
    membership_values: np.array, shape = (n_samples,)  # 每个输出值对应的隶属度
    """
    numerator = np.sum(x_values * membership_values)
    denominator = np.sum(membership_values)
    if denominator == 0:
        return 0.0  # 避免除零
    return numerator / denominator
# -----------------------
# 测试数据推理流程
# -----------------------

def load_test_data(file_path):
    """加载测试文件，数据形状 (N,6)"""
    data = sio.loadmat(file_path)
    X_test = data['data']  # 每行包含6个输入：注意力, 警觉度, 认知负荷, 人员绩效预测, 环境, 任务紧急度
    return X_test


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def check_convergence(prev_state, current_state, epsilon=1e-5):
    # 判断所有节点的状态变化是否小于阈值
    return np.all(np.abs(current_state - prev_state) < epsilon)


def fcm_update(state, weight_matrix):
    # 使用 FCM 更新公式计算下一个状态
    return sigmoid(np.dot(weight_matrix, state))






class fuzzy_inference_test():

    def __init__(self):
        self.W = np.array([0.221611, 0.059092, 0.064770, 0.156009, 0.070445,
                  -0.238028, 0.658603, 0.302264, -0.362164, 0.563610])
        print('模型加载完成')

    def predict_batch(self, W, X_input):  # X_input shape = (N, 9)
        bias = W[0]
        weights = W[1:]
        return X_input @ weights + bias
    def run_fcm(self, initial_state):
        state = np.array(initial_state)
        state = self.predict_batch(self.W, state)
        return sigmoid(state)

    def fuzzy_inference_for_sample_init(self, sample):
        """
        对单个样本进行模糊推理：
          输入sample为长度为6的一维数组，依次代表：
          1. 注意力分配
          2. 警觉度
          3. 认知负荷
          4. 人员绩效预测值
          5. 环境因素
          6. 任务紧急程度
        """
        # 1. 模糊化各输入
        att = get_triangular_memberships(sample[0])
        alert = get_triangular_memberships(sample[1])
        cog = get_cog_memberships(sample[2])
        ext = get_triangular_memberships(sample[3])
        env = get_environment_memberships(sample[4])
        urg = get_urgency_memberships(sample[5])

        # 2. 内部节点推理
        # 节点8：任务难度感知（由认知负荷和环境决定）
        diff = fuzzy_rule_difficulty(cog, ext, env, urg)
        # diff_value = defuzzify_scanning(diff)
        diff_value = defuzzify_centroid(np.array([0.1, 0.5, 0.9]),
                                        np.array([diff['low'], diff['medium'], diff['high']]))
        # 节点7：人员绩效评价
        perf = fuzzy_rule_performance(att, alert, cog, ext, diff)
        # perf_value = defuzzify_scanning(perf)
        perf_value = defuzzify_centroid(np.array([0.1, 0.5, 0.9]),
                                        np.array([perf['low'], perf['medium'], perf['high']]))
        # 节点9：决策信心（由人员绩效评价和任务难度感知决定）
        conf = fuzzy_rule_confidence(perf, diff)
        # conf_value = defuzzify_scanning(conf)
        conf_value = defuzzify_centroid(np.array([0.1, 0.5, 0.9]),
                                        np.array([conf['low'], conf['medium'], conf['high']]))

        return np.array([perf_value, diff_value, conf_value])

    def fuzzy_inference_for_sample(self, last, sample):
        """
        对单个样本进行模糊推理：
          输入sample为长度为6的一维数组，依次代表：
          1. 注意力分配
          2. 警觉度
          3. 认知负荷
          4. 人员绩效预测值
          5. 环境因素
          6. 任务紧急程度
          7. 人员绩效评价
          8. 任务难度感知
          9. 决策信心
        """
        # 1. 模糊化各输入
        att = get_triangular_memberships(sample[0])
        alert = get_triangular_memberships(sample[1])
        cog = get_cog_memberships(sample[2])
        ext = get_triangular_memberships(sample[3])
        env = get_environment_memberships(sample[4])
        urg = get_urgency_memberships(sample[5])
        perf = get_triangular_memberships(sample[6])
        diff = get_diff_conf_memberships(sample[7])
        conf = get_diff_conf_memberships(sample[8])

        # 3. 输出节点推理（节点10：机器扫描速率），结合任务紧急程度、决策信心和人员绩效评价
        scan_membership = fuzzy_rule_scanning(urg, conf, perf, diff)

        # 4. 去模糊化输出，得到一个数值（0～1之间）
        output_value = defuzzify_scanning(scan_membership)

        output_value2 = (last + output_value) / 2
        # output_value2 = output_value
        # 若需要离散标签（例如：0代表低扫描速率，1代表高扫描速率），可以设置阈值 0.5
        discrete_output = 1 if output_value2 >= 0.5 else 0
        # discrete_output = 1 if last >= 0.5 else 0
        return discrete_output, output_value2

    def run(self, X):

        """对所有样本进行模糊推理，返回预测标签数组"""
        N = X.shape[0]
        state_init = np.zeros((N, 10))
        state_init[:, :6] = X
        for i in range(N):
            state_init[i, 6:9] = self.fuzzy_inference_for_sample_init(X[i, :])

        final_state = self.run_fcm(state_init[:, :9])

        predictions = np.zeros((N,))
        predictions_pr = np.zeros((N,))
        for i in range(N):
            predictions[i], predictions_pr[i] = self.fuzzy_inference_for_sample(final_state[i,], state_init[i, :])
        state_init[:, -1] = predictions
        return predictions, state_init


# def save_predictions(predictions, output_file):
#     sio.savemat(output_file, {'predict_label': predictions})


# -----------------------
# 主函数
# -----------------------

if __name__ == '__main__':

    fcm_model = fuzzy_inference_test()

    # mat = sio.loadmat('./hm_data/test3.mat')
    # X_test = mat['data'][:, :-1]
    # y_test = mat['data'][:, -1]
    # # 进行模糊推理，得到预测标签（离散：0或1）
    # predicted_labels, _ = fcm_model.run(X_test[:, :6])
    #
    # from sklearn.metrics import accuracy_score
    #
    # print(f"Accuracy score: {accuracy_score(y_test, predicted_labels)}")



