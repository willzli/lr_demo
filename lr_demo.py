# 环境: python3.6
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split

def get_prf(label_score_list, threshold, pos_label):
    acc_cnt = 0
    pred_cnt = 0
    label_cnt = 0
    for label, score in label_score_list:
        if label == pos_label:
            label_cnt += 1
        if score > threshold:
            pred_cnt += 1
        if label == pos_label and score > threshold:
            acc_cnt += 1
    p = float(acc_cnt) / float(pred_cnt) if pred_cnt > 0 else -1.0
    r = float(acc_cnt) / float(label_cnt) if label_cnt > 0 else -1.0
    f = (2.0 * p * r) / (p + r) if p > 0.0 and r > 0.0 else -1.0

    return p, r, f


if __name__ == "__main__":

    # 鸢尾花数据集获取，实际项目不用这块
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    # 转换为ctr/cvr预估样本(多分类->二分类)
    for i in range(len(Y)):
        if Y[i] == 2:
            Y[i] = 1
        else:
            Y[i] = 0
    print(X, type(X))
    #print(Y, type(Y))

    # 拆分训练集和测试集，只是举个例子，实际项目可以离线先把训练集、验证集处理好，会有一些数据策略，单独处理比较好
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    #print(X_test, Y_test)

    # 分类器初始化
    # penalty惩罚项, 一般用l2
    # C 过拟合系数, 越大过拟合风险越大
    # verbose=1显示训练过程
    logreg = linear_model.LogisticRegression(solver='liblinear', penalty='l2', C=1.0, n_jobs=-1, verbose=0)

    # 分类器训练
    logreg.fit(X_train, Y_train)

    # 分类器预测
    prepro = logreg.predict_proba(X_test)
    print(prepro, type(prepro))

    # 模型评估acc
    acc = logreg.score(X_test, Y_test)
    print(acc)

    # 效果评估这块是重点，决定了整个项目的前进方向
    # 实际CTR/CVR预估项目中，acc(预测对的/预测次数)不重要，要看准/召，不同的阈值会产生不同的准确(p)/召回(r)/F1值，准确的概念：预测为正样本且对的/预测为正样本的，召回的概念：预测为正样本且对的/测试集中正样本
    label_score_list = []
    for i in range(len(Y_test)):
        label_score_list.append((Y_test[i], prepro[i][1]))
    # from 0.05 to 0.95
    threshold = 0.0
    print("p/r/f1:")
    for i in range(19):
        threshold = threshold + 0.05
        p, r, f = get_prf(label_score_list, threshold, 1)
        prf_str = "thd: %.2f, p: %.5f, r: %.5f, f: %.5f" % (threshold, p, r, f)
        print(prf_str)



