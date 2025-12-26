/* 2452214 霍宇哲 大数据 */
#include "./ID3.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

/*
该文件只调用ID3.cpp和ID3.h提供的ID3算法(具体实现在对应文件中完成)
同时该文件会读取数据集文件并用ID3进行分类。
*/

// 读取CSV文件
vector<vector<string>> readCSV(const string& filename, char delimiter = ',') {
    vector<vector<string>> data;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return data;
    }

    while (getline(file, line)) {
        // 跳过空行
        if (line.empty()) continue;

        vector<string> row;
        stringstream ss(line);
        string cell;

        while (getline(ss, cell, delimiter)) {
            row.push_back(cell);
        }

        data.push_back(row);
    }

    file.close();
    return data;
}

// 将连续属性离散化（简单的等宽分箱）
vector<string> discretizeNumericAttribute(const vector<double>& values, int bins = 3) {
    vector<string> discretizedValues;

    if (values.empty()) return discretizedValues;

    // 找到最小值和最大值
    double minVal = *min_element(values.begin(), values.end());
    double maxVal = *max_element(values.begin(), values.end());
    double binWidth = (maxVal - minVal) / bins;

    // 对每个值进行分箱
    for (double val : values) {
        if (val == maxVal) {
            discretizedValues.push_back("bin_" + to_string(bins - 1));
        }
        else {
            int binIndex = static_cast<int>((val - minVal) / binWidth);
            discretizedValues.push_back("bin_" + to_string(binIndex));
        }
    }

    return discretizedValues;
}

// 准备数据集：将连续属性离散化
vector<vector<string>> prepareDataset(const vector<vector<string>>& rawData) {
    if (rawData.empty()) return {};

    vector<vector<string>> preparedData;
    int numFeatures = rawData[0].size() - 1; // 最后一列是类别

    // 为每个特征（除了类别）存储数值
    vector<vector<double>> numericFeatures(numFeatures);

    // 收集数值
    for (const auto& row : rawData) {
        if (row.size() < numFeatures + 1) continue;

        for (int i = 0; i < numFeatures; i++) {
            try {
                double value = stod(row[i]);
                numericFeatures[i].push_back(value);
            }
            catch (const std::exception& e) {
                numericFeatures[i].push_back(0.0);
            }
        }
    }

    // 离散化每个特征
    vector<vector<string>> discretizedFeatures(numFeatures);
    for (int i = 0; i < numFeatures; i++) {
        discretizedFeatures[i] = discretizeNumericAttribute(numericFeatures[i], 3);
    }

    // 构建离散化后的数据集
    for (size_t rowIdx = 0; rowIdx < rawData.size(); rowIdx++) {
        const auto& row = rawData[rowIdx];
        if (row.size() < numFeatures + 1) continue;

        vector<string> newRow;

        // 添加离散化后的特征值
        for (int i = 0; i < numFeatures; i++) {
            newRow.push_back(discretizedFeatures[i][rowIdx]);
        }

        // 添加类别（最后一个元素）
        newRow.push_back(row.back());

        preparedData.push_back(newRow);
    }

    return preparedData;
}

// 划分训练集和测试集（80%训练，20%测试）
void splitDataset(const vector<vector<string>>& data,
    vector<vector<string>>& trainData,
    vector<vector<string>>& testData,
    float trainRatio = 0.8) {
    if (data.empty()) return;

    // 复制数据并打乱顺序
    vector<vector<string>> shuffledData = data;
    random_shuffle(shuffledData.begin(), shuffledData.end());

    // 计算训练集大小
    size_t trainSize = static_cast<size_t>(shuffledData.size() * trainRatio);

    // 划分数据集
    trainData.assign(shuffledData.begin(), shuffledData.begin() + trainSize);
    testData.assign(shuffledData.begin() + trainSize, shuffledData.end());
}

// 打印数据集信息
void printDatasetInfo(const vector<vector<string>>& data, const string& name) {
    cout << "数据集: " << name << endl;
    cout << "样本数量: " << data.size() << endl;
    if (!data.empty()) {
        cout << "特征数量: " << data[0].size() - 1 << endl;
        cout << "类别数量: ";

        // 统计类别
        vector<string> classes;
        for (const auto& row : data) {
            string className = row.back();
            if (find(classes.begin(), classes.end(), className) == classes.end()) {
                classes.push_back(className);
            }
        }
        cout << classes.size() << " (";
        for (size_t i = 0; i < classes.size(); i++) {
            cout << classes[i];
            if (i < classes.size() - 1) cout << ", ";
        }
        cout << ")" << endl;
    }
    cout << "------------------------" << endl;
}

// 测试分类器性能
void testClassifier(ID3& classifier, const vector<vector<string>>& testData) {
    if (testData.empty()) return;

    int correct = 0;
    int total = testData.size();

    for (const auto& sample : testData) {
        // 准备特征向量（去掉类别标签）
        vector<string> features(sample.begin(), sample.end() - 1);
        string trueLabel = sample.back();

        // 预测
        string predictedLabel = classifier.predict(features);

        // 检查是否正确
        if (predictedLabel == trueLabel) {
            correct++;
        }
    }

    double accuracy = static_cast<double>(correct) / total * 100;
    cout << "测试结果:" << endl;
    cout << "  总样本数: " << total << endl;
    cout << "  正确分类: " << correct << endl;
    cout << "  准确率: " << accuracy << "%" << endl;
}

int main() {
    // 1. 读取Iris数据集
    cout << "正在读取Iris数据集..." << endl;
    vector<vector<string>> rawData = readCSV("iris.data");

    if (rawData.empty()) {
        cerr << "错误: 无法读取数据集或数据集为空！" << endl;
        cerr << "请确保iris.data文件位于当前目录下。" << endl;
        return 1;
    }

    cout << "成功读取 " << rawData.size() << " 条数据" << endl;

    // 2. 预处理数据（离散化连续特征）
    cout << "\n正在预处理数据（离散化连续特征）..." << endl;
    vector<vector<string>> dataset = prepareDataset(rawData);

    if (dataset.empty()) {
        cerr << "错误: 数据预处理失败！" << endl;
        return 1;
    }

    // 3. 划分训练集和测试集
    cout << "\n划分训练集和测试集（80%/20%）..." << endl;
    vector<vector<string>> trainData, testData;
    splitDataset(dataset, trainData, testData, 0.8);

    // 4. 打印数据集信息
    printDatasetInfo(trainData, "训练集");
    printDatasetInfo(testData, "测试集");

    // 5. 准备训练数据（特征和标签分开）
    vector<vector<string>> trainFeatures;
    vector<string> trainLabels;

    for (const auto& sample : trainData) {
        // 特征（去掉最后一个元素）
        vector<string> features(sample.begin(), sample.end() - 1);
        trainFeatures.push_back(features);

        // 标签（最后一个元素）
        trainLabels.push_back(sample.back());
    }

    // 6. 创建并训练ID3决策树
    cout << "\n正在训练ID3决策树..." << endl;
    ID3 classifier;

    // 定义特征名称（根据Iris数据集）
    vector<string> featureNames = {
        "sepal_length", "sepal_width", "petal_length", "petal_width"
    };

    // 训练分类器
    classifier.train(trainFeatures, trainLabels, featureNames);

    // 7. 显示决策树结构（可选）
    cout << "\n决策树结构:" << endl;
    classifier.printTree();

    // 8. 在测试集上评估分类器
    cout << "\n在测试集上评估分类器性能..." << endl;
    testClassifier(classifier, testData);

    // 9. 示例：对新样本进行分类
    cout << "\n示例预测:" << endl;

    // 示例1：典型的Iris-setosa
    vector<string> sample1 = { "bin_0", "bin_1", "bin_0", "bin_0" };
    cout << "样本1 [短花萼, 中花萼宽, 短花瓣, 窄花瓣]: "
        << classifier.predict(sample1) << endl;

    // 示例2：典型的Iris-versicolor
    vector<string> sample2 = { "bin_1", "bin_1", "bin_1", "bin_1" };
    cout << "样本2 [中花萼, 中花萼宽, 中花瓣, 中花瓣宽]: "
        << classifier.predict(sample2) << endl;

    // 示例3：典型的Iris-virginica
    vector<string> sample3 = { "bin_2", "bin_1", "bin_2", "bin_2" };
    cout << "样本3 [长花萼, 中花萼宽, 长花瓣, 宽花瓣]: "
        << classifier.predict(sample3) << endl;

    cout << "\nID3分类完成！" << endl;

    return 0;
}