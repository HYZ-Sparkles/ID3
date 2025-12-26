/* 2452214 霍宇哲 大数据 */
#include "ID3.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <iomanip>
#include <map>

using namespace std;

const int SEED = 42;    //分解数据集成为训练集和测试集的种子
// 函数声明
vector<vector<string>> loadData(const string& filename, vector<string>& attributeNames);
vector<string> splitLine(const string& line, char delimiter);
double calculateAccuracy(const ID3& tree, const vector<vector<string>>& testData, int targetIndex);
string discretizeGrade(const string& gradeStr);
void printConfusionMatrix(const ID3& tree, const vector<vector<string>>& testData, int targetIndex);

int main() {
	cout << "==========================================" << endl;
	cout << "       学生成绩预测系统 - ID3决策树" << endl;
	cout << "       使用数学课程数据 (student-mat)" << endl;
	cout << "==========================================" << endl;

	// 1. 加载数据
	cout << "\n[1] 正在加载数据..." << endl;
	vector<string> attributeNames;
	vector<vector<string>> allData = loadData("student-mat.csv", attributeNames);

	if (allData.empty()) {
		cerr << "错误: 无法加载数据文件!" << endl;
		return 1;
	}

	cout << "     成功加载 " << allData.size() << " 条学生记录" << endl;
	cout << "     属性数量: " << attributeNames.size() << endl;

	// 显示属性列表
	cout << "     属性列表: ";
	for (size_t i = 0; i < min((size_t)10, attributeNames.size()); i++) {
		cout << attributeNames[i] << " ";
	}
	if (attributeNames.size() > 10) cout << "...";
	cout << endl;

	// 2. 预处理数据
	cout << "\n[2] 正在预处理数据..." << endl;

	// 设置目标属性为最终成绩G3（第32列，索引32）
	string targetAttribute = "G3";
	int targetIndex = -1;
	for (size_t i = 0; i < attributeNames.size(); i++) {
		if (attributeNames[i] == targetAttribute) {
			targetIndex = (int)i;
			break;
		}
	}

	if (targetIndex == -1) {
		cerr << "错误: 找不到目标属性 '" << targetAttribute << "'" << endl;
		cerr << "可用的属性: ";
		for (const auto& attr : attributeNames) {
			cerr << attr << " ";
		}
		cerr << endl;
		return 1;
	}

	cout << "     目标属性: " << targetAttribute << " (索引: " << targetIndex << ")" << endl;
	cout << "     目标属性位置: 第" << targetIndex + 1 << "列（从1开始计数）" << endl;

	// 对G3成绩进行离散化
	map<string, int> gradeDistribution;
	for (auto& row : allData) {
		string originalGrade = row[targetIndex];
		string discretizedGrade = discretizeGrade(originalGrade);
		row[targetIndex] = discretizedGrade;
		gradeDistribution[discretizedGrade]++;
	}

	cout << "     已将G3成绩离散化为: 不及格(0-9), 中等(10-14), 优秀(15-20)" << endl;
	cout << "     成绩分布统计:" << endl;
	for (const auto& pair : gradeDistribution) {
		double percentage = (double)pair.second / allData.size() * 100;
		cout << "        " << pair.first << ": " << pair.second << "人 ("
			<< fixed << setprecision(1) << percentage << "%)" << endl;
	}

	// 3. 划分训练集和测试集
	cout << "\n[3] 正在划分训练集和测试集..." << endl;

	// 为了可重复性，使用固定种子
	unsigned int seed = SEED; // 固定种子以确保可重复性
	mt19937 g(seed);
	shuffle(allData.begin(), allData.end(), g); //随机划分allData成为训练集和测试集

	// 80%训练，20%测试
	size_t trainSize = allData.size() * 0.8;
	vector<vector<string>> trainData(allData.begin(), allData.begin() + trainSize);
	vector<vector<string>> testData(allData.begin() + trainSize, allData.end());

	cout << "     训练集大小: " << trainData.size() << " 条记录" << endl;
	cout << "     测试集大小: " << testData.size() << " 条记录" << endl;
	cout << "     划分比例: 80%训练 / 20%测试" << endl;

	// 4. 训练决策树
	cout << "\n[4] 正在训练ID3决策树..." << endl;
	ID3 decisionTree;
	try {
		decisionTree.train(trainData, attributeNames, targetAttribute);
		cout << "     决策树训练完成!" << endl;
	}
	catch (const exception& e) {
		cerr << "     训练失败: " << e.what() << endl;
		return 1;
	}

	// 5. 测试模型性能
	cout << "\n[5] 正在测试模型性能..." << endl;
	double accuracy = calculateAccuracy(decisionTree, testData, targetIndex);
	cout << fixed << setprecision(2);
	cout << "     测试集准确率: " << accuracy * 100 << "%" << endl;

	// 显示混淆矩阵
	printConfusionMatrix(decisionTree, testData, targetIndex);

	// 6. 进行一些预测示例
	cout << "\n[6] 预测示例:" << endl;
	cout << "------------------------------------------" << endl;

	if (!testData.empty()) {
		// 预测前5个测试样本
		for (int i = 0; i < min(5, (int)testData.size()); i++) {
			// 创建预测用的样本（包含所有属性，目标属性保持原值）
			vector<string> predictionSample = testData[i];
			string actual = testData[i][targetIndex];

			string predicted = decisionTree.predict(predictionSample);

			cout << "样本 " << (i + 1) << ":" << endl;
			cout << "  实际G3等级: " << actual << endl;
			cout << "  预测G3等级: " << predicted;

			if (predicted == actual) {
				cout << "  正确" << endl;
			}
			else if (predicted.find("Error") != string::npos ||
				predicted.find("错误") != string::npos ||
				predicted.find("未知") != string::npos ||
				predicted.find("Empty") != string::npos) {
				cout << "  预测错误: " << predicted << endl;
			}
			else {
				cout << "  错误 (预测为" << predicted << ")" << endl;
			}

			// 显示一些关键特征
			cout << "  关键特征: ";
			cout << "学校=" << testData[i][0] << ", ";
			cout << "性别=" << testData[i][1] << ", ";
			cout << "年龄=" << testData[i][2] << ", ";
			cout << "学习时间=" << testData[i][13] << ", ";
			cout << "失败次数=" << testData[i][14] << ", ";
			cout << "缺勤=" << testData[i][29] << ", ";
			cout << "G1成绩=" << testData[i][30] << ", ";
			cout << "G2成绩=" << testData[i][31] << endl;
			cout << endl;
		}
	}

	cout << "\n==========================================" << endl;
	cout << "           程序执行完毕" << endl;
	cout << "==========================================" << endl;

	return 0;
}

// 加载CSV数据
vector<vector<string>> loadData(const string& filename, vector<string>& attributeNames) {
	vector<vector<string>> data;
	ifstream file(filename, ios::in);

	if (!file.is_open()) {
		cerr << "错误: 无法打开文件 " << filename << endl;
		cerr << "请确保文件存在于当前目录: " << endl;
		return data;
	}

	string line;
	int lineCount = 0;

	// 读取属性名（第一行）
	if (getline(file, line)) {
		lineCount++;
		attributeNames = splitLine(line, ';');
		cout << "     读取到 " << attributeNames.size() << " 个属性名" << endl;
	}

	// 读取数据行
	while (getline(file, line)) {
		lineCount++;
		vector<string> row = splitLine(line, ';');

		// 检查列数是否匹配
		if (row.size() != attributeNames.size()) {
			cerr << "警告: 第 " << lineCount << " 行有 " << row.size()
				<< " 列，但期望 " << attributeNames.size() << " 列" << endl;
			cerr << "行内容: " << line << endl;

			// 尝试修复：如果列数不足，填充空值
			while (row.size() < attributeNames.size()) {
				row.push_back("");
			}

			// 如果列数过多，截断
			if (row.size() > attributeNames.size()) {
				row.resize(attributeNames.size());
			}
		}

		data.push_back(row);
	}

	file.close();
	cout << "     成功读取 " << data.size() << " 行数据" << endl;
	return data;
}

// 分割CSV行
vector<string> splitLine(const string& line, char delimiter) {
	vector<string> tokens;
	string token;
	istringstream tokenStream(line);

	while (getline(tokenStream, token, delimiter)) {
		// 移除可能的引号
		if (!token.empty()) {
			if (token.front() == '"' && token.back() == '"') {
				token = token.substr(1, token.size() - 2);
			}
			else if (token.front() == '\'' && token.back() == '\'') {
				token = token.substr(1, token.size() - 2);
			}
		}
		tokens.push_back(token);
	}

	return tokens;
}

// 离散化成绩
string discretizeGrade(const string& gradeStr) {
	if (gradeStr.empty()) {
		return "未知";
	}

	int grade = stoi(gradeStr);
	if (grade >= 0 && grade <= 9) {
		return "不及格";
	}
	else if (grade >= 10 && grade <= 14) {
		return "中等";
	}
	else if (grade >= 15 && grade <= 20) {
		return "优秀";
	}
	else {
		return "无效成绩";
	}
}

// 计算准确率
double calculateAccuracy(const ID3& tree, const vector<vector<string>>& testData, int targetIndex) {
	if (testData.empty()) return 0.0;

	int correct = 0;
	int total = 0;

	for (const auto& sample : testData) {
		string actual = sample[targetIndex];
		string predicted = tree.predict(sample);

		// 检查预测是否有效
		if (predicted == "Empty Tree" ||
			predicted.find("Error") != string::npos ||
			predicted.find("错误") != string::npos ||
			predicted.find("未知") != string::npos) {
			// 预测失败，跳过
			continue;
		}

		if (predicted == actual) {
			correct++;
		}
		total++;
	}

	return total > 0 ? (double)correct / total : 0.0;
}

// 打印混淆矩阵
void printConfusionMatrix(const ID3& tree, const vector<vector<string>>& testData, int targetIndex) {
	if (testData.empty()) return;

	map<string, map<string, int>> confusionMatrix;
	vector<string> categories = { "不及格", "中等", "优秀" };

	// 初始化混淆矩阵
	for (const auto& actual : categories) {
		for (const auto& predicted : categories) {
			confusionMatrix[actual][predicted] = 0;
		}
	}

	int totalPredictions = 0;
	int validPredictions = 0;

	for (const auto& sample : testData) {
		totalPredictions++;
		string actual = sample[targetIndex];
		string predicted = tree.predict(sample);

		// 只统计有效的预测
		if (predicted == "不及格" || predicted == "中等" || predicted == "优秀") {
			confusionMatrix[actual][predicted]++;
			validPredictions++;
		}
	}

	cout << "\n     混淆矩阵 (基于" << validPredictions << "/" << totalPredictions << "个有效预测):" << endl;
	cout << "实际\\预测    不及格    中等    优秀" << endl;

	for (const auto& actual : categories) {
		cout << actual;
		if (actual == "不及格") cout << "   ";
		else if (actual == "中等") cout << "     ";
		else if (actual == "优秀") cout << "     ";

		for (const auto& predicted : categories) {
			cout << setw(8) << confusionMatrix[actual][predicted];
		}
		cout << endl;
	}
}