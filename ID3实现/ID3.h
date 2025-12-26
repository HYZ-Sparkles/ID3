/* 2452214 霍宇哲 大数据 */
#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
//#include <algorithm>
#include <set>

using namespace std;

const double EPSILON = 1e-4;

struct TreeNode {
	string Attribute;                // 分裂属性的名字
	map<string, TreeNode*> children; // 分支：值 -> 子节点
	string label;                    // 如果是叶子节点，这里存类别
	bool isLeaf;

	TreeNode() : isLeaf(false) {}
};

class ID3 {
public:
	ID3();
	~ID3(); //动态内存释放

	// 对外接口：传入数据构建树
	// 注意：假设 data 的每一行顺序与 attributeNames 对应，target 是目标列的名称
	void train(const vector<vector<string>>& data, const vector<string>& attributeNames, const string& target);

	// 对外接口：预测新数据
	string predict(const vector<string>& sample) const;

private:
	TreeNode* root;
	vector<string> attr_name;                // 属性名列表
	map<string, vector<string>> attr_values; // 记录每个属性所有可能的取值（用于生成完整分支）
	map<string, int> attr_index_map;         // 属性名 -> 列索引
	int target_index;                        // 目标列的索引

	// 释放树的内存
	void destroyTree(TreeNode* node);

	// 核心递归函数
	TreeNode* buildTree(vector<vector<string>> currentData, vector<bool> usedAttributes);

	// 计算信息熵
	double calculateEntropy(const vector<vector<string>>& data) const;

	// 计算信息增益，返回最佳属性的索引，如果没有正增益返回 -1
	int getBestAttribute(const vector<vector<string>>& data, const vector<bool>& usedAttributes) const;

	// 分割数据：返回第 axis 列的值等于 value 的那些行（并不真正删除列，只是筛选行）
	vector<vector<string>> splitData(const vector<vector<string>>& data, int axis, const string& value) const;

	// 辅助：获取数据集中出现最多的类别（用于处理无法继续分裂的情况）
	string getMajorityLabel(const vector<vector<string>>& data) const;
};