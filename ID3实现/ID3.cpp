/* 2452214 霍宇哲 大数据 */
#include "ID3.h"

// 构造函数
ID3::ID3() : root(nullptr), target_index(-1) {}

// 析构函数
ID3::~ID3() {
	destroyTree(root);
}

// 递归释放内存
void ID3::destroyTree(TreeNode* node) {
	if (node) {
		for (auto& pair : node->children) {
			destroyTree(pair.second);
		}
		delete node;
	}
}

// 训练入口
void ID3::train(const vector<vector<string>>& data, const vector<string>& attributeNames, const string& target) {
	// 1. 初始化元数据
	this->attr_name = attributeNames;
	this->target_index = -1;
	this->attr_index_map.clear();
	this->attr_values.clear();

	// 2. 建立 属性名 -> 索引 的映射，并找到 Target 列
	for (int i = 0; i < attributeNames.size(); ++i) {
		attr_index_map[attributeNames[i]] = i;
		if (attributeNames[i] == target) {
			target_index = i;
		}
	}

	if (target_index == -1) {
		cerr << "目标属性没有出现在属性中" << endl;
		return;
	}

	// 3. 预处理：记录每个属性所有可能出现的唯一值
	// 这对于处理测试集中出现、但训练集当前分支没出现的属性值很重要
	for (int j = 0; j < attributeNames.size(); ++j) {
		if (j == target_index)
			continue; // 跳过目标列
		set<string> unique_vals;
		for (const auto& row : data) {
			unique_vals.insert(row[j]);
		}
		// 转存到 vector
		for (const string& v : unique_vals) {
			attr_values[attributeNames[j]].push_back(v);
		}
	}

	// 4. 初始化属性使用状态 (全部为 false)
	vector<bool> usedAttributes(attributeNames.size(), false);
	usedAttributes[target_index] = true; // 目标列本身不参与分裂

	// 5. 开始递归构建
	root = buildTree(data, usedAttributes);
}

// 核心递归构建函数
TreeNode* ID3::buildTree(vector<vector<string>> currentData, vector<bool> usedAttributes) {
	TreeNode* node = new TreeNode();

	// --- 1. 终止条件检查 ---

	// A. 如果数据集为空
	if (currentData.empty()) {
		node->isLeaf = true;
		node->label = "数据集是空的"; // 或者取父节点的多数类
		return node;
	}

	// B. 如果数据集所有样本属于同一类 (纯净)
	string firstLabel = currentData[0][target_index];
	bool isPure = true;
	for (const auto& row : currentData) {
		if (row[target_index] != firstLabel) {
			isPure = false;
			break;
		}
	}
	if (isPure) {
		node->isLeaf = true;
		node->label = firstLabel;
		return node;
	}

	// C. 如果所有属性都用完了，或者没有可用的特征
	bool allUsed = true;
	for (bool u : usedAttributes) {
		if (!u) {
			allUsed = false;
			break;
		}
	}
	if (allUsed) {
		node->isLeaf = true;
		node->label = getMajorityLabel(currentData); // 多数表决
		return node;
	}

	// --- 2. 寻找最佳分裂属性 ---
	int bestAttrIndex = getBestAttribute(currentData, usedAttributes);

	// 如果无法找到有增益的属性 (增益极小)，也停止
	if (bestAttrIndex == -1) {
		node->isLeaf = true;
		node->label = getMajorityLabel(currentData);
		return node;
	}

	// --- 3. 构建节点 ---
	node->Attribute = attr_name[bestAttrIndex];

	// 标记该属性已使用 (注意：usedAttributes 是按值传递的，所以这里修改只影响当前子树)
	usedAttributes[bestAttrIndex] = true;

	// --- 4. 递归生成子节点 ---
	// 遍历该属性的所有可能取值 (从全局元数据 attr_values 中取，而不是只取当前数据的 unique)
	// 这样可以防止测试数据中有 valid 的值，但当前分支的训练数据里正好缺失该值导致的 Crash
	const vector<string>& allPossibleValues = attr_values[attr_name[bestAttrIndex]];

	for (const string& val : allPossibleValues) {
		// 分割数据
		vector<vector<string>> subData = splitData(currentData, bestAttrIndex, val);

		if (subData.empty()) {
			// 如果这个值在当前数据集中没有样本，创建一个叶子节点，类别为父集合的多数类
			TreeNode* leafChild = new TreeNode();
			leafChild->isLeaf = true;
			leafChild->label = getMajorityLabel(currentData);
			node->children[val] = leafChild;
		}
		else {
			// 递归构建子树
			node->children[val] = buildTree(subData, usedAttributes);
		}
	}

	return node;
}

// 计算信息熵
double ID3::calculateEntropy(const vector<vector<string>>& data) const {
	if (data.empty())
		return 0.0;

	map<string, int> labelCounts;
	for (const auto& row : data) {
		labelCounts[row[target_index]]++;
	}

	double entropy = 0.0;
	double total = (double)data.size();

	for (auto const& pair : labelCounts) {
		double p = pair.second / total;
		entropy -= p * log2(p);
	}
	return entropy;
}

// 获取最佳分裂属性 (计算最大信息增益)
int ID3::getBestAttribute(const vector<vector<string>>& data, const vector<bool>& usedAttributes) const {
	double baseEntropy = calculateEntropy(data);
	double maxGain = 0.0;
	int bestAttr = -1;

	// 遍历每一个属性
	for (int i = 0; i < attr_name.size(); ++i) {
		// 如果该属性已经被用过了(目标列已经处理)，跳过
		if (usedAttributes[i])
			continue;

		// 计算条件熵
		double newEntropy = 0.0;
		map<string, vector<vector<string>>> subSets;

		// 按照当前属性 i 的值将数据临时分组
		for (const auto& row : data) {
			subSets[row[i]].push_back(row);
		}

		// 累加条件熵: Sum ( Sv/S * H(Sv) )
		for (auto const& pair : subSets) {
			double prob = (double)pair.second.size() / data.size();
			newEntropy += prob * calculateEntropy(pair.second);
		}

		double gain = baseEntropy - newEntropy;

		// 更新最大增益
		if (gain > maxGain) {
			maxGain = gain;
			bestAttr = i;
		}
	}

	// 如果增益太小 (小于阈值)，认为没有区分度，返回 -1
	if (maxGain < EPSILON)
		return -1;

	return bestAttr;
}

// 分割数据
vector<vector<string>> ID3::splitData(const vector<vector<string>>& data, int axis, const string& value) const {
	vector<vector<string>> subSet;
	for (const auto& row : data) {
		if (row[axis] == value) {
			subSet.push_back(row);
		}
	}
	return subSet;
}

// 获取多数类
string ID3::getMajorityLabel(const vector<vector<string>>& data) const {
	map<string, int> counts;
	for (const auto& row : data) {
		counts[row[target_index]]++;
	}

	string majorityLabel;
	int maxCount = -1;
	for (auto const& pair : counts) {
		if (pair.second > maxCount) {
			maxCount = pair.second;
			majorityLabel = pair.first;
		}
	}
	return majorityLabel;
}

// 预测函数
string ID3::predict(const vector<string>& sample) const {
	if (!root)
		return "Empty Tree";
	if (sample.size() > attr_name.size())
	{
		return "测试集的维数错误";
	}

	TreeNode* currentNode = root;

	while (!currentNode->isLeaf) {
		string attrName = currentNode->Attribute;

		// 找到该属性在 sample 中的位置
		if (attr_index_map.find(attrName) == attr_index_map.end()) {
			return "Error: 未知的属性";
		}
		int idx = attr_index_map.at(attrName);
		string val = sample[idx];

		// 查找对应的子节点
		if (currentNode->children.find(val) == currentNode->children.end()) {
			// 遇到训练集中没见过的特征值，无法继续走
			// 简单的策略：返回未知，或者这里可以做得更复杂(比如返回父节点多数类)
			return "未知的特征值: " + val;
		}

		currentNode = currentNode->children.at(val);
	}

	return currentNode->label;
}