# LeetCode

斐波那契数列加速办法：升维。

我们可以使用如下递推式来计算斐波那契数列中的每一项：

$ \begin{bmatrix}F(n) & F(n-1)\end{bmatrix} = \begin{bmatrix}1 & 1\\1 & 0\end{bmatrix} \cdot \begin{bmatrix}F(n-1) \\ F(n-2)\end{bmatrix}$

令矩阵$A = \begin{bmatrix}1 & 1\\ 1 & 0\end{bmatrix}$，则有 $F(n) = A^{n-1} \cdot \begin{bmatrix}1\\ 1\end{bmatrix}$。

对于求解矩阵的幂次，可以使用快速幂算法。因此，要实现时间复杂度为$\log n$的斐波那契数列求解，我们只需要实现以下三个部分：

1. 矩阵的乘法运算；
2. 快速幂算法；
3. 斐波那契数列的通项公式；

斐波那契数列的通项公式为：$F(n) = \frac{1}{\sqrt{5}}\left(\left(\frac{1+\sqrt{5}}{2}\right)^n - \left(\frac{1-\sqrt{5}}{2}\right)^n\right)$

```c++
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

const int MAXN = 2;

struct Matrix
{
    ll a[MAXN][MAXN];

    Matrix() { memset(a, 0, sizeof(a)); }

    Matrix operator *(const Matrix &b) const
    {
        Matrix res;
        for (int i = 0; i < MAXN; i++)
            for (int j = 0; j < MAXN; j++)
                for (int k = 0; k < MAXN; k++)
                    res.a[i][j] += a[i][k] * b.a[k][j];
        return res;
    }
};

Matrix quick_pow(Matrix a, ll b)
{
    Matrix res;
    for (int i = 0; i < MAXN; i++) res.a[i][i] = 1;
    while (b)
    {
        if (b & 1) res = res * a;
        a = a * a;
        b >>= 1;
    }
    return res;
}

ll fib(ll n)
{
    if (n == 0) return 0;
    if (n == 1 || n == 2) return 1;

    Matrix base;
    base.a[0][0] = 1;
    base.a[0][1] = 1;
    base.a[1][0] = 1;

    Matrix res = quick_pow(base, n - 2);

    return res.a[0][0] + res.a[0][1];
}

int main()
{
    ll n;
    cin >> n;
    cout << fib(n) << endl;
    return 0;
}
```



## Array / Linked List

### 206. Reverse Linked List
```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur, prev = head, None
        while cur:
            cur.next, prev, cur = prev, cur, cur.next
        return prev
```
```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* new_head = nullptr;
        ListNode* rest;
        while (head) {
            rest = head->next;
            head->next = new_head;
            new_head = head;
            head = rest; 
        }
        return new_head;
    }
};
```

### 141. Linked List Cycle

```
Brute Force:
	0.5s\1s touch None
Set:
	O(n)
Two pointers:
	The Hare & the Tortoise
	O(n)
	https://en.wikipedia.org/wiki/Cycle_detection
```

```c++
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode* fast = head;
        ListNode* slow = head;
        while (fast && fast->next) {	// Condition
            slow = slow->next;
            fast = fast->next->next;
            if (fast == slow) {
                return true;
            }
        }
        return false;
    }
};
```

### 24. Swap Nodes in Pairs
```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre, pre.next = self, head
        while pre.next and pre.next.next:
            a = pre.next
            b = a.next
            pre.next, a.next, b.next = b, b.next, a
            pre = a
        return self.next
```
```c++
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (head && head->next) {
            auto rest = head->next;
            head->next = swapPairs(rest->next);
            rest->next = head;
            return rest;
        }
        return head;
    }
};

class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (!head || !head->next) {
            return head;
        }
        ListNode* curr = head;
        ListNode* prev = head;
        head = head->next;
        while (curr && curr->next) {
            ListNode* next_pair = curr->next->next;
            curr->next->next = curr;
            prev->next = curr->next;
            curr->next = next_pair;
            prev = curr;
            curr = next_pair;
        }
        return head;
    }
};
```

### 142. Linked List Cycle II

```
Slow moves 1 step at a time, fast moves 2 steps at a time.
When slow and fast meet each other, they must be on the cycle
x denotes the length of the linked list before starting the circle
y denotes the distance from the start of the cycle to where slow and fast met
C denotes the length of the cycle
When they meet, slow traveled (x + y) steps while fast traveled 2 * (x + y) steps, and the extra distance (x + y) must be a multiple of the circle length C
so we have x + y = N * C, let slow continue to travel from y and after x more steps, slow will return to the start of the cycle.(x + y + x = x + N * C)
At the same time, according to the definition of x, head will also reach the start of the cycle after moving x steps.
so if head and slow start to move at the same time, they will meet at the start of the cycle, that is the answer.
```

```c++
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        if (!head || !head->next) {
            return nullptr;
        }
        auto slow = head;
        auto fast = head;
        while (fast && fast->next) {	// Condition
            slow = slow->next;
            fast = fast->next->next;
            if (fast == slow) {
                while (head != slow) {
                    slow = slow->next;
                    head = head->next;
                }
                return head;
            }
        }
        return nullptr;
    }
};
```

### 25. Reverse Nodes in k-Group
```c++
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        auto rest = head;
        for (int i = 0; i < k; ++i) {
            if (!rest) {
                return head;
            } 
            rest = rest->next;
        }
        ListNode* new_head = reverseKGroup(rest, k);
        ListNode* cur = head;
        for (auto i = 0; i < k; ++i) {
            rest = cur->next;
            cur->next = new_head;
            new_head = cur;
            cur = rest;
        }
        return new_head;
    }
};
```
## Stack / Queue

### 232. Implement Queue using Stacks
```c++
class MyQueue {
public:
    stack<int> input;
    stack<int> output; 

    MyQueue() {}
    
    void push(int x) {
        input.push(x);
    }
    
    int pop() {
        int x = peek();
        output.pop();
        return x;
    }
    
    int peek() {
        if (output.empty()) {
            while (!input.empty()) {
                output.push(input.top());
                input.pop();
            }
        }
        return output.top();
    }
    
    bool empty() {
        return output.empty() && input.empty();
    }
};
```

### 225. Implement Stack using Queues

```
Use one queue, O(n) push, O(1) other.
```

```c++
class MyStack {
public:
    queue<int> queue;
    
    MyStack() {}
    
    void push(int x) {
        auto size = queue.size();
        queue.push(x);
        for (auto i = 0; i < size; ++i) {	// the elements in the queue are ordered
            queue.push(queue.front());
            queue.pop();
        }
    }
    
    int pop() {
        auto x = queue.front();
        queue.pop();
        return x;
    }
    
    int top() {
        return queue.front();
    }
    
    bool empty() {
        return queue.empty();
    }
};
```

### 20. Valid Parentheses

```c++
class Solution {
public:
    bool isValid(string s) {
        map<char, char> map {
            {')', '('}, 
            {']', '['}, 
            {'}', '{'}
        };
        stack<char> stack;
        for (const auto& c : s) {
            if (map.count(c)) {
                if (stack.empty() || stack.top() != map[c]) {
                    return false;
                } else {
                    stack.pop();
                }
            } else {
                stack.push(c);	// left parenthesis
            }
        }
        return stack.empty();
    }
};
```

## Priority Queue

### 703. Kth Largest Element in a Stream

```c++
class KthLargest {
public:
    priority_queue<int, vector<int>, std::greater<int>> pq;
    int size;
    KthLargest(int k, vector<int>& nums) : size(k) {
        for (const auto& n : nums) {
            add(n);
        }
    }
    
    int add(int val) {
        if (pq.size() < size) {
            pq.push(val);
        } else if (pq.top() < val) {
            pq.pop();
            pq.push(val);
        }
        return pq.top();
    }
};
```

### 239. Sliding Window Maximum

```c++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> window;
        vector<int> res;
        auto size = nums.size();
        for (auto i = 0; i < size; ++i) {
            if (i >= k && i - window[0] >= k) {
                window.pop_front();
            }
            while(!window.empty() && nums[window[0]] <= nums[i]) {	// save index in window
                window.pop_back();
            }
            window.push_back(i);
            if (i >= k - 1) {
                res.push_back(nums[window[0]]);
            }
        }
        return res;
    }
};
```

## Map / Set

### 242. Valid Anagram

```
Sort:
	O(nlogn)
Map:
	O(n)
```

```c++
class Solution {
public:
    bool isAnagram(string s, string t) {
        unordered_map<char, int> map;
        if (s.size() != t.size()) {
            return false;
        }
        for (const auto& c : s) {
            map[c]++;
        }
        for (const auto& c : t) {
            map[c]--;
        }
        for (const auto& [k, v] : map) {
            if (v != 0) {
                return false;
            }
        }
        return true;
    }
};
```

### 1. Two Sum

```
sort + two pointers:
	O(nlogn)
map:
	O(n)
```

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> map;
        auto size = nums.size();
        for (auto i = 0; i < size; ++i) {
            int elem = nums[i];
            int complement = target - elem;
            if (map.count(complement)) {
                return {map[complement], i};
            } else {
                map[elem] = i;
            }
        }
        return {};
    }
};
```

### 15. 3Sum

```
3 loops:
	O(n^3)
2 loops + map:
	O(n^2) 
	space:O(n)
Sort + two pointers:
	O(n^2) 
	space:O(1)
```

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int len = nums.size();
        if (len < 3 || nums[0] > 0) {
            return {};
        }
        unordered_map<int, int> map;
        for (int i = 0; i < len; ++i) {
            map[nums[i]] = i;
        }
        vector<vector<int>> res;
        for (int i = 0; i < len - 2; ++i) {
            if (nums[i] > 0) {
                break;
            }
            for (int j = i + 1; j < len - 1; ++j) {
                int required = -nums[i] - nums[j];
                if (map.count(required) && map[required] > j) {
                    res.push_back({nums[i], nums[j], required});
                }
                j = map[nums[j]];
            }
            i = map[nums[i]];
        }
        return res;
    }
};
```

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int size = nums.size();
        std::sort(nums.begin(), nums.end());
        vector<vector<int>> res;
        for (int i = 0; i < size - 2; ++i) {
            int n = nums[i];
            if (n > 0) {
                break;
            }
            if (i > 0 && n == nums[i - 1]) {    // skip the repeated numbers
                continue;
            }
            int l = i + 1;
            int r = size - 1;
            while (l < r) {
                int sum = n + nums[l] + nums[r];
                if (sum > 0) {
                    --r;
                } else if (sum < 0) {
                    ++l;
                } else {
                    res.push_back({n, nums[l], nums[r]});
                    --r;    // update index
                    ++l;
                    while (l < r && nums[r] == nums[r + 1]) {
                        --r;
                    }
                    while (l < r && nums[l] == nums[l - 1]) {
                        ++l;
                    }
                }
            }
        }
        return res;
    }
};
```

### 18. 4Sum

```c++
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> res;
        vector<int> path;
        std::sort(nums.begin(), nums.end());
        kSum(4, nums, 0, nums.size(), path, static_cast<long>(target), res);    // size() - 1 is the largest index
        return res;
    }
private:
    void kSum(int k, vector<int>& nums, int l, int r, vector<int>& path, long target, vector<vector<int>>& res) {
        if (r - l < k || k < 2) {
            return;
        }
        if (k == 2) {
            int left = l, right = r - 1;
            while (left < right) {  // scan the nums from l to r - 1
                long sum = static_cast<long>(nums[left]) + static_cast<long>(nums[right]);
                if (sum == target) {
                    path.push_back(nums[left]);
                    path.push_back(nums[right]);
                    res.push_back(path);
                    path.pop_back();
                    path.pop_back();
                    ++left;
                    --right;
                    while (left < right && nums[left] == nums[left - 1]) {
                        ++left;
                    }
                    while (left < right && nums[right] == nums[right + 1]) {
                        --right;
                    }
                } else if (sum < target) {
                    ++left;
                } else {
                    --right;
                }
            }
        } else {
            for (int i = l; i < r; ++i) {
                if (i > l && nums[i] == nums[i - 1]) {  // skip repeated nums
                    continue;
                }
                path.push_back(nums[i]);
                kSum(k - 1, nums, i + 1, r, path, target - static_cast<long>(nums[i]), res);
                path.pop_back();
            }
        }
    }
};
```

### 49. Group Anagrams

```c++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> map;
        for (const auto& str : strs) {
            string s = str;
            sort(s.begin(), s.end());
            map[s].push_back(str);
        }
        vector<vector<string>> res;
        for (const auto& [k, v] : map) {
            res.push_back(v);
        }
        return res;
    }
};
```

## Tree

### 98. Validate Binary Search Tree

```
In-order:
	array asc
	O(n)
Recursion:
	validate(..., min, max)
	O(n)
```

```c++
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        return isValidBST(root, LONG_MIN, LONG_MAX);
    }
private:
    bool isValidBST(TreeNode* root, long l, long r) {
        if (root == nullptr) {
            return true;
        }
        int val = root->val;
        if (val >= r || val <= l) {
            return false;
        }
        return isValidBST(root->left, l, val) && isValidBST(root->right, val, r); 
    }
};
```

```c++
class Solution {
public:
    TreeNode* prev;
    bool isValidBST(TreeNode* root) {
        prev = nullptr;
        return isValid(root);
    }
    bool isValid(TreeNode* curr) {
        if (!curr) {
            return true;
        }
        if (!isValid(curr->left)) {
            return false;
        }
        if (prev && prev->val >= curr->val) {
            return false;
        }
        prev = curr;
        return isValid(curr->right);
    }
};
```

### 235. Lowest Common Ancestor of a Binary Search Tree

```c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        while (root) {
            if (root->val > p->val && root->val > q->val) {
                root = root->left;
            } else if (root->val < p->val && root->val < q->val) {
                root = root->right;
            } else {
                break;
            }
        }
        return root;
    }
};
```

### 236. Lowest Common Ancestor of a Binary Tree

```c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (root == nullptr || root == p || root == q) {
            return root;
        }
        TreeNode* l = lowestCommonAncestor(root->left, p, q);
        TreeNode* r = lowestCommonAncestor(root->right, p, q);
        if (l && r) {
            return root;
        }
        return l ? l : r;
    }
};
```

## Recursion, Divide & Conquer

### 50. Pow(x, n)

```
调库:
	O(1)
Brute Force:
	O(n)
x^(n/2):
	O(logn)
```

```c++
class Solution {
public:
    double myPow(double x, long n) {
        if (n == 0) {
            return 1;
        } else if (n < 0) {
            return 1 / myPow(x, -n);
        } else if (n % 2) {
            return myPow(x, n - 1) * x;
        } else {
            return myPow(x * x, n / 2);
        }
    }
};
```

```c++
class Solution {
public:
    double myPow(double x, long n) {
        if (n < 0) {
            n = -n;
            x = 1 / x;
        }
        double pow = 1;
        while (n) {
            if (n & 1) {
                pow *= x;
            }
            x = x * x;
            n >>= 1;    // equal to n = n >> 1;
        }
        return pow;
    }
};
```

### 169. Majority Element

```
Brute Force:
	O(n^2)
Map:
	O(n)
Sort:
	O(nlogn)
Divide & Conquer:
	O(nlogn)
Boyer-Moore Majority Vote Algorithm:
	O(n)
	http://www.cs.utexas.edu/~moore/best-ideas/mjrty/
```

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        unordered_map<int, int> map;
        int half = nums.size() / 2;
        for (const auto& n : nums) {
            if (++map[n] > half) {
                return n;
            }
        }
        throw;
    }
};
```

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        nth_element(nums.begin(), nums.begin() + nums.size() / 2, nums.end());
        return nums[nums.size() / 2];
    }
};
```

```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int elem, cnt = 0;
        for (const auto& n : nums) {
            if (!cnt) {
                elem = n;
                cnt = 1;
            } else
                cnt += (n == elem) ? 1 : -1;
        }
        return elem;
    }
};
```

### 53. Maximum Subarray

```c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int max = INT_MIN, cur = INT_MIN;
        for (const auto& n : nums) {
            cur = cur < 0 ? n : cur + n;    // cur < 0 : adding any previous element to n would not increase the sum
            max = cur > max ? cur : max;
        }
        return max;
    }
};
```

### 438. Find All Anagrams in a String

```c++
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        vector<int> hash(26);
        vector<int> cur(26);
        vector<int> res;
        for (const auto& c : p) {
            ++hash[c - 'a'];
        }
        int size = s.size();
        int k = p.size();
        for (int i = 0; i < size; ++i) {
            ++cur[s[i] - 'a'];
            if (i >= k) {
                --cur[s[i - k] - 'a'];
            }
            if (cur == hash) {
                res.push_back(i - k + 1);
            } 
        }
        return res;
    }
};
```

## Greedy Algorithms

### 122. Best Time to Buy and Sell Stock II

```
持有1股 可以买卖无数次 无交易手续费
DFS:
	O(2^n)
Greedy:
	O(n)
DP:
	O(n)
```

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int profit = 0;
        int old = prices[0];
        for (const auto& p : prices) {
            if (p > old) {
                profit += p - old;
            }
            old = p;
        }
        return profit;
    }
};
```

### 860. Lemonade Change

```c++
class Solution {
public:
    bool lemonadeChange(vector<int>& bills) {
        int five = 0, ten = 0;
        for (const auto& b : bills) {
            if (b == 5) {
                ++five;
            } else if (b == 10) {
                if (five <= 0) {
                    return false;
                }
                --five;
                ++ten;
            } else {
                if (five > 0 && ten > 0) {
                    --five;
                    --ten;
                } else if (five > 2) {
                    five -= 3;
                } else {
                    return false;
                }
            }
        }
        return true;
    }
};
```

### 455. Assign Cookies

```c++
class Solution {
public:
    int findContentChildren(vector<int>& g, vector<int>& s) {
        sort(g.begin(), g.end());
        sort(s.begin(), s.end());
        int g_size = g.size(), s_size = s.size();
        int i = 0, j = 0, n = 0;
        while (i < g_size && j < s_size) {
            if (g[i] <= s[j]) {
                ++n;
                ++i;
                ++j;
            } else {
                ++j;
            }
        }
        return n;
    }
};
```

### 874. Walking Robot Simulation

```c++
class Solution {
public:
    int robotSim(vector<int>& commands, vector<vector<int>>& obstacles) {
        set<pair<int, int>> set;
        for (const auto& obstacle : obstacles) {
            set.insert({obstacle[0], obstacle[1]});
        }
        vector<vector<int>> dir = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int d = 0, x = 0, y = 0, res = 0;
        for (const int& cmd : commands) {
            if (cmd < 0) {
                if (cmd == -1) {
                    ++d;
                } else if (cmd == -2) {
                    --d;
                    d += 4;
                }
                d %= 4;
            } else {
                for (int i = 0; i < cmd; ++i) {
                    x += dir[d][0];
                    y += dir[d][1];
                    if (set.count({x, y})) {
                        x -= dir[d][0];
                        y -= dir[d][1];
                        break;
                    }
                    res = max(x * x + y * y, res);
                }
            }
        }
        return res;
    }
};
class Solution {
public:
    int robotSim(vector<int>& commands, vector<vector<int>>& obstacles) {
        int x = 0, y = 0, dx = 0, dy = 1, dis = 0;
        set<pair<int, int>> obs;
        for (const auto& pos : obstacles) {
            obs.emplace(pos[0], pos[1]);
        }
        for (const auto& c : commands) {
            if (c == -1) {
                int t = dy;
                dy = dy ? dx : -dx;
                dx = t;
            } else if (c == -2) {
                int t = dx;
                dx = dx ? dy : -dy;
                dy = t;
            } else {
                for (int i = 0; i < c; ++i) {
                    x += dx;
                    y += dy;
                    if (obs.count({x, y})) {
                        x -= dx;
                        y -= dy;
                        break;
                    }
                }
                dis = max(dis, x*x + y*y);
            }
        }
        return dis;
    }
};
```

## Breadth First Search / Depth First Search

### 102. Binary Tree Level Order Traversal

```
BFS:
	level->queue ×
	batch process √
	O(n)
DFS:
	O(n)
```

```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if (!root) {
            return res;
        }
        queue<pair<TreeNode*, int>> q;
        q.emplace(root, 0);
        while (!q.empty()) {
            const auto& [node, level] = q.front();
            if (res.size() <= level) {
                res.emplace_back(std::initializer_list<int>{node->val});
            } else {
                res[level].emplace_back(node->val);
            }
            if (node->left) {
                q.emplace(node->left, level + 1);
            }
            if (node->right) {
                q.emplace(node->right, level + 1);
            }
            q.pop();
        }
        return res;
    }
};
```

```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if (!root) {
            return res;
        }
        queue<TreeNode*> q;
        q.emplace(root);
        while (!q.empty()) {
            int level_size = q.size();
            vector<int> level;
            while (level_size) {
                const auto& node = q.front();
                level.push_back(node->val);
                if (node->left) {
                    q.emplace(node->left);
                }
                if (node->right) {
                    q.emplace(node->right);
                }
                q.pop();
                --level_size;
            }
            res.push_back(move(level));
        }
        return res;
    }
};
```

```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<vector<int>> levelOrder(TreeNode* root) {
        _dfs(root, 0);
        return res;
    }
    void _dfs(TreeNode* node, int level) {
        if (!node) {
            return;
        }
        if (res.size() == level) {
            res.push_back({});
        }
        res[level].push_back(node->val);
        _dfs(node->left, level + 1);
        _dfs(node->right, level + 1);
    }
};
```

### 104. Maximum Depth of Binary Tree

```
Recursion:
	×
DFS:
	O(n)
BFS:
	O(n)
```

```c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) {
            return 0;
        }
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }
};
```

```c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) {
            return 0;
        }
        queue<TreeNode*> q;
        q.push(root);
        int d = 0;
        while (!q.empty()) {
            int level = q.size();
            ++d;
            while (level) {
                auto node = q.front();
                if (node->left) {
                    q.push(node->left);
                }
                if (node->right) {
                    q.push(node->right);
                }
                --level;
                q.pop();
            }
        }
        return d;
    }
};
```

### 111. Minimum Depth of Binary Tree

```c++
class Solution {
public:
    int minDepth(TreeNode* root) {
        if (!root) {
            return 0;
        }
        if (!root->left) {  // no left subtree
            return minDepth(root->right) + 1;
        }
        if (!root->right) {
            return minDepth(root->left) + 1;
        }
        return min(minDepth(root->left), minDepth(root->right)) + 1;
    }
};
```

```c++
class Solution {
   public:
    int minDepth(TreeNode* root) {
        if (!root) {
            return 0;
        }
        int l = minDepth(root->left);
        int r = minDepth(root->right);
        return (l && r) ? min(l, r) + 1 : l + r + 1;
    }
};
```

```c++
class Solution {
public:
    int minDepth(TreeNode* root) {
        if (!root) {
            return 0;
        }
        queue<TreeNode*> q;
        q.push(root);
        int d = 0;
        while (!q.empty()) {
            int level = q.size();
            ++d;
            while (level) {
                auto node = q.front();
                if (node->left) {
                    q.push(node->left);
                }
                if (node->right) {
                    q.push(node->right);
                }
                if (!node->right && !node->left) {
                    return d;
                }
                --level;
                q.pop();
            }
        }
        return d;
    }
};
```

### 22. Generate Parentheses

```
DFS:
	O(2^2n)
剪枝:
	O(2^2n)
	局部不合法, 不再递归
	left_used, right_used
```

```c++
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> res;
        gen(n, n, res, "");
        return res;
    }

    void gen(int l, int r, vector<string>& output, string s) {
        if (!l && !r) {
            output.push_back(move(s));
            return;
        } 
        if (l > 0) {
            gen(l - 1, r, output, s + '(');
        } 
        if (r > l) {
            gen(l, r - 1, output, s + ')');
        } 
    }
};
```

## Pruning

### 51. N-Queens

```
Save the row / column and the diagonals that each queen occupies. 
The columns can be saved on a boolean one dimentional array. 
The diagonals can be also saved on two boolean one dimentional array and accessed with x - y and x + y numbers. The diagonals accesed with x - y are those with positive slope whereas x + y accessed the negative slope diagonals. 
T(n) = n*(T(n-1) + O(1)) which translates to O(N!) time complexity approximately.
```

![image-20230302113158326](https://raw.githubusercontent.com/Christina0031/Images/main/202303241705369.png)

```c++
class Solution {
public:
    unordered_set<int> col;
    unordered_set<int> pie;
    unordered_set<int> na;
    string padding;
    vector<string> board;
    vector<vector<string>> res;

    vector<vector<string>> solveNQueens(int n) {
        padding.assign(n, '.');
        board.assign(n, padding);
        dfs(0, n);
        return res;
    }

    void dfs(int x, int n) {
        if (x >= n) {
            res.push_back(board);
            return;
        }
        for (int y = 0; y < n; ++y) {
            if (!col.count(y) && !pie.count(x + y) && !na.count(x - y)) {
                board[x][y] = 'Q';
                col.insert(y);
                pie.insert(x + y);
                na.insert(x - y);
                dfs(x + 1, n);
                col.erase(y);
                pie.erase(x + y);
                na.erase(x - y);
                board[x][y] = '.';
            }
        }
    }
};
```

```c++
class Solution {
public:
    vector<bool> col;
    vector<bool> neg_slope;
    vector<bool> pos_slope;
    string padding;
    vector<string> board;
    vector<vector<string>> res;

    vector<vector<string>> solveNQueens(int n) {
        padding.assign(n, '.');
        board.assign(n, padding);
        neg_slope.assign(2 * n - 1, false);
        pos_slope.assign(2 * n - 1, false);
        col.assign(n, false);
        dfs(0, n);
        return res;
    }

    void dfs(int x, int n) {
        if (x >= n) {
            res.push_back(board);
            return;
        }
        for (int y = 0; y < n; ++y) {
            if (!col[y] && !neg_slope[x + y] && !pos_slope[x - y + n - 1]) {
                board[x][y] = 'Q';
                col[y] = true;
                neg_slope[x + y] = true;
                pos_slope[x - y + n - 1] = true;
                dfs(x + 1, n);
                col[y] = false;
                neg_slope[x + y] = false;
                pos_slope[x - y + n - 1] = false;
                board[x][y] = '.';
            }
        }
    }
};
```

### 52. N-Queens II

```c++
class Solution {
public:
    vector<bool> col;
    vector<bool> neg_slope;
    vector<bool> pos_slope;
    int num;
    int totalNQueens(int n) {
        neg_slope.assign(2 * n - 1, false);
        pos_slope.assign(2 * n - 1, false);
        col.assign(n, false);
        dfs(0, n);
        return num;
    }
    void dfs(int x, int n) {
        if (x >= n) {
            ++num;
            return;
        }
        for (int y = 0; y < n; ++y) {
            if (!col[y] && !neg_slope[x + y] && !pos_slope[x - y + n - 1]) {
                col[y] = true;
                neg_slope[x + y] = true;
                pos_slope[x - y + n - 1] = true;
                dfs(x + 1, n);
                col[y] = false;
                neg_slope[x + y] = false;
                pos_slope[x - y + n - 1] = false;
            }
        }
    }
};
```

```c++
class Solution {
public:
    int num = 0;
    int size;
    int totalNQueens(int n) {
        size = n;
        dfs(0, 0, 0, 0);
        return num;
    }

    void dfs(int row, int col, int neg, int pos) {
        if (row >= size) {
            ++num;
            return;
        }
        int bits = (~(col | neg | pos)) & ((1 << size) - 1); // calculate valid positions for queen on this row using bitwise operations
        while (bits > 0) {
            int p = bits & -bits; // get lowest valid position (the first bit set to 1)
            dfs(row + 1, col | p, (neg | p) >> 1, (pos | p) << 1);
            bits = bits & (bits - 1); // remove the lowest valid position to consider next valid position
        }
    }
};
```

### 36. Valid Sudoku

```c++
class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        vector<vector<bool>> row(9, vector<bool>(9, false));
        vector<vector<bool>> col = row;
        vector<vector<bool>> block = row;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (board[i][j] == '.') {
                    continue;
                }
                int val = board[i][j] - '1';
                if (row[i][val] || col[j][val] || block[i / 3 + j / 3 * 3][val]) {
                    return false;
                } else {
                    row[i][val] = true;
                    col[j][val] = true;
                    block[i / 3 + j / 3 * 3][val] = true;
                }
            }
        }
        return true;
    }
};
```

### 37. Sudoku Solver

```
Naive DFS:
	loop board
	枚举1-9, check valid
加速:
	1 对N*N格预处理, 统计可选的值
	  先枚举选项比较少的空格, 减少搜索空间
	2 高级数据结构, Dancing Links
```

```c++
class Solution {
public:
    array<array<bool, 9>, 9> row= {false};
    array<array<bool, 9>, 9> col = row;
    array<array<bool, 9>, 9> block = row;
    void solveSudoku(vector<vector<char>>& board) {
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if (board[i][j] == '.') {
                    continue;
                } else {
                    int val = board[i][j] - '1';
                    row[i][val] = true;
                    col[j][val] = true;
                    block[i / 3 + j / 3 * 3][val] = true;
                }
            }
        }
        solve(board, 0, 0);
    }

    bool solve(vector<vector<char>>& board, int i, int j) {
        if (j == 9) {
            j = 0;
            ++i;
            if (i == 9) {
                return true;
            }
        } 
        if (board[i][j] == '.') {
            for (int val = 0; val < 9; ++val) {
                if (!(row[i][val] || col[j][val] || block[i / 3 + j / 3 * 3][val])) {
                    board[i][j] = val + '1';
                    row[i][val] = true;
                    col[j][val] = true;
                    block[i / 3 + j / 3 * 3][val] = true;
                    if (solve(board, i, j + 1)) {
                        return true;
                    }
                    board[i][j] = '.';
                    row[i][val] = false;
                    col[j][val] = false;
                    block[i / 3 + j / 3 * 3][val] = false;
                }
            }
            return false;
        } else {
            return solve(board, i, j + 1);
        }
    }

};
```

## Binary Search

1. Sorted（单调递增或者递减）
2. Bounded（存在上下界）
3. Accessible by index（能够通过索引访问）

第一种写法：定义 target 是在一个在左闭右闭的区间[left, right]里

- while (left <= right) 要使用 <= ，因为left == right有意义
- if (nums[middle] > target) right 要赋值为 middle - 1，因为nums[middle]不是target，要查找的左区间结束下标位置就是 middle - 1

```c++
// 版本一
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size() - 1; // 定义target在左闭右闭的区间里，[left, right]
        while (left <= right) { // 当left==right，区间[left, right]依然有效，所以用 <=
            int middle = left + ((right - left) / 2); // 防止溢出 等同于(left + right)/2
            if (nums[middle] > target) {
                right = middle - 1; // target 在左区间，所以[left, middle - 1]
            } else if (nums[middle] < target) {
                left = middle + 1; // target 在右区间，所以[middle + 1, right]
            } else { // nums[middle] == target
                return middle; // 数组中找到目标值，直接返回下标
            }
        }
        return -1; // 未找到目标值
    }
};

```

第二种写法：定义 target 在左闭右开的区间[left, right)里 

- while (left < right)，使用 < , 因为left == right在区间[left, right)是没有意义的
- if (nums[middle] > target) right 更新为 middle。当前nums[middle]不等于target，去左区间继续寻找，而寻找区间是左闭右开区间，所以right更新为middle，即：下一个查询区间不会去比较nums[middle]

```c++
// 版本二
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size(); // 定义target在左闭右开的区间里，即：[left, right)
        while (left < right) { // 因为left == right的时候，在[left, right)是无效的空间，所以使用 <
            int middle = left + ((right - left) >> 1);
            if (nums[middle] > target) {
                right = middle; // target 在左区间，在[left, middle)中
            } else if (nums[middle] < target) {
                left = middle + 1; // target 在右区间，在[middle + 1, right)中
            } else { // nums[middle] == target
                return middle; // 数组中找到目标值，直接返回下标
            }
        }
        return -1; // 未找到目标值
    }
};
```

### 69. Sqrt(x)

```
二分法: y = x^2单调递增
牛顿迭代法
```

$$
x_{n+1} = x_{n} - \frac{f(x_n)}{f'(x_n)}\\
x_{n+1} = x_{n} - \frac{x_n^2 - y_0}{2x_n} = \frac{x_n + y_0/x_n}{2}
$$

[Fast Inverse Square Root](https://www.beyond3d.com/content/articles/8/)

```c++
class Solution {
public:
    int mySqrt(int x) {
        if (x == 0 || x == 1) {
            return x;
        }
        int l = 1, r = x, res;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (m == x / m) {
                return m;
            } else if (m > x / m) {
                r = m - 1;
            } else {
                l = m + 1;
                res = m;
            }
        }
        return res;
    }
};
```

```c++
class Solution {
public:
    int mySqrt(int x) {
        if (x == 0) {
            return x;
        }
        int r = x;
        while (r > x / r) {
            r  = (static_cast<long>(r) +  x / r) / 2;
        }
        return r;
    }
};
```

### 367. Valid Perfect Square

```c++
class Solution {
public:
    bool isPerfectSquare(int num) {
        if (num == 0 || num == 1) {
            return true;
        }
        int r = num;
        while (r > num / r) {
            r = (static_cast<long>(r) + num / r) / 2;
        }
        return r * r == num;
    }
};
```

```c++
class Solution {
public:
    bool isPerfectSquare(int num) {
        if (num == 0 || num == 1) {
            return true;
        }
        int l = 1, r = num;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (m > num / m) {
                r = m - 1;
            } else if (m < num / m) {
                l = m + 1;
            } else {
                return m * m == num;
            }
        }
        return false;
    }
};
```



## Trie

用路径上走过的边来表示信息（用边存储字母）。

从根节点到某一节点路径上经过的字符连接起来，为该节点对应的字符串。

### 208. Implement Trie (Prefix Tree)

```c++
class Trie {
    class TrieNode {
        friend class Trie;
        bool is_key = false;
        array<unique_ptr<TrieNode>, 26> next;
    };
public:
    TrieNode root;
    Trie() {}
    
    void insert(string word) {
        TrieNode* node = &root;
        for(const char& c : word) {
            int i = c - 'a';
            if (node->next[i] == nullptr) {
                node->next[i] = make_unique<TrieNode>();
            } 
            node = node->next[i].get();
        }
        node->is_key = true;
    }
    
    bool search(string word) {
        return searchPrefix(word, true);
    }
    
    bool startsWith(string prefix) {
        return searchPrefix(prefix, false);
    }
    
private:
    bool searchPrefix(const string& word, bool exact_match) {
        TrieNode* node = &root;
        for(const char& c : word) {
            int i = c - 'a';
            if (node->next[i] == nullptr) {
                return false;
            } 
            node = node->next[i].get();
        }
        return !exact_match || node->is_key;
    }
};
```

```c++
class Trie {
    struct TrieNode {
        bool is_key = false;
        TrieNode* next[26] = {nullptr};
    };
    
public:
    TrieNode* root = new TrieNode();
    
    void insert(const string& word) {
        TrieNode* node = root;
        for (const char& c : word) {
            int i = c - 'a';
            if (!node->next[i]) {
                node->next[i] = new TrieNode();
            }
            node = node->next[i];
        }
        node->is_key = true;
    }
    
    bool search(const string& word) {
        return searchPrefix(word, true);
    }
    
    bool startsWith(const string& prefix) {
        return searchPrefix(prefix, false);
    }
    
    ~Trie() {
        cleanup(root);
    }
    
private:
    bool searchPrefix(const string& word, bool exactMatch) {
        TrieNode* node = root;
        for (const char& c : word) {
            int i = c - 'a';
            if (!node->next[i]) {
                return false;
            }
            node = node->next[i];
        }
        return !exactMatch || node->is_key;
    }
    
    void cleanup(TrieNode* node) {
        for (int i = 0; i < 26; ++i) {
            if (node->next[i]) {
                cleanup(node->next[i]);
            }
        }
        delete node;
    }
};
```

### 212. Word Search II

```c++
class Solution {
public:
    struct Trie {
        bool is_word = false;
        Trie* next[26] = {nullptr};
        string word;
        ~Trie() {
            for (int i = 0; i < 26; ++i) {
                if (next[i]) {
                    delete next[i]; // call destructer recursively
                }
            }
        }
        
    };

    int x, y;
    vector<string> res;
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        Trie root;
        string s;
        for (const auto& word : words) {
            Trie* node = &root;
            for (const auto& c : word) {
                if (node->next[c - 'a'] == nullptr) {
                    node->next[c - 'a'] = new Trie();
                }
                node = node->next[c - 'a'];
            }
            node->is_word = true;
            node->word = word;
        }
        x = board.size();
        y = board[0].size();
        for (int i = 0; i < x; ++i) {
            for (int j = 0; j < y; ++j) {
                dfs(board, i, j, &root);
            }
        }
        return res;
    }

    void dfs(vector<vector<char>>& board, int i, int j, Trie* node) {
        if (i < 0 || j < 0 || i >= x || j >= y || board[i][j] == '#') {
            return;
        }
        char c = board[i][j];
        node = node->next[c - 'a'];
        if (node == nullptr) {
            return;
        }
        if (node->is_word) {
            res.push_back(node->word);
            node->is_word = false;
        }
        board[i][j] = '#';
        dfs(board, i + 1, j, node);
        dfs(board, i - 1, j, node);
        dfs(board, i, j + 1, node);
        dfs(board, i, j - 1, node);
        board[i][j] = c;
    }
};
```

## Bitwise operations

<img src="https://raw.githubusercontent.com/Christina0031/Images/main/202303241705370.png" alt="image-20230301235027143" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/Christina0031/Images/main/202303241705371.png" alt="image-20230301235056313" style="zoom: 50%;" />![image-20230301235113397](https://raw.githubusercontent.com/Christina0031/Images/main/202303241705372.png)

<img src="https://raw.githubusercontent.com/Christina0031/Images/main/202303241705373.png" alt="image-20230301235134527" style="zoom:33%;" />

### 191. Number of 1 Bits

```
x % 2, x = x >> 1
	32 ops
x = x & (x - 1)
	i & (i-1) clears the least significant bit in i
	number of 1-bit ops
```

```c++
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int num = 0;
        for (int i = 0; i < 32; ++i) {
            num += n & 1;
            n >>= 1;
        }
        return num;
    }
};
```

```c++
class Solution {
 public:
  int hammingWeight(uint32_t n) {
    int num = 0;
    while (n != 0) {
      n &= n - 1;
      ++num;
    }
    return num;
  }
};
```

### 231. Power of Two

```c++
mod 2
log_2
bit operations
```

```c++
class Solution {
public:
    bool isPowerOfTwo(int n) {
        return n > 0 && !(n & (n - 1)); // n > 0
    }
};
```

### 338. Counting Bits

```
i & (i-1) clears the least significant bit in i, and the result is used as an index to lookup the number of set bits in the corresponding element of bits. This value is then incremented by 1 and stored in bits[i]. 
```

```c++
class Solution {
public:
    vector<int> countBits(int n) {
        vector<int> bits(n + 1, 0);
        for(int i = 1; i <= n; ++i) {
            bits[i] = bits[i & (i - 1)] + 1;
        }
        return bits;
    }
};
```

## LRU

### 146. LRU Cache

```c++
class LRUCache {
public:
    std::list<pair<int, int>> cache;
    unordered_map<int, std::list<pair<int, int>>::iterator> location_map;
    int max_capacity;

    LRUCache(int capacity) : max_capacity(capacity) {}

    int get(int key) {
        auto map_iter = location_map.find(key);
        if (map_iter != location_map.end()) {
            auto cache_iter = map_iter->second;
            cache.splice(cache.begin(), cache, cache_iter); // move item to front of cache
            return cache_iter->second; // return value of item
        }
        return -1;
    }

    void put(int key, int value) {
        auto map_iter = location_map.find(key);
        if (map_iter != location_map.end()) {
            auto cache_iter = map_iter->second;
            cache_iter->second = value;
            cache.splice(cache.begin(), cache, cache_iter);
        } else {
            if (cache.size() == max_capacity) {
                location_map.erase(cache.back().first);
                cache.pop_back();
            }
            cache.emplace_front(key, value);
            location_map[key] = cache.begin();
        }
    }
};
```

## Union Find

### 200. Number of Islands

```
Flood Fill:
	DFS or BFS: if node == 1, count++, 将node和附近陆地结点变为0
Union Find:
	遍历, 1则相邻合并, 0跳过, 统计roots数量
```

```c++
class Solution {
 public:
  array<int, 4> dx = {0, 1, -1, 0};
  array<int, 4> dy = {1, 0, 0, -1};
  int m, n;
  int numIslands(vector<vector<char>>& grid) {
    int count = 0;
    n = grid.size();
    m = grid[0].size();
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        if (grid[i][j] != '0') {
          ++count;
          dfs(grid, i, j);
        }
      }
    }
    return count;
  }
  void dfs(vector<vector<char>>& grid, int i, int j) {
    if (i < 0 || j < 0 || i >= n || j >= m || grid[i][j] == '0') {
      return;
    } else {
      grid[i][j] = '0';
      for (int d = 0; d < 4; ++d) {
        dfs(grid, i + dx[d], j + dy[d]);
      }
    }
  }
};
```

```c++
class Solution {
  struct UnionFind {
    vector<int> roots;
    int count = 0;
    UnionFind(vector<vector<char>>& grid, int n, int m) : roots(m * n) {
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
          if (grid[i][j] == '1') {
            ++count;
            roots[i * m + j] = i * m + j;
          } else {
            roots[i * m + j] = -1;
          }
        }
      }
    }
    void unionPQ(int p, int q) {
      int rootp = findRoot(p);
      int rootq = findRoot(q);
      if (rootp != rootq) {
        --count;
      }
      roots[rootp] = rootq;
    }
    int findRoot(int i) {
      int root = i;
      while (root != roots[root]) {
        root = roots[root];
      }
      while (i != roots[i]) {
        int tmp = i;
        i = roots[i];
        roots[tmp] = root;
      }
      return root;
    }
    bool isConnected(int p, int q) { return findRoot(p) == findRoot(q); }
  };
  array<int, 4> dx = {0, 1, -1, 0};
  array<int, 4> dy = {1, 0, 0, -1};
  int n, m;

 public:
  int numIslands(vector<vector<char>>& grid) {
    n = grid.size();
    m = grid[0].size();
    UnionFind uf(grid, n, m);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        if (grid[i][j] == '1') {
          for (int d = 0; d < 4; ++d) {
            int ni = i + dx[d], nj = j + dy[d];
            if (ni >= 0 && nj >= 0 && ni < n && nj < m && grid[ni][nj] == '1') {
              uf.unionPQ(i * m + j, ni * m + nj);
            }
          }
        }
      }
    }
    return uf.count;
  }
};
```

### 547. Number of Provinces

```c++
class Solution {
public:
    int findCircleNum(vector<vector<int>>& isConnected) {
        int n = isConnected.size();
        vector<bool> visited(n, false);
        int count = 0;
        function<void(int)> dfs = [&](int node) {
            visited[node] = true;
            for (int neighbor = 0; neighbor < n; ++neighbor) {
                if (isConnected[node][neighbor] == 1 && !visited[neighbor]) {
                    dfs(neighbor);
                }
            }
        };
        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                dfs(i);
                count++;
            }
        }
        return count;
    }
};

```

## Dynamic Programming

```
DP状态定义
DP方程
状态压缩: 复用低维数组
```

### 70. Climbing Stairs

```
dp[n] 到n级台阶的总走法个数
dp[n] = dp[n - 1] + dp[n - 2]
```

```c++
class Solution {
 public:
  int climbStairs(int n) {
    vector<int> dp(n + 1, 1);
    for (int i = 2; i <= n; ++i) {
      dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
  }
};
```

```c++
class Solution {
 public:
  int climbStairs(int n) {
    int one = 1, two = 1;
    for (int i = 2; i <= n; ++i) {
      int tmp = one + two;
      two = one;
      one = tmp;
    }
    return one;
  }
};
```

### 120. Triangle

```
Recursion: O(2^n)
DP: O(m*n)
dp[i][j]: bottom -> i, j min path sum
dp[i][j] = min(dp[i+1][j],dp[i+1][j+1]) + triangle[i,j];
dp[m-1,j]=triangle[m-1,j];
```

```c++
class Solution {
 public:
  int minimumTotal(vector<vector<int>>& triangle) {
    vector<int> minimum = triangle[triangle.size() - 1];
    for (int i = triangle.size() - 2; i >= 0; --i) {
      for (int j = 0; j < triangle[i].size(); ++j) {
        minimum[j] = triangle[i][j] + min(minimum[j], minimum[j + 1]);
      }
    }
    return minimum[0];
  }
};
```

### 152. Maximum Product Subarray

```
dp[x][0] stores the maximum product subarray ending at index x,
dp[x][1] stores the minimum product subarray ending at index x.
curMax = curMax*num[i] if num[i] > 0; else curMin*num[i]
curMin = curMin*num[i] if num[i] > 0; else curMax*num[i]
result = max(dp[x][0])
```

```c++
class Solution {
 public:
  int maxProduct(vector<int>& nums) {
    int dp[2][2] = {{nums[0], nums[0]}};
    int result = nums[0];
    for (int i = 1; i < nums.size(); i++) {
      int curr = nums[i];
      int x = i % 2; // rolling
      int y = (i - 1) % 2;
      dp[x][0] = max({dp[y][0] * curr, dp[y][1] * curr, curr});
      dp[x][1] = min({dp[y][0] * curr, dp[y][1] * curr, curr});
      result = max(result, dp[x][0]);
    }
    return result;
  }
};
```

### 121. Best Time to Buy and Sell Stock

```c++
class Solution {
 public:
  int maxProfit(vector<int>& prices) {
    int profit = 0, minimum = prices[0];
    for (const auto& price : prices) {
      profit = max(price - minimum, profit);
      minimum = min(price, minimum);
    }
    return profit;
  }
};
```

`mp[0][0]`, `mp[0][1]`, and `mp[0][2]` represent the maximum profit when holding no stock, holding one stock, and holding no stock after selling, respectively.

```c++
class Solution {
 public:
  int maxProfit(vector<int>& prices) {
    // Initialize a 2D array to store maximum profits at each day
    int mp[2][3] = {{0, -prices[0], 0}};
    // Get the size of the prices array
    int n = prices.size();
    // Iterate through the prices array
    for (int i = 1; i < n; ++i) {
      // Calculate the indices of the current and previous rows in the 2D array
      int x = (i + 1) % 2, y = i % 2;
      // Update the maximum profit for not holding any stock
      mp[y][0] = mp[x][0];
      // Update the maximum profit for holding a stock
      mp[y][1] = max(mp[x][1], mp[x][0] - prices[i]);
      // Update the maximum profit for selling a stock
      mp[y][2] = max(mp[x][2], mp[x][1] + prices[i]);
    }
    // Get the index of the last row in the 2D array
    int end = (n - 1) % 2;
    // Return the maximum profit that can be made by either not holding any stock or selling a stock
    return max(mp[end][2], mp[end][0]);
  }
};
```

### 122. Best Time to Buy and Sell Stock II

```c++
class Solution {
 public:
  int maxProfit(vector<int>& prices) {
    int mp[2][2] = {{0, -prices[0]}};
    int n = prices.size();
    for (int i = 1; i < n; ++i) {
      // Calculate the indices of the current and previous rows in the 2D array
      int x = (i + 1) % 2, y = i % 2;
      // Update the maximum profit for not holding any stock
      mp[y][0] = max(mp[x][0], mp[x][1] + prices[i]);
      // Update the maximum profit for holding a stock
      mp[y][1] = max(mp[x][1], mp[x][0] - prices[i]);
    }
    // Return the maximum profit that can be made by not holding any stock at the end of the period
    return mp[(n - 1) % 2][0];
  }
};
```

### 123. Best Time to Buy and Sell Stock III

```c++
class Solution {
 public:
  int maxProfit(vector<int>& prices) {
    // Initialize 3D array to store max profit for each transaction and day.
    int mp[2][3][2] = {0};
 
    // Set initial values based on price of first day.
    mp[0][0][1] = mp[0][1][1] = -prices[0];
    
    int n = prices.size();
    for (int i = 1; i < n; ++i) {
      int x = (i + 1) % 2, y = i % 2;
      // Calculate max profit for no transaction today with stock.
      mp[y][0][1] = max(mp[x][0][1], -prices[i]);
      // Calculate max profit for one transaction today with no stock or with stock.
      mp[y][1][0] = max(mp[x][1][0], mp[x][0][1] + prices[i]);
      mp[y][1][1] = max(mp[x][1][1], mp[x][1][0] - prices[i]);
      // Calculate max profit for two transactions today with no stock.
      mp[y][2][0] = max(mp[x][2][0], mp[x][1][1] + prices[i]);
    }
    
    // Only care about the final state.
    int end = (n - 1) % 2;
    return max({mp[end][2][0], mp[end][1][0], 0});
  }
};
```

```c++
class Solution {
public:
    int maxProfit(const vector<int>& prices) {
        int n = prices.size();
        if (n < 2) return 0;
        
        // First transaction
        int buy1 = -prices[0];
        int sell1 = 0;
        
        // Second transaction
        int buy2 = -prices[0];
        int sell2 = 0;
        
        for (int i = 1; i < n; i++) {
            // Update first transaction
            buy1 = max(buy1, -prices[i]);
            sell1 = max(sell1, buy1 + prices[i]);
            
            // Update second transaction with profit from first transaction
            buy2 = max(buy2, sell1 - prices[i]);
            sell2 = max(sell2, buy2 + prices[i]);
        }
        
        return sell2;
    }
};
```

### 188. Best Time to Buy and Sell Stock IV

```
到 i 的 max profit (mp[i][j][k])
i 天数
j 是否拥有股票
k 之前交易过多少次
for i
	for k
		mp[i,k,0] = max(mp[i-1,k,0], mp[i-1,k-1,1]+a[i])
		mp[i,k,1] = max(mp[i-1,k,1], mp[i-1,k-1,0]-a[i])
return max(mp[n-1,j={0-k},0])
O(n*k)

cooldown: k 换成 0和1

如果股票可以累加: 0和1 换成 j(0-x)
for i
	for k
		for j
            mp[i,k,j] = max(mp[i-1,k,j], mp[i-1,k-1,j+1]+a[i], mp[i-1,k-1,j-1]-a[i])
O(n*k*x)
```

```c++
class Solution {
 public:
  int maxProfit(int k, vector<int>& prices) {
    int n = prices.size();
    vector<vector<int>> hold0(n, vector<int>(k + 1, 0));
    vector<vector<int>> hold1(n, vector<int>(k + 1, -prices[0]));
    for (int i = 1; i < n; ++i) {
      for (int j = 1; j < k + 1; ++j) {
        hold1[i][j] = std::max(hold1[i - 1][j], hold0[i - 1][j - 1] - prices[i]);
        hold0[i][j] = std::max(hold0[i - 1][j], hold1[i - 1][j] + prices[i]);
      }
    }
    return *std::max_element(hold0[n - 1].begin(), hold0[n - 1].end());
  }
};
```

```c++
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int n = prices.size();
        if (k >= n / 2) { // if k is large enough, we can make unlimited transactions
            int max_profit = 0;
            for (int i = 1; i < n; i++) {
                if (prices[i] > prices[i-1]) { // if price increases, buy and sell on the same day
                    max_profit += prices[i] - prices[i-1];
                }
            }
            return max_profit;
        }
        vector<vector<int>> dp(k+1, vector<int>(n));
        for (int i = 1; i <= k; i++) { // iterate over the number of transactions allowed
            int max_diff = -prices[0]; // maximum difference between the current price and previous prices seen so far
            for (int j = 1; j < n; j++) { // iterate over days
                dp[i][j] = max(dp[i][j-1], prices[j]+max_diff); // compare profit from previous transaction vs. buying and selling on current day
                max_diff = max(max_diff, dp[i-1][j-1]-prices[j]); // calculate the maximum difference up to this day, taking into account the previous transactions
            }
        }
        return dp[k][n-1]; // return the maximum profit with k transactions up to the last day
    }
};
```

### 309. Best Time to Buy and Sell Stock with Cooldown ⭐

```c++
s0[i] = max(s0[i - 1], s2[i - 1]);  
// Stay at s0, or rest from s2, no stock now, and the max profit should be 'last no stock profit or 'last rest profit'
s1[i] = max(s1[i - 1], s0[i - 1] - prices[i]);  
// Stay at s1, or buy from s0 have stock, and the profit should be 'last stock profit' or 'last no stock but buy this time'
s2[i] = s1[i - 1] + prices[i];  
// Only one way from s1, we should sell then take a rest
```

```c++
class Solution {
 public:
  int maxProfit(vector<int>& prices) {
    int mp[2][3] = {{0, -prices[0], 0}};
    int n = prices.size();
    for (int i = 1; i < n; ++i) {
      int x = (i + 1) % 2, y = i % 2;
      mp[y][0] = max(mp[x][0], mp[x][2]);
      mp[y][1] = max(mp[x][1], mp[x][0] - prices[i]);
      mp[y][2] = mp[x][1] + prices[i];
    }
    return max(mp[(n - 1) % 2][0], mp[(n - 1) % 2][2]);
  }
};
```

### 714. Best Time to Buy and Sell Stock with Transaction Fee

```c++
class Solution {
 public:
  int maxProfit(vector<int>& prices, int fee) {
    int mp[2][2] = {{0, -prices[0]}};
    int n = prices.size();
    for (int i = 1; i < n; ++i) {
      int x = (i + 1) % 2, y = i % 2;
      mp[y][0] = max(mp[x][0], mp[x][1] + prices[i] - fee);
      mp[y][1] = max(mp[x][1], mp[x][0] - prices[i]);
    }
    return mp[(n - 1) % 2][0];
  }
};
```

### 300. Longest Increasing Subsequence

```
dp[i] 从0-i, 包含i的子序列最长长度
for i = 0 - n-1
	dp[i] = max{dp[j]} + 1
	j = 0 - i - 1 && a[j] < a[i]
dp: O(n^2)
二分: O(nlogn)
```

```c++
class Solution {
 public:
  int lengthOfLIS(vector<int>& nums) {
    int n = nums.size();
    vector<int> dp(n, 1);
    int ans = 1;
    for (int i = 1; i < n; i++) {
      for (int j = 0; j < i; j++) {
        if (nums[i] > nums[j]) {
          dp[i] = max(dp[i], dp[j] + 1);
        }
      }
      ans = max(ans, dp[i]);
    }
    return ans;
  }
};
```

```c++
class Solution {
 public:
  int lengthOfLIS(vector<int>& nums) {
    vector<int> LIS;
    for (int num : nums) {
      auto it = lower_bound(LIS.begin(), LIS.end(), num); // first elem greater or equal to num
      if (it == LIS.end()) {
        LIS.push_back(num);
      } else {
        *it = num;
      }
    }
    return LIS.size();
  }
};
```

### 322. Coin Change

```
dp[i] 到达i面值最小硬币数
dp[i] = min{dp[i-coins[j]]} + 1
coins[j] = 1, 2, 5
return dp[x] 
O(X*N)
```

```c++
class Solution {
 public:
  int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, amount + 1);
    dp[0] = 0;
    for (int i = 1; i <= amount; ++i) {
      for (const auto& coin : coins) {
        if (i - coin >= 0) {
          dp[i] = min(dp[i], dp[i - coin] + 1);
        }
      }
    }
    return dp[amount] > amount ? -1 : dp[amount];
  }
};
```

### 72. Edit Distance

```
BFS
dp[i][j]
i: word1的字符位
j: word2的字符位
word1的前i字符替换到word2的前j个字符, 最少需要步数
return dp[m][n]

dp[i][j] = 
    if w1[i] == w2[j]
        dp[i-1][j-1]
    else 
    	1 + min(insert, delete, replace)
    	dp[i][j-1], dp[i-1][j], dp[i-1][j-1]
    	

```

```c++
class Solution {
 public:
  int minDistance(string word1, string word2) {
    int m = word1.size(), n = word2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1)); // m * n matrix
    for (int i = 0; i <= m; ++i) { // <=
      dp[i][0] = i;
    }
    for (int j = 0; j <= n; ++j) {
      dp[0][j] = j;
    }
    for (int i = 1; i <= m; ++i) {
      for (int j = 1; j <= n; ++j) {
        if (word1[i - 1] == word2[j - 1]) { // i - 1, j - 1
          dp[i][j] = dp[i - 1][j - 1];
        } else {
          dp[i][j] = min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]}) + 1;
        }
      }
    }
    return dp[m][n];
  }
};
```

## 面试准备

把面试官当成未来的同事，而不是监考老师。

### 切题姿势

1. Clarification（询问题目细节、边界条件、可能的极端错误情况）
   - 数组范围
   - 数据范围
2. Possible Solution（所有可能的解法都和面试官沟通一遍）
  - Compare Time & Space Complexity
  - Optimal Solution（最优解）
3. Coding
4. Test Cases

## 精通一个领域

- Chunk it up（切碎知识点）
- Deliberate practicing（刻意练习）
- Feedback（获得反馈）

### 获得反馈

- 即时反馈
- 主动型反馈（自己去找）
  - 高手代码 (GitHub, LeetCode, etc.)
  - 第一视角直播
- 被动式反馈（高手给你指点）
  - Code Review