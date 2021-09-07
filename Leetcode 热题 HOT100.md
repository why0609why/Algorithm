# Leetcode 热题 HOT100



# <font color='red'>6. Z字型变换</font>

## 题意

将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：

P   A   H   N
A P L S I I G
Y   I   R
之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。

请你实现这个将字符串进行指定行数变换的函数：

string convert(string s, int numRows);



## 示例1

```
输入：s = "PAYPALISHIRING", numRows = 3
输出："PAHNAPLSIIGYIR"
```



## 示例2

```
输入：s = "PAYPALISHIRING", numRows = 4
输出："PINALSIGYAHRPI"
解释：
P     I    N
A   L S  I G
Y A   H R
P     I
```



## 示例3

```
输入：s = "A", numRows = 1
输出："A"
```



## 解法

参考链接

https://leetcode-cn.com/problems/zigzag-conversion/solution/zzi-xing-bian-huan-by-jyd/

既然要把每一行的内容都记录下来，比如说一共n行，那就创建n个字符串（采用字符串数组实现），然后遍历字符串，把读到的字符放到正确属于他的那一行---也就是那一个字符串里面。

```java
class Solution {
    public String convert(String s, int n) {
        if (n < 2) {
            return s;
        }

        //创建一个字符串数组，里面包括n个字符串，每个字符串用来保存一行的信息
        List<StringBuffer> res = new ArrayList<>();
        //初始化字符串数组
        for (int i = 0; i < n; i++) {
            res.add(new StringBuffer());
        }
        //flag表示i如何变化
        int flag = -1;
        //i表示行号
        int i = 0;
        //最后返回的结果
        StringBuffer ans = new StringBuffer();

        //z字打印，然后分别用每一个字符串存储每一行的内容
        for (char c : s.toCharArray()) {
            res.get(i).append(c);
            //如果到了两端，就改变方向
            if (i == 0 || i == n - 1) {
                flag *= -1;
            }
            i += flag;
        }

        //拼接每个字符串
        for (StringBuffer ss : res) {
            ans.append(ss);
        }

        return ans.toString();
    }

}
```















# <font color='red'>11. 盛水最多的容器</font>

## 题意

给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

说明：你不能倾斜容器。



## 示例1

![image-20210501104813190](C:\Users\84334\AppData\Roaming\Typora\typora-user-images\image-20210501104813190.png)

```
输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
```



## 示例2

```
输入：height = [1,1]
输出：1
```



## 示例3

```
输入：height = [4,3,2,1,4]
输出：16
```



## 示例4

```
输入：height = [1,2,1]
输出：2
```



## 解法1 暴力（超时）

既然要求找出最大面积，那我就找出所有的两根柱子可以组成的最大面积。从中取最大

```java
lass Solution {
    public int maxArea(int[] nums) {
        int ans = 0;
        int len = nums.length;

        //遍历所有面积
        for (int i = 0; i < len; i++) {
            for (int j = i + 1; j < len; j++) {
                //因为两根柱子不一定谁高，所以找出两根柱子里面的低的那个，j-i是底
                int temp_area = Math.min(nums[i], nums[j]) * (j - i);
                //更新最大值
                ans = Math.max(ans, temp_area);
            }
        }

        return ans;
    }
}
```



## 解法2 双指针

参考链接 https://leetcode-cn.com/problems/container-with-most-water/solution/on-shuang-zhi-zhen-jie-fa-li-jie-zheng-que-xing-tu/

![image-20210501110720654](C:\Users\84334\AppData\Roaming\Typora\typora-user-images\image-20210501110720654.png)

我们举个例子，假如我们从最旁边的两根柱子开始选择，我们可以看到，1柱子比5柱子小，那么1柱子和2,3,4柱子的组合产生的面积一定比1和5产生的面积小。

为什么？

我们可以举例子，2,3,4柱子无非比5高，要么比5矮。

- 比5高的柱子，比如4柱子，由于还是和1柱子组合，最终的矩形面积高还是取决于1柱子，高不变，而由于矩形面积的宽变小了，所以面积小了。
- 比5矮的柱子，比如柱子3，那这个就很明显了，矩形面积的高取决于柱子3了，高变小了，而且宽也变小了，面积更小了。

所以经过我们的分析后，我们发现，只要柱子1比柱子5小，那么柱子1和剩下的柱子的组合我们都可以不考虑了，所以我们可以舍弃柱子1，考虑柱子2的问题。因为此时，如果你想要更大的面积，只能从柱子1到柱子5里面去找新的面积了。



所以我们可以总结一个规律：

我们从两边取两个柱子，柱子a与柱子b，发现一个柱子比另一个柱子小，比如柱子a小于柱子b，那么a柱子和剩下柱子的组合就不考虑了，我们看a柱子的右边，如果b柱子小，那么我们就看b柱子的左边。

```java
class Solution {
    public int maxArea(int[] nums) {
        int ans = 0;
        int len = nums.length;
        //双指针
        int i = 0;
        int j = len - 1;

        while (i < j) {
            //如果左面的柱子小
            if (nums[i] < nums[j]) {
                //先更新一下面积
                ans = Math.max(ans, Math.min(nums[i], nums[j]) * (j - i));
                //舍弃掉左面的柱子
                i++;
            } else {
                ans = Math.max(ans, Math.min(nums[i], nums[j]) * (j - i));
                //否则就是舍弃掉右面的柱子
                j--;
            }
        }

        return ans;
    }
}
```







# <font color='red'>19 删除链表的倒数第N个节点</font>

## 示例1

```
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

进阶：你能尝试使用一趟扫描实现吗？
```

## 示例2

```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
```

## 示例3

```
输入：head = [1], n = 1
输出：[]
```

## 提示

```
输入：head = [1,2], n = 1
输出：[1]
```



## 解法1，遍历一遍，求长度，然后找倒数的节点

求长度，找到倒数第N个节点的上一个节点，然后删除就好了

```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        int len = 0;
        ListNode dummy = new ListNode();
        dummy.next = head;

        //求长度
        ListNode cur = dummy;
        while (cur.next != null) {
            len++;
            cur = cur.next;
        }

        //找倒数第N个节点的上一个
        cur = dummy;
        int times = len - n;
        while (times > 0) {
            times--;
            cur = cur.next;
        }

        cur.next = cur.next.next;
        return dummy.next;
    }
}
```



## 解法2，用栈找到倒数节点的前一个节点

先把所有节点放到栈里面，然后弹出n个节点，此时栈顶的节点就是倒数第N个节点的上一个节点了。

```java
import java.util.*;

class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode();
        dummy.next = head;

        //注意这里需要把
        Stack<ListNode> stack = new Stack<>();
        ListNode cur = dummy;
        while (cur != null) {
            stack.push(cur);
            cur = cur.next;
        }

        while (n > 0) {
            n--;
            stack.pop();
        }

        if (stack.isEmpty()) {
            return null;
        }
        cur = stack.peek();
        cur.next = cur.next.next;
        return dummy.next;
    }
}
```



## 解法3，快慢指针

快指针先走n步，然后慢指针不动，当快指针走到最后一个的时候，慢指针就走到倒数第N个节点的上一个了。

```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode();
        dummy.next = head;

        ListNode fast = dummy;
        ListNode low = dummy;

		//快指针先走n步
        while (n > 0) {
            n--;
            fast = fast.next;
        }

		//当快指针走到最后一个的时候，慢指针就走到了倒数节点的前面了
        while (fast.next != null) {
            fast = fast.next;
            low = low.next;
        }

		//删除节点。
        low.next = low.next.next;
        return dummy.next;
    }
}
```





# <font color='red'>31下一个排列</font>

## 题意

```
实现获取 下一个排列 的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须 原地 修改，只允许使用额外常数空间。
```



## 示例1

```
输入：nums = [1,2,3]
输出：[1,3,2]
```

## 示例2

```
输入：nums = [3,2,1]
输出：[1,2,3]
```

## 示例3

```
输入：nums = [1,1,5]
输出：[1,5,1]
```

## 示例4

```
输入：nums = [1]
输出：[1]
```

## 提示

```
1 <= nums.length <= 100
0 <= nums[i] <= 100
```



## 解法

![image-20210602174408014](C:\Users\84334\AppData\Roaming\Typora\typora-user-images\image-20210602174408014.png)

想想全排列的逻辑，其实最重要的也就是两个数，一个是上面说的较小的数，另一个是较大的数。以4,5,2,6,3,1举例子，每次book数组更新的一定是从小往大更新的。

感觉下一个排列就像是一个公式，你只需要找到较小数，索引为i，和较大数索引为j，然后交换两者的位置，然后倒置[i+1,n)就行了。

```java
class Solution {
    public void nextPermutation(int[] nums) {
        if (nums.length == 1) {
            return;
        }

        //从后往前找第一个对a[i]<a[i+1]
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }

		//i如果小于0就说明当前序列是最大的排列
        if (i >= 0) {
            //此时的i就是较小数的地方
            int j = nums.length - 1;
            //从后往前找逆序数
            while (j >= i + 1 && nums[i] >= nums[j]) {
                j--;
            }
            swap(nums, i, j);
        }

        reverse(nums, i + 1);
    }

    public void swap(int[] nums, int i, int j) {
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }

    public void reverse(int[] nums, int start) {
        int i = start;
        int j = nums.length - 1;
        while (i < j) {
            int t = nums[i];
            nums[i] = nums[j];
            nums[j] = t;

            i++;
            j--;
        }
    }
}
```







# <font color='red'>56 合并区间</font>

## 题意

```
以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间
```



## 示例1

```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
```



## 示例2

```
输入：intervals = [[1,4],[4,5]]
输出：[[1,5]]
解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。
```



## 解法1 手写快排+模拟

首先先把区间数组按照开始地点升序排序，这里我采用的是快排的板子

```java
public class Solution {
    private void quickSort(int[] nums, int l, int r) {
        int q;
        if (l < r) {
            q = partition(nums, l, r);
            quickSort(nums, l, q - 1);
            quickSort(nums, q + 1, r);
        }
    }

    private int partition(int[] nums, int l, int r) {
        int x, i, j, t;
        x = nums[r];
        i = l - 1;
        //从j区间开始选择数
        for (j = l; j < r; j++) {
            //如果发现这是i区间的数
            if (nums[j] < x) {
                //就把数放到i区间里面
                i++;
                t = nums[i];
                nums[i] = nums[j];
                nums[j] = t;
            }

        }
        //基准数归位
        t = nums[i + 1];
        nums[i + 1] = nums[r];
        nums[r] = t;
        return i + 1;
    }

    public static void main(String[] args) {
        int[] nums = {1, 5, 3, 2, 6, 4, 7,-3213123,3242342,4645,323523};
        Solution solution = new Solution();
        solution.quickSort(nums, 0, nums.length - 1);
        
        for (int num : nums) {
            System.out.println(num);
        }
    }
}
```

首先，我们将列表中的区间按照左端点升序排序。然后我们将第一个区间加入 `merged` 数组中，并按顺序依次考虑之后的每个区间：

- 如果当前区间的左端点在数组 `merged` 中最后一个区间的右端点之后，那么它们不会重合，我们可以直接将这个区间加入数组 `merged` 的末尾；
- 否则，它们重合，我们需要用当前区间的右端点更新数组 `merged` 中最后一个区间的右端点，将其置为二者的较大值。

按照上面说的算法，就可以进行区间整合了



下面是手写的快排

```java
import java.util.*;

class Solution {
    public int[][] merge(int[][] nums) {
        //按照首位排序
       sort(nums, 0, nums.length - 1);

        //把第一个区间直接放到ans里面
        List<int[]> ans = new LinkedList<>();
        ans.add(new int[]{nums[0][0], nums[0][1]});

        //区间合并的过程
        for (int i = 1; i < nums.length; i++) {
            if (nums[i][0] > ans.get(ans.size() - 1)[1]) {
                ans.add(new int[]{nums[i][0], nums[i][1]});
            } else {
                ans.get(ans.size() - 1)[1] =
                        Math.max(nums[i][1], ans.get(ans.size() - 1)[1]);
            }
        }

        //这个参数其实就是list要放到什么样的数组里面
        return ans.toArray(new int[ans.size()][]);
    }

    public void sort(int[][] nums, int l, int r) {
        int q = 0;
        if (l < r) {
            q = partition(nums, l, r);
            sort(nums, l, q - 1);
            sort(nums, q + 1, r);
        }

    }

    public int partition(int[][] nums, int l, int r) {
        //x是基准数
        int[] x = new int[2];
        int[] t = new int[2];
        int i, j;

        //基准数赋值
        x = nums[r];
        i = l - 1;
        for (j = l; j < r; j++) {
            if (nums[j][0] < x[0]) {
                i++;
                t = nums[i];
                nums[i] = nums[j];
                nums[j] = t;
            }

        }

        t = nums[i + 1];
        nums[i + 1] = nums[r];
        nums[r] = t;

        return i + 1;
    }
}
```



## 解法2 库的快排+模拟

原理同解法1，只不过排序的方法是重写Comparator

这里注意，优先级，Comparator的compare方法的概述

![image-20210601211017645](C:\Users\84334\AppData\Roaming\Typora\typora-user-images\image-20210601211017645.png)

返回-1表示第一个数比第二个数小，返回0表示相等，返回1表示第一个数比第二个数大。

而java默认是最小堆，-1表示优先级高。

下面的写法返回o1-o2，假如o1比o2小，那么返回的就是-1，那么优先级就是o1比o2小，这正好是我们排序要的规则（从小到大排）。



```java
import java.util.*;

class Solution {
    public int[][] merge(int[][] nums) {
        //按照首位排序
        Arrays.sort(nums, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });

        //把第一个区间直接放到ans里面
        List<int[]> ans = new LinkedList<>();
        ans.add(new int[]{nums[0][0], nums[0][1]});

        //区间合并的过程
        for (int i = 1; i < nums.length; i++) {
            if (nums[i][0] > ans.get(ans.size() - 1)[1]) {
                ans.add(new int[]{nums[i][0], nums[i][1]});
            } else {
                ans.get(ans.size() - 1)[1] =
                        Math.max(nums[i][1], ans.get(ans.size() - 1)[1]);
            }
        }

        //这个参数其实就是list要放到什么样的数组里面
        return ans.toArray(new int[ans.size()][]);
    }

    public void sort(int[][] nums, int l, int r) {
        int q = 0;
        if (l < r) {
            q = partition(nums, l, r);
            sort(nums, l, q - 1);
            sort(nums, q + 1, r);
        }

    }

    public int partition(int[][] nums, int l, int r) {
        //x是基准数
        int[] x = new int[2];
        int[] t = new int[2];
        int i, j;

        //基准数赋值
        x = nums[r];
        i = l - 1;
        for (j = l; j < r; j++) {
            if (nums[j][0] < x[0]) {
                i++;
                t = nums[i];
                nums[i] = nums[j];
                nums[j] = t;
            }

        }

        t = nums[i + 1];
        nums[i + 1] = nums[r];
        nums[r] = t;

        return i + 1;
    }
}
```





# <font color='red'>62. 不同路径</font>

## 题意

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？



## 示例1

```
输入：m = 3, n = 7
输出：28
```

![img](https://assets.leetcode.com/uploads/2018/10/22/robot_maze.png)



## 示例2

```
输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。

1. 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右
3. 向下 -> 向右 -> 向下
```



## 示例3

```
输入：m = 7, n = 3
输出：28
```



## 示例4

```
输入：m = 3, n = 3
输出：6
```



## 提示

```
1 <= m, n <= 100
题目数据保证答案小于等于 2 * 109
```



## 解法1 递归（超时）

从坐标（1,1）开始往下或者往右递归。

只要不越界，计数器就加一，因为是递归，所以路径不可能重复，不用考虑重复的问题。



```java
class Solution {
    //记录的结果
    int ans = 0;

    public int uniquePaths(int m, int n) {
        dfs(m, n, 1, 1);
        return ans;
    }

    public void dfs(int m, int n, int i, int j) {
        //只要不越界
        if (i > m || j > n) {
            return;
        }

        //如果递归到左下角
        if (i == m && j == n) {
            ans++;
        }

        //往下或者往右递归
        dfs(m, n, i + 1, j);
        dfs(m, n, i, j + 1);
    }
}
```





## 解法2 dp

这题很明显有动态规划方程

```java
dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
```

当前这一格只能从上面走下来或者从左面走过来。



```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        
        //排除特殊情况
        if (m == 1 && n == 1) {
            return 1;
        } else {
            //先给第一行的所有走法赋值1
            for (int i = 0; i < m; i++)
                dp[i][0] = 1;
            //再给第一列的所有走法赋值1
            for (int j = 0; j < n; j++)
                dp[0][j] = 1;

            //剩下的格子就是上面的动态规划
            for (int i = 1; i < m; i++) {
                for (int j = 1; j < n; j++) {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }

            return dp[m - 1][n - 1];
        }
    }
}
```



# <font color='red'>102 二叉树的层序遍历</font>

## 题意

```
给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）。

 

示例：
二叉树：[3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回其层序遍历结果：

[
  [3],
  [9,20],
  [15,7]
]
```



## 解法 常规的层序遍历

```java
import java.util.*;

class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        //非空判断
        if (root == null)
            return new LinkedList<>();

        //存放结果的集合
        List<List<Integer>> ans = new LinkedList<>();
        Queue<TreeNode> q = new LinkedList<>();

        //常规bfs
        q.add(root);
        while (!q.isEmpty()) {
            List<Integer> temp = new LinkedList<>();
            int times = q.size();
            for (int i = 0; i < times; i++) {
                TreeNode remove = q.remove();
                temp.add(remove.val);
                if (remove.left != null) {
                    q.add(remove.left);
                }
                if (remove.right != null) {
                    q.add(remove.right);
                }
            }
            ans.add(temp);
        }
        return ans;
    }
}
```









# <font color='red'>128 最长连续序列</font>

## 题意

给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

 

进阶：你可以设计并实现时间复杂度为 O(n) 的解决方案吗？



## 示例1

```
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
```



## 示例2

```
输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9
```



## 提示

```
0 <= nums.length <= 104
-109 <= nums[i] <= 109
```



## 解法 哈希表+暴力优化

既然要在乱序的数组里面找到最长的连续序列，至少得把数据存储到哈希表里面吧？至少我们得知道一个数是不是在这个数组里面存在。

然后自然可以想到的是，直接遍历每一个数，拿每一个数做起点，然后往后一个一个的找连续的数字是不是存储在数组里面（找的过程用哈希表找）并且找的过程中不断更新最大长度。

这个是暴力的做法，但是有点太暴力了，如果对于每一个数都进行找的话，假如一个数组长度是1^9这么长的连续序列，那么岂不是每个数都跑了炒鸡长的循环。

我们其实可以发现一个规律，因为题目要求的是连续的数字，假如说数组里面是1,2,3,4,5，这5个数字，我们拿1做起点的时候，就找了一次1,2,3,4,5，如果那2做起点的话，最多就是2,3,4,5，长度肯定是比1小的。所以说我们知道了，假设num是当前的起点，如果num-1存在的话，就不用拿num做起点了。



具体的证明可以看官方的题解

https://leetcode-cn.com/problems/longest-consecutive-sequence/solution/zui-chang-lian-xu-xu-lie-by-leetcode-solution/	





# <font color='red'>142 环形链表2</font>

## 题意

```
给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅是用于标识环的情况，并不会作为参数传递到函数中。

说明：不允许修改给定的链表。
```

## 示例1

```
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。
```

## 示例2

```
输入：head = [1,2], pos = 0
输出：返回索引为 0 的链表节点
解释：链表中有一个环，其尾部连接到第一个节点。
```

## 示例3

```
输入：head = [1], pos = -1
输出：返回 null
解释：链表中没有环。
```

## 提示

```
链表中节点的数目范围在范围 [0, 104] 内
-105 <= Node.val <= 105
pos 的值为 -1 或者链表中的一个有效索引
```





## 解法1 哈希表

哈希表存储走过的节点，如果当前节点在哈希表里面存在，那就说明这个节点就是环的入口

```java
import java.util.*;

public class Solution {
    public ListNode detectCycle(ListNode head) {
        //哈希表存储
        Set<ListNode> visited = new HashSet<>();

        ListNode cur = head;
        while (cur != null) {
            //如果哈希表存储，那么当前节点就是环的入口
            if (visited.contains(cur)) {
                return cur;
            } else {
                //如果没走过，那就记录下地址
                visited.add(cur);
                cur = cur.next;
            }
        }
        
        return null;
    }
}
```



## 解法2 公式法

![image-20210602231954628](C:\Users\84334\AppData\Roaming\Typora\typora-user-images\image-20210602231954628.png)

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode slow;
        ListNode fast;

        slow = head;
        fast = head;

        //先让fast和slow一直走
        while (fast != null && slow != null) {
            slow = slow.next;
            if (fast.next == null) {
                return null;    
            }

            fast = fast.next.next;
            //当fast和slow相遇的时候
            if (fast == slow) {
                //拿一个新的指针ptr，ptr从头走
                ListNode ptr = head;
                //让ptr与slow相遇
                while (ptr != slow) {
                    ptr = ptr.next;
                    slow = slow.next;
                }

                //按照公式，相遇的地方就是环的入口
                return ptr;
            }
        }
        return null;
    }
}
```



# <font color='red'>152 乘积最大子数组</font>

## 题意

```
给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
```

## 示例1

```
输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
```

## 示例2

```
输入: [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
```



## 解法 动态规划

![image-20210603213249039](C:\Users\84334\AppData\Roaming\Typora\typora-user-images\image-20210603213249039.png)



![image-20210603213306532](C:\Users\84334\AppData\Roaming\Typora\typora-user-images\image-20210603213306532.png)

说几个点吧，因为存在负数的情况，所以需要有两个dp数组，分别存储最大和最小的情况，即代码里面的max数组和min数组，更新最大值的时候不仅要计算`max[i-1]*nums[i]`还要计算 `min[i-1]*nums[i]`，还有 `nums[i]`，这三个进行找最大，同理找最小的也是一样的道理，就是找负数的可能性。









# <font color='red'>200 岛屿数量</font>

## 题意

```
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围
```

## 示例1

```
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```

## 示例2

```
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
```



## 解法 递归

 遍历格子，只要碰到是1的格子，就进行一次dfs，并且每次dfs都朝4个方向去dfs，每次dfs都把走过的格子加上标记就可以了。

```java
class Solution {
    //岛屿数量
    private int ans;
    //标记数组
    private int[][] book;

    public int numIslands(char[][] nums) {
        int row = nums.length;
        int col = nums[0].length;
        book = new int[row][col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                book[i][j] = 0;
            }
        }

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                //如果是陆地并且没有走过
                if (nums[i][j] == '1' && book[i][j] == 0) {
                    dfs(nums, row, col, i, j);
                    ans++;
                }
            }
        }

        return ans;
    }

    private void dfs(char[][] nums, int row, int col, int i, int j) {
        //越界判断
        if (i >= row || j >= col || i < 0 || j < 0) {
            return;
        }

        //走过了
        if (book[i][j] == 1) {
            return;
        }

        //如果不是陆地
        if (nums[i][j] == '0') {
            return;
        }
        
        //标记为走过
        book[i][j] = 1;
        //朝4个方向走
        dfs(nums, row, col, i + 1, j);
        dfs(nums, row, col, i - 1, j);
        dfs(nums, row, col, i, j + 1);
        dfs(nums, row, col, i, j - 1);
    }

    public static void main(String[] args) {
        char[][] nums = new char[][]{{'1', '1', '0', '0', '0'},
                {'1', '1', '0', '0', '0'}, {'0', '0', '1', '0', '0'},
                {'0', '0', '0', '1', '1'}};

        new Solution().numIslands(nums);
    }
}
```







# <font color='red'>215 数组中的第K大最大元素</font>

## 题意

```
在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
```

## 示例1

```
输入: [3,2,1,5,6,4] 和 k = 2
输出: 5
```

## 示例2

```
输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4
```



## 解法1 手写快排，直接从索引拿

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        sort(nums, 0, nums.length - 1);
        return nums[k - 1];
    }

    //手写快排
    private void sort(int[] nums, int l, int r) {
        if (l < r) {
            int q = partition(nums, l, r);
            sort(nums, l, q - 1);
            sort(nums, q + 1, r);
        }
    }

    private int partition(int[] nums, int l, int r) {
        int i, j, x, t;
        x = nums[r];
        i = l - 1;

        for (j = l; j < r; j++) {
            //因为要求的是从大到小排的，所以这里是大于号
            if (nums[j] > x) {
                i++;
                t = nums[i];
                nums[i] = nums[j];
                nums[j] = t;
            }
        }

        t = nums[i + 1];
        nums[i + 1] = nums[r];
        nums[r] = t;
        return i + 1;
    }
}
```





## 解法2 堆

java默认是最小堆，也就是谁小谁的优先级大，谁小谁在堆顶

```java
import java.util.*;

class Solution {
    public int findKthLargest(int[] nums, int k) {
    	//造一个最大堆
        PriorityQueue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }

        });

		//所有元素入堆
        for (int num : nums) {
            queue.add(num);
        }

		//因为要第K大，所以pop出k-1个元素后，堆顶元素就是
        while (k > 1) {
            k--;
            queue.remove();
        }
        return queue.peek();
    }

}
```



## 解法3 二分partition

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        int q, l, r;
        l = 0;
        r = nums.length - 1;
        //二分partition
        while (l < r) {
            q = partition(nums, l, r);
            //如果找到了第k-1的，也就是
            if (q == k - 1) {
                break;
            }

            //如果此次归为在k-1(其实是第K大元素的索引)的左面，那就在右面的区间partition
            if (q < k - 1) {
                l = q + 1;
                q = partition(nums, l, r);
                //否则就是在左面partition
            } else {
                r = q - 1;
                q = partition(nums, l, r);
            }
        }

        return nums[k - 1];
    }


    //从大到小的partition操作
    private int partition(int[] nums, int l, int r) {
        int i, j, x, t;
        x = nums[r];
        i = l - 1;

        for (j = l; j < r; j++) {
            //因为要求的是从大到小排的，所以这里是大于号
            if (nums[j] > x) {
                i++;
                t = nums[i];
                nums[i] = nums[j];
                nums[j] = t;
            }
        }

        t = nums[i + 1];
        nums[i + 1] = nums[r];
        nums[r] = t;
        return i + 1;
    }

    public static void main(String[] args) {
        int[] nums = new int[]{3, 2, 1, 5, 6, 4};
        System.out.println(new Solution().findKthLargest(nums, 2));
    }
}
```





# <font color='red'>236 二叉树的最近公共祖先</font>

## 题意

```
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
```

## 示例1

```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。
```

## 示例2

```
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出：5
解释：节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。
```

## 提示

```
树中节点数目在范围 [2, 105] 内。
-109 <= Node.val <= 109
所有 Node.val 互不相同 。
p != q
p 和 q 均存在于给定的二叉树中。
```



## 解法1 递归

![image-20210604114039787](C:\Users\84334\AppData\Roaming\Typora\typora-user-images\image-20210604114039787.png)

```java
class Solution {
    //公共祖先节点
    TreeNode ans = null;

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        dfs(root, p, q);
        return ans;
    }

    //如果cur的子树里面有p，或者q，都返回true
    private boolean dfs(TreeNode cur, TreeNode p, TreeNode q) {
        if (cur == null) {
            return false;
        }

        //看看cur的左子树里面有没有p或者q
        boolean flag_left = dfs(cur.left, p, q);
        //看看cur的右子树里面有没有p或者q
        boolean flag_right = dfs(cur.right, p, q);

        //因为p和q不是一个节点
        //这个条件是说cur的左子树里面有p或者q，cur的右子树里面有p或者q，那么cur一定是答案了
        if (flag_left && flag_right ||
            //这个条件是说cur的左子树或者右子树有一个有p或者q，并且cur是p或者q
            (flag_left || flag_right) && (cur.val == p.val || cur.val == q.val)) {
            ans = cur;
        }

        //如果自己的字数有p或者q，或者自己本身就是p或q，就得返回true。
        return flag_left || flag_right || cur.val == p.val || cur.val == q.val;
    }
}
```





## 解法2 存储父节点

先用哈希表存储每个节点的父节点，键值对是当前节点的值和当前节点的父节点的地址，我们根据当前节点的值val，就可以得到当前节点的父节点的地址，然后根据父节点的地址，得到父节点的值，在去哈希表里找，得到父节点的父节点的地址，如此，最后可以找到根节点。

存储了每个节点的父节点，那就可以把p节点的所有父节点知道了，并且是从下往上的。可以把p的所有父节点放到set里面，然后遍历q的所有父节点（注意是从下往上），每找到一个新的父节点就去set里面找，第一个找到的公共的节点就是二叉树的最近公共祖先。

```java
import java.util.*;

class Solution {
    Set<Integer> visited;
    Map<Integer, TreeNode> map;


    public Solution() {
        visited = new HashSet<>();
        map = new HashMap<>();
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        //遍历一遍
        dfs(root);
        TreeNode cur = p;
        //先把p本身加进去
        visited.add(cur.val);
        while (cur != null) {
            //拿到p的父节点
            cur = map.get(cur.val);
            //再把p的所有父节点都放进去
            if (cur != null)
                visited.add(cur.val);
        }
        //把p的左子树都放到set里面标记一下

        cur = q;
        while (cur != null) {
            if (visited.contains(cur.val)) {
                return cur;
            }
            cur = map.get(cur.val);
        }
        return null;
    }

    private void dfs(TreeNode cur) {
        if (cur == null) {
            return;
        }

        if (cur.left != null) {
            //map里面存储的是左节点的值左节点的父节点的值
            //然后我们可以通过父节点的值拿到父节点的地址
            map.put(cur.left.val, cur);
            dfs(cur.left);
        }
        //右子树同理
        if (cur.right != null) {
            map.put(cur.right.val, cur);
            dfs(cur.right);
        }

    }

}
```







# <font color='red'>437 路径总和 III</font>

## 题意

```
给定一个二叉树，它的每个结点都存放着一个整数值。

找出路径和等于给定数值的路径总数。

路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

二叉树不超过1000个节点，且节点数值范围是 [-1000000,1000000] 的整数。
```

## 示例

```
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

返回 3。和等于 8 的路径有:

1.  5 -> 3
2.  5 -> 2 -> 1
3.  -3 -> 11
```



## 解法1 双层递归

首先先说第二层递归，对于当前节点记录从当前节点往其他节点的跑的前缀和（这个用一个局部变量sum就可以实现），然后每到一个节点，看一下当前的sum是不是目标的target，如果是的话计数器就加一。

第二层递归只能求以某个节点为起点的所有路径和是不是target。所以就需要第一层递归，前序遍历所有的节点，让每个节点都当一次起点。然后自然就可以找到，整棵树中的所有路径。

```java
class Solution {
    private int ans;

    public Solution() {
        ans = 0;

    }

    public int pathSum(TreeNode root, int targetSum) {
        recurse(root, targetSum);
        return ans;
    }

    public void recurse(TreeNode cur, int target) {
        if (cur == null) {
            return;
        }

        //对于每个节点都跑一次
        dfs(cur, target, 0);
        recurse(cur.left, target);
        recurse(cur.right, target);
    }

    public void dfs(TreeNode cur, int target, int sum) {
        if (cur == null) {
            return;
        }

        if (sum + cur.val == target) {
            ans++;
        }

        dfs(cur.left, target, sum + cur.val);
        dfs(cur.right, target, sum + cur.val);
    }

}
```



## 解法2 前缀和+哈希表

前缀和的解法https://leetcode-cn.com/problems/path-sum-iii/solution/qian-zhui-he-di-gui-hui-su-by-shi-huo-de-xia-tian/

想想，求和为target的路径，那么一条路径和A，一条路径和为B，如果B比A长，并且B-A==target，那么和为A、B的两条路径和是不是就是题目所求的路径。

我们可以拿一个sum，用来表示到达当前节点的前缀和（这个是可以实现的），再用一个哈希表，用来存储到达当前节点的所有路径和，key是路径和，value是出现的次数（因为你存储的是路径和，路径和有很多种可能，但是他们都是一条路径上的，所以要记录次数）



```java
import java.util.*;

class Solution {
    Map<Integer, Integer> map;
    int target;

    public Solution() {
        map = new HashMap<>();
    }

    public int pathSum(TreeNode root, int targetSum) {
        target = targetSum;
        map.put(0, 1);
        return dfs(root, 0);

    }

    private int dfs(TreeNode cur, int sum) {
        if (cur == null) {
            return 0;
        }

        //更新前缀和
        sum += cur.val;

        //先看一下是不是存在这么一条路径
        //sum表示到当前的节点的前缀和，那么sum-target就是在当前节点cur上面的一条路径和
        int ans = 0;
        if (map.containsKey(sum - target))
            ans += map.get(sum - target);

        //把前缀和放到map里面
        if (!map.containsKey(sum)) {
            map.put(sum, 1);
        } else {
            map.put(sum, map.get(sum) + 1);
        }

        //统计左右子树的结果
        ans = ans + dfs(cur.left, sum) + dfs(cur.right, sum);

        //再把数据从哈希表里面拿出来,这个是保证从上到下路径的内容
        map.put(sum, map.get(sum) - 1);
        
        return ans;
    }
}
```

