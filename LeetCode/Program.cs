using Microsoft.VisualBasic.CompilerServices;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Data.Common;
using System.Linq;
using System.Numerics;
using System.Security;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace LeetCode
{
    class Program
    {
        private static AutoResetEvent event_1 = new AutoResetEvent(true);
        private static AutoResetEvent event_2 = new AutoResetEvent(false);
        static void Main(string[] args)
        {
            //ListNode aa = new ListNode(2) { next = new ListNode(4) { next = new ListNode(3) } };
            //ListNode bb = new ListNode(5) { next = new ListNode(6) { next = new ListNode(4) } };
            //ListNode cc = AddTwoNumbers(aa, bb);
            //int aac = LengthOfLongestSubstring("pwwbcb");
            //string aacc = LongestPalindrome("pweewbcb");
            //IsPalindrome(123343621);
            //int[] nums = { 3, 2, 3, 1, 3, 3, 1 };
            //Console.WriteLine(removeElement2(nums, 3));
            //char[][] grid = new char[4][] { new char[3] { '1', '1', '1' }, new char[3] { '0', '1', '0' }, new char[3] { '1', '0', '0' }, new char[3] { '1', '0', '1' } };
            //char[][] grid = new char[4][] { new char[5] { '1', '1', '1', '1', '0' }, new char[5] { '1', '1', '0', '1', '0' }, new char[5] { '1', '1', '0', '0', '0' }, new char[5] { '0', '0', '0', '0', '0' } };
            //Console.WriteLine(NumIslands(grid, 3));
            //Foo foo = new Foo();
            //Action A = () => { Console.Write("first"); };
            //Action B = () => { Console.Write("second"); };
            //Action C = () => { Console.Write("third"); };
            //Thread t1 = new Thread(() => foo.First(A));
            //t1.Start();
            //Thread t3 = new Thread(() => foo.Third(C));
            //t3.Start();
            //Thread t2 = new Thread(() => foo.Second(B));
            //t2.Start();
            //int[] nums = { 1, 2, 5, 9 }; int threshold = 6;
            //smallestDivisor(nums, threshold);
            //Console.WriteLine(StrStr("hello", "ll"));
            //Console.WriteLine("-----------------------------------------------------------");
            //var test = new FooBar(2);
            //Thread t1 = new Thread(() => test.Foo(() => { Console.Write("foo"); }));
            //Thread t2 = new Thread(() => test.Bar(() => { Console.Write("bar"); }));
            //t1.Start();
            //t2.Start();
            //Action A = () => { Console.Write("H"); };
            //Action B = () => { Console.Write("O"); };
            //Action C = () => { Console.Write("H"); };
            //var h20 = new H2O();
            //Thread t1 = new Thread(() => h20.Hydrogen(A));
            //t1.Start();
            //Thread t2 = new Thread(() => h20.Oxygen(B));
            //t2.Start();
            //Thread t3 = new Thread(() => h20.Hydrogen(C));
            //t3.Start();
            //int[] nums1 = new int[] { 1, 2, 3, 0, 0, 0 };
            //Merge(nums1, 3, new int[] { 2, 5, 6 }, 3);
            //Console.WriteLine(SearchInsert(new int[] { 1, 3, 5, 6 }, 2));
            //Console.WriteLine(Convert("LEFT", 2));
            //Console.WriteLine(NumSquares(7));

            //LRUCache cache = new LRUCache(2 /* 缓存容量 */ );
            //cache.put(1, 1);
            //cache.put(2, 2);
            //cache.get(1);       // 返回  1
            //cache.put(3, 3);    // 该操作会使得密钥 2 作废
            //cache.get(2);       // 返回 -1 (未找到)
            //cache.put(4, 4);    // 该操作会使得密钥 1 作废
            //cache.get(1);       // 返回 -1 (未找到)
            //cache.get(3);       // 返回  3
            //cache.get(4);       // 返回  4
            //var aa = new int[][] { new int[] { 5, 4 }, new int[] { 6, 4 }, new int[] { 6, 7 }, new int[] { 2, 3 } };
            //Console.WriteLine(MaxEnvelopes(aa));
            //Console.WriteLine(LengthOfLIS(new int[] { 10, 9, 2, 5, 3, 7, 101, 18 }));
            //Console.WriteLine(BackspaceCompare("e##e#o##oyof##q", "e##e#fq##o##oyof##q"));
            //ListNode head = new ListNode() { val = 1, next = new ListNode(2, new ListNode(3, new ListNode(4, new ListNode(5)))) };
            //ReorderList(head);
            //Console.WriteLine(IsLongPressedName("leelee", "lleeelee"));
            //Console.WriteLine(PartitionLabels("ababc").ToString());
            //ListNode head = new ListNode() { val = 1, next = new ListNode(2, new ListNode(3, new ListNode(2, new ListNode(1)))) };
            //Console.WriteLine(IsPalindrome(head));
            //var clips = new int[][] { new int []{ 0, 2 }, new int[] { 4, 6}, new int[] { 8, 10},
            //new int[] { 1, 9}, new int[] { 1, 5}, new int[] { 5, 9}};
            //Console.WriteLine(VideoStitching(clips,10));
            //Console.WriteLine(LongestMountain(new int[] { 2, 1, 4, 7, 3, 2, 5 }));
            //Console.WriteLine(LongestMountain(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }));
            //Console.WriteLine(SmallerNumbersThanCurrent(new int[] { 8, 1, 2, 2, 3 }));
            //Console.WriteLine(PreorderTraversal(new TreeNode(1, null, new TreeNode(2, new TreeNode(3)))));
            //Console.WriteLine(SumNumbers(new TreeNode(1, new TreeNode(2, new TreeNode(4)), new TreeNode(3))));
            //Console.WriteLine(IslandPerimeter(new int[][] {
            //    new int[] { 0,1,0,0 },
            //    new int[] { 1,1,1,0 },
            //    new int[] { 0,1,0,0 },
            //    new int[] { 1,1,0,0 },
            //}));

            /**
            // 初始化一个空的集合。
            RandomizedCollection collection = new RandomizedCollection();

            // 向集合中插入 1 。返回 true 表示集合不包含 1 。
            collection.Insert(1);

            // 向集合中插入另一个 1 。返回 false 表示集合包含 1 。集合现在包含 [1,1] 。
            collection.Insert(1);

            // 向集合中插入 2 ，返回 true 。集合现在包含 [1,1,2] 。
            collection.Insert(2);

            // getRandom 应当有 2/3 的概率返回 1 ，1/3 的概率返回 2 。
            collection.GetRandom();

            // 从集合中删除 1 ，返回 true 。集合现在包含 [1,2] 。
            collection.Remove(1);

            // getRandom 应有相同概率返回 1 和 2 。
            collection.GetRandom();
            */
            //var result = WordBreak("catsanddog", new string[5] { "cat", "cats", "and", "sand", "dog" }.ToList());
            //Console.WriteLine(ValidMountainArray(new int[] { 0, 2, 3, 4, 5, 2, 1, 0 }));
            //Console.WriteLine(Insert(new int[][] { new int[] { 1, 3 }, new int[] { 6, 9 } }, new int[] { 2, 5 }));
            //Console.WriteLine(LadderLength("hit", "cog", new List<string> { "hot", "dot", "dog", "lot", "log", "cog" }));
            //Console.WriteLine(SortByBits(new int[] { 2, 3, 5, 7, 11, 13, 17, 19 }));
            //Console.WriteLine(CountRangeSum(new int[] { -2, 5, -1 }, -2, 2));
            //var aa=KClosest(new int[][] { new int[] { 1, 3 }, new int[] { -2, 2 } },1);
            //Console.WriteLine(FindRotateSteps("godding", "gd"));
            //var a = SortArrayByParityII(new int[] { 4, 2, 5, 7 });
            //ListNode head = new ListNode() { val = 1, next = new ListNode(2, new ListNode(3, new ListNode(4, new ListNode(5)))) };
            //Console.WriteLine(OddEvenList(head));
            //SelectSort(new int[] { 1, 0, 5, 2, 7 });
            //var aa = RelativeSortArray(new int[] { 2, 3, 1, 3, 2, 4, 6, 19, 9, 2, 7 }, new int[] { 2, 1, 4, 3, 9, 6 });
            //Console.WriteLine(RemoveKdigits("12345264", 5));
            //Console.WriteLine(ReconstructQueue(new int[][] { new int[] {7, 0 }, new int[] { 4, 4 }, new int[] { 7,1 }, new int[] { 5, 0 }, new int[] { 6, 1 }, new int[] { 5, 2 } })); 
            //Console.WriteLine(AllCellsDistOrder(3, 3, 1, 1));
            //Console.WriteLine(CanCompleteCircuit(new int[] { 1, 2, 3, 4, 5 }, new int[] { 3, 4, 5, 1, 2 }));
            //MoveZeroes(new int[] { 0, 1, 0, 3, 12 });
            //var aa = InsertionSortList(new ListNode(-1, new ListNode(5, new ListNode(3, new ListNode(4, new ListNode(0))))));
            //var aa = SortList(new ListNode(-1, new ListNode(5, new ListNode(3, new ListNode(4, new ListNode(0))))));
            //Console.WriteLine(IsAnagram("anagram", "nagaram"));
            //Console.WriteLine(FindMinArrowShots(new int[][] { new int[] { 10, 16 }, new int[] { 2, 8 }, new int[] { 1, 6 }, new int[] { 7, 12 } }));
            //Console.WriteLine(FindMinArrowShots(new int[][] { new int[] { -2147483646, -2147483645 }, new int[] { 2147483646, 2147483647 } }));
            //Console.WriteLine(CountNodes(new TreeNode(1, new TreeNode(2, new TreeNode(3)), new TreeNode(4))));
            //Console.WriteLine(SortString("leetcode"));
            //Console.WriteLine(MaximumGap(new int[] { 3, 6, 9, 1 }));
            //Console.WriteLine(FourSumCount(new int[] { 1, 2 }, new int[] { -2, -1 }, new int[] { -1, 2 }, new int[] { 0, 2 }));
            Console.WriteLine(LargestPerimeter(new int[] { 3, 9, 2, 5, 2, 19 }));
            Console.ReadKey();
        }

        /// <summary>
        /// 三角形的最大周长
        /// </summary>
        /// <param name="A"></param>
        /// <returns></returns>
        public static int LargestPerimeter(int[] A)
        {
            //假设三角形的边长满足a<=b<=c，那么这三条边组成面积不为零的三角形的充分必要条件为 a+b>c
            //int max = 0;
            //Array.Sort(A);
            //for (int i = 0; i < A.Length; i++)
            //{
            //    if (i + 3 <= A.Length && A[i] + A[i + 1] > A[i + 2] && A[i] - A[i + 1] < A[i + 2])
            //    {
            //        max = Math.Max(max, A[i] + A[i + 1] + A[i + 2]);
            //    }
            //}
            //return max;

            Array.Sort(A);
            for (int i = A.Length - 1; i >= 2; --i)
            {
                if (A[i - 2] + A[i - 1] > A[i])
                {
                    return A[i - 2] + A[i - 1] + A[i];
                }
            }
            return 0;
        }

        /// <summary>
        /// 翻转对
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public static int ReversePairs(int[] nums)
        {
            if (nums.Length == 0)
            {
                return 0;
            }
            return ReversePairsRecursive(nums, 0, nums.Length - 1);
        }

        public static int ReversePairsRecursive(int[] nums, int left, int right)
        {
            if (left == right)
            {
                return 0;
            }
            else
            {
                int mid = (left + right) / 2;
                int n1 = ReversePairsRecursive(nums, left, mid);
                int n2 = ReversePairsRecursive(nums, mid + 1, right);
                int ret = n1 + n2;

                // 首先统计下标对的数量
                int i = left;
                int j = mid + 1;
                while (i <= mid)
                {
                    while (j <= right && (long)nums[i] > 2 * (long)nums[j])
                    {
                        j++;
                    }
                    ret += j - mid - 1;
                    i++;
                }

                // 随后合并两个排序数组
                int[] sorted = new int[right - left + 1];
                int p1 = left, p2 = mid + 1;
                int p = 0;
                while (p1 <= mid || p2 <= right)
                {
                    if (p1 > mid)
                    {
                        sorted[p++] = nums[p2++];
                    }
                    else if (p2 > right)
                    {
                        sorted[p++] = nums[p1++];
                    }
                    else
                    {
                        if (nums[p1] < nums[p2])
                        {
                            sorted[p++] = nums[p1++];
                        }
                        else
                        {
                            sorted[p++] = nums[p2++];
                        }
                    }
                }
                for (int k = 0; k < sorted.Length; k++)
                {
                    nums[left + k] = sorted[k];
                }
                return ret;
            }
        }
        /// <summary>
        /// 四数相加 II
        /// </summary>
        /// <param name="A"></param>
        /// <param name="B"></param>
        /// <param name="C"></param>
        /// <param name="D"></param>
        /// <returns></returns>
        public static int FourSumCount(int[] A, int[] B, int[] C, int[] D)
        {
            //分组 + 哈希表
            Dictionary<int, int> countAB = new Dictionary<int, int>();
            foreach (var u in A)
            {
                foreach (var v in B)
                {
                    if (!countAB.ContainsKey(u + v))
                    {
                        countAB.Add(u + v, 1);
                    }
                    else
                    {
                        countAB[u + v]++;
                    }

                }
            }
            int ans = 0;
            foreach (var u in C)
            {
                foreach (var v in D)
                {
                    if (countAB.ContainsKey(-u - v))
                    {
                        ans += countAB[-u - v];
                    }
                }
            }
            return ans;
        }

        /// <summary>
        /// 最大间距
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public static int MaximumGap(int[] nums)
        {
            //if (nums.Length < 2)
            //{
            //    return 0;
            //}
            //Array.Sort(nums);
            //int front = nums[0];
            //int max = 0;
            //foreach (var item in nums)
            //{
            //    max = Math.Max(max, item - front);
            //    front = item;
            //}
            //return max;

            int n = nums.Length;
            if (n < 2)
            {
                return 0;
            }
            int minVal = nums.Min();
            int maxVal = nums.Max();
            int d = Math.Max(1, (maxVal - minVal) / (n - 1));
            int bucketSize = (maxVal - minVal) / d + 1;

            int[][] bucket = new int[bucketSize][];

            for (int i = 0; i < bucketSize; ++i)
            {
                bucket[i] = new int[] { -1, -1 };// 存储 (桶内最小值，桶内最大值) 对， (-1, -1) 表示该桶是空的
            }
            for (int i = 0; i < n; i++)
            {
                int idx = (nums[i] - minVal) / d;
                if (bucket[idx][0] == -1)
                {
                    bucket[idx][0] = bucket[idx][1] = nums[i];
                }
                else
                {
                    bucket[idx][0] = Math.Min(bucket[idx][0], nums[i]);
                    bucket[idx][1] = Math.Max(bucket[idx][1], nums[i]);
                }
            }

            int ret = 0;
            int prev = -1;
            for (int i = 0; i < bucketSize; i++)
            {
                if (bucket[i][0] == -1)
                {
                    continue;
                }
                if (prev != -1)
                {
                    ret = Math.Max(ret, bucket[i][0] - bucket[prev][1]);
                }
                prev = i;
            }
            return ret;
        }

        /// <summary>
        /// 上升下降字符串
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static string SortString(string s)
        {
            StringBuilder sb = new StringBuilder();
            int[] check = new int[26];
            for (int i = 0; i < s.Length; i++)
            {
                check[s[i] - 'a']++;
            }
            while (sb.Length != s.Length)
            {
                for (int i = 0; i < check.Length; i++)
                {
                    if (check[i] > 0)
                    {
                        sb.Append((char)(i + 'a'));
                        check[i]--;
                    }
                }
                for (int i = check.Length - 1; i >= 0; i--)
                {
                    if (check[i] > 0)
                    {
                        sb.Append((char)(i + 'a'));
                        check[i]--;
                    }
                }
            }
            return sb.ToString();
        }

        static int n = 0;
        /// <summary>
        /// 完全二叉树的节点个数
        /// </summary>
        /// <param name="root"></param>
        /// <returns></returns>
        public static int CountNodes(TreeNode root)
        {
            //if (root != null)
            //{
            //    n++;
            //}
            //if (root?.left != null)
            //{
            //    CountNodes(root.left);
            //}
            //if (root?.right != null)
            //{
            //    CountNodes(root.right);
            //}
            //return n;

            if (root == null)
            {
                return 0;
            }
            int left = CountNodes(root.left);
            int right = CountNodes(root.right);

            return left + right + 1;
        }

        /// <summary>
        /// 用最少数量的箭引爆气球
        /// </summary>
        /// <param name="points"></param>
        /// <returns></returns>
        public static int FindMinArrowShots(int[][] points)
        {
            if (points.Length == 0)
            {
                return 0;
            }
            Array.Sort(points, (point1, point2) =>
            {
                if (point1[1] > point2[1])
                {
                    return 1;
                }
                else if (point1[1] < point2[1])
                {
                    return -1;
                }
                else
                {
                    return 0;
                }
                //return point1[1] - point2[1]; //错误
            });
            int pos = points[0][1];
            int ans = 1;
            foreach (var balloon in points)
            {
                if (balloon[0] > pos)
                {
                    pos = balloon[1];
                    ++ans;
                }
            }
            return ans;
        }

        /// <summary>
        /// 有效的字母异位词
        /// </summary>
        /// <param name="s"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        public static bool IsAnagram(string s, string t)
        {
            //如果长度不一样则肯定不是相同字符构成的
            if (s.Length != t.Length)
            {
                return false;
            }

            //创建26只桶，A-Z
            int[] check = new int[26];

            for (int i = 0; i < s.Length; i++)
            {
                check[s[i] - 'a']++;
                check[t[i] - 'a']--;
            }

            foreach (int i in check)
            {
                if (i != 0)
                {
                    return false;
                }
            }
            return true;
        }

        /// <summary>
        /// 排序链表
        /// </summary>
        /// <param name="head"></param>
        /// <returns></returns>
        public static ListNode SortList(ListNode head)
        {
            return SortList(head, null);
        }

        public static ListNode SortList(ListNode head, ListNode tail)
        {
            if (head == null)
            {
                return head;
            }
            if (head.next == tail)
            {
                head.next = null;
                return head;
            }
            ListNode slow = head, fast = head;
            while (fast != tail)
            {
                slow = slow.next;
                fast = fast.next;
                if (fast != tail)
                {
                    fast = fast.next;
                }
            }
            ListNode mid = slow;
            ListNode list1 = SortList(head, mid);
            ListNode list2 = SortList(mid, tail);
            ListNode sorted = Merge(list1, list2);
            return sorted;
        }

        public static ListNode Merge(ListNode head1, ListNode head2)
        {
            ListNode dummyHead = new ListNode(0);
            ListNode temp = dummyHead, temp1 = head1, temp2 = head2;
            while (temp1 != null && temp2 != null)
            {
                if (temp1.val <= temp2.val)
                {
                    temp.next = temp1;
                    temp1 = temp1.next;
                }
                else
                {
                    temp.next = temp2;
                    temp2 = temp2.next;
                }
                temp = temp.next;
            }
            if (temp1 != null)
            {
                temp.next = temp1;
            }
            else if (temp2 != null)
            {
                temp.next = temp2;
            }
            return dummyHead.next;
        }

        /// <summary>
        /// 对链表进行插入排序
        /// </summary>
        /// <param name="head"></param>
        /// <returns></returns>
        public static ListNode InsertionSortList(ListNode head)
        {
            if (head == null)
            {
                return head;
            }
            ListNode dummyHead = new ListNode(0);
            dummyHead.next = head;
            ListNode lastSorted = head, curr = head.next;
            while (curr != null)
            {
                if (lastSorted.val <= curr.val)
                {
                    lastSorted = lastSorted.next;
                }
                else
                {
                    ListNode prev = dummyHead;
                    while (prev.next.val <= curr.val)
                    {
                        prev = prev.next;
                    }
                    lastSorted.next = curr.next;
                    curr.next = prev.next;
                    prev.next = curr;
                }
                curr = lastSorted.next;
            }
            return dummyHead.next;
        }

        /// <summary>
        /// 移动零
        /// </summary>
        /// <param name="nums"></param>
        public static void MoveZeroes(int[] nums)
        {
            //var indexs = new Queue<int>();
            //for (int i = 0; i < nums.Length; i++)
            //{
            //    if (nums[i] == 0)
            //    {
            //        indexs.Enqueue(i);
            //    }
            //    else if (indexs.Any())
            //    {
            //        var index = indexs.Dequeue();
            //        indexs.Enqueue(i);
            //        nums[index] = nums[i];
            //    }
            //}
            //for (int i = 0; i < indexs.Count; i++)
            //{
            //    nums[nums.Length - i - 1] = 0;
            //}

            int n = nums.Length, left = 0, right = 0;
            while (right < n)
            {
                if (nums[right] != 0)
                {
                    Swap(nums, left, right);
                    left++;
                }
                right++;
            }
        }

        /// <summary>
        /// 加油站
        /// </summary>
        /// <param name="gas"></param>
        /// <param name="cost"></param>
        /// <returns></returns>
        public static int CanCompleteCircuit(int[] gas, int[] cost)
        {
            int n = gas.Length;
            int i = 0;
            while (i < n)
            {
                int sumOfGas = 0, sumOfCost = 0;
                int cnt = 0;
                while (cnt < n)
                {
                    int j = (i + cnt) % n;
                    sumOfGas += gas[j];
                    sumOfCost += cost[j];
                    if (sumOfCost > sumOfGas)
                    {
                        break;
                    }
                    cnt++;
                }
                if (cnt == n)
                {
                    return i;
                }
                else
                {
                    i = i + cnt + 1;
                }
            }
            return -1;
        }

        /// <summary>
        /// 距离顺序排列矩阵单元格
        /// </summary>
        /// <param name="R"></param>
        /// <param name="C"></param>
        /// <param name="r0"></param>
        /// <param name="c0"></param>
        /// <returns></returns>
        public static int[][] AllCellsDistOrder(int R, int C, int r0, int c0)
        {
            //int[][] ret = new int[R * C][];
            //for (int i = 0; i < R; i++)
            //{
            //    for (int j = 0; j < C; j++)
            //    {
            //        ret[i * C + j] = new int[] { i, j };
            //    }
            //}
            //Array.Sort(ret, (p1, p2) =>
            //{
            //    return (Math.Abs(r0 - p1[0]) + Math.Abs(c0 - p1[1])) - (Math.Abs(r0 - p2[0]) + Math.Abs(c0 - p2[1]));
            //});

            //return ret;

            int[] dr = { 1, 1, -1, -1 };
            int[] dc = { 1, -1, -1, 1 };
            int maxDist = Math.Max(r0, R - 1 - r0) + Math.Max(c0, C - 1 - c0);
            int[][] ret = new int[R * C][];
            int row = r0, col = c0;
            int index = 0;
            ret[index++] = new int[] { row, col };
            for (int dist = 1; dist <= maxDist; dist++)
            {
                row--;
                for (int i = 0; i < 4; i++)
                {
                    while ((i % 2 == 0 && row != r0) || (i % 2 != 0 && col != c0))
                    {
                        if (row >= 0 && row < R && col >= 0 && col < C)
                        {
                            ret[index++] = new int[] { row, col };
                        }
                        row += dr[i];
                        col += dc[i];
                    }
                }
            }
            return ret;
        }

        /// <summary>
        /// 根据身高重建队列
        /// </summary>
        /// <param name="people"></param>
        /// <returns></returns>
        public static int[][] ReconstructQueue(int[][] people)
        {
            /*
           排序+插入
           1. 排序：按照先H高度降序，K个数升序排序
           2. 插入：把矮个插入到 k 位置
            */

            Array.Sort(people, (p1, p2) =>
            {
                return p1[0] == p2[0] ? p1[1] - p2[1] : p2[0] - p1[0];
            });

            List<int[]> ans = new List<int[]>();
            foreach (int[] i in people)
            {
                ans.Insert(i[1], i);
            }

            return ans.ToArray();
        }

        /// <summary>
        /// 移掉K位数字
        /// </summary>
        /// <param name="num"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public static string RemoveKdigits(string num, int k)
        {
            var stack = new LinkedList<char>();
            foreach (var digit in num)
            {
                while (stack.Any() && k > 0 && stack.Last() > digit)
                {
                    stack.RemoveLast();
                    k--;
                }
                stack.AddLast(digit);
            }
            for (int i = 0; i < k; i++) stack.RemoveLast();
            bool isLeadingZero = true;
            StringBuilder ans = new StringBuilder();
            foreach (var digit in stack)
            {
                if (isLeadingZero && digit == '0') continue;
                isLeadingZero = false;
                ans.Append(digit);
            }
            if (ans.Length == 0) return "0";
            return ans.ToString();
        }

        /// <summary>
        /// 数组的相对排序
        /// </summary>
        /// <param name="arr1"></param>
        /// <param name="arr2"></param>
        /// <returns></returns>
        public static int[] RelativeSortArray(int[] arr1, int[] arr2)
        {
            //List<int> dest = new List<int>();
            //List<int> arr2List = new List<int>(arr1);
            //foreach (var item2 in arr2)
            //{
            //    for (int i = 0; i < arr2List.Count; i++)
            //    {
            //        if (arr2List[i] == item2)
            //        {
            //            dest.Add(item2);
            //            arr2List.RemoveAt(i);
            //            i--;
            //        }
            //    }
            //}
            //dest.AddRange(SelectSort(arr2List.ToArray()));
            //return dest.ToArray();

            int[] arr = new int[1001];
            int[] res = new int[arr1.Length];
            int index = 0;

            foreach (var item in arr1)
            {
                arr[item]++;
            }

            foreach (var item in arr2)
            {
                while (arr[item]-- > 0)
                {
                    res[index++] = item;
                }
            }

            for (int i = 0; i < 1001; i++)
            {
                while (arr[i]-- > 0)
                {
                    res[index++] = i;
                }
            }

            return res;
        }

        /// <summary>
        /// 选择排序
        /// </summary>
        /// <param name="list"></param>
        public static int[] SelectSort(int[] list)
        {
            int min, temp;
            for (int i = 0; i < list.Length - 1; i++)
            {
                min = i;
                for (int j = i + 1; j <= list.Length - 1; j++)
                {
                    if (list[min] > list[j])
                    {
                        min = j;
                    }
                }
                temp = list[i];
                list[i] = list[min];
                list[min] = temp;
            }
            return list;
        }

        /// <summary>
        /// 奇偶链表
        /// </summary>
        /// <param name="head"></param>
        /// <returns></returns>
        public static ListNode OddEvenList(ListNode head)
        {
            if (head == null)
            {
                return head;
            }
            ListNode evenHead = head.next;
            ListNode odd = head, even = evenHead;
            while (even != null && even.next != null)
            {
                odd.next = even.next;
                odd = odd.next;
                even.next = odd.next;
                even = even.next;
            }
            odd.next = evenHead;
            return head;
        }

        /// <summary>
        /// 按奇偶排序数组 II
        /// </summary>
        /// <param name="A"></param>
        /// <returns></returns>
        public static int[] SortArrayByParityII(int[] A)
        {
            //int[] AA = new int[A.Length];
            //int ii = 0; int jj = 1;
            //for (int i = 0; i < A.Length; i++)
            //{
            //    if (A[i] % 2 == 0)
            //    {
            //        AA[ii] = A[i];
            //        ii += 2;
            //    }
            //    else
            //    {
            //        AA[jj] = A[i];
            //        jj += 2;
            //    }
            //}
            //return AA;

            int n = A.Length;
            int j = 1;
            for (int i = 0; i < n; i += 2)
            {
                if (A[i] % 2 == 1)
                {
                    while (A[j] % 2 == 1)
                    {
                        j += 2;
                    }
                    Swap(A, i, j);
                }
            }
            return A;
        }

        public static void Swap(int[] A, int i, int j)
        {
            int temp = A[i];
            A[i] = A[j];
            A[j] = temp;
        }

        /// <summary>
        /// 自由之路
        /// </summary>
        /// <param name="ring"></param>
        /// <param name="key"></param>
        /// <returns></returns>
        public static int FindRotateSteps(string ring, string key)
        {
            //Hashtable ht = new Hashtable();
            //for (int i = 0; i < ring.Length; i++)
            //{
            //    if (ht.ContainsKey(ring[i]))
            //    {
            //        ((List<int>)ht[ring[i]]).Add(i);
            //    }
            //    else
            //    {
            //        List<int> list = new List<int>();
            //        list.Add(i);
            //        ht.Add(ring[i], list);
            //    }
            //}
            //int[] steps = new int[ring.Length];
            //for (int i = 0; i < steps.Length; i++)
            //{
            //    steps[i] = int.MaxValue;
            //}
            //List<int> list1 = (List<int>)ht[key[0]];
            //foreach (int item in list1)
            //    steps[item] = Math.Min(Math.Abs(item - 0), ring.Length - Math.Abs(item - 0));
            //for (int i = 1; i < key.Length; i++)
            //{
            //    List<int> list = (List<int>)ht[key[i]];
            //    foreach (int item in list)
            //    {
            //        List<int> prelist = (List<int>)ht[key[i - 1]];
            //        int stepTemp = int.MaxValue;
            //        foreach (int preItem in prelist)
            //            stepTemp = Math.Min(stepTemp, Math.Min(Math.Abs(item - preItem), ring.Length - Math.Abs(item - preItem)) + steps[preItem]);
            //        steps[item] = stepTemp;
            //    }
            //}
            //int minStep = int.MaxValue;
            //List<int> list2 = (List<int>)ht[key[key.Length - 1]];
            //foreach (int item in list2)
            //    minStep = Math.Min(minStep, steps[item]);
            //return minStep + key.Length;

            int ans = int.MaxValue;
            Queue<int[]> queue = new Queue<int[]>();
            Dictionary<char, List<int>> dic = new Dictionary<char, List<int>>();
            int[,] min = new int[ring.Length, key.Length];
            for (int i = 0; i < ring.Length; ++i)
            {
                char c = ring[i];
                if (!dic.ContainsKey(c)) dic.Add(c, new List<int>());
                dic[c].Add(i);
            }
            for (int i = 0; i < min.GetLength(0); ++i) for (int j = 0; j < min.GetLength(1); ++j) min[i, j] = int.MaxValue;
            queue.Enqueue(new int[] { 0, 0, 0 });
            while (queue.Count != 0)
            {
                int cur = queue.Peek()[0];
                int index = queue.Peek()[1];
                int steps = queue.Peek()[2];
                queue.Dequeue();
                if (index == key.Length)
                {
                    ans = Math.Min(ans, steps);
                    continue;
                }
                foreach (int i in dic[key[index]])
                {
                    int step = Math.Min(Math.Abs(cur - i), ring.Length - Math.Abs(cur - i));
                    if (steps + step + 1 < min[i, index])
                    {
                        queue.Enqueue(new int[] { i, index + 1, steps + step + 1 });
                        min[i, index] = steps + step + 1;
                    }
                }
            }
            return ans;
        }

        /// <summary>
        /// 最接近原点的 K 个点
        /// </summary>
        /// <param name="points"></param>
        /// <param name="K"></param>
        /// <returns></returns>
        public static int[][] KClosest(int[][] points, int K)
        {
            Array.Sort(points, new IntArryComparer());
            int[][] result = new int[K][];
            Array.Copy(points, result, K);
            return result;

            //Array.Sort<int[]>(points, (x, y) => (x[0] * x[0] + x[1] * x[1]).CompareTo(y[0] * y[0] + y[1] * y[1]));
            //int[][] result = new int[K][];
            //Array.Copy(points, result, K);
            //return result;
        }

        public class IntArryComparer : IComparer<int[]>
        {
            public int Compare(int[] point1, int[] point2)
            {
                return (point1[0] * point1[0] + point1[1] * point1[1]) - (point2[0] * point2[0] + point2[1] * point2[1]);
            }
        }

        /// <summary>
        /// 区间和的个数
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="lower"></param>
        /// <param name="upper"></param>
        /// <returns></returns>
        public static int CountRangeSum(int[] nums, int lower, int upper)
        {
            long s = 0;
            long[] sum = new long[nums.Length + 1];
            for (int i = 0; i < nums.Length; ++i)
            {
                s += nums[i];
                sum[i + 1] = s;
            }
            return CountRangeSumRecursive(sum, lower, upper, 0, sum.Length - 1);
        }

        public static int CountRangeSumRecursive(long[] sum, int lower, int upper, int left, int right)
        {
            if (left == right)
            {
                return 0;
            }
            else
            {
                int mid = (left + right) / 2;
                int n1 = CountRangeSumRecursive(sum, lower, upper, left, mid);
                int n2 = CountRangeSumRecursive(sum, lower, upper, mid + 1, right);
                int ret = n1 + n2;

                // 首先统计下标对的数量
                int i = left;
                int l = mid + 1;
                int r = mid + 1;
                while (i <= mid)
                {
                    while (l <= right && sum[l] - sum[i] < lower)
                    {
                        l++;
                    }
                    while (r <= right && sum[r] - sum[i] <= upper)
                    {
                        r++;
                    }
                    ret += r - l;
                    i++;
                }

                // 随后合并两个排序数组
                int[] sorted = new int[right - left + 1];
                int p1 = left, p2 = mid + 1;
                int p = 0;
                while (p1 <= mid || p2 <= right)
                {
                    if (p1 > mid)
                    {
                        sorted[p++] = (int)sum[p2++];
                    }
                    else if (p2 > right)
                    {
                        sorted[p++] = (int)sum[p1++];
                    }
                    else
                    {
                        if (sum[p1] < sum[p2])
                        {
                            sorted[p++] = (int)sum[p1++];
                        }
                        else
                        {
                            sorted[p++] = (int)sum[p2++];
                        }
                    }
                }
                for (int j = 0; j < sorted.Length; j++)
                {
                    sum[left + j] = sorted[j];
                }
                return ret;
            }
        }

        /// <summary>
        /// 根据数字二进制下 1 的数目排序
        /// </summary>
        /// <param name="arr"></param>
        /// <returns></returns>
        public static int[] SortByBits(int[] arr)
        {
            //var v = from a in arr
            //        orderby System.Convert.ToString(a, 2).Count(x => x == '1')
            //        group a by System.Convert.ToString(a, 2).Count(x => x == '1') into g
            //        select g.OrderBy(x => x);
            //return v.SelectMany(x => x).ToArray();

            //var list = arr.Select(a => new { a, b = System.Convert.ToString(a, 2).Replace("0", "").Length }).OrderBy(g => g.b).ThenBy(g => g.a);
            //return list.Select(a => a.a).ToArray();

            int[] newArr = new int[arr.Length];
            for (int i = 0; i < arr.Length; i++)
            {
                newArr[i] = HammingWeight((uint)arr[i]) * 100000 + arr[i];
            }
            Array.Sort(newArr);
            for (int i = 0; i < newArr.Length; i++)
            {
                newArr[i] = newArr[i] % 100000;
            }
            return newArr;

        }

        public static int HammingWeight(uint n)
        {
            int result = 0;
            while (n != 0)
            {
                n = n & (n - 1);
                result++;
            }
            return result;
        }

        /// <summary>
        /// 单词接龙
        /// </summary>
        /// <param name="beginWord"></param>
        /// <param name="endWord"></param>
        /// <param name="wordList"></param>
        /// <returns></returns>
        public static int LadderLength(string beginWord, string endWord, List<string> wordList)
        {
            // 第 1 步：先将 wordList 放到哈希表里，便于判断某个单词是否在 wordList 里
            HashSet<string> wordSet = new HashSet<string>(wordList);
            if (wordSet.Count == 0 || !wordSet.Contains(endWord))
            {
                return 0;
            }
            wordSet.Remove(beginWord);

            // 第 2 步：图的广度优先遍历，必须使用队列和表示是否访问过的 visited 哈希表
            Queue<string> queue = new Queue<string>();
            queue.Enqueue(beginWord);
            HashSet<string> visited = new HashSet<string>();
            visited.Add(beginWord);

            // 第 3 步：开始广度优先遍历，包含起点，因此初始化的时候步数为 1
            int step = 1;
            while (queue.Count > 0)
            {
                int currentSize = queue.Count;
                for (int i = 0; i < currentSize; i++)
                {
                    // 依次遍历当前队列中的单词
                    string currentWord = queue.Dequeue();
                    // 如果 currentWord 能够修改 1 个字符与 endWord 相同，则返回 step + 1
                    if (ChangeWordEveryOneLetter(currentWord, endWord, queue, visited, wordSet))
                    {
                        return step + 1;
                    }
                }
                step++;
            }
            return 0;


            //if (!wordList.Contains(endWord) || beginWord == endWord)
            //    return 0;
            //List<string> words = new List<string>(wordList);
            //int reuslt = 1;
            //int startIndex = 0;

            //Queue wordQ = new Queue();
            //wordQ.Enqueue(beginWord);
            //while (wordQ.Count > 0)
            //{
            //    Queue tempQ = new Queue();
            //    reuslt++;
            //    while (wordQ.Count > 0)
            //    {
            //        string value = (string)wordQ.Dequeue();
            //        for (int k = startIndex; k < words.Count; k++)
            //        {
            //            if (words[k].Length != value.Length)
            //                continue;
            //            int count = 0;
            //            for (int i = 0; i < words[k].Length; i++)
            //            {
            //                if (words[k][i] != value[i])
            //                    count++;
            //                if (count > 1)
            //                    break;
            //            }
            //            if (count == 1)
            //            {
            //                if (words[k] == endWord)
            //                    return reuslt;
            //                tempQ.Enqueue(words[k]);
            //                words[k] = words[startIndex++];
            //            }
            //        }
            //    }
            //    wordQ = tempQ;
            //}
            //return 0;
        }

        /**
         * 尝试对 currentWord 修改每一个字符，看看是不是能与 endWord 匹配
         *
         * @param currentWord
         * @param endWord
         * @param queue
         * @param visited
         * @param wordSet
         * @return
         */
        private static bool ChangeWordEveryOneLetter(string currentWord, string endWord,
                                                 Queue<string> queue, HashSet<string> visited, HashSet<string> wordSet)
        {
            char[] charArray = currentWord.ToCharArray();
            for (int i = 0; i < endWord.Length; i++)
            {
                // 先保存，然后恢复
                char originChar = charArray[i];
                for (char k = 'a'; k <= 'z'; k++)
                {
                    if (k == originChar)
                    {
                        continue;
                    }
                    charArray[i] = k;
                    string nextWord = string.Join("", charArray);
                    if (wordSet.Contains(nextWord))
                    {
                        if (nextWord.Equals(endWord))
                        {
                            return true;
                        }
                        if (!visited.Contains(nextWord))
                        {
                            queue.Enqueue(nextWord);
                            // 注意：添加到队列以后，必须马上标记为已经访问
                            visited.Add(nextWord);
                        }
                    }
                }
                // 恢复
                charArray[i] = originChar;
            }
            return false;
        }

        /// <summary>
        /// 插入区间  
        /// </summary>
        /// <param name="intervals"></param>
        /// <param name="newInterval"></param>
        /// <returns></returns>
        public static int[][] Insert(int[][] intervals, int[] newInterval)
        {
            int left = newInterval[0];
            int right = newInterval[1];
            bool placed = false;
            List<int[]> ansList = new List<int[]>();
            foreach (var item in intervals)
            {
                if (item[0] > right)
                {
                    if (!placed)
                    {
                        ansList.Add(new int[] { left, right });
                        placed = true;
                    }
                    ansList.Add(item);
                }
                else if (item[1] < left)
                {
                    ansList.Add(item);
                }
                else
                {
                    left = Math.Min(left, item[0]);
                    right = Math.Max(right, item[1]);
                }
            }
            if (!placed)
            {
                ansList.Add(new int[] { left, right });
            }
            return ansList.ToArray();
        }

        /// <summary>
        /// 有效的山脉数组
        /// </summary>
        /// <param name="A"></param>
        /// <returns></returns>
        public static bool ValidMountainArray(int[] A)
        {
            int N = A.Length;
            int i = 0;

            // 递增扫描
            while (i + 1 < N && A[i] < A[i + 1])
            {
                i++;
            }

            // 最高点不能是数组的第一个位置或最后一个位置
            if (i == 0 || i == N - 1)
            {
                return false;
            }

            // 递减扫描
            while (i + 1 < N && A[i] > A[i + 1])
            {
                i++;
            }

            return i == N - 1;
        }

        public static int[] Intersection(int[] nums1, int[] nums2)
        {
            return nums1.Intersect(nums2).ToArray();
        }

        public static IList<string> WordBreak(string s, IList<string> wordDict)
        {
            return Partition(s, 0, wordDict);
        }

        private static IDictionary<int, IList<string>> _cache = new Dictionary<int, IList<string>>();

        /// <summary>
        /// 切分子字符串
        /// </summary>
        /// <param name="s"></param>
        /// <param name="start">子字符串起始位置</param>
        private static IList<string> Partition(string s, int start, IList<string> wordDict)
        {
            if (_cache.ContainsKey(start)) return _cache[start];

            var results = new List<string>();
            //字符串末尾是唯一出口，祖先节点已经切分好的前缀即为此分支的唯一结果
            if (start == s.Length) results.Add(null);

            string sub;
            //对字符串截前1字符、前2字符...前n字符，为此字符串的子节点
            for (int i = start; i < s.Length; i++)
            {
                sub = s.Substring(start, i - start + 1);
                if (wordDict.Contains(sub))
                {
                    foreach (var x in Partition(s, i + 1, wordDict))
                    {
                        results.Add(sub + (x == null ? "" : " " + x));
                    }
                }
                //若截取的不在字典，则此分支剪断
            }
            _cache.Add(start, results);
            return results;
        }

        public class RandomizedCollection1
        {
            List<int> list = new List<int>();
            IDictionary<int, int> maps = new Dictionary<int, int>();
            Random rd = new Random();
            /** Initialize your data structure here. */
            public RandomizedCollection1()
            {

            }

            /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
            public bool Insert(int val)
            {
                bool isExists = maps.ContainsKey(val);
                if (!isExists)
                    maps.Add(val, 0);
                list.Add(val);
                maps[val]++;
                return isExists;

            }

            /** Removes a value from the collection. Returns true if the collection contained the specified element. */
            public bool Remove(int val)
            {
                if (!maps.ContainsKey(val)) return false;

                list.Remove(val);
                maps[val]--;
                if (maps[val] == 0)
                    maps.Remove(val);
                return true;
            }

            /** Get a random element from the collection. */
            public int GetRandom()
            {
                int index = rd.Next(0, list.Count);
                return list[index];
            }
        }

        public class RandomizedCollection
        {
            // 字典内使用 HashSet 是为了实现 O(1) 删除索引，而且索引不会重复
            Dictionary<int, HashSet<int>> IndexDictionary { get; set; } = new Dictionary<int, HashSet<int>>();
            List<int> Values { get; set; } = new List<int>();

            /** Initialize your data structure here. */
            public RandomizedCollection()
            {
                /* 列表随机插入是 0(n)，因为要后移插入位置之后的元素，但追加在列表尾部是O(1)
                 * 列表按索引查找是 O(1)，按值查找是 O(n)
                 */

            }

            /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
            public bool Insert(int val)
            {
                // 使用一个字典维护某个数字的所有索引列表
                if (!IndexDictionary.TryGetValue(val, out var indexSet))
                {
                    indexSet = new HashSet<int>();
                    IndexDictionary.Add(val, indexSet);
                }

                Values.Add(val);
                indexSet.Add(Values.Count - 1);
                return true;
            }

            /** Removes a value from the collection. Returns true if the collection contained the specified element. */
            public bool Remove(int val)
            {
                // 字典不存在某个数字的索引，也就不存在某个数字
                if (!IndexDictionary.TryGetValue(val, out var valIndexList) ||
                    valIndexList.Count == 0)
                    return false;

                var lastValue = Values.Last();
                if (lastValue == val)
                {
                    // 直接在列表最后移除
                    valIndexList.Remove(Values.Count - 1);
                    Values.RemoveAt(Values.Count - 1);
                    return true;
                }

                // 将Values最后一个元素的值替换要删除的元素的最后一个索引的位置，然后Values删除最后一个元素，即为0(1)删除
                // 被替换的值的索引列表应该移除其索引值（Values的长度-1），并追加一个被删除元素所在的索引值（因为被替换的值被替换到这个索引了）
                var valLastIndex = valIndexList.Last();
                var exchangeIndexList = IndexDictionary[lastValue];

                valIndexList.Remove(valLastIndex);
                exchangeIndexList.Remove(Values.Count - 1);
                exchangeIndexList.Add(valLastIndex);

                Values[valLastIndex] = lastValue;
                Values.RemoveAt(Values.Count - 1);
                return true;
            }

            /** Get a random element from the collection. */
            public int GetRandom()
            {
                return Values[new Random().Next(0, Values.Count)];
            }
        }

        static int[] dx = { 0, 1, 0, -1 };
        static int[] dy = { 1, 0, -1, 0 };
        /// <summary>
        /// 岛屿的周长
        /// </summary>
        /// <param name="grid"></param>
        /// <returns></returns>
        public static int IslandPerimeter(int[][] grid)
        {
            //int n = grid.Length, m = grid[0].Length;
            //int ans = 0;
            //for (int i = 0; i < n; ++i)
            //{
            //    for (int j = 0; j < m; ++j)
            //    {
            //        if (grid[i][j] == 1)
            //        {
            //            int cnt = 0;
            //            for (int k = 0; k < 4; ++k)
            //            {
            //                int tx = i + dx[k];
            //                int ty = j + dy[k];
            //                if (tx < 0 || tx >= n || ty < 0 || ty >= m || grid[tx][ty] == 0)
            //                {
            //                    cnt += 1;
            //                }
            //            }
            //            ans += cnt;
            //        }
            //    }
            //}
            //return ans;

            //int n = grid.Length, m = grid[0].Length;
            //int ans = 0;
            //for (int i = 0; i < n; ++i)
            //{
            //    for (int j = 0; j < m; ++j)
            //    {
            //        if (grid[i][j] == 1)
            //        {
            //            ans += DFS(i, j, grid, n, m);
            //        }
            //    }
            //}
            //return ans;

            int m = grid.Length;
            int n = grid[0].Length;
            int ans = 0;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                {
                    if (grid[i][j] == 1)
                    {
                        if (i == 0)
                            ++ans;
                        else
                            ans += grid[i - 1][j] == 0 ? 1 : 0;
                        if (j == 0)
                            ++ans;
                        else
                            ans += grid[i][j - 1] == 0 ? 1 : 0;
                    }
                }
            return ans << 1;
        }

        public static int DFS(int x, int y, int[][] grid, int n, int m)
        {
            if (x < 0 || x >= n || y < 0 || y >= m || grid[x][y] == 0)
            {
                return 1;
            }
            if (grid[x][y] == 2)
            {
                return 0;
            }
            grid[x][y] = 2;
            int res = 0;
            for (int i = 0; i < 4; ++i)
            {
                int tx = x + dx[i];
                int ty = y + dy[i];
                res += DFS(tx, ty, grid, n, m);
            }
            return res;
        }

        /// <summary>
        /// 求根到叶子节点数字之和
        /// </summary>
        /// <param name="root"></param>
        /// <returns></returns>
        public static int SumNumbers(TreeNode root)
        {
            if (root == null)
            {
                return 0;
            }
            int sum = 0;
            Queue<TreeNode> nodeQueue = new Queue<TreeNode>();
            Queue<int> numQueue = new Queue<int>();
            nodeQueue.Enqueue(root);
            numQueue.Enqueue(root.val);
            while (nodeQueue.Count > 0)
            {
                TreeNode node = nodeQueue.Dequeue();
                int num = numQueue.Dequeue();
                TreeNode left = node.left, right = node.right;
                if (left == null && right == null)
                {
                    sum += num;
                }
                else
                {
                    if (left != null)
                    {
                        nodeQueue.Enqueue(left);
                        numQueue.Enqueue(num * 10 + left.val);
                    }
                    if (right != null)
                    {
                        nodeQueue.Enqueue(right);
                        numQueue.Enqueue(num * 10 + right.val);
                    }
                }
            }
            return sum;
            //return DFS(root, 0);
        }

        public static int DFS(TreeNode root, int prevSum)
        {
            if (root == null)
            {
                return 0;
            }
            int sum = prevSum * 10 + root.val;
            if (root.left == null && root.right == null)
            {
                return sum;
            }
            else
            {
                return DFS(root.left, sum) + DFS(root.right, sum);
            }
        }

        /// <summary>
        /// 独一无二的出现次数
        /// </summary>
        /// <param name="arr"></param>
        /// <returns></returns>
        public static bool UniqueOccurrences(int[] arr)
        {
            Dictionary<int, int> occur = new Dictionary<int, int>();
            foreach (var item in arr)
            {
                if (occur.ContainsKey(item))
                {
                    occur[item]++;
                }
                else
                {
                    occur.Add(item, 1);
                }
            }
            //return occur.Values.Distinct().Count() == occur.Count;
            HashSet<int> times = new HashSet<int>();
            foreach (var item in occur)
            {
                times.Add(item.Value);
            }
            return times.Count() == occur.Count;
        }

        static IList<int> values = new List<int>();
        public static IList<int> PreorderTraversal(TreeNode root)
        {
            //if (root != null)
            //{
            //    values.Add(root.val);
            //    PreorderTraversal(root.left);
            //    PreorderTraversal(root.right);
            //}
            //return values;

            IList<int> res = new List<int>();
            if (root == null)
            {
                return res;
            }
            Stack<TreeNode> stack = new Stack<TreeNode>();
            TreeNode node = root;
            while (stack.Count > 0 || node != null)
            {
                while (node != null)
                {
                    res.Add(node.val);
                    stack.Push(node);
                    node = node.left;
                }
                node = stack.Pop();
                node = node.right;
            }
            return res;
        }

        public class TreeNode
        {
            public int val;
            public TreeNode left;
            public TreeNode right;
            public TreeNode(int val = 0, TreeNode left = null, TreeNode right = null)
            {
                this.val = val;
                this.left = left;
                this.right = right;
            }
        }

        /// <summary>
        /// 有多少小于当前数字的数字
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public static int[] SmallerNumbersThanCurrent(int[] nums)
        {
            //计数排序等八大排序
            int[] cnt = new int[101];
            int n = nums.Length;
            for (int i = 0; i < n; i++)
            {
                cnt[nums[i]]++;
            }
            for (int i = 1; i <= 100; i++)
            {
                cnt[i] += cnt[i - 1];
            }
            int[] ret = new int[n];
            for (int i = 0; i < n; i++)
            {
                ret[i] = nums[i] == 0 ? 0 : cnt[nums[i] - 1];
            }
            return ret;
        }


        /// <summary>
        /// 数组中的最长山脉
        /// </summary>
        /// <param name="A"></param>
        /// <returns></returns>
        public static int LongestMountain(int[] A)
        {
            //[2,1,4,7,3,2,5] 
            //[1,4,7,3,2]
            //5
            int uphill = 0, downhill = 0;
            int last = 0;
            int max = 0;
            if (A.Length < 3) return 0;
            for (int i = 1; i < A.Length; i++)
            {
                if (A[i] > A[i - 1])
                {
                    if (downhill == 1 || uphill == 0)
                    {
                        downhill = 0;
                        last = i - 1;
                    }
                    uphill = 1;
                }
                if (A[i] == A[i - 1])
                {
                    uphill = 0;
                    downhill = 0;
                }
                if (A[i] < A[i - 1]) downhill = 1;
                if (uphill == 1 && downhill == 1)
                {
                    if (max < i - last + 1)
                    {
                        max = i - last + 1;
                    }
                }
            }
            return max;

            //int n = A.Length;
            //int ans = 0;
            //int left = 0;
            //while (left + 2 < n)
            //{
            //    int right = left + 1;
            //    if (A[left] < A[left + 1])
            //    {
            //        while (right + 1 < n && A[right] < A[right + 1])
            //        {
            //            ++right;
            //        }
            //        if (right < n - 1 && A[right] > A[right + 1])
            //        {
            //            while (right + 1 < n && A[right] > A[right + 1])
            //            {
            //                ++right;
            //            }
            //            ans = Math.Max(ans, right - left + 1);
            //        }
            //        else
            //        {
            //            ++right;
            //        }
            //    }
            //    left = right;
            //}
            //return ans;
        }

        /// <summary>
        /// 视频拼接
        /// </summary>
        /// <param name="clips"></param>
        /// <param name="T"></param>
        /// <returns></returns>
        public static int VideoStitching(int[][] clips, int T)
        {
            // [[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]], T = 10
            //[0,2], [8,10], [1,9] 

            int[] dp = new int[T + 1];
            Array.Fill(dp, int.MaxValue - 1);
            dp[0] = 0;
            for (int i = 1; i <= T; i++)
            {
                foreach (var clip in clips)
                {
                    if (clip[0] < i && i <= clip[1])
                    {
                        dp[i] = Math.Min(dp[i], dp[clip[0]] + 1);
                    }
                }
            }
            return dp[T] == int.MaxValue - 1 ? -1 : dp[T];


            //int[] maxn = new int[T];
            //int last = 0, ret = 0, pre = 0;
            //foreach (var clip in clips)
            //{
            //    if (clip[0] < T)
            //    {
            //        maxn[clip[0]] = Math.Max(maxn[clip[0]], clip[1]);
            //    }
            //}
            //for (int i = 0; i < T; i++)
            //{
            //    last = Math.Max(last, maxn[i]);
            //    if (i == last)
            //    {
            //        return -1;
            //    }
            //    if (i == pre)
            //    {
            //        ret++;
            //        pre = last;
            //    }
            //}
            //return ret;
        }

        /// <summary>
        /// 回文链表
        /// </summary>
        /// <param name="head"></param>
        /// <returns></returns>
        public static bool IsPalindrome(ListNode head)
        {
            //12455421
            //2 12 212
            //1221
            List<int> values = new List<int>();
            while (head != null)
            {
                values.Add(head.val);
                head = head.next;
            }
            int front = 0, end = values.Count - 1;
            while (front < end)
            {
                if (values[front] != values[end])
                {
                    return false;
                }
                front++;
                end++;
            }
            return true;
            //frontPointer = head;
            //return recursivelyCheck(head);
        }

        private static ListNode frontPointer;

        private static bool recursivelyCheck(ListNode currentNode)
        {
            if (currentNode != null)
            {
                if (!recursivelyCheck(currentNode.next))
                {
                    return false;
                }
                if (currentNode.val != frontPointer.val)
                {
                    return false;
                }
                frontPointer = frontPointer.next;
            }
            return true;
        }

        /// <summary>
        /// 划分字母区间
        /// </summary>
        /// <param name="S"></param>
        /// <returns></returns>
        public static IList<int> PartitionLabels(string S)
        {
            //使用“ last['b'] = 5”之类的映射来帮助您扩展分区的宽度
            int[] last = new int[26];
            int length = S.Length;
            for (int i = 0; i < length; i++)
            {
                last[S.ElementAt(i) - 'a'] = i;
            }
            List<int> partition = new List<int>();
            int start = 0, end = 0;
            for (int i = 0; i < length; i++)
            {
                end = Math.Max(end, last[S.ElementAt(i) - 'a']);
                if (i == end)
                {
                    partition.Add(end - start + 1);
                    start = end + 1;
                }
            }
            return partition;
        }

        /// <summary>
        /// 长按键入
        /// </summary>
        /// <param name="name"></param>
        /// <param name="typed"></param>
        /// <returns></returns>
        public static bool IsLongPressedName(string name, string typed)
        {
            int i = 0, j = 0;
            while (j < typed.Length)
            {
                if (i < name.Length && name.ElementAt(i) == typed.ElementAt(j))
                {
                    i++;
                    j++;
                }
                else if (j > 0 && name.ElementAt(j) == typed.ElementAt(j - 1))
                {
                    j++;
                }
                else
                {
                    return false;
                }
            }
            return i == name.Length;
        }

        /// <summary>
        ///  重排链表
        /// </summary>
        /// <param name="head"></param>
        public static void ReorderList(ListNode head)
        {
            //给定链表 1->2->3->4->5, 重新排列为 1->5->2->4->3
            //画图表示，链表指针移动，链表细节可以忽略。
            //1   2   3   4   5
            //15              52
            //15  24      43  52
            if (head == null)
            {
                return;
            }
            List<ListNode> list = new List<ListNode>();
            ListNode node = head;
            while (node != null)
            {
                list.Add(node);
                node = node.next;
            }
            int i = 0, j = list.Count() - 1;
            while (i < j)
            {
                list.ElementAt(i).next = list.ElementAt(j);
                i++;
                if (i == j)
                {
                    break;
                }
                list.ElementAt(j).next = list.ElementAt(i);
                j--;
            }
            list.ElementAt(i).next = null;


            //if (head == null)
            //{
            //    return;
            //}
            //ListNode mid = middleNode(head);
            //ListNode l1 = head;
            //ListNode l2 = mid.next;
            //mid.next = null;
            //l2 = reverseList(l2);
            //mergeList(l1, l2);

        }

        public static ListNode middleNode(ListNode head)
        {
            ListNode slow = head;
            ListNode fast = head;
            while (fast.next != null && fast.next.next != null)
            {
                slow = slow.next;
                fast = fast.next.next;
            }
            return slow;
        }

        public static ListNode reverseList(ListNode head)
        {
            ListNode prev = null;
            ListNode curr = head;
            while (curr != null)
            {
                ListNode nextTemp = curr.next;
                curr.next = prev;
                prev = curr;
                curr = nextTemp;
            }
            return prev;
        }

        public static void mergeList(ListNode l1, ListNode l2)
        {
            ListNode l1_tmp;
            ListNode l2_tmp;
            while (l1 != null && l2 != null)
            {
                l1_tmp = l1.next;
                l2_tmp = l2.next;

                l1.next = l2;
                l1 = l1_tmp;

                l2.next = l1;
                l2 = l2_tmp;
            }
        }

        /// <summary>
        /// 比较含退格的字符串
        /// </summary>
        /// <param name="S"></param>
        /// <param name="T"></param>
        /// <returns></returns>
        public static bool BackspaceCompare(string S, string T)
        {
            //StringBuilder sbS = new StringBuilder(S);
            //StringBuilder sbT = new StringBuilder(T);
            //if (S.Contains("#"))
            //{
            //    sbS = GetSb(sbS);
            //}
            //if (T.Contains("#"))
            //{
            //    sbT = GetSb(sbT);
            //}
            //if (sbS.ToString().Equals(sbT.ToString()))
            //{
            //    return true;
            //}
            //else
            //{
            //    return false;
            //}

            //var stackS = GetSb(S);
            //var stackT = GetSb(T);
            //if (stackS.ToString().Equals(stackT.ToString()))
            //{
            //    return true;
            //}
            //else
            //{
            //    return false;
            //}

            int i = S.Length - 1, j = T.Length - 1;
            int skipS = 0, skipT = 0;

            while (i >= 0 || j >= 0)
            {
                while (i >= 0)
                {
                    if (S.ElementAt(i) == '#')
                    {
                        skipS++;
                        i--;
                    }
                    else if (skipS > 0)
                    {
                        skipS--;
                        i--;
                    }
                    else
                    {
                        break;
                    }
                }
                while (j >= 0)
                {
                    if (T.ElementAt(j) == '#')
                    {
                        skipT++;
                        j--;
                    }
                    else if (skipT > 0)
                    {
                        skipT--;
                        j--;
                    }
                    else
                    {
                        break;
                    }
                }
                if (i >= 0 && j >= 0)
                {
                    if (S.ElementAt(i) != T.ElementAt(j))
                    {
                        return false;
                    }
                }
                else
                {
                    if (i >= 0 || j >= 0)
                    {
                        return false;
                    }
                }
                i--;
                j--;
            }
            return true;
        }

        public static Stack<char> GetSb(string str)
        {
            var stack = new Stack<char>();
            for (int i = 0; i < str.Length; i++)
            {
                if (str[i] != '#')
                {
                    stack.Push(str[i]);
                }
                else if (stack.Count > 0)
                {
                    stack.Pop();
                }
            }
            return stack;
        }

        //public static StringBuilder GetSb(StringBuilder str)
        //{
        //    for (int i = 0; i < str.Length; i++)
        //    {
        //        if (str[i] == '#')
        //        {
        //            if (i == 0)
        //            {
        //                str.Remove(0, 1);
        //                i -= 1;
        //            }
        //            else
        //            {
        //                str.Remove(i - 1, 2);
        //                i -= 2;
        //            }
        //        }
        //    }
        //    return str;
        //}

        /// <summary>
        /// 俄罗斯套娃信封问题
        /// </summary>
        /// <param name="envelopes"></param>
        /// <returns></returns>
        public static int MaxEnvelopes(int[][] envelopes)
        {
            // sort on increasing in first dimension and decreasing in second
            Array.Sort(envelopes, new IntComparer());
            int[] secondDim = new int[envelopes.Length];
            for (int i = 0; i < envelopes.Length; ++i) secondDim[i] = envelopes[i][1];
            return LengthOfLIS(secondDim);

        }

        /// <summary>
        /// 最长上升子序列
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public static int LengthOfLIS(int[] nums)
        {
            // O(n^2) dp

            //int[] dp = new int[nums.Length];
            //int len = 0;
            //foreach (int num in nums)
            //{
            //    int i = Array.BinarySearch(dp, 0, len, num);
            //    if (i < 0)
            //    {
            //        i = -(i + 1);
            //    }
            //    dp[i] = num;
            //    if (i == len)
            //    {
            //        len++;
            //    }
            //}
            //return len;
            var res = new int[nums.Length];
            int len = 0;

            foreach (int num in nums)
            {
                var index = Array.BinarySearch(res, 0, len, num);
                index = index < 0 ? ~index : index;
                res[index] = num;
                len = index == len ? len + 1 : len;
            }

            return len;

            // O(nlogn) binarySearch
            if (nums.Length == 0)
            {
                return 0;
            }
            int[] dp = new int[nums.Length];
            dp[0] = 1;
            int maxans = 1;
            for (int i = 1; i < dp.Length; i++)
            {
                int maxval = 0;
                for (int j = 0; j < i; j++)
                {
                    if (nums[i] > nums[j])
                    {
                        maxval = Math.Max(maxval, dp[j]);
                        //dp[i] = Math.Max(dp[i], dp[j] + 1);
                    }
                }
                dp[i] = maxval + 1;
                maxans = Math.Max(maxans, dp[i]);
            }
            return maxans;
            //  return dp.Max();
        }

        public class IntComparer : IComparer<int[]>
        {
            public int Compare(int[] arr1, int[] arr2)
            {
                // 按宽度升序排列，如果宽度⼀样，则按⾼度降序排列，避免相同宽度的错误情况
                if (arr1[0] == arr2[0])
                {
                    return arr2[1] - arr1[1];
                }
                else
                {
                    return arr1[0] - arr2[0];
                }
            }
        }

        public class LRUCache
        {
            Dictionary<int, int> dic = new Dictionary<int, int>();
            LinkedList<int> list = new LinkedList<int>();
            //Hashtable hs = new Hashtable();
            int _capacity;
            public LRUCache(int capacity)
            {

                _capacity = capacity;
            }

            public int get(int key)
            {
                if (!dic.ContainsKey(key))
                {
                    return -1;
                }
                list.Remove(key);
                list.AddLast(key);
                return dic[key];
            }

            public void put(int key, int value)
            {
                if (dic.ContainsKey(key))
                {
                    dic[key] = value;
                    list.Remove(key);
                    list.AddLast(key);
                    return;
                }
                if (_capacity == dic.Count)
                {
                    int removeKey = list.First.Value;
                    list.Remove(removeKey);
                    list.AddLast(key);
                    dic.Remove(removeKey);
                    dic.Add(key, value);
                }
                else
                {
                    dic.Add(key, value);
                    list.AddLast(key);
                }

                //错误：只考虑个别用例，应先排除最先发生的情况，比如散列表有相同的键。
                //if (_capacity <= dic.Count)
                //{
                //    if (!dic.ContainsKey(key))
                //    {
                //        int removeKey = list.First.Value;
                //        list.Remove(removeKey);
                //        dic.Remove(removeKey);
                //        return;
                //    }
                //}
                //if (dic.ContainsKey(key))
                //{
                //    dic[key] = value;
                //}
                //else
                //{
                //    dic.Add(key, value);
                //}
                //list.Remove(key);
                //list.AddLast(key);
            }
        }

        /// <summary>
        /// 完全平方数
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        public static int NumSquares(int n)
        {
            List<int> square_nums = new List<int>();
            for (int i = 1; i * i <= n; ++i)
            {
                square_nums.Add(i * i);
            }

            HashSet<int> queue = new HashSet<int>();
            queue.Add(n);

            int level = 0;
            while (queue.Count() > 0)
            {
                level += 1;
                HashSet<int> next_queue = new HashSet<int>();
                foreach (int remainder in queue)
                {
                    foreach (int square in square_nums)
                    {
                        if (remainder.Equals(square))
                        {
                            return level;
                        }
                        else if (remainder < square)
                        {
                            break;
                        }
                        else
                        {
                            next_queue.Add(remainder - square);
                        }
                    }
                }
                queue = next_queue;
            }
            return level;
        }

        /// <summary>
        /// Z 字形变换
        /// </summary>
        /// <param name="s"></param>
        /// <param name="numRows"></param>
        /// <returns></returns>
        public static string Convert(string s, int numRows)
        {
            if (numRows == 1)
            {
                return s;
            }
            List<StringBuilder> sbList = new List<StringBuilder>();
            for (int i = 0; i < Math.Min(numRows, s.Length); i++)
            {
                sbList.Add(new StringBuilder());
            }
            int updown = -1;
            int index = 0;
            char[] ch = s.ToCharArray();
            for (int i = 0; i < ch.Count(); i++)
            {
                sbList[index].Append(ch[i]);
                if (index == 0 || index == sbList.Count - 1)
                {
                    updown = -updown;
                }
                index += updown;
            }
            return string.Join("", sbList.Select(a => a.ToString()));
        }

        public static int SearchInsert(int[] nums, int target)
        {
            int n = nums.Length;
            int l = 0, r = n - 1;
            while (l <= r)
            {
                int mid = (l + r) / 2;
                if (nums[mid] < target)
                    l = mid + 1;
                else r = mid - 1;
            }
            return l;
            //return nums.Append(target).OrderBy(a => a).Select((a, i) => new { index = i, target = a }).First(a => a.target == target).index;
        }

        /// <summary>
        /// 合并两个有序数组
        /// </summary>
        /// <param name="nums1"></param>
        /// <param name="m"></param>
        /// <param name="nums2"></param>
        /// <param name="n"></param>
        public static void Merge(int[] nums1, int m, int[] nums2, int n)
        {
            int[] nums1_copy = new int[m];
            Array.Copy(nums1, 0, nums1_copy, 0, m);
            int p1 = 0;
            int p2 = 0;
            int p = 0;
            while (p1 < m && p2 < n)
            {
                nums1[p++] = nums1_copy[p1] < nums2[p2] ? nums1_copy[p1++] : nums2[p2++];
            }
            if (p1 < m)
            {
                Array.Copy(nums1_copy, p1, nums1, p1 + p2, m + n - p1 - p2);
            }
            if (p2 < n)
            {
                Array.Copy(nums2, p2, nums1, p1 + p2, m + n - p1 - p2);
            }

        }



        public class H2O
        {
            private SemaphoreSlim SemaphoreH = new SemaphoreSlim(2, 2);
            private SemaphoreSlim SemaphoreO = new SemaphoreSlim(0, 1);
            public H2O()
            {

            }

            public void Hydrogen(Action releaseHydrogen)
            {
                SemaphoreH.Wait();
                // releaseHydrogen() outputs "H". Do not change or remove this line.
                releaseHydrogen();
                if (SemaphoreH.CurrentCount == 0)
                {
                    SemaphoreO.Release();
                }

            }
            //HHO
            public void Oxygen(Action releaseOxygen)
            {
                SemaphoreO.Wait();
                // releaseOxygen() outputs "O". Do not change or remove this line.
                releaseOxygen();
                SemaphoreH.Release(2);
            }
        }

        /// <summary>
        /// 闭包（定义在函数内部的函数）
        /// 调用函数内部局部变量的函数（编译器使T1生成类，匿名函数为成员函数）
        /// 如果一个氧线程到达屏障时没有氢线程到达，它必须等候直到两个氢线程到达。
        /// </summary>

        public class Closer
        {
            public Func<int> T1()
            {
                var n = 99;
                return () =>
                {
                    Console.WriteLine(n);
                    return n;
                };
            }
        }

        public class TCloser1
        {
            public Func<int> T1()
            {
                var n = 10;
                return () =>
                {
                    return n;
                };
            }

            public Func<int> T4()
            {
                return () =>
                {
                    var n = 10;
                    return n;
                };
            }
        }

        public class FooBar
        {
            //AutoResetEvent reset1 = new AutoResetEvent(false);
            //AutoResetEvent reset2 = new AutoResetEvent(false);
            private SemaphoreSlim foo = new SemaphoreSlim(1);
            private SemaphoreSlim bar = new SemaphoreSlim(0);
            private int n;

            public FooBar(int n)
            {
                this.n = n;
            }

            public void Foo(Action printFoo)
            {
                for (int i = 0; i < n; i++)
                {
                    //printFoo();
                    //reset1.Set();
                    //reset2.WaitOne();
                    foo.Wait();
                    printFoo();
                    bar.Release();
                }
            }

            public void Bar(Action printBar)
            {

                for (int i = 0; i < n; i++)
                {
                    //reset1.WaitOne();
                    //printBar();
                    //reset2.Set();
                    bar.Wait();
                    printBar();
                    foo.Release();
                }
            }
        }

        /// <summary>
        /// 使结果不超过阈值的最小除数
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="threshold"></param>
        /// <returns></returns>
        public static int smallestDivisor(int[] nums, int threshold)
        {
            //输入：nums = [1, 2, 5, 9], threshold = 6
            //输出：5
            //解释：如果除数为 1 ，我们可以得到和为 17 （1 + 2 + 5 + 9）。
            //如果除数为 4 ，我们可以得到和为 7(1 + 1 + 2 + 3) 。如果除数为 5 ，和为 5(1 + 1 + 1 + 2)。

            int max = nums.Max();
            int left = 1, right = max;
            while (left < right)
            {
                int mid = (left + right) / 2, sum = 0;
                foreach (var num in nums)
                {
                    sum += (num + mid - 1) / mid;
                }
                if (sum <= threshold)
                    right = mid;
                else
                    left = mid + 1;
            }
            return right;
        }

        public class WordsFrequency
        {
            Dictionary<string, int> bookDic = new Dictionary<string, int>();
            public WordsFrequency(string[] book)
            {
                foreach (var item in book)
                {
                    if (bookDic.ContainsKey(item))
                    {
                        bookDic[item]++;
                    }
                    else
                    {
                        bookDic.Add(item, 1);
                    }
                }
            }

            public int Get(string word)
            {
                bookDic.TryGetValue(word, out int value);
                return value;
            }
        }

        public static int StrStr(string haystack, string needle)
        {
            return haystack.IndexOf(needle);
        }
        public class Foo
        {
            AutoResetEvent reset1 = new AutoResetEvent(false);
            AutoResetEvent reset2 = new AutoResetEvent(false);
            public Foo()
            {

            }

            public void First(Action printFirst)
            {
                // printFirst() outputs "first". Do not change or remove this line.
                printFirst();
                reset1.Set();
            }

            public void Second(Action printSecond)
            {
                reset1.WaitOne();
                // printSecond() outputs "second". Do not change or remove this line.
                printSecond();
                reset2.Set();
            }

            public void Third(Action printThird)
            {
                reset2.WaitOne();
                // printThird() outputs "third". Do not change or remove this line.
                printThird();
            }
        }

        public static int RemoveElement(int[] nums, int val)
        {
            //if (nums.Length == 0)
            //{
            //    return 0;
            //}
            //int L = 0, R = nums.Length - 1;
            //while (L <= R)
            //{
            //    while (L < R && nums[R] == val)
            //    {
            //        R--;
            //    }
            //    if (L <= R && nums[L] == val)
            //    {
            //        nums[L] = nums[R];
            //        R--;
            //    }
            //    L++;
            //}
            //return R + 1;

            int L = 0, R = nums.Length;
            while (L < R)
            {
                if (nums[L] == val)
                {
                    nums[L] = nums[R - 1];
                    R--;
                }
                else
                {
                    L++;
                }
            }
            return R;
        }

        public int removeElement1(int[] nums, int val)
        {
            int i = 0;
            for (int j = 0; j < nums.Length; j++)
            {
                if (nums[j] != val)
                {
                    nums[i] = nums[j];
                    i++;
                }
            }
            return i;
        }

        public static int removeElement2(int[] nums, int val)
        {
            int i = 0;
            int n = nums.Length;
            while (i < n)
            {
                if (nums[i] == val)
                {
                    nums[i] = nums[n - 1];
                    // reduce array size by one
                    n--;
                }
                else
                {
                    i++;
                }
            }
            return n;
        }

        public class ListNode
        {
            public int val;
            public ListNode next;
            public ListNode(int x) { val = x; }
            public ListNode(int val = 0, ListNode next = null)
            {
                this.val = val;
                this.next = next;
            }
        }
        public static ListNode AddTwoNumbers(ListNode l1, ListNode l2)
        {
            ListNode head = new ListNode(0);
            ListNode p = l1, q = l2, curr = head;
            int carry = 0;
            while (p != null || q != null)
            {
                int x = p != null ? p.val : 0;
                int y = q != null ? q.val : 0;
                int sum = carry + x + y;
                carry = sum / 10;
                curr.next = new ListNode(sum % 10);
                curr = curr.next;
                if (p != null)
                {
                    p = p.next;
                }
                if (q != null)
                {
                    q = q.next;
                }
            }
            if (carry > 0)
            {
                curr.next = new ListNode(carry);
            }
            return head.next;
        }

        public static int LengthOfLongestSubstring(string s)
        {

            //List<string> result = new List<string>();
            //int start = 0; int end = 1;
            //while (start < s.Length)
            //{
            //    //重复则找重复位置重新再找
            //    while (end < s.Length && !s.Substring(start, end - start).Contains(s.Substring(end, 1)))
            //    {
            //        end++;
            //    }
            //    result.Add(s.Substring(start, end - start));
            //    start++;
            //    end = start + 1;
            //}
            //return result.Any()?result.Max(a => a.Length):0;

            /*滑动窗口，左右指针计算；重复次数则选用散列表（哈希表）*/
            // 哈希集合，记录每个字符是否出现过
            string abc = "pwwbcb";
            HashSet<char> occ = new HashSet<char>();
            int n = s.Length;
            // 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
            int rk = -1, ans = 0;
            for (int i = 0; i < n; ++i)
            {
                if (i != 0)
                {
                    // 左指针向右移动一格，移除一个字符
                    occ.Remove(s.ElementAt(i - 1));
                }
                while (rk + 1 < n && !occ.Contains(s.ElementAt(rk + 1)))
                {
                    // 不断地移动右指针
                    occ.Add(s.ElementAt(rk + 1));
                    ++rk;
                }
                // 第 i 到 rk 个字符是一个极长的无重复字符子串
                ans = Math.Max(ans, rk - i + 1);
            }
            return ans;

        }

        /// <summary>
        /// 最长的回文子串
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static string LongestPalindrome(string s)
        {
            if (s == null || s.Length < 1) return "";
            int start = 0, end = 0;
            for (int i = 0; i < s.Length; i++)
            {
                int len1 = expandAroundCenter(s, i, i);
                int len2 = expandAroundCenter(s, i, i + 1);
                int len = Math.Max(len1, len2);
                if (len > end - start)
                {
                    start = i - (len - 1) / 2;
                    end = i + len / 2;
                }
            }
            return s.Substring(start, end + 1 - start);
        }

        private static int expandAroundCenter(string s, int left, int right)
        {
            int L = left, R = right;
            while (L >= 0 && R < s.Length && s.ElementAt(L) == s.ElementAt(R))
            {
                L--;
                R++;
            }
            return R - L - 1;
        }

        public static bool IsPalindrome(int x)
        {
            //string s = x.ToString();
            //for (int i = 0; i < s.Length; i++)
            //{
            //    if (s[i]!=s[s.Length-i-1])
            //    {
            //        return false;
            //    }
            //}
            //return true;

            // 特殊情况：
            // 如上所述，当 x < 0 时，x 不是回文数。
            // 同样地，如果数字的最后一位是 0，为了使该数字为回文，
            // 则其第一位数字也应该是 0
            // 只有 0 满足这一属性
            if (x < 0 || (x % 10 == 0 && x != 0))
            {
                return false;
            }

            int revertedNumber = 0;
            while (x > revertedNumber)
            {
                revertedNumber = revertedNumber * 10 + x % 10;
                x /= 10;
            }

            // 当数字长度为奇数时，我们可以通过 revertedNumber/10 去除处于中位的数字。
            // 例如，当输入为 12321 时，在 while 循环的末尾我们可以得到 x = 12，revertedNumber = 123，
            // 由于处于中位的数字不影响回文（它总是与自己相等），所以我们可以简单地将其去除。
            return x == revertedNumber || x == revertedNumber / 10;
        }

        public static void DFS(char[][] grid, int r, int c)
        {
            int nr = grid.Length;
            int nc = grid[0].Length;
            if (r < 0 || c < 0 || r >= nr || c >= nc || grid[r][c] == '0')
            {
                return;
            }
            grid[r][c] = '0';
            DFS(grid, r - 1, c);
            DFS(grid, r + 1, c);
            DFS(grid, r, c - 1);
            DFS(grid, r, c + 1);
        }

        /// <summary>
        /// 深度优先
        /// </summary>
        /// <param name="grid"></param>
        /// <returns></returns>
        public static int NumIslands(char[][] grid, int type)
        {
            switch (type)
            {
                // 深度优先
                case 1:
                    {
                        if (grid == null || grid.Length == 0)
                        {
                            return 0;
                        }
                        int nr = grid.Length;
                        int nc = grid[0].Length;
                        int num_islands = 0;
                        for (int r = 0; r < nr; r++)
                        {
                            for (int c = 0; c < nc; c++)
                            {
                                if (grid[r][c] == '1')
                                {
                                    ++num_islands;
                                    DFS(grid, r, c);
                                }
                            }
                        }
                        return num_islands;
                    }

                case 2:
                    {
                        if (grid == null || grid.Length == 0)
                        {
                            return 0;
                        }
                        int nr = grid.Length;
                        int nc = grid[0].Length;
                        int num_islands = 0;
                        for (int r = 0; r < nr; r++)
                        {
                            for (int c = 0; c < nc; c++)
                            {
                                if (grid[r][c] == '1')
                                {
                                    ++num_islands;
                                    grid[r][c] = '0';
                                    Queue<int> neighbors = new Queue<int>();
                                    neighbors.Enqueue(r * nc + c);
                                    while (neighbors.Count > 0)
                                    {
                                        int id = neighbors.Dequeue();
                                        int row = id / nc;
                                        int col = id % nc;
                                        if (row - 1 >= 0 && grid[row - 1][col] == '1')
                                        {
                                            neighbors.Enqueue((row - 1) * nc + col);
                                            grid[row - 1][col] = '0';
                                        }
                                        if (row + 1 < nr && grid[row + 1][col] == '1')
                                        {
                                            neighbors.Enqueue((row + 1) * nc + col);
                                            grid[row + 1][col] = '0';
                                        }
                                        if (col - 1 >= 0 && grid[row][col - 1] == '1')
                                        {
                                            neighbors.Enqueue(row * nc + col - 1);
                                            grid[row - 1][col] = '0';
                                        }
                                        if (col + 1 < nc && grid[row][col + 1] == '1')
                                        {
                                            neighbors.Enqueue(row * nc + col + 1);
                                            grid[row + 1][col] = '0';
                                        }
                                    }
                                }
                            }
                        }
                        return num_islands;
                    }
                case 3:
                    {
                        if (grid == null || grid.Length == 0)
                        {
                            return 0;
                        }

                        int nr = grid.Length;
                        int nc = grid[0].Length;
                        UnionFind uf = new UnionFind(grid);
                        for (int r = 0; r < nr; ++r)
                        {
                            for (int c = 0; c < nc; ++c)
                            {
                                if (grid[r][c] == '1')
                                {
                                    grid[r][c] = '0';
                                    if (r - 1 >= 0 && grid[r - 1][c] == '1')
                                    {
                                        uf.Union(r * nc + c, (r - 1) * nc + c);
                                    }
                                    if (r + 1 < nr && grid[r + 1][c] == '1')
                                    {
                                        uf.Union(r * nc + c, (r + 1) * nc + c);
                                    }
                                    if (c - 1 >= 0 && grid[r][c - 1] == '1')
                                    {
                                        uf.Union(r * nc + c, r * nc + c - 1);
                                    }
                                    if (c + 1 < nc && grid[r][c + 1] == '1')
                                    {
                                        uf.Union(r * nc + c, r * nc + c + 1);
                                    }
                                }
                            }
                        }
                        return uf.getCount();
                    }
                default:
                    break;
            }
            return 0;
        }

        public class UnionFind
        {
            int count = 0;
            int[] parent;
            int[] rank;
            public UnionFind(char[][] grid)
            {
                count = 0;
                int m = grid.Length;
                int n = grid[0].Length;
                parent = new int[m * n];
                rank = new int[m * n];
                for (int i = 0; i < m; ++i)
                {
                    for (int j = 0; j < n; ++j)
                    {
                        if (grid[i][j] == '1')
                        {
                            parent[i * n + j] = i * n + j;
                            ++count;
                        }
                        rank[i * n + j] = 0;
                    }
                }
            }

            public int Find(int i)
            {
                if (parent[i] != i) parent[i] = Find(parent[i]);
                return parent[i];
            }

            public void Union(int x, int y)
            {
                int rootx = Find(x);
                int rooty = Find(y);
                if (rootx != rooty)
                {
                    if (rank[rootx] > rank[rooty])
                    {
                        parent[rooty] = rootx;
                    }
                    else if (rank[rootx] < rank[rooty])
                    {
                        parent[rootx] = rooty;
                    }
                    else
                    {
                        //当前查询节点赋值父节点值，这样做也是为了重复减去count的值，如第二行第一列也为1看的很明显。
                        parent[rooty] = rootx;
                        //找到第一次可以归并的累加归并集合的根节点值
                        rank[rootx] += 1;
                    }
                    --count;
                }


            }

            public int getCount()
            {
                return count;
            }
        }
    }
}
