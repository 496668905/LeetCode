using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
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
            //给定链表 1->2->3->4->5, 重新排列为 1->5->2->4->3.
            ListNode head = new ListNode() { val = 1, next = new ListNode(2, new ListNode(3, new ListNode(4, new ListNode(5)))) };
            ReorderList(head);
            Console.WriteLine();
            Console.ReadKey();
        }

        /// <summary>
        ///  重排链表
        /// </summary>
        /// <param name="head"></param>
        public static void ReorderList(ListNode head)
        {
            //if (head == null)
            //{
            //    return;
            //}
            //List<ListNode> list = new List<ListNode>();
            //ListNode node = head;
            //while (node != null)
            //{
            //    list.Add(node);
            //    node = node.next;
            //}
            //int i = 0, j = list.Count() - 1;
            //while (i < j)
            //{
            //    list.ElementAt(i).next = list.ElementAt(j);
            //    i++;
            //    if (i == j)
            //    {
            //        break;
            //    }
            //    list.ElementAt(j).next = list.ElementAt(i);
            //    j--;
            //}
            //list.ElementAt(i).next = null;


            if (head == null)
            {
                return;
            }
            ListNode mid = middleNode(head);
            ListNode l1 = head;
            ListNode l2 = mid.next;
            mid.next = null;
            l2 = reverseList(l2);
            mergeList(l1, l2);

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
