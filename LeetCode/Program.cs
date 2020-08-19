using System;
using System.Collections.Generic;
using System.Linq;
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
            //Console.WriteLine(StrStr("hello", "ll"));
            //Console.WriteLine("-----------------------------------------------------------");
            var test = new FooBar(2);
            Thread t1 = new Thread(() => test.Foo(() => { Console.Write("foo"); }));
            Thread t2 = new Thread(() => test.Bar(() => { Console.Write("bar"); }));
            t1.Start();
            t2.Start();
            Console.ReadKey();
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

        public class WordsFrequency
        {
            Dictionary<string, int> bookDic=new Dictionary<string, int>();
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
