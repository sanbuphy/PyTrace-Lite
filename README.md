# PyTrace-Lite
A simple tool helps you trace the python programs.	  

It can display the execution sequence and corresponding annotations and parameters of the specified function.

![](./trace_file/Sample.png)

I believe the [official python trace document](https://docs.python.org/3/library/trace.html) is not written well, so I rewrite the code.	  

The official filter seems to not be woking, so I write my own filter code.	  

My demo is used to trace the MMDetection code, you can change it.

As the saying goes,read the code is worse than read the execution.—— Yanyan Jiang 

子曾经曰过，读代码不如读执行，优雅调试前提是舒服 —— [蒋炎岩](https://cs.nju.edu.cn/ics/people/yanyanjiang/index.html)

PyTrace 可以 trace 所有的 python 程序（能结束，非死循环，你只要把程序调用入口放到main(),不要忘记修改程序需要import的内容，而import的内容不能放入main()）。

你可以筛选输出后的csv文件获取想要的信息，帮助你更好的理解代码数据流动。

## Usage
1. First, added the functions you want to the main() function.

2. Run it and wait!
```shell
python powerful_trace.py
```
if successful, you will see the output information.


You can search the function in main() to find the specific output location in csv.

I suggest you reshape the csv by setting row height and column width to 20.  

You can also delete the 'file_name' column to read more comfortably.




