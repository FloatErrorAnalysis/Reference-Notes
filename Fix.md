# Fix
**本文档用于纠正和制定计划目标**
## 目标
在给定输入函数f和输入[x0, x1..., xn]，寻找出误差

## 训练数据集
f的字节码和对应误差集合

<table>
<tr><th>f</th><th>[{x: error}]</th></tr>
<tr><th>字节码</th><th>[{0: 1}, {1: 2}....]</th></tr>
</table>

先准备1w份数据，对sqrt_minus函数，改变x如sqrt(2 * x) - sqrt(2 * x - 1)作为训练和测试