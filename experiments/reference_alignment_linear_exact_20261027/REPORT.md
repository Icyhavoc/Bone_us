# 六路 Beam 投票参考协议对齐结果

状态：complete；完成时间：2026-10-27T14:12:10+08:00

测试正确数：56/67
测试 accuracy：83.5821%
测试 AUC：0.912758
目标 61/67：未达到

重采样：reference_linear；其余对齐共享全段 max1、双 Beam 融合头、
双 Beam 交叉熵、参考五折、验证 AUC 选 epoch、BF16、fused AdamW、参考学习率和五折概率平均。
