from scipy.special import comb

n = 200  # 总元素数
k = 20   # 选择的元素数

combination = comb(n, k)
print("C(200, 20) =", combination)
