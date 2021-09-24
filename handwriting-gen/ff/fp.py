from scipy import stats
ff = [85.07, 86.76, 85.74, 84.8, 84.24]
lstm = [85.19, 87.61, 86.92, 85.76, 85.74]
t, p = stats.ttest_ind(ff, lstm, equal_var=False)
print(t)
print(p)
