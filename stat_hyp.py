def stat_hypothesis(data1,data2,val, alt, alpha):
    
    from statsmodels.stats.weightstats import ztest

    if data2 == None:
      ztest_Score, p_value = ztest(data1, value = val, alternative=alt)
    else:
      ztest_Score, p_value = ztest(data1,data2, value = val, alternative=alt)

    print('Z-statistics is equal {:.6f}'.format(ztest_Score))
    print('P-value is equal {:.6f}'.format(p_value))

    if (p_value < alpha):
      print('Відхилити h0')
    else:
      print('H0 не може бути відхилена')

    return ztest_Score, p_value