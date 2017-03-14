if i - N < 0:
    b = 0
else:
    b = i - N + 1
rsvarr = array[b:i + 1, 0:5]
rsv = (float(rsvarr[-1, -1]) - float(min(rsvarr[:, 3]))) / (float(max(rsvarr[:, 2])) - float(min(rsvarr[:, 3]))) * 100
if i == 0:
    k = rsv
    d = rsv
else:
    k = 1 / float(M1) * rsv + (float(M1) - 1) / M1 * float(kdjarr[-1][2])
    d = 1 / float(M2) * k + (float(M2) - 1) / M2 * float(kdjarr[-1][3])
j = 3 * k - 2 * d
kdjarr.append(list((rsvarr[-1, 0], rsv, k, d, j)))