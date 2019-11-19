import numpy as np
import sys

if len(sys.argv) != 5:
    print("""Usage: four command line arguments, in the following order, must be included with call:
    n (int), the number of samples desired for dataset
    m (int), the dimension of each input sample
    std (float), the standard deviation of the positive and negative distributions
    mean (float), the distance from zero the mean of each distribution is
    """)
    sys.exit()

# dimension of each input sample
n, m, std, mean  = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])

# draw values from negative distribution to create negative samples
x_neg = np.random.normal(loc=-mean, scale=std, size=(int(n/2), m))
y_neg = np.empty((int(n/2), 1))
y_neg.fill(0)
data_neg = np.concatenate((x_neg, y_neg), axis=1)
# draw values from positive distribution to create positive samples
x_pos = np.random.normal(loc=mean, scale=std, size=(int(n/2), m))
y_pos = np.empty((int(n/2), 1))
y_pos.fill(1)
data_pos = np.concatenate((x_pos, y_pos), axis=1)
# concatenate negative and positive data and shuffle it
data = np.concatenate((data_neg, data_pos), axis=0)
np.random.shuffle(data)

fname = '../data_n{}_m{}_mu{}.csv'.format(n, m, mean)
header = '{},{}'.format(n, m)
np.savetxt(fname, data, delimiter=",", fmt='%.3f', header=header)
