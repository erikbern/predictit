from matplotlib import pyplot
import numpy
import re
import requests
import scipy.optimize
import scipy.stats
import sys

id = int(re.search('/(\d+)/', sys.argv[1]).group(1))
res = requests.get('https://www.predictit.org/api/Market/%d/Contracts' % id)

los, his = [], []
yps, nps = [], []
yqs, nqs = [], []
contracts = []
for c in res.json():
    name = c['contractName']
    contracts.append(name)
    if name.endswith('or fewer'):
        name = '0 - ' + name.replace(' or fewer', '')
    if name.endswith('or lower'):
        name = '0 - ' + name.replace(' or lower', '')
    elif name.endswith('or more'):
        name = name.replace(' or more', '') + ' - 999999'
    elif name.endswith('or higher'):
        name = name.replace(' or higher', '') + ' - 100%'
    lo, hi = map(lambda z: float(z.strip('%')), name.split(' - '))
    coalesce = lambda z, d: z if z is not None else d
    los.append(lo)
    his.append(hi)
    yps.append(coalesce(c['bestYesPrice'], 0))
    nps.append(coalesce(c['bestNoPrice'], 1))
    yqs.append(coalesce(c['bestYesQuantity'], 0))
    nqs.append(coalesce(c['bestNoQuantity'], 0))

if all(int(z) == z for z in los):
    # It's a bunch of integers: let's model as a negative binomial distribution
    los, his, yps, nps, yqs, nqs = (numpy.array(z) for z in (los, his, yps, nps, yqs, nqs))

    grid = [(n, p) for n in numpy.exp(numpy.linspace(0, 12, 1000)) for p in numpy.linspace(0, 1, 100)]

    def cdf(x, z):
        n, p = z
        return scipy.stats.nbinom.cdf(x, n, p)

    get_probs = lambda z: cdf(his, z) - cdf(numpy.maximum(los-1, 0), z)
    get_range = lambda: list(range(int(max(los)*1.5)))
    plot_pdf = lambda z: (get_range(), scipy.stats.nbinom.pmf(get_range(), *z))
else:
    # It's a percentage: let's model it as a Beta distribution

    # First, replace every limit by the midpoints
    los2 = [(hi_prev + lo_cur)/200 for lo_cur, hi_prev in zip(los, [0] + his[:-1])]
    his2 = [(hi_cur + lo_next)/200 for hi_cur, lo_next in zip(his, los[1:] + [1])]
    los, his, yps, nps, yqs, nqs = (numpy.array(z) for z in (los2, his2, yps, nps, yqs, nqs))

    # Generate a grid to search on
    g = numpy.exp(numpy.linspace(0, 12, 1000))
    grid = [(p, q) for p in g for q in g if his[0]*0.7 <= p/(p+q) <= los[-1]*1.4]
    print('checking', len(grid), 'params')

    def cdf(x, z):
        a, b = z
        return scipy.stats.beta.cdf(x, a, b)

    get_probs = lambda z: cdf(his, z) - cdf(los, z)
    plot_pdf = lambda z: (numpy.linspace(0, 1), scipy.stats.beta.pdf(numpy.linspace(0, 1), *z))

#####################

def loss(z, p=False):
    ps = get_probs(z)
    y_loss = numpy.maximum((ps - yps)*yqs, 0)
    n_loss = numpy.maximum(((1-ps) - nps)*nqs, 0)
    l = numpy.sum(y_loss + n_loss)
    if p:
        print('loss:', l)
        print(y_loss)
        print(n_loss)
        for contract, p in zip(contracts, ps):
            print('%20s %.4f' % (contract, p))
        tips = []
        for gains, prices, worths, side in [(ps-yps, yps, ps, 'yes'), ((1-ps)-nps, nps, 1-ps, 'no')]:
            for gain, price, worth, contract in zip(gains, prices, worths, contracts):
                if gain > 0:
                    tips.append((gain, contract, side, price, worth))
        tips.sort(reverse=True)
        for gain, contract, side, price, worth in tips:
            print('%.4f: buy %20s %3s @ %.2f worth %.4f' % (gain, contract, side, price, worth))
    return l

# Grid search to find solution
s = min(grid, key=loss)
print('Grid min loss:', s, '->', loss(s))
# s = scipy.optimize.minimize(loss, x0=s).x
# print('Fine tuned loss:', s, '->', loss(s))
loss(s, True)
xs, ys = plot_pdf(s)
pyplot.plot(xs, ys)
pyplot.grid(True)
pyplot.show()
