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
    yps.append(coalesce(c['bestYesPrice'], 1))
    nps.append(coalesce(c['bestNoPrice'], 1))
    yqs.append(coalesce(c['bestYesQuantity'], 0))
    nqs.append(coalesce(c['bestNoQuantity'], 0))

if all(int(z) == z for z in los) and all(int(z) == z for z in his):
    # It's a bunch of integers: let's model as a negative binomial distribution
    los, his, yps, nps, yqs, nqs = (numpy.array(z) for z in (los, his, yps, nps, yqs, nqs))

    grid = [(n, p) for n in numpy.exp(numpy.linspace(0, 12, 1000)) for p in numpy.linspace(0, 1, 100)]

    cdf = lambda x, z: scipy.stats.nbinom.cdf(x, *z)
    get_probs = lambda z: cdf(his, z) - cdf(numpy.maximum(los-1, 0), z)
    get_range = lambda z: list(range(int(scipy.stats.nbinom.ppf(0.99, *z))))
    plot_pdf = lambda z: (get_range(z), scipy.stats.nbinom.pmf(get_range(z), *z))
else:
    # It's a percentage: let's model it as a Beta distribution

    # First, replace every limit by the midpoints
    los2 = [(hi_prev + lo_cur)/200 for lo_cur, hi_prev in zip(los, [0] + his[:-1])]
    his2 = [(hi_cur + lo_next)/200 for hi_cur, lo_next in zip(his, los[1:] + [1])]
    los, his, yps, nps, yqs, nqs = (numpy.array(z) for z in (los2, his2, yps, nps, yqs, nqs))

    # Generate a grid to search on
    g = numpy.exp(numpy.linspace(0, 12, 1000))
    grid = [(p, q) for p in g for q in g if his[0]*0.7 <= p/(p+q) <= los[-1]*1.4]

    cdf = lambda x, z: scipy.stats.beta.cdf(x, *z)
    get_probs = lambda z: cdf(his, z) - cdf(los, z)
    get_range = lambda z: numpy.linspace(scipy.stats.beta.ppf(0.01, *z), scipy.stats.beta.ppf(0.99, *z), 1000)
    plot_pdf = lambda z: (get_range(z), scipy.stats.beta.pdf(get_range(z), *z))

#####################

def margin(w):
    # Kind of relu, if no contracts are in the money then maximize margin instead
    return numpy.maximum(w, w*1e-3)

def loss(z):
    ps = get_probs(z)
    y_loss = margin((ps - yps)*yqs)
    n_loss = margin(((1-ps) - nps)*nqs)
    l = numpy.sum(y_loss + n_loss)
    return l

def print_loss(z):
    ps = get_probs(z)
    for contract, p in zip(contracts, ps):
        print('%20s %.4f' % (contract, p))
    tips = []
    options = [(yps, ps, 'yes'),
               (nps, 1-ps, 'no'),
               (1-nps+0.01, ps, 'yes (limit)'),
               (1-yps+0.01, 1-ps, 'no (limit)')]
    for prices, worths, side in options:
        for price, worth, contract in zip(prices, worths, contracts):
            tips.append((worth/price, contract, side, price, worth))
    tips.sort(reverse=True)
    for gain, contract, side, price, worth in tips:
        print('%+9.2f%%: buy %30s @ %.2f worth %.4f' % ((gain-1)*100, contract + ' ' + side, price, worth))

print('Grid searching', len(grid), 'combinations')
z = min(grid, key=loss)
print('Grid min loss:', z, '->', loss(z))
# z = scipy.optimize.minimize(loss, x0=s).x
# print('Fine tuned loss:', z, '->', loss(z))
print_loss(z)
xs, ys = plot_pdf(z)
pyplot.plot(xs, ys)
pyplot.grid(True)
pyplot.show()
