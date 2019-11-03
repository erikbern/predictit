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
case = 'neg-binomial'
for c in res.json():
    name = c['contractName']
    contracts.append(name)
    if name.endswith('or fewer'):
        name = '0 - ' + name.replace(' or fewer', '')
    if name.endswith('or lower'):
        name = '0 - ' + name.replace(' or lower', '')
    elif name.endswith('or more'):
        name = name.replace(' or more', '') + ' - inf'
    elif name.endswith('or higher'):
        name = name.replace(' or higher', '') + ' - 100%'
    if ' - ' in name:
        lo, hi = map(lambda z: float(z.strip('%')), name.split(' - '))
    else:
        lo = hi = int(name)
    if name.endswith('%'):
        case = 'beta'
    coalesce = lambda z, d: z if z is not None else d
    los.append(lo)
    his.append(hi)
    yps.append(coalesce(c['bestYesPrice'], 1))
    nps.append(coalesce(c['bestNoPrice'], 1))
    yqs.append(coalesce(c['bestYesQuantity'], 0))
    nqs.append(coalesce(c['bestNoQuantity'], 0))

if case == 'beta':
    # First, replace every limit by the midpoints
    los2 = [(hi_prev + lo_cur)/200 for lo_cur, hi_prev in zip(los, [0] + his[:-1])]
    his2 = [(hi_cur + lo_next)/200 for hi_cur, lo_next in zip(his, los[1:] + [1])]
    los, his = los2, his2

los, his, yps, nps, yqs, nqs = (numpy.array(z) for z in (los2, his2, yps, nps, yqs, nqs))
grid_range_a, grid_range_b = 2*his[0]-los[-1], 2*los[-1]-his[0]
print('Constraining grid search to %.2f-%.2f' % (grid_range_a, grid_range_b))

if case == 'neg-binomial':
    # It's a bunch of integers: let's model as a negative binomial distribution
    los, his, yps, nps, yqs, nqs = (numpy.array(z) for z in (los, his, yps, nps, yqs, nqs))
    grid = [(n, p) for n in numpy.exp(numpy.linspace(0, 12, 2000)) for p in numpy.linspace(0, 1, 2000)
            if grid_range_a <= p*n/(1-p) <= grid_range_b]
    cdf = lambda x, z: scipy.stats.nbinom.cdf(x, *z)
    get_probs = lambda z: cdf(his, z) - cdf(los-1, z)
    get_range = lambda z: list(range(int(scipy.stats.nbinom.ppf(0.999, *z))))
    plot_pdf = lambda z: (get_range(z), scipy.stats.nbinom.pmf(get_range(z), *z))
    # transform = lambda w: (numpy.exp(w[0]), 1.0 / (1 + numpy.exp(-w[1])))
else:
    # It's a percentage: let's model it as a Beta distribution
    g = numpy.exp(numpy.linspace(0, 12, 2000))
    grid = [(p, q) for p in g for q in g
            if grid_range_a <= p/(p+q) <= grid_range_b]
    cdf = lambda x, z: scipy.stats.beta.cdf(x, *z)
    get_probs = lambda z: cdf(his, z) - cdf(los, z)
    get_range = lambda z: numpy.linspace(scipy.stats.beta.ppf(0.001, *z), scipy.stats.beta.ppf(0.999, *z), 1000)
    plot_pdf = lambda z: (get_range(z), scipy.stats.beta.pdf(get_range(z), *z))
    # transform = lambda w: numpy.exp(w)

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
    for prices, probs, side in options:
        for price, prob, contract in zip(prices, probs, contracts):
            gain = prob * 0.9 * (1 - price) + (1-prob)*(-price)  # 10% transaction cost
            tips.append((gain/price, contract, side, price, prob))
    tips.sort(reverse=True)
    for gain, contract, side, price, worth in tips:
        print('%+9.2f%%: buy %30s @ %.2f worth %.4f' % (gain*100, contract + ' ' + side, price, worth))

print('Grid searching', len(grid), 'combinations')
z = min(grid, key=loss)
print('Grid min loss:', z, '->', loss(z))
print_loss(z)
xs, ys = plot_pdf(z)
pyplot.plot(xs, ys)
pyplot.grid(True)
pyplot.show()
