from matplotlib import pyplot
import json
import numpy
import random
import re
import requests
import scipy.optimize
import scipy.stats
import sys

id = int(re.search('/(\d+)', sys.argv[1]).group(1))

session = requests.Session()
credentials = json.load(open('credentials.json'))
res = session.get('https://www.predictit.org/dashboard')
res.raise_for_status()
res = session.post('https://www.predictit.org/api/Account/token',
                   data={'grant_type': 'password', 'rememberMe': 'true', **credentials})
res.raise_for_status()
access_token = res.json()['access_token']
res = session.get('https://www.predictit.org/api/Market/%d/Contracts' % id,
                  headers={'Authorization': 'Bearer %s' % access_token})
res.raise_for_status()

los, his = [], []
yps, nps = [], []
yqs, nqs = [], []
yus, nus = [], []
contracts = []
case = 'neg-binomial'
for c in res.json():
    name = c['contractName']
    contracts.append(name)
    if name.startswith('Less than '):
        name = '0 - ' + name.replace('Less than ', '')
    elif name.endswith('or fewer'):
        name = '0 - ' + name.replace(' or fewer', '')
    elif name.endswith('or lower'):
        name = '0 - ' + name.replace(' or lower', '')
    elif name.endswith('or more'):
        name = name.replace(' or more', '') + ' - inf'
    elif name.endswith('or higher'):
        name = name.replace(' or higher', '') + ' - 100%'
    if ' to ' in name:
        name = name.replace(' to ', ' - ')
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
    yus.append(c['userQuantity'] if c['userPrediction'] == 1 else 0)
    nus.append(c['userQuantity'] if c['userPrediction'] == 0 else 0)

if case == 'beta':
    # First, replace every limit by the midpoints
    los2 = [(hi_prev + lo_cur)/200 for lo_cur, hi_prev in zip(los, [0] + his[:-1])]
    his2 = [(hi_cur + lo_next)/200 for hi_cur, lo_next in zip(his, los[1:] + [1])]
    los, his = los2, his2

los, his, yps, nps, yqs, nqs, yus, nus = (numpy.array(z) for z in (los, his, yps, nps, yqs, nqs, yus, nus))
grid_range_a, grid_range_b = 2*his[0]-los[-1], 2*los[-1]-his[0]
print('Constraining grid search to %.2f-%.2f' % (grid_range_a, grid_range_b))

if case == 'neg-binomial':
    # It's a bunch of integers: let's model as a negative binomial distribution
    ns = sorted(set(map(int, numpy.exp(numpy.linspace(0, 6, 400)))))
    ps = numpy.linspace(0, 1, 400)
    grid = [(n, p) for n in ns for p in ps]

    # if grid_range_a <= p*n/(1-p) <= grid_range_b]
    cdf = lambda x, z: scipy.stats.nbinom.cdf(x-z[0], z[0], z[1])
    get_probs = lambda z: cdf(his, z) - cdf(los-1, z)
    get_range = lambda z: list(range(int(scipy.stats.nbinom.ppf(0.999, z[0], z[1])) + z[0]))
    plot_pdf = lambda z: (get_range(z), scipy.stats.nbinom.pmf(numpy.array(get_range(z)) - z[0], z[0], z[1]))
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
    y_loss = margin((ps - yps)*yqs**0.25)
    n_loss = margin(((1-ps) - nps)*nqs**0.25)
    l = numpy.sum(y_loss + n_loss)
    return l

def utility(z, ybs, nbs, new_balance):
    probs = get_probs(z)
    total_no = sum(nus) + sum(nbs)
    total_log_utility = 0.0
    # TODO: add transaction costs
    # new_balance = balance - numpy.dot(yps_limit, ybs) - numpy.dot(nps_limit, nbs)
    payouts = new_balance + (total_no - nus - nbs) + yus + ybs
    return numpy.dot(probs, numpy.log(payouts))

#X, Y = numpy.meshgrid(ns, ps)
#Z = numpy.zeros(X.shape)
#for (i, j), _ in numpy.ndenumerate(X):
#    Z[i][j] = loss((X[i][j], Y[i][j]))
#print(Z)
#fig, ax = pyplot.subplots()
#CS = ax.pcolormesh(X, Y, Z)
#pyplot.show()

#####################

print('Grid searching', len(grid), 'combinations')
best_z, best_score = None, float('inf')
random.shuffle(grid)
for i, z in enumerate(grid):
    score = loss(z)
    if score < best_score:
        print(i, z, score)
        best_z, best_score = z, score
z = best_z

#####################

for contract, p in zip(contracts, get_probs(z)):
    print('%20s %.4f' % (contract, p))

for contract, bs, yu, nu, yp, np in zip(contracts, numpy.eye(yus.shape[0]), yus, nus, yps, nps):
    best_utility, best_side, best_quantity, best_price, best_action = 0, 0, 0, 0, None
    if yu > 0:  # buy more yes, or sell yes
        options = [('buy', 'yes', q, bs*q, bs*0, 1-np+0.01) for q in range(100)] + \
            [('sell', 'yes', q, bs*-q, bs*0, -(yp-0.01)) for q in range(1, yu+1)]
    elif nu > 0:
        options = [('buy', 'no', q, bs*0, bs*q, 1-yp+0.01) for q in range(100)] + \
            [('sell', 'no', q, bs*0, bs*-q, -(np-0.01)) for q in range(1, nu+1)]
    else:
        options = [('buy', 'yes', q, bs*q, bs*0, 1-np+0.01) for q in range(1000)] + \
            [('buy', 'no', q, bs*0, bs*q, 1-yp+0.01) for q in range(1000)]
    for action, side, quantity, ybs, nbs, price in options:
        new_balance = 10 - quantity*price
        new_utility = utility(z, ybs, nbs, new_balance)
        # print(action, side, quantity, '@', price, '->', new_utility)
        if new_utility > best_utility:
            best_utility, best_side, best_quantity, best_price, best_action = new_utility, side, quantity, price, action
    if best_quantity > 0:
        print('%20s %4s @ %+.2f %6d %3s -> %6.4f' % (contract, best_action, best_price, best_quantity, best_side, best_utility))

#xs, ys = plot_pdf(z)
#pyplot.plot(xs, ys)
#pyplot.grid(True)
#pyplot.show()
