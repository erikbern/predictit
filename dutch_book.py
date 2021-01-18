from matplotlib import pyplot
import re
import time
from lib import PredictItScraper

predict_it = PredictItScraper()

result = predict_it.get('/api/Browse/FilteredMarkets/0?itemsPerPage=999&page=1&filterIds=&sort=traded&sortParameter=TODAY')
payoffs_buy = []
payoffs_sell = []
for market in result['markets']:
    time.sleep(5.0)
    id = market['marketId']
    try:
        contract_data = predict_it.get('/api/Market/%d/Contracts' % id)
    except:
        print('failed fetching data for', market['marketName'])
        continue
    contract_data = [c for c in contract_data if c['bestNoPrice'] is not None and c['bestYesPrice'] is not None]
    contract_data.sort(key=lambda k: k['bestNoPrice'])
    price_buy = 0
    price_sell = 0
    contracts = []
    for i, c in enumerate(contract_data):
        price_buy += c['bestNoPrice']
        price_sell += 1 - c['bestYesPrice']
        payoffs_buy.append(i / price_buy)
        payoffs_sell.append(i / price_sell)
        contracts.append(c['contractName'])
        if i > 0:
            price = price_buy / i
            if price < 0.99:
                print(price, price_buy, '->', i, ':', market['marketName'], contracts)

    pyplot.clf()
    pyplot.plot(sorted(payoffs_buy), label='buy')
    pyplot.plot(sorted(payoffs_sell), label='sell')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.xlabel('Contract rank')
    pyplot.ylabel('Guaranteed ROI')
    pyplot.savefig('prices.png')
