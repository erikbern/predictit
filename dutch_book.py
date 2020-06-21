import re
import time
from lib import PredictItScraper

predict_it = PredictItScraper()

result = predict_it.get('/api/Browse/FilteredMarkets/0?itemsPerPage=999&page=1&filterIds=&sort=traded&sortParameter=TODAY')
for market in result['markets']:
    time.sleep(5.0)
    id = market['marketId']
    contract_data = predict_it.get('/api/Market/%d/Contracts' % id)
    contract_data = [c for c in contract_data if c['bestNoPrice'] is not None]
    contract_data.sort(key=lambda k: k['bestNoPrice'])
    cost = 0
    contracts = []
    for i, c in enumerate(contract_data):
        p = c['bestNoPrice']
        cost += p
        payoff = i/cost
        contracts.append(c['contractName'])
        if payoff > 1/0.99:
            print(payoff, cost, '->', i, ':', market['marketName'], contracts)
