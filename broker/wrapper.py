

# IBKR http calls will happen here

def sellTicker(ticker:str, count, price = None):
    if price:
        print(f"Sold {count} {ticker} each for {price}")
    else:
        print(f"Sold {count} {ticker} each for market value!")

def buyTicker(ticker:str, count, price = None):
    if price:
        print(f"Sold {count} {ticker} each for {price}")
    else:
        print(f"Sold {count} {ticker} each for market value!")

def getCurrentPortfolio():
    pass