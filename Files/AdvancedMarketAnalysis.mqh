//+------------------------------------------------------------------+
//|                                          AdvancedMarketAnalysis.mqh|
//|                     Advanced Market Analysis and Optimization     |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Advanced Market Analysis"
#property link      ""
#property version   "1.00"

//--- Market microstructure data
struct MarketMicrostructure
{
    double bidAskSpread;
    double spreadPercentage;
    double orderFlowImbalance;
    double buyVolume;
    double sellVolume;
    double largeOrdersRatio;
    double liquidityScore;
    double executionRisk;
    double avgTradeSize;
    double blockTradesCount;
    double deltaVolume;       // Buy volume - Sell volume
    double cumulativeDelta;
    double vpin;             // Volume-synchronized Probability of Informed Trading
    double toxicity;         // Market toxicity indicator
};

//--- Market regime data
struct MarketRegime
{
    enum REGIME_TYPE
    {
        REGIME_TRENDING_UP,
        REGIME_TRENDING_DOWN,
        REGIME_RANGING,
        REGIME_VOLATILE,
        REGIME_QUIET,
        REGIME_BREAKOUT,
        REGIME_REVERSAL
    };

    REGIME_TYPE currentRegime;
    REGIME_TYPE previousRegime;
    double regimeStrength;
    double transitionProbability;
    datetime regimeStartTime;
    int regimeDuration;
    double avgVolatilityInRegime;
    double successRateInRegime;
};

//--- Correlation data
struct CorrelationData
{
    string symbol1;
    string symbol2;
    double correlation;
    double cointegration;
    double beta;
    double spreadMean;
    double spreadStdDev;
    double halfLife;
    bool isTradeable;
    double zScore;
};

//--- Sentiment indicators
struct MarketSentiment
{
    double putCallRatio;
    double vix;              // Fear index
    double advanceDecline;   // Market breadth
    double highLowRatio;     // New highs vs new lows
    double upDownVolume;     // Volume in advancing vs declining stocks
    double smartMoney;       // Large trader positioning
    double retailSentiment;  // Retail positioning (fade indicator)
    double optionSkew;       // Put vs call implied volatility
    double termStructure;    // Short vs long term IV
};

//--- Order flow analysis
struct OrderFlowAnalysis
{
    double poc;              // Point of Control (highest volume price)
    double vah;              // Value Area High
    double val;              // Value Area Low
    double delta;            // Cumulative delta
    double absorption;       // How well market absorbs selling/buying
    double initiative;       // Aggressive buyers vs sellers
    double responsive;       // Passive buyers vs sellers
    double imbalances[];     // Price levels with order imbalances
    double liquidityVoids[]; // Areas with no trading
};

//--- Kelly Criterion position sizing
struct KellyPosition
{
    double winProbability;
    double avgWin;
    double avgLoss;
    double kellyPercentage;
    double adjustedKelly;    // With safety factor
    double maxDrawdownRisk;
    double optimalLeverage;
    double sharpeRatio;
    double sortinoRatio;
    double calmarRatio;
};

//+------------------------------------------------------------------+
//| Calculate market microstructure                                  |
//+------------------------------------------------------------------+
void CalculateMarketMicrostructure(string symbol, MarketMicrostructure &micro)
{
// Get tick data
    MqlTick ticks[];
    int copied = CopyTicks(symbol, ticks, COPY_TICKS_ALL, 0, 1000);

    if(copied <= 0) return;

// Calculate bid-ask spread
    double totalSpread = 0;
    double buyVol = 0, sellVol = 0;
    double largeOrders = 0;

    for(int i = 0; i < copied; i++)
    {
        double spread = ticks[i].ask - ticks[i].bid;
        totalSpread += spread;

        // Classify volume by aggressor side
        if(ticks[i].flags & TICK_FLAG_BUY)
            buyVol += ticks[i].volume;
        else if(ticks[i].flags & TICK_FLAG_SELL)
            sellVol += ticks[i].volume;

        // Track large orders
        if(ticks[i].volume > micro.avgTradeSize * 3)
            largeOrders++;
    }

    micro.bidAskSpread = totalSpread / copied;
    micro.spreadPercentage = micro.bidAskSpread / SymbolInfoDouble(symbol, SYMBOL_BID) * 100;
    micro.buyVolume = buyVol;
    micro.sellVolume = sellVol;
    micro.orderFlowImbalance = (buyVol - sellVol) / (buyVol + sellVol + 1);
    micro.deltaVolume = buyVol - sellVol;
    micro.largeOrdersRatio = largeOrders / copied;

// Calculate VPIN (Volume-synchronized Probability of Informed Trading)
    CalculateVPIN(ticks, copied, micro.vpin);

// Liquidity score based on spread and volume
    double avgVolume = (buyVol + sellVol) / copied;
    micro.liquidityScore = avgVolume / (micro.spreadPercentage + 0.01);

// Execution risk
    micro.executionRisk = micro.spreadPercentage * (1 + micro.largeOrdersRatio);

// Market toxicity (adverse selection risk)
    micro.toxicity = CalculateToxicity(micro);
}

//+------------------------------------------------------------------+
//| Calculate VPIN indicator                                         |
//+------------------------------------------------------------------+
void CalculateVPIN(const MqlTick &ticks[], int count, double &vpin)
{
// Simplified VPIN calculation
    double volumeBuckets[];
    ArrayResize(volumeBuckets, 50);

    double bucketSize = 0;
    for(int i = 0; i < count; i++)
        bucketSize += ticks[i].volume;
    bucketSize /= 50;

    int currentBucket = 0;
    double currentVolume = 0;
    double buyVolume = 0, sellVolume = 0;

    for(int i = 0; i < count && currentBucket < 50; i++)
    {
        currentVolume += ticks[i].volume;

        if(ticks[i].flags & TICK_FLAG_BUY)
            buyVolume += ticks[i].volume;
        else
            sellVolume += ticks[i].volume;

        if(currentVolume >= bucketSize)
        {
            volumeBuckets[currentBucket] = MathAbs(buyVolume - sellVolume) / (buyVolume + sellVolume + 1);
            currentBucket++;
            currentVolume = 0;
            buyVolume = 0;
            sellVolume = 0;
        }
    }

// Calculate average VPIN
    vpin = 0;
    for(int i = 0; i < currentBucket; i++)
        vpin += volumeBuckets[i];
    vpin /= currentBucket;
}

//+------------------------------------------------------------------+
//| Calculate market toxicity                                        |
//+------------------------------------------------------------------+
double CalculateToxicity(const MarketMicrostructure &micro)
{
// Toxicity increases with:
// - High order flow imbalance
// - Wide spreads
// - Low liquidity
// - High VPIN

    double toxicity = 0;

// Order flow component (0-30)
    toxicity += MathAbs(micro.orderFlowImbalance) * 30;

// Spread component (0-25)
    toxicity += MathMin(micro.spreadPercentage * 10, 25);

// Liquidity component (0-25)
    toxicity += MathMax(0, 25 - micro.liquidityScore / 10);

// VPIN component (0-20)
    toxicity += micro.vpin * 20;

    return MathMin(toxicity, 100);
}

//+------------------------------------------------------------------+
//| Detect market regime                                             |
//+------------------------------------------------------------------+
void DetectMarketRegime(string symbol, MarketRegime &regime)
{
    double close[];
    double high[], low[];
    ArraySetAsSeries(close, true);
    ArraySetAsSeries(high, true);
    ArraySetAsSeries(low, true);

    CopyClose(symbol, PERIOD_CURRENT, 0, 100, close);
    CopyHigh(symbol, PERIOD_CURRENT, 0, 100, high);
    CopyLow(symbol, PERIOD_CURRENT, 0, 100, low);

// Calculate various metrics
    double atr = CalculateATR(high, low, close, 14);
    double adx = CalculateADX(high, low, close, 14);
    double volatility = CalculateVolatility(close, 20);
    double trendStrength = CalculateTrendStrength(close, 50);
    double rangeRatio = CalculateRangeRatio(high, low, 20);

// Determine regime
    MarketRegime::REGIME_TYPE newRegime;

    if(adx > 30 && trendStrength > 0.7)
    {
        if(close[0] > close[20])
            newRegime = MarketRegime::REGIME_TRENDING_UP;
        else
            newRegime = MarketRegime::REGIME_TRENDING_DOWN;
    }
    else if(adx < 20 && rangeRatio < 1.5)
    {
        newRegime = MarketRegime::REGIME_RANGING;
    }
    else if(volatility > CalculateVolatility(close, 50) * 1.5)
    {
        newRegime = MarketRegime::REGIME_VOLATILE;
    }
    else if(volatility < CalculateVolatility(close, 50) * 0.5)
    {
        newRegime = MarketRegime::REGIME_QUIET;
    }
    else if(IsBreakoutPattern(high, low, close))
    {
        newRegime = MarketRegime::REGIME_BREAKOUT;
    }
    else if(IsReversalPattern(high, low, close))
    {
        newRegime = MarketRegime::REGIME_REVERSAL;
    }
    else
    {
        newRegime = MarketRegime::REGIME_RANGING;
    }

// Update regime info
    if(newRegime != regime.currentRegime)
    {
        regime.previousRegime = regime.currentRegime;
        regime.currentRegime = newRegime;
        regime.regimeStartTime = TimeCurrent();
        regime.regimeDuration = 0;
    }
    else
    {
        regime.regimeDuration++;
    }

// Calculate regime strength
    regime.regimeStrength = CalculateRegimeStrength(regime.currentRegime, adx, volatility, trendStrength);

// Calculate transition probability
    regime.transitionProbability = CalculateTransitionProbability(regime);

// Track performance in regime
    regime.avgVolatilityInRegime = volatility;
}

//+------------------------------------------------------------------+
//| Calculate ATR                                                    |
//+------------------------------------------------------------------+
double CalculateATR(const double &high[], const double &low[], const double &close[], int period)
{
    double atr[];
    ArraySetAsSeries(atr, true);
    int handle = iATR(_Symbol, _Period, period);
    CopyBuffer(handle, 0, 0, 1, atr);
    return atr[0];
}

//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Calculate ADX                                                    |
//+------------------------------------------------------------------+
double CalculateADX(const double &high[], const double &low[], const double &close[], int period)
{
    double adx[];
    ArraySetAsSeries(adx, true);
    int handle = iADX(_Symbol, _Period, period);
    CopyBuffer(handle, 0, 0, 1, adx);
    return adx[0];
}

//+------------------------------------------------------------------+
//| Calculate volatility                                             |
//+------------------------------------------------------------------+
double CalculateVolatility(const double &close[], int period)
{
    double returns[];
    ArrayResize(returns, period - 1);

    for(int i = 0; i < period - 1; i++)
        returns[i] = MathLog(close[i] / close[i + 1]);

    double mean = 0;
    for(int i = 0; i < period - 1; i++)
        mean += returns[i];
    mean /= (period - 1);

    double variance = 0;
    for(int i = 0; i < period - 1; i++)
        variance += MathPow(returns[i] - mean, 2);
    variance /= (period - 1);

    return MathSqrt(variance * 252);  // Annualized volatility
}

//+------------------------------------------------------------------+
//| Calculate trend strength                                         |
//+------------------------------------------------------------------+
double CalculateTrendStrength(const double &close[], int period)
{
// Linear regression R-squared
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;

    for(int i = 0; i < period; i++)
    {
        sumX += i;
        sumY += close[i];
        sumXY += i * close[i];
        sumX2 += i * i;
        sumY2 += close[i] * close[i];
    }

    double n = period;
    double r = (n * sumXY - sumX * sumY) /
               MathSqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    return r * r;  // R-squared
}

//+------------------------------------------------------------------+
//| Calculate range ratio                                            |
//+------------------------------------------------------------------+
double CalculateRangeRatio(const double &high[], const double &low[], int period)
{
    double maxHigh = high[0];
    double minLow = low[0];

    for(int i = 1; i < period; i++)
    {
        if(high[i] > maxHigh) maxHigh = high[i];
        if(low[i] < minLow) minLow = low[i];
    }

    double range = maxHigh - minLow;
    double avgRange = 0;

    for(int i = 0; i < period; i++)
        avgRange += high[i] - low[i];
    avgRange /= period;

    return range / (avgRange * period);
}

//+------------------------------------------------------------------+
//| Check for breakout pattern                                       |
//+------------------------------------------------------------------+
bool IsBreakoutPattern(const double &high[], const double &low[], const double &close[])
{
// Check for range breakout
    double rangeHigh = high[1];
    double rangeLow = low[1];

    for(int i = 2; i < 20; i++)
    {
        if(high[i] > rangeHigh) rangeHigh = high[i];
        if(low[i] < rangeLow) rangeLow = low[i];
    }

    double range = rangeHigh - rangeLow;

// Breakout conditions
    if(close[0] > rangeHigh + range * 0.1 || close[0] < rangeLow - range * 0.1)
        return true;

    return false;
}

//+------------------------------------------------------------------+
//| Check for reversal pattern                                       |
//+------------------------------------------------------------------+
bool IsReversalPattern(const double &high[], const double &low[], const double &close[])
{
// Check for key reversal patterns

// Double top/bottom
    double tolerance = (high[0] - low[0]) * 0.1;

    for(int i = 10; i < 30; i++)
    {
        if(MathAbs(high[0] - high[i]) < tolerance && close[0] < high[0] - tolerance)
            return true;  // Double top

        if(MathAbs(low[0] - low[i]) < tolerance && close[0] > low[0] + tolerance)
            return true;  // Double bottom
    }

// Exhaustion gap
    if(low[0] > high[1] + tolerance || high[0] < low[1] - tolerance)
        return true;

    return false;
}

//+------------------------------------------------------------------+
//| Calculate regime strength                                        |
//+------------------------------------------------------------------+
double CalculateRegimeStrength(MarketRegime::REGIME_TYPE regime, double adx,
                               double volatility, double trendStrength)
{
    double strength = 0;

    switch(regime)
    {
    case MarketRegime::REGIME_TRENDING_UP:
    case MarketRegime::REGIME_TRENDING_DOWN:
        strength = adx / 100 * 0.5 + trendStrength * 0.5;
        break;

    case MarketRegime::REGIME_RANGING:
        strength = (1 - adx / 100) * 0.5 + (1 - trendStrength) * 0.5;
        break;

    case MarketRegime::REGIME_VOLATILE:
        strength = MathMin(volatility / 0.5, 1.0);
        break;

    case MarketRegime::REGIME_QUIET:
        strength = MathMin(0.1 / volatility, 1.0);
        break;

    default:
        strength = 0.5;
    }

    return strength;
}

//+------------------------------------------------------------------+
//| Calculate transition probability                                 |
//+------------------------------------------------------------------+
double CalculateTransitionProbability(const MarketRegime &regime)
{
// Base probability increases with regime duration
    double baseProbability = 1 - MathExp(-regime.regimeDuration / 20.0);

// Adjust based on regime strength
    double probability = baseProbability * (1 - regime.regimeStrength * 0.5);

    return MathMin(probability, 0.8);
}

//+------------------------------------------------------------------+
//| Calculate correlation between symbols                            |
//+------------------------------------------------------------------+
void CalculateCorrelation(string symbol1, string symbol2, int period, CorrelationData &corr)
{
    double close1[], close2[];
    ArraySetAsSeries(close1, true);
    ArraySetAsSeries(close2, true);

    CopyClose(symbol1, PERIOD_CURRENT, 0, period, close1);
    CopyClose(symbol2, PERIOD_CURRENT, 0, period, close2);

// Calculate returns
    double returns1[], returns2[];
    ArrayResize(returns1, period - 1);
    ArrayResize(returns2, period - 1);

    for(int i = 0; i < period - 1; i++)
    {
        returns1[i] = MathLog(close1[i] / close1[i + 1]);
        returns2[i] = MathLog(close2[i] / close2[i + 1]);
    }

// Calculate correlation
    double mean1 = 0, mean2 = 0;
    for(int i = 0; i < period - 1; i++)
    {
        mean1 += returns1[i];
        mean2 += returns2[i];
    }
    mean1 /= (period - 1);
    mean2 /= (period - 1);

    double cov = 0, var1 = 0, var2 = 0;
    for(int i = 0; i < period - 1; i++)
    {
        cov += (returns1[i] - mean1) * (returns2[i] - mean2);
        var1 += MathPow(returns1[i] - mean1, 2);
        var2 += MathPow(returns2[i] - mean2, 2);
    }

    corr.correlation = cov / MathSqrt(var1 * var2);

// Calculate beta (regression coefficient)
    corr.beta = cov / var1;

// Calculate spread statistics
    double spread[];
    ArrayResize(spread, period);

    for(int i = 0; i < period; i++)
        spread[i] = close1[i] - corr.beta * close2[i];

// Spread mean and std dev
    corr.spreadMean = 0;
    for(int i = 0; i < period; i++)
        corr.spreadMean += spread[i];
    corr.spreadMean /= period;

    corr.spreadStdDev = 0;
    for(int i = 0; i < period; i++)
        corr.spreadStdDev += MathPow(spread[i] - corr.spreadMean, 2);
    corr.spreadStdDev = MathSqrt(corr.spreadStdDev / period);

// Current z-score
    corr.zScore = (spread[0] - corr.spreadMean) / corr.spreadStdDev;

// Calculate half-life (mean reversion speed)
    corr.halfLife = CalculateSpreadHalfLife(spread, period);

// Determine if pair is tradeable
    corr.isTradeable = (MathAbs(corr.correlation) > 0.7 &&
                        corr.halfLife > 0 && corr.halfLife < 30 &&
                        corr.spreadStdDev > 0);

    corr.symbol1 = symbol1;
    corr.symbol2 = symbol2;
}

//+------------------------------------------------------------------+
//| Calculate spread half-life                                       |
//+------------------------------------------------------------------+
double CalculateSpreadHalfLife(const double &spread[], int period)
{
// Ornstein-Uhlenbeck process
    double y[], x[];
    ArrayResize(y, period - 1);
    ArrayResize(x, period - 1);

    for(int i = 0; i < period - 1; i++)
    {
        y[i] = spread[i] - spread[i + 1];
        x[i] = spread[i + 1];
    }

// Linear regression
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for(int i = 0; i < period - 1; i++)
    {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
    }

    double beta = (period * sumXY - sumX * sumY) / (period * sumX2 - sumX * sumX);

    if(beta >= 0) return -1;  // No mean reversion

    return -MathLog(2) / beta;
}

//+------------------------------------------------------------------+
//| Calculate Kelly Criterion position size                          |
//+------------------------------------------------------------------+
void CalculateKellyPosition(double winRate, double avgWin, double avgLoss,
                            double currentBalance, KellyPosition &kelly)
{
    kelly.winProbability = winRate;
    kelly.avgWin = avgWin;
    kelly.avgLoss = avgLoss;

// Basic Kelly formula: f = (p*b - q) / b
// where p = win probability, q = loss probability, b = win/loss ratio
    double b = avgWin / avgLoss;
    double p = winRate;
    double q = 1 - winRate;

    kelly.kellyPercentage = (p * b - q) / b;

// Apply safety factor (typically use 25% of Kelly)
    kelly.adjustedKelly = kelly.kellyPercentage * 0.25;

// Calculate maximum drawdown risk
    double z = 1.96;  // 95% confidence
    double n = 100;   // Sample size
    double stdError = MathSqrt(p * q / n);
    double worstWinRate = p - z * stdError;

    kelly.maxDrawdownRisk = CalculateMaxDrawdown(worstWinRate, b);

// Optimal leverage calculation
    kelly.optimalLeverage = kelly.adjustedKelly * b;

// Risk-adjusted performance metrics
    kelly.sharpeRatio = (kelly.avgWin * kelly.winProbability - kelly.avgLoss * (1 - kelly.winProbability)) /
                        MathSqrt(kelly.avgWin * kelly.avgWin * kelly.winProbability +
                                 kelly.avgLoss * kelly.avgLoss * (1 - kelly.winProbability));

// Sortino ratio (downside deviation)
    double downsideDeviation = kelly.avgLoss * MathSqrt(1 - kelly.winProbability);
    kelly.sortinoRatio = (kelly.avgWin * kelly.winProbability - kelly.avgLoss * (1 - kelly.winProbability)) /
                         downsideDeviation;
}

//+------------------------------------------------------------------+
//| Calculate maximum drawdown                                       |
//+------------------------------------------------------------------+
double CalculateMaxDrawdown(double winRate, double winLossRatio)
{
// Monte Carlo simulation for max drawdown
    int simulations = 1000;
    int trades = 100;
    double maxDD = 0;

    for(int sim = 0; sim < simulations; sim++)
    {
        double equity = 1.0;
        double peak = 1.0;
        double drawdown = 0;

        for(int i = 0; i < trades; i++)
        {
            if(MathRand() / 32767.0 < winRate)
                equity *= (1 + winLossRatio * 0.01);  // 1% risk per trade
            else
                equity *= (1 - 0.01);

            if(equity > peak)
                peak = equity;

            drawdown = (peak - equity) / peak;
            if(drawdown > maxDD)
                maxDD = drawdown;
        }
    }

    return maxDD;
}

//+------------------------------------------------------------------+
//| Calculate market sentiment                                       |
//+------------------------------------------------------------------+
void CalculateMarketSentiment(string symbol, MarketSentiment &sentiment)
{
// This would typically connect to external data sources
// For now, we'll calculate what we can from price action

    double close[], high[], low[], volume[];
    ArraySetAsSeries(close, true);
    ArraySetAsSeries(high, true);
    ArraySetAsSeries(low, true);
    ArraySetAsSeries(volume, true);

    CopyClose(symbol, PERIOD_D1, 0, 20, close);
    CopyHigh(symbol, PERIOD_D1, 0, 20, high);
    CopyLow(symbol, PERIOD_D1, 0, 20, low);
    CopyTickVolume(symbol, PERIOD_D1, 0, 20, volume);

// Advance/Decline approximation
    int advances = 0, declines = 0;
    double upVolume = 0, downVolume = 0;

    for(int i = 0; i < 19; i++)
    {
        if(close[i] > close[i + 1])
        {
            advances++;
            upVolume += volume[i];
        }
        else
        {
            declines++;
            downVolume += volume[i];
        }
    }

    sentiment.advanceDecline = (double)advances / (advances + declines);
    sentiment.upDownVolume = upVolume / (upVolume + downVolume);

// High/Low ratio
    int newHighs = 0, newLows = 0;
    double highest = high[0], lowest = low[0];

    for(int i = 1; i < 20; i++)
    {
        if(high[i] > highest)
        {
            highest = high[i];
            newHighs++;
        }
        if(low[i] < lowest)
        {
            lowest = low[i];
            newLows++;
        }
    }

    sentiment.highLowRatio = (double)newHighs / (newHighs + newLows + 1);

// Simplified VIX calculation (using ATR as proxy)
    double atr = CalculateATR(high, low, close, 14);
    double avgPrice = (close[0] + close[1] + close[2]) / 3;
    sentiment.vix = (atr / avgPrice) * 100 * 16;  // Annualized

// Smart money vs retail (using volume patterns)
    double avgVolume = 0;
    for(int i = 0; i < 20; i++)
        avgVolume += volume[i];
    avgVolume /= 20;

// High volume = smart money, low volume = retail
    sentiment.smartMoney = volume[0] > avgVolume * 1.5 ?
                           (close[0] > close[1] ? 1 : -1) : 0;

    sentiment.retailSentiment = volume[0] < avgVolume * 0.7 ?
                                (close[0] > close[1] ? 1 : -1) : 0;
}

//+------------------------------------------------------------------+
//| Calculate order flow analysis                                    |
//+------------------------------------------------------------------+
void CalculateOrderFlow(string symbol, OrderFlowAnalysis &flow)
{
    MqlTick ticks[];
    int copied = CopyTicks(symbol, ticks, COPY_TICKS_ALL, 0, 5000);

    if(copied <= 0) return;

// Build volume profile
    double priceStep = SymbolInfoDouble(symbol, SYMBOL_POINT) * 10;
    int priceLevels = 100;
    double volumeProfile[];
    double deltaProfile[];
    ArrayResize(volumeProfile, priceLevels);
    ArrayResize(deltaProfile, priceLevels);
    ArrayInitialize(volumeProfile, 0);
    ArrayInitialize(deltaProfile, 0);

    double minPrice = ticks[0].bid;
    double maxPrice = ticks[0].bid;

// Find price range
    for(int i = 0; i < copied; i++)
    {
        if(ticks[i].bid < minPrice) minPrice = ticks[i].bid;
        if(ticks[i].bid > maxPrice) maxPrice = ticks[i].bid;
    }

    double priceRange = maxPrice - minPrice;
    double levelSize = priceRange / priceLevels;

// Build profiles
    for(int i = 0; i < copied; i++)
    {
        int level = (int)((ticks[i].bid - minPrice) / levelSize);
        if(level >= 0 && level < priceLevels)
        {
            volumeProfile[level] += ticks[i].volume;

            if(ticks[i].flags & TICK_FLAG_BUY)
                deltaProfile[level] += ticks[i].volume;
            else if(ticks[i].flags & TICK_FLAG_SELL)
                deltaProfile[level] -= ticks[i].volume;
        }
    }

// Find Point of Control (highest volume)
    double maxVolume = 0;
    int pocLevel = 0;

    for(int i = 0; i < priceLevels; i++)
    {
        if(volumeProfile[i] > maxVolume)
        {
            maxVolume = volumeProfile[i];
            pocLevel = i;
        }
    }

    flow.poc = minPrice + pocLevel * levelSize;

// Calculate Value Area (70% of volume)
    double totalVolume = 0;
    for(int i = 0; i < priceLevels; i++)
        totalVolume += volumeProfile[i];

    double targetVolume = totalVolume * 0.7;
    double vaVolume = volumeProfile[pocLevel];
    int vaHigh = pocLevel, vaLow = pocLevel;

    while(vaVolume < targetVolume && (vaHigh < priceLevels - 1 || vaLow > 0))
    {
        double upVolume = (vaHigh < priceLevels - 1) ? volumeProfile[vaHigh + 1] : 0;
        double downVolume = (vaLow > 0) ? volumeProfile[vaLow - 1] : 0;

        if(upVolume >= downVolume && vaHigh < priceLevels - 1)
        {
            vaHigh++;
            vaVolume += upVolume;
        }
        else if(vaLow > 0)
        {
            vaLow--;
            vaVolume += downVolume;
        }
    }

    flow.vah = minPrice + vaHigh * levelSize;
    flow.val = minPrice + vaLow * levelSize;

// Calculate cumulative delta
    flow.delta = 0;
    for(int i = 0; i < priceLevels; i++)
        flow.delta += deltaProfile[i];

// Find imbalances
    ArrayResize(flow.imbalances, 0);
    for(int i = 0; i < priceLevels; i++)
    {
        if(MathAbs(deltaProfile[i]) > volumeProfile[i] * 0.7)
        {
            int size = ArraySize(flow.imbalances);
            ArrayResize(flow.imbalances, size + 1);
            flow.imbalances[size] = minPrice + i * levelSize;
        }
    }

// Calculate absorption and initiative
    CalculateAbsorptionInitiative(ticks, copied, flow);
}

//+------------------------------------------------------------------+
//| Calculate absorption and initiative                              |
//+------------------------------------------------------------------+
void CalculateAbsorptionInitiative(const MqlTick &ticks[], int count, OrderFlowAnalysis &flow)
{
    double buyInitiative = 0, sellInitiative = 0;
    double buyResponsive = 0, sellResponsive = 0;

    for(int i = 1; i < count; i++)
    {
        double priceDiff = ticks[i].bid - ticks[i-1].bid;

        if(ticks[i].flags & TICK_FLAG_BUY)
        {
            if(priceDiff > 0)
                buyInitiative += ticks[i].volume;  // Aggressive buying
            else
                buyResponsive += ticks[i].volume;  // Passive buying
        }
        else if(ticks[i].flags & TICK_FLAG_SELL)
        {
            if(priceDiff < 0)
                sellInitiative += ticks[i].volume;  // Aggressive selling
            else
                sellResponsive += ticks[i].volume;  // Passive selling
        }
    }

    flow.initiative = (buyInitiative - sellInitiative) / (buyInitiative + sellInitiative + 1);
    flow.responsive = (buyResponsive - sellResponsive) / (buyResponsive + sellResponsive + 1);

// Absorption: market's ability to absorb selling/buying pressure
    flow.absorption = 1 - MathAbs(flow.initiative);  // High absorption = low net initiative
}

//+------------------------------------------------------------------+
//| Execute TWAP order                                               |
//+------------------------------------------------------------------+
bool ExecuteTWAP(string symbol, double volume, int direction, int durationMinutes)
{
// Time-Weighted Average Price execution
    int slices = MathMax(10, (int)(volume / 0.01));  // Number of order slices
    double sliceSize = volume / slices;
    int intervalSeconds = (durationMinutes * 60) / slices;

    datetime startTime = TimeCurrent();
    datetime endTime = startTime + durationMinutes * 60;

    Print("Starting TWAP execution: ", volume, " lots over ", durationMinutes, " minutes");

// This would typically be handled by a timer event
// For demonstration, showing the structure

    for(int i = 0; i < slices; i++)
    {
        datetime executeTime = startTime + i * intervalSeconds;

        // Schedule order for executeTime
        // In real implementation, this would use OnTimer()

        MqlTradeRequest request = {};
        MqlTradeResult result = {};

        request.action = TRADE_ACTION_DEAL;
        request.symbol = symbol;
        request.volume = NormalizeDouble(sliceSize, 2);
        request.type = direction > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
        request.price = direction > 0 ?
                        SymbolInfoDouble(symbol, SYMBOL_ASK) :
                        SymbolInfoDouble(symbol, SYMBOL_BID);
        request.comment = StringFormat("TWAP %d/%d", i+1, slices);

        // Would execute at scheduled time
        // OrderSend(request, result);
    }

    return true;
}

//+------------------------------------------------------------------+
//| Execute VWAP order                                               |
//+------------------------------------------------------------------+
bool ExecuteVWAP(string symbol, double volume, int direction, double participationRate)
{
// Volume-Weighted Average Price execution
// Tracks market volume and participates proportionally

    MqlTick ticks[];
    int copied = CopyTicks(symbol, ticks, COPY_TICKS_ALL, 0, 1000);

    if(copied <= 0) return false;

// Calculate average volume pattern
    double volumePattern[24];  // Hourly volume distribution
    ArrayInitialize(volumePattern, 0);

    for(int i = 0; i < copied; i++)
    {
        MqlDateTime dt;
        TimeToStruct(ticks[i].time, dt);
        volumePattern[dt.hour] += ticks[i].volume;
    }

// Normalize pattern
    double totalVolume = 0;
    for(int i = 0; i < 24; i++)
        totalVolume += volumePattern[i];

    for(int i = 0; i < 24; i++)
        volumePattern[i] /= totalVolume;

// Current hour
    MqlDateTime currentTime;
    TimeToStruct(TimeCurrent(), currentTime);

// Calculate slice size based on historical volume pattern
    double expectedVolumeThisHour = volumePattern[currentTime.hour];
    double sliceSize = volume * expectedVolumeThisHour * participationRate;

    Print("VWAP execution: ", sliceSize, " lots this hour (",
          expectedVolumeThisHour * 100, "% of daily volume)");

// Execute order
    MqlTradeRequest request = {};
    MqlTradeResult result = {};

    request.action = TRADE_ACTION_DEAL;
    request.symbol = symbol;
    request.volume = NormalizeDouble(sliceSize, 2);
    request.type = direction > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
    request.price = direction > 0 ?
                    SymbolInfoDouble(symbol, SYMBOL_ASK) :
                    SymbolInfoDouble(symbol, SYMBOL_BID);
    request.comment = "VWAP";

    return OrderSend(request, result);
}

//+------------------------------------------------------------------+
//| Calculate portfolio heat map                                     |
//+------------------------------------------------------------------+
void CalculatePortfolioHeatMap(string& symbols[], double &correlationMatrix[])
{
    int symbolCount = ArraySize(symbols);
    int matrixSize = symbolCount * symbolCount;
    ArrayResize(correlationMatrix, symbolCount);

    for(int i = 0; i < symbolCount; i++)
    {
        for(int j = 0; j < symbolCount; j++)
        {
            int index = i * symbolCount + j;
            if(i == j)
            {
                correlationMatrix[index] = 1.0;
            }
            else if(j > i)
            {
                CorrelationData corr;
                CalculateCorrelation(symbols[i], symbols[j], 100, corr);
                correlationMatrix[index] = corr.correlation;
                correlationMatrix[j * symbolCount + i] = corr.correlation;
            }
        }
    }

// Print heat map
    Print("Portfolio Correlation Heat Map:");
    for(int i = 0; i < symbolCount; i++)
    {
        string row = symbols[i] + ": ";
        for(int j = 0; j < symbolCount; j++)
        {
            int index = i * symbolCount + j;
            row += StringFormat("%.2f ", correlationMatrix[index]);
        }
        Print(row);
    }
}

//+------------------------------------------------------------------+
//| Calculate optimal portfolio weights (Markowitz)                  |
//+------------------------------------------------------------------+
void CalculateOptimalPortfolio(string &symbols[], double &expectedReturns[],
                               double &weights[], double targetReturn)
{
    int n = ArraySize(symbols);
    ArrayResize(weights, n);

// Get correlation matrix (as 1d array)
    double correlationMatrix[];
    ArrayResize(correlationMatrix, n * n);
    CalculatePortfolioHeatMap(symbols, correlationMatrix);

// Calculate covariance matrix (as 1d array)
    double covMatrix[];
    ArrayResize(covMatrix, n * n);

    for(int i = 0; i < n; i++)
    {
        ArrayResize(covMatrix[i], n);

        for(int j = 0; j < n; j++)
        {
            int idx = i * n + j;
            // Simplified: using correlation as covariance proxy
            covMatrix[idx] = correlationMatrix[idx] * 0.01;  // Assuming 1% vol
        }
    }

// Simplified mean-variance optimization
// In practice, would use quadratic programming solver

// Equal weight as starting point
    for(int i = 0; i < n; i++)
        weights[i] = 1.0 / n;

// Adjust weights based on expected returns and correlations
    double totalReturn = 0;
    for(int i = 0; i < n; i++)
        totalReturn += weights[i] * expectedReturns[i];

// Normalize to target return
    if(totalReturn > 0)
    {
        double scale = targetReturn / totalReturn;
        for(int i = 0; i < n; i++)
            weights[i] *= scale;
    }

// Ensure weights sum to 1
    double sum = 0;
    for(int i = 0; i < n; i++)
        sum += weights[i];

    for(int i = 0; i < n; i++)
        weights[i] /= sum;
}
//+------------------------------------------------------------------+
