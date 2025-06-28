//+------------------------------------------------------------------+
//|                                          EnhancedMLFeatures.mqh   |
//|                    Enhanced Machine Learning Features Library    |
//+------------------------------------------------------------------+
#property copyright "Enhanced ML Features"
#property link      ""

//+------------------------------------------------------------------+
//| Market regime classification                                     |
//+------------------------------------------------------------------+
enum ENUM_MARKET_REGIME
{
   REGIME_TRENDING_UP,
   REGIME_TRENDING_DOWN,
   REGIME_RANGING,
   REGIME_VOLATILE,
   REGIME_QUIET
};

//+------------------------------------------------------------------+
//| Enhanced feature structure for ML                                |
//+------------------------------------------------------------------+
struct EnhancedFeatures
{
   // Time-based features
   double sessionOverlap;        // Major session overlap indicator
   double newsImpact;           // Upcoming news event impact
   double weeklyPattern;        // Weekly seasonal pattern
   
   // Market structure
   double marketRegime;         // Current market regime
   double trendStrength;        // ADX-based trend strength
   double marketBreadth;        // Multiple pairs correlation
   
   // Multi-timeframe confluence
   double htfTrend;            // Higher timeframe trend
   double htfSupport;          // Distance to HTF support
   double htfResistance;       // Distance to HTF resistance
   
   // Volume and liquidity
   double relativeVolume;      // Volume vs average
   double spreadRatio;         // Current spread vs average
   double liquidityScore;      // Market depth indicator
   
   // Technical indicators
   double rsiDivergence;       // RSI divergence score
   double bollingerPosition;   // Position within Bollinger Bands
   double pivotDistance;       // Distance to daily pivot
   
   // Sentiment indicators
   double orderFlowImbalance;  // Buy/sell pressure
   double priceAcceleration;   // Rate of price change
   double volatilityRegime;    // Current vs historical volatility
};

//+------------------------------------------------------------------+
//| Calculate market regime                                          |
//+------------------------------------------------------------------+
ENUM_MARKET_REGIME GetMarketRegime(string symbol, ENUM_TIMEFRAMES timeframe)
{
   // Calculate ADX for trend strength
   double adx[], plus_di[], minus_di[];
   ArraySetAsSeries(adx, true);
   ArraySetAsSeries(plus_di, true);
   ArraySetAsSeries(minus_di, true);
   
   int adx_handle = iADX(symbol, timeframe, 14);
   CopyBuffer(adx_handle, 0, 0, 50, adx);
   CopyBuffer(adx_handle, 1, 0, 50, plus_di);
   CopyBuffer(adx_handle, 2, 0, 50, minus_di);
   
   // Calculate ATR for volatility
   double atr[];
   ArraySetAsSeries(atr, true);
   int atr_handle = iATR(symbol, timeframe, 14);
   CopyBuffer(atr_handle, 0, 0, 50, atr);
   
   // Determine regime
   double avg_adx = 0;
   double avg_atr = 0;
   for(int i = 0; i < 20; i++)
   {
      avg_adx += adx[i];
      avg_atr += atr[i];
   }
   avg_adx /= 20;
   avg_atr /= 20;
   
   // Historical ATR for comparison
   double hist_atr = 0;
   for(int i = 20; i < 50; i++)
      hist_atr += atr[i];
   hist_atr /= 30;
   
   // Classify regime
   if(avg_adx > 25)
   {
      if(plus_di[0] > minus_di[0])
         return REGIME_TRENDING_UP;
      else
         return REGIME_TRENDING_DOWN;
   }
   else if(avg_atr > hist_atr * 1.5)
      return REGIME_VOLATILE;
   else if(avg_atr < hist_atr * 0.7)
      return REGIME_QUIET;
   else
      return REGIME_RANGING;
}

//+------------------------------------------------------------------+
//| Calculate session overlap score                                  |
//+------------------------------------------------------------------+
double GetSessionOverlapScore()
{
   datetime current = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(current, dt);
   
   // Convert to GMT hours
   int hour = dt.hour;
   
   // Define session times (GMT)
   // Sydney: 22:00 - 07:00
   // Tokyo: 00:00 - 09:00
   // London: 08:00 - 17:00
   // New York: 13:00 - 22:00
   
   double score = 0;
   
   // Tokyo-London overlap (08:00-09:00 GMT)
   if(hour >= 8 && hour < 9)
      score = 0.8;
   
   // London-New York overlap (13:00-17:00 GMT)
   else if(hour >= 13 && hour < 17)
      score = 1.0;  // Most liquid time
   
   // Single session activity
   else if((hour >= 0 && hour < 9) ||    // Tokyo
           (hour >= 8 && hour < 17) ||   // London
           (hour >= 13 && hour < 22))    // New York
      score = 0.6;
   
   // Low activity periods
   else
      score = 0.3;
   
   return score;
}

//+------------------------------------------------------------------+
//| Calculate multi-timeframe confluence                             |
//+------------------------------------------------------------------+
void GetMultiTimeframeData(string symbol, EnhancedFeatures &features)
{
   ENUM_TIMEFRAMES current_tf = _Period;
   ENUM_TIMEFRAMES higher_tf;
   
   // Determine higher timeframe
   switch(current_tf)
   {
      case PERIOD_M1:
      case PERIOD_M5:
         higher_tf = PERIOD_M30;
         break;
      case PERIOD_M15:
      case PERIOD_M30:
         higher_tf = PERIOD_H4;
         break;
      case PERIOD_H1:
      case PERIOD_H4:
         higher_tf = PERIOD_D1;
         break;
      default:
         higher_tf = PERIOD_W1;
   }
   
   // Get higher timeframe trend
   double htf_ma_fast[], htf_ma_slow[];
   ArraySetAsSeries(htf_ma_fast, true);
   ArraySetAsSeries(htf_ma_slow, true);
   
   int ma_fast = iMA(symbol, higher_tf, 20, 0, MODE_EMA, PRICE_CLOSE);
   int ma_slow = iMA(symbol, higher_tf, 50, 0, MODE_EMA, PRICE_CLOSE);
   
   CopyBuffer(ma_fast, 0, 0, 3, htf_ma_fast);
   CopyBuffer(ma_slow, 0, 0, 3, htf_ma_slow);
   
   // Calculate trend direction
   if(htf_ma_fast[0] > htf_ma_slow[0])
      features.htfTrend = 1.0;  // Uptrend
   else
      features.htfTrend = -1.0; // Downtrend
   
   // Find HTF support/resistance
   double htf_high[], htf_low[];
   ArraySetAsSeries(htf_high, true);
   ArraySetAsSeries(htf_low, true);
   
   CopyHigh(symbol, higher_tf, 0, 20, htf_high);
   CopyLow(symbol, higher_tf, 0, 20, htf_low);
   
   double current_price = SymbolInfoDouble(symbol, SYMBOL_BID);
   
   // Find nearest support
   double nearest_support = 0;
   for(int i = 1; i < 20; i++)
   {
      if(htf_low[i] < current_price)
      {
         if(nearest_support == 0 || htf_low[i] > nearest_support)
            nearest_support = htf_low[i];
      }
   }
   
   // Find nearest resistance
   double nearest_resistance = DBL_MAX;
   for(int i = 1; i < 20; i++)
   {
      if(htf_high[i] > current_price)
      {
         if(htf_high[i] < nearest_resistance)
            nearest_resistance = htf_high[i];
      }
   }
   
   // Calculate distances as percentages
   features.htfSupport = (current_price - nearest_support) / current_price * 100;
   features.htfResistance = (nearest_resistance - current_price) / current_price * 100;
}

//+------------------------------------------------------------------+
//| Calculate volatility-adjusted position size                      |
//+------------------------------------------------------------------+
double GetVolatilityAdjustedSize(string symbol, double base_risk_percent)
{
   // Get current and historical volatility
   double atr[];
   ArraySetAsSeries(atr, true);
   int atr_handle = iATR(symbol, _Period, 14);
   CopyBuffer(atr_handle, 0, 0, 100, atr);
   
   double current_atr = atr[0];
   double avg_atr = 0;
   
   for(int i = 20; i < 100; i++)
      avg_atr += atr[i];
   avg_atr /= 80;
   
   // Volatility ratio
   double vol_ratio = current_atr / avg_atr;
   
   // Adjust position size inversely to volatility
   double adjusted_risk;
   if(vol_ratio > 1.5)
      adjusted_risk = base_risk_percent * 0.5;  // Half size in high volatility
   else if(vol_ratio > 1.2)
      adjusted_risk = base_risk_percent * 0.75;
   else if(vol_ratio < 0.8)
      adjusted_risk = base_risk_percent * 1.25; // Larger size in low volatility
   else
      adjusted_risk = base_risk_percent;
   
   return adjusted_risk;
}

//+------------------------------------------------------------------+
//| Calculate comprehensive feature vector                           |
//+------------------------------------------------------------------+
void CalculateEnhancedFeatures(string symbol, EnhancedFeatures &features)
{
   // Time-based features
   features.sessionOverlap = GetSessionOverlapScore();
   
   // Market regime
   ENUM_MARKET_REGIME regime = GetMarketRegime(symbol, _Period);
   features.marketRegime = (double)regime / 4.0; // Normalize to 0-1
   
   // Multi-timeframe data
   GetMultiTimeframeData(symbol, features);
   
   // Volume analysis
   long volume[];
   ArraySetAsSeries(volume, true);
   CopyTickVolume(symbol, _Period, 0, 50, volume);
   
   double avg_volume = 0;
   for(int i = 10; i < 50; i++)
      avg_volume += volume[i];
   avg_volume /= 40;
   
   features.relativeVolume = volume[0] / avg_volume;
   
   // Spread analysis
   double current_spread = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   features.spreadRatio = current_spread / SymbolInfoDouble(symbol, SYMBOL_POINT);
   
   // RSI divergence
   double rsi[];
   ArraySetAsSeries(rsi, true);
   int rsi_handle = iRSI(symbol, _Period, 14, PRICE_CLOSE);
   CopyBuffer(rsi_handle, 0, 0, 50, rsi);
   
   // Simple divergence check
   double price_change = (iClose(symbol, _Period, 0) - iClose(symbol, _Period, 10)) / iClose(symbol, _Period, 10);
   double rsi_change = rsi[0] - rsi[10];
   
   if((price_change > 0 && rsi_change < 0) || (price_change < 0 && rsi_change > 0))
      features.rsiDivergence = MathAbs(price_change - rsi_change/100);
   else
      features.rsiDivergence = 0;
   
   // Bollinger Bands position
   double bb_upper[], bb_lower[], bb_middle[];
   ArraySetAsSeries(bb_upper, true);
   ArraySetAsSeries(bb_lower, true);
   ArraySetAsSeries(bb_middle, true);
   
   int bb_handle = iBands(symbol, _Period, 20, 0, 2, PRICE_CLOSE);
   CopyBuffer(bb_handle, 1, 0, 1, bb_upper);
   CopyBuffer(bb_handle, 2, 0, 1, bb_lower);
   CopyBuffer(bb_handle, 0, 0, 1, bb_middle);
   
   double current_price = iClose(symbol, _Period, 0);
   double bb_range = bb_upper[0] - bb_lower[0];
   features.bollingerPosition = (current_price - bb_lower[0]) / bb_range;
   
   // Liquidity score (simplified - based on spread and volume)
   features.liquidityScore = features.relativeVolume / (1 + features.spreadRatio/100);
   
   // Price acceleration
   double price_1 = iClose(symbol, _Period, 1);
   double price_5 = iClose(symbol, _Period, 5);
   double price_10 = iClose(symbol, _Period, 10);
   
   double recent_change = (current_price - price_5) / price_5;
   double prev_change = (price_5 - price_10) / price_10;
   features.priceAcceleration = recent_change - prev_change;
}

//+------------------------------------------------------------------+
//| Score calculation with enhanced features                         |
//+------------------------------------------------------------------+
double CalculateEnhancedScore(const RangeData &range, const EnhancedFeatures &features)
{
   double score = 0;
   
   // Base score from range quality
   score += range.qualityScore * 0.3;
   
   // Market regime bonus
   if(features.marketRegime < 0.75) // Not in strong trend
      score += 20;
   
   // Session overlap bonus
   score += features.sessionOverlap * 15;
   
   // HTF alignment bonus
   if(features.htfTrend > 0 && range.breakoutDirection > 0)
      score += 15;
   else if(features.htfTrend < 0 && range.breakoutDirection < 0)
      score += 15;
   
   // Volume confirmation
   if(features.relativeVolume > 1.2)
      score += 10;
   
   // Liquidity bonus
   score += features.liquidityScore * 5;
   
   // Volatility penalty
   if(features.marketRegime == REGIME_VOLATILE)
      score -= 10;
   
   // Distance to HTF levels bonus
   if(features.htfSupport < 2.0 || features.htfResistance < 2.0)
      score += 10; // Near important levels
   
   // RSI divergence bonus
   if(features.rsiDivergence > 0.1)
      score += 15;
   
   return MathMin(score, 100);
}