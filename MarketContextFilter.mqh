//+------------------------------------------------------------------+
//|                                          MarketContextFilter.mqh  |
//|                        Advanced Market Context Filtering         |
//+------------------------------------------------------------------+
#property copyright "Market Context Filter"
#property link      ""

//+------------------------------------------------------------------+
//| News impact levels                                               |
//+------------------------------------------------------------------+
enum ENUM_NEWS_IMPACT
{
   NEWS_NONE,
   NEWS_LOW,
   NEWS_MEDIUM,
   NEWS_HIGH
};

//+------------------------------------------------------------------+
//| Market context structure                                         |
//+------------------------------------------------------------------+
struct MarketContext
{
   bool isValidSession;         // Trading session active
   bool isNewsTime;            // High-impact news nearby
   bool isEndOfWeek;           // Friday afternoon
   bool isHoliday;             // Market holiday
   bool isLowLiquidity;        // Low liquidity period
   double correlationScore;     // Correlation with related pairs
   double marketSentiment;      // Overall market sentiment
   bool isMajorLevel;          // Near major S/R level
};

//+------------------------------------------------------------------+
//| Check if current time is valid for trading                      |
//+------------------------------------------------------------------+
bool IsValidTradingTime()
{
   datetime current = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(current, dt);
   
   // Skip weekends
   if(dt.day_of_week == 0 || dt.day_of_week == 6)
      return false;
   
   // Skip Friday after 20:00 GMT (end of week)
   if(dt.day_of_week == 5 && dt.hour >= 20)
      return false;
   
   // Skip Monday before 01:00 GMT (market opening)
   if(dt.day_of_week == 1 && dt.hour < 1)
      return false;
   
   // Skip low liquidity hours (22:00 - 01:00 GMT)
   if(dt.hour >= 22 || dt.hour < 1)
      return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Calculate correlation with related pairs                         |
//+------------------------------------------------------------------+
double CalculateMarketCorrelation(string main_symbol)
{
   string correlated_pairs[];
   
   // Define correlated pairs based on main symbol
   if(StringFind(main_symbol, "EUR") >= 0)
   {
      ArrayResize(correlated_pairs, 3);
      correlated_pairs[0] = "EURUSD";
      correlated_pairs[1] = "EURGBP";
      correlated_pairs[2] = "EURJPY";
   }
   else if(StringFind(main_symbol, "GBP") >= 0)
   {
      ArrayResize(correlated_pairs, 3);
      correlated_pairs[0] = "GBPUSD";
      correlated_pairs[1] = "EURGBP";
      correlated_pairs[2] = "GBPJPY";
   }
   else if(StringFind(main_symbol, "USD") >= 0)
   {
      ArrayResize(correlated_pairs, 3);
      correlated_pairs[0] = "EURUSD";
      correlated_pairs[1] = "GBPUSD";
      correlated_pairs[2] = "USDJPY";
   }
   else
   {
      return 0.5; // Default neutral correlation
   }
   
   // Calculate correlation score
   double correlation_sum = 0;
   int valid_pairs = 0;
   
   for(int i = 0; i < ArraySize(correlated_pairs); i++)
   {
      if(SymbolSelect(correlated_pairs[i], true))
      {
         // Get price changes
         double main_change = (iClose(main_symbol, _Period, 0) - iClose(main_symbol, _Period, 10)) / iClose(main_symbol, _Period, 10);
         double pair_change = (iClose(correlated_pairs[i], _Period, 0) - iClose(correlated_pairs[i], _Period, 10)) / iClose(correlated_pairs[i], _Period, 10);
         
         // Simple correlation check
         if((main_change > 0 && pair_change > 0) || (main_change < 0 && pair_change < 0))
            correlation_sum += 1.0;
         else
            correlation_sum += 0.0;
         
         valid_pairs++;
      }
   }
   
   return valid_pairs > 0 ? correlation_sum / valid_pairs : 0.5;
}

//+------------------------------------------------------------------+
//| Check if near major support/resistance level                    |
//+------------------------------------------------------------------+
bool IsNearMajorLevel(string symbol, double &nearest_level)
{
   double current_price = SymbolInfoDouble(symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   
   // Calculate daily pivot points
   double yesterday_high = iHigh(symbol, PERIOD_D1, 1);
   double yesterday_low = iLow(symbol, PERIOD_D1, 1);
   double yesterday_close = iClose(symbol, PERIOD_D1, 1);
   
   double pivot = (yesterday_high + yesterday_low + yesterday_close) / 3;
   double r1 = 2 * pivot - yesterday_low;
   double s1 = 2 * pivot - yesterday_high;
   double r2 = pivot + (yesterday_high - yesterday_low);
   double s2 = pivot - (yesterday_high - yesterday_low);
   
   // Check weekly levels
   double week_high = iHigh(symbol, PERIOD_W1, 1);
   double week_low = iLow(symbol, PERIOD_W1, 1);
   
   // Round number levels
   double round_level = MathRound(current_price * 100) / 100;
   
   // Find nearest level
   double levels[];
   ArrayResize(levels, 8);
   levels[0] = pivot;
   levels[1] = r1;
   levels[2] = s1;
   levels[3] = r2;
   levels[4] = s2;
   levels[5] = week_high;
   levels[6] = week_low;
   levels[7] = round_level;
   
   nearest_level = 0;
   double min_distance = DBL_MAX;
   
   for(int i = 0; i < ArraySize(levels); i++)
   {
      double distance = MathAbs(current_price - levels[i]);
      if(distance < min_distance)
      {
         min_distance = distance;
         nearest_level = levels[i];
      }
   }
   
   // Check if within 20 pips of major level
   return (min_distance < 20 * point);
}

//+------------------------------------------------------------------+
//| Calculate market sentiment based on multiple indicators          |
//+------------------------------------------------------------------+
double CalculateMarketSentiment(string symbol)
{
   double sentiment = 0;
   
   // Fear & Greed indicators
   // 1. RSI across multiple timeframes
   double rsi_m15 = iRSI(symbol, PERIOD_M15, 14, PRICE_CLOSE);
   double rsi_h1 = iRSI(symbol, PERIOD_H1, 14, PRICE_CLOSE);
   double rsi_h4 = iRSI(symbol, PERIOD_H4, 14, PRICE_CLOSE);
   
   int rsi_handle_m15 = iRSI(symbol, PERIOD_M15, 14, PRICE_CLOSE);
   int rsi_handle_h1 = iRSI(symbol, PERIOD_H1, 14, PRICE_CLOSE);
   int rsi_handle_h4 = iRSI(symbol, PERIOD_H4, 14, PRICE_CLOSE);
   
   double rsi_val_m15[], rsi_val_h1[], rsi_val_h4[];
   ArraySetAsSeries(rsi_val_m15, true);
   ArraySetAsSeries(rsi_val_h1, true);
   ArraySetAsSeries(rsi_val_h4, true);
   
   CopyBuffer(rsi_handle_m15, 0, 0, 1, rsi_val_m15);
   CopyBuffer(rsi_handle_h1, 0, 0, 1, rsi_val_h1);
   CopyBuffer(rsi_handle_h4, 0, 0, 1, rsi_val_h4);
   
   // RSI sentiment (normalized -1 to 1)
   double rsi_sentiment = ((rsi_val_m15[0] + rsi_val_h1[0] + rsi_val_h4[0]) / 3 - 50) / 50;
   sentiment += rsi_sentiment * 0.3;
   
   // 2. Price position relative to moving averages
   double ma_20[], ma_50[], ma_200[];
   ArraySetAsSeries(ma_20, true);
   ArraySetAsSeries(ma_50, true);
   ArraySetAsSeries(ma_200, true);
   
   int ma20_handle = iMA(symbol, _Period, 20, 0, MODE_EMA, PRICE_CLOSE);
   int ma50_handle = iMA(symbol, _Period, 50, 0, MODE_EMA, PRICE_CLOSE);
   int ma200_handle = iMA(symbol, _Period, 200, 0, MODE_EMA, PRICE_CLOSE);
   
   CopyBuffer(ma20_handle, 0, 0, 1, ma_20);
   CopyBuffer(ma50_handle, 0, 0, 1, ma_50);
   CopyBuffer(ma200_handle, 0, 0, 1, ma_200);
   
   double current_price = SymbolInfoDouble(symbol, SYMBOL_BID);
   
   double ma_score = 0;
   if(current_price > ma_20[0]) ma_score += 0.33;
   if(current_price > ma_50[0]) ma_score += 0.33;
   if(current_price > ma_200[0]) ma_score += 0.34;
   
   sentiment += (ma_score - 0.5) * 2 * 0.3; // Normalize to -1 to 1
   
   // 3. Momentum
   double momentum[];
   ArraySetAsSeries(momentum, true);
   int momentum_handle = iMomentum(symbol, _Period, 14, PRICE_CLOSE);
   CopyBuffer(momentum_handle, 0, 0, 1, momentum);
   
   double momentum_sentiment = (momentum[0] - 100) / 10; // Normalize around 100
   sentiment += momentum_sentiment * 0.2;
   
   // 4. Volume pressure
   long volume[];
   ArraySetAsSeries(volume, true);
   CopyTickVolume(symbol, _Period, 0, 20, volume);
   
   double avg_volume = 0;
   for(int i = 0; i < 20; i++)
      avg_volume += volume[i];
   avg_volume /= 20;
   
   double volume_pressure = (volume[0] - avg_volume) / avg_volume;
   sentiment += volume_pressure * 0.2;
   
   // Normalize to -1 to 1 range
   return MathMax(-1, MathMin(1, sentiment));
}

//+------------------------------------------------------------------+
//| Get comprehensive market context                                 |
//+------------------------------------------------------------------+
void GetMarketContext(string symbol, MarketContext &context)
{
   // Session validity
   context.isValidSession = IsValidTradingTime();
   
   // Check for end of week
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   context.isEndOfWeek = (dt.day_of_week == 5 && dt.hour >= 16);
   
   // Low liquidity check
   double spread = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
   double avg_spread = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE) * 10; // Approximate average
   context.isLowLiquidity = (spread > avg_spread * 2);
   
   // Correlation check
   context.correlationScore = CalculateMarketCorrelation(symbol);
   
   // Market sentiment
   context.marketSentiment = CalculateMarketSentiment(symbol);
   
   // Major level check
   double nearest_level;
   context.isMajorLevel = IsNearMajorLevel(symbol, nearest_level);
   
   // News check (simplified - would need news calendar API in production)
   context.isNewsTime = false; // Placeholder
   
   // Holiday check (simplified)
   context.isHoliday = false; // Placeholder
}

//+------------------------------------------------------------------+
//| Filter trades based on market context                           |
//+------------------------------------------------------------------+
bool ShouldTakeTradeBasedOnContext(const MarketContext &context, double base_score)
{
   // Hard filters - reject trade
   if(!context.isValidSession)
      return false;
   
   if(context.isEndOfWeek)
      return false;
   
   if(context.isNewsTime)
      return false;
   
   if(context.isLowLiquidity)
      return false;
   
   // Soft filters - require higher score
   double required_score = 60; // Base requirement
   
   // Adjust based on market sentiment
   if(MathAbs(context.marketSentiment) < 0.2) // Neutral market
      required_score += 10;
   
   // Adjust based on correlation
   if(context.correlationScore < 0.3) // Low correlation
      required_score += 10;
   
   // Bonus for major levels
   if(context.isMajorLevel)
      required_score -= 15;
   
   return base_score >= required_score;
}

//+------------------------------------------------------------------+
//| Calculate dynamic stop loss based on market conditions           |
//+------------------------------------------------------------------+
double CalculateDynamicStopLoss(string symbol, double entry_price, int direction)
{
   double atr[];
   ArraySetAsSeries(atr, true);
   int atr_handle = iATR(symbol, _Period, 14);
   CopyBuffer(atr_handle, 0, 0, 5, atr);
   
   // Base stop loss = 2 ATR
   double base_sl = atr[0] * 2;
   
   // Adjust for market conditions
   ENUM_MARKET_REGIME regime = GetMarketRegime(symbol, _Period);
   
   switch(regime)
   {
      case REGIME_VOLATILE:
         base_sl *= 1.5; // Wider stop in volatile markets
         break;
      case REGIME_QUIET:
         base_sl *= 0.8; // Tighter stop in quiet markets
         break;
   }
   
   // Find recent swing points for stop placement
   double swing_point = 0;
   
   if(direction > 0) // Long trade
   {
      // Find recent swing low
      double lowest = DBL_MAX;
      for(int i = 5; i < 20; i++)
      {
         double low = iLow(symbol, _Period, i);
         if(low < lowest)
         {
            // Check if it's a swing low
            if(iLow(symbol, _Period, i+1) > low && iLow(symbol, _Period, i-1) > low)
               lowest = low;
         }
      }
      swing_point = lowest;
      
      // Use the more conservative stop
      double atr_stop = entry_price - base_sl;
      return MathMin(swing_point - 5 * SymbolInfoDouble(symbol, SYMBOL_POINT), atr_stop);
   }
   else // Short trade
   {
      // Find recent swing high
      double highest = 0;
      for(int i = 5; i < 20; i++)
      {
         double high = iHigh(symbol, _Period, i);
         if(high > highest)
         {
            // Check if it's a swing high
            if(iHigh(symbol, _Period, i+1) < high && iHigh(symbol, _Period, i-1) < high)
               highest = high;
         }
      }
      swing_point = highest;
      
      // Use the more conservative stop
      double atr_stop = entry_price + base_sl;
      return MathMax(swing_point + 5 * SymbolInfoDouble(symbol, SYMBOL_POINT), atr_stop);
   }
}