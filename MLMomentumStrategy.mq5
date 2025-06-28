//+------------------------------------------------------------------+
//|                                          MLMomentumStrategy.mq5   |
//|                    Machine Learning Momentum Trading Strategy    |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "ML Momentum Strategy"
#property link      ""
#property version   "1.00"
#property description "Advanced momentum strategy with ML optimization and pattern recognition"

#include "EnhancedMLFeatures.mqh"
#include "MarketContextFilter.mqh"
#include "AdvancedRiskManagement.mqh"

//--- Input parameters
input group "Strategy Settings"
input int      InpMomentumPeriod = 20;           // Momentum calculation period
input int      InpLookbackPeriod = 100;          // Pattern lookback period
input double   InpMinMomentumScore = 65;         // Minimum momentum score to trade
input int      InpPatternHistory = 1000;         // Bars to analyze for patterns

input group "Machine Learning"
input bool     InpEnableML = true;               // Enable machine learning
input int      InpMLUpdateFreq = 50;             // Update ML model every N trades
input double   InpLearningRate = 0.15;           // ML learning rate
input string   InpModelFile = "MLMomentum.bin";  // Model save file

input group "Risk Management"
input double   InpRiskPerTrade = 1.0;            // Risk per trade (%)
input double   InpMaxDailyRisk = 3.0;            // Maximum daily risk (%)
input int      InpMaxPositions = 3;              // Maximum open positions
input bool     InpUseATRStop = true;             // Use ATR-based stops
input double   InpATRMultiplier = 2.0;           // ATR multiplier for stops

//--- Momentum patterns
enum ENUM_MOMENTUM_PATTERN
{
   PATTERN_NONE,
   PATTERN_BULLISH_DIVERGENCE,
   PATTERN_BEARISH_DIVERGENCE,
   PATTERN_HIDDEN_BULL_DIV,
   PATTERN_HIDDEN_BEAR_DIV,
   PATTERN_MOMENTUM_BURST,
   PATTERN_MOMENTUM_FADE,
   PATTERN_VOLUME_SURGE,
   PATTERN_VOLATILITY_BREAKOUT
};

//--- Momentum signal structure
struct MomentumSignal
{
   datetime time;
   double price;
   double momentum;
   double strength;
   ENUM_MOMENTUM_PATTERN pattern;
   int direction;        // 1 for bullish, -1 for bearish
   double score;        // ML-calculated score
   double riskReward;   // Expected risk/reward ratio
};

//--- ML Model for momentum
struct MomentumMLModel
{
   double weights[40];          // Feature weights
   double patternWeights[10];   // Pattern-specific weights
   double bias;
   double performance;
   int signalsAnalyzed;
   int successfulSignals;
   double avgReturn;
   datetime lastUpdate;
};

//--- Global variables
MomentumMLModel mlModel;
MomentumSignal currentSignals[];
RiskParameters riskParams;
int signalCount = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("ML Momentum Strategy initialized");
   
   // Initialize risk parameters
   riskParams.maxRiskPerTrade = InpRiskPerTrade;
   riskParams.maxDailyRisk = InpMaxDailyRisk;
   riskParams.maxOpenTrades = InpMaxPositions;
   riskParams.useTrailingStop = true;
   riskParams.trailingStopATR = 1.5;
   
   // Load or initialize ML model
   if(!LoadMomentumModel())
   {
      InitializeMomentumModel();
      Print("New momentum model initialized");
   }
   else
   {
      Print("Momentum model loaded. Performance: ", mlModel.performance);
   }
   
   // Initial pattern analysis if ML enabled
   if(InpEnableML)
   {
      AnalyzeHistoricalPatterns();
   }
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   SaveMomentumModel();
   ObjectsDeleteAll(0, "MOM_");
   
   Print("ML Momentum Strategy stopped");
   Print(GenerateRiskReport());
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new bar
   static datetime lastBarTime = 0;
   datetime currentBarTime = iTime(_Symbol, _Period, 0);
   
   if(currentBarTime == lastBarTime) return;
   lastBarTime = currentBarTime;
   
   // Update trailing stops
   UpdateAllTrailingStops();
   
   // Get market context
   MarketContext context;
   GetMarketContext(_Symbol, context);
   
   if(!context.isValidSession || context.isLowLiquidity)
      return;
   
   // Scan for momentum signals
   ScanMomentumSignals();
   
   // Check for trade opportunities
   if(ArraySize(currentSignals) > 0)
   {
      EvaluateTradingOpportunities(context);
   }
   
   // Update ML model periodically
   if(InpEnableML && signalCount >= InpMLUpdateFreq)
   {
      UpdateMomentumModel();
      signalCount = 0;
   }
}

//+------------------------------------------------------------------+
//| Scan for momentum signals                                        |
//+------------------------------------------------------------------+
void ScanMomentumSignals()
{
   ArrayResize(currentSignals, 0);
   
   // Calculate momentum indicators
   double momentum[], rsi[], macd_main[], macd_signal[], volume[];
   ArraySetAsSeries(momentum, true);
   ArraySetAsSeries(rsi, true);
   ArraySetAsSeries(macd_main, true);
   ArraySetAsSeries(macd_signal, true);
   ArraySetAsSeries(volume, true);
   
   // Get indicator values
   int momentum_handle = iMomentum(_Symbol, _Period, InpMomentumPeriod, PRICE_CLOSE);
   int rsi_handle = iRSI(_Symbol, _Period, 14, PRICE_CLOSE);
   int macd_handle = iMACD(_Symbol, _Period, 12, 26, 9, PRICE_CLOSE);
   
   CopyBuffer(momentum_handle, 0, 0, InpLookbackPeriod, momentum);
   CopyBuffer(rsi_handle, 0, 0, InpLookbackPeriod, rsi);
   CopyBuffer(macd_handle, 0, 0, InpLookbackPeriod, macd_main);
   CopyBuffer(macd_handle, 1, 0, InpLookbackPeriod, macd_signal);
   CopyTickVolume(_Symbol, _Period, 0, InpLookbackPeriod, volume);
   
   // Look for momentum patterns
   for(int i = 1; i < InpLookbackPeriod - 20; i++)
   {
      MomentumSignal signal;
      signal.time = iTime(_Symbol, _Period, i);
      signal.price = iClose(_Symbol, _Period, i);
      signal.momentum = momentum[i];
      
      // Check for various momentum patterns
      ENUM_MOMENTUM_PATTERN pattern = DetectMomentumPattern(i, momentum, rsi, macd_main, macd_signal, volume);
      
      if(pattern != PATTERN_NONE)
      {
         signal.pattern = pattern;
         signal.direction = GetPatternDirection(pattern);
         signal.strength = CalculateMomentumStrength(i, momentum, rsi, volume);
         
         // Calculate ML score if enabled
         if(InpEnableML)
         {
            signal.score = CalculateMLMomentumScore(signal, i);
         }
         else
         {
            signal.score = signal.strength * 100;
         }
         
         // Only keep high-scoring signals
         if(signal.score >= InpMinMomentumScore)
         {
            signal.riskReward = EstimateRiskReward(signal, i);
            
            int size = ArraySize(currentSignals);
            ArrayResize(currentSignals, size + 1);
            currentSignals[size] = signal;
         }
      }
   }
   
   // Sort signals by score
   SortSignalsByScore(currentSignals);
   
   // Display top signals
   DisplayMomentumSignals();
}

//+------------------------------------------------------------------+
//| Detect momentum patterns                                         |
//+------------------------------------------------------------------+
ENUM_MOMENTUM_PATTERN DetectMomentumPattern(int bar, const double &momentum[], 
                                           const double &rsi[], const double &macd_main[], 
                                           const double &macd_signal[], const long &volume[])
{
   // Price action
   double price_current = iClose(_Symbol, _Period, bar);
   double price_prev = iClose(_Symbol, _Period, bar + 5);
   double price_change = (price_current - price_prev) / price_prev * 100;
   
   // Check for divergences
   if(CheckBullishDivergence(bar, price_current, momentum, rsi))
      return PATTERN_BULLISH_DIVERGENCE;
   
   if(CheckBearishDivergence(bar, price_current, momentum, rsi))
      return PATTERN_BEARISH_DIVERGENCE;
   
   // Check for momentum burst
   if(CheckMomentumBurst(bar, momentum, volume))
      return PATTERN_MOMENTUM_BURST;
   
   // Check for volume surge
   if(CheckVolumeSurge(bar, volume, momentum))
      return PATTERN_VOLUME_SURGE;
   
   // Check for volatility breakout
   if(CheckVolatilityBreakout(bar))
      return PATTERN_VOLATILITY_BREAKOUT;
   
   // Check for hidden divergences
   if(CheckHiddenBullishDivergence(bar, price_current, momentum))
      return PATTERN_HIDDEN_BULL_DIV;
   
   if(CheckHiddenBearishDivergence(bar, price_current, momentum))
      return PATTERN_HIDDEN_BEAR_DIV;
   
   return PATTERN_NONE;
}

//+------------------------------------------------------------------+
//| Check for bullish divergence                                     |
//+------------------------------------------------------------------+
bool CheckBullishDivergence(int bar, double price, const double &momentum[], const double &rsi[])
{
   // Look for lower lows in price but higher lows in indicators
   int lookback = 20;
   
   // Find recent low
   double recentLow = price;
   int recentLowBar = bar;
   
   for(int i = bar + 1; i < bar + lookback && i < ArraySize(momentum); i++)
   {
      double checkPrice = iClose(_Symbol, _Period, i);
      if(checkPrice < recentLow)
      {
         recentLow = checkPrice;
         recentLowBar = i;
      }
   }
   
   // Need significant price difference
   if(recentLowBar == bar || MathAbs(price - recentLow) < 10 * _Point)
      return false;
   
   // Check if current price is lower than recent low (lower low)
   if(price < recentLow)
   {
      // Check if momentum is making higher low
      if(momentum[bar] > momentum[recentLowBar] && rsi[bar] > rsi[recentLowBar])
      {
         // Confirm with price action
         if(iClose(_Symbol, _Period, bar - 1) > iOpen(_Symbol, _Period, bar - 1))
            return true;
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Check for bearish divergence                                     |
//+------------------------------------------------------------------+
bool CheckBearishDivergence(int bar, double price, const double &momentum[], const double &rsi[])
{
   // Look for higher highs in price but lower highs in indicators
   int lookback = 20;
   
   // Find recent high
   double recentHigh = price;
   int recentHighBar = bar;
   
   for(int i = bar + 1; i < bar + lookback && i < ArraySize(momentum); i++)
   {
      double checkPrice = iClose(_Symbol, _Period, i);
      if(checkPrice > recentHigh)
      {
         recentHigh = checkPrice;
         recentHighBar = i;
      }
   }
   
   // Need significant price difference
   if(recentHighBar == bar || MathAbs(price - recentHigh) < 10 * _Point)
      return false;
   
   // Check if current price is higher than recent high (higher high)
   if(price > recentHigh)
   {
      // Check if momentum is making lower high
      if(momentum[bar] < momentum[recentHighBar] && rsi[bar] < rsi[recentHighBar])
      {
         // Confirm with price action
         if(iClose(_Symbol, _Period, bar - 1) < iOpen(_Symbol, _Period, bar - 1))
            return true;
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Check for momentum burst                                         |
//+------------------------------------------------------------------+
bool CheckMomentumBurst(int bar, const double &momentum[], const long &volume[])
{
   // Calculate average momentum
   double avgMomentum = 0;
   double avgVolume = 0;
   
   for(int i = bar + 5; i < bar + 25 && i < ArraySize(momentum); i++)
   {
      avgMomentum += MathAbs(momentum[i] - 100);
      avgVolume += volume[i];
   }
   avgMomentum /= 20;
   avgVolume /= 20;
   
   // Check for momentum burst
   double currentMomentum = MathAbs(momentum[bar] - 100);
   
   if(currentMomentum > avgMomentum * 2.0 && volume[bar] > avgVolume * 1.5)
   {
      // Confirm with consecutive bars
      if(MathAbs(momentum[bar + 1] - 100) > avgMomentum * 1.5)
         return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Check for volume surge                                           |
//+------------------------------------------------------------------+
bool CheckVolumeSurge(int bar, const long &volume[], const double &momentum[])
{
   // Calculate average volume
   double avgVolume = 0;
   for(int i = bar + 5; i < bar + 30 && i < ArraySize(volume); i++)
   {
      avgVolume += volume[i];
   }
   avgVolume /= 25;
   
   // Check for volume surge with momentum confirmation
   if(volume[bar] > avgVolume * 2.5 && volume[bar - 1] > avgVolume * 2.0)
   {
      // Momentum should be strong in same direction
      if(MathAbs(momentum[bar] - 100) > 2.0)
         return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Check for volatility breakout                                    |
//+------------------------------------------------------------------+
bool CheckVolatilityBreakout(int bar)
{
   // Get ATR values
   double atr[];
   ArraySetAsSeries(atr, true);
   int atr_handle = iATR(_Symbol, _Period, 14);
   CopyBuffer(atr_handle, 0, bar, 30, atr);
   
   // Calculate average ATR
   double avgATR = 0;
   for(int i = 10; i < 30; i++)
   {
      avgATR += atr[i];
   }
   avgATR /= 20;
   
   // Check for volatility expansion
   if(atr[0] > avgATR * 1.5 && atr[1] > avgATR * 1.3)
   {
      // Confirm with price movement
      double range = iHigh(_Symbol, _Period, bar) - iLow(_Symbol, _Period, bar);
      double avgRange = 0;
      
      for(int i = bar + 5; i < bar + 20; i++)
      {
         avgRange += iHigh(_Symbol, _Period, i) - iLow(_Symbol, _Period, i);
      }
      avgRange /= 15;
      
      if(range > avgRange * 2.0)
         return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Check for hidden bullish divergence                              |
//+------------------------------------------------------------------+
bool CheckHiddenBullishDivergence(int bar, double price, const double &momentum[])
{
   // In uptrend: higher lows in price, lower lows in momentum
   double ma50[];
   ArraySetAsSeries(ma50, true);
   int ma_handle = iMA(_Symbol, _Period, 50, 0, MODE_EMA, PRICE_CLOSE);
   CopyBuffer(ma_handle, 0, bar, 1, ma50);
   
   // Must be in uptrend
   if(price < ma50[0])
      return false;
   
   // Find recent higher low in price
   for(int i = bar + 10; i < bar + 30 && i < ArraySize(momentum); i++)
   {
      double prevPrice = iClose(_Symbol, _Period, i);
      
      // Is this a higher low?
      if(prevPrice < price && prevPrice > ma50[0])
      {
         // Check if momentum made lower low
         if(momentum[bar] < momentum[i])
            return true;
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Check for hidden bearish divergence                              |
//+------------------------------------------------------------------+
bool CheckHiddenBearishDivergence(int bar, double price, const double &momentum[])
{
   // In downtrend: lower highs in price, higher highs in momentum
   double ma50[];
   ArraySetAsSeries(ma50, true);
   int ma_handle = iMA(_Symbol, _Period, 50, 0, MODE_EMA, PRICE_CLOSE);
   CopyBuffer(ma_handle, 0, bar, 1, ma50);
   
   // Must be in downtrend
   if(price > ma50[0])
      return false;
   
   // Find recent lower high in price
   for(int i = bar + 10; i < bar + 30 && i < ArraySize(momentum); i++)
   {
      double prevPrice = iClose(_Symbol, _Period, i);
      
      // Is this a lower high?
      if(prevPrice > price && prevPrice < ma50[0])
      {
         // Check if momentum made higher high
         if(momentum[bar] > momentum[i])
            return true;
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Get pattern direction                                            |
//+------------------------------------------------------------------+
int GetPatternDirection(ENUM_MOMENTUM_PATTERN pattern)
{
   switch(pattern)
   {
      case PATTERN_BULLISH_DIVERGENCE:
      case PATTERN_HIDDEN_BULL_DIV:
         return 1;
         
      case PATTERN_BEARISH_DIVERGENCE:
      case PATTERN_HIDDEN_BEAR_DIV:
         return -1;
         
      case PATTERN_MOMENTUM_BURST:
      case PATTERN_VOLUME_SURGE:
      case PATTERN_VOLATILITY_BREAKOUT:
         // Determine direction from price action
         double close = iClose(_Symbol, _Period, 0);
         double open = iOpen(_Symbol, _Period, 0);
         return close > open ? 1 : -1;
   }
   
   return 0;
}

//+------------------------------------------------------------------+
//| Calculate momentum strength                                      |
//+------------------------------------------------------------------+
double CalculateMomentumStrength(int bar, const double &momentum[], 
                                const double &rsi[], const long &volume[])
{
   double strength = 0;
   
   // Momentum strength (deviation from 100)
   double momStrength = MathAbs(momentum[bar] - 100);
   strength += NormalizeValue(momStrength, 0, 10) * 30;
   
   // RSI extremes
   if(rsi[bar] > 70 || rsi[bar] < 30)
      strength += 20;
   
   // Volume confirmation
   double avgVolume = 0;
   for(int i = bar + 5; i < bar + 20; i++)
      avgVolume += volume[i];
   avgVolume /= 15;
   
   double volRatio = volume[bar] / avgVolume;
   strength += NormalizeValue(volRatio, 0.5, 2.5) * 20;
   
   // Consistency check (multiple bars showing same direction)
   int consistentBars = 0;
   double direction = momentum[bar] > 100 ? 1 : -1;
   
   for(int i = bar; i < bar + 5 && i < ArraySize(momentum); i++)
   {
      if((momentum[i] > 100 && direction > 0) || (momentum[i] < 100 && direction < 0))
         consistentBars++;
   }
   
   strength += consistentBars * 6;
   
   return MathMin(strength, 100);
}

//+------------------------------------------------------------------+
//| Calculate ML momentum score                                      |
//+------------------------------------------------------------------+
double CalculateMLMomentumScore(const MomentumSignal &signal, int bar)
{
   double score = mlModel.bias;
   
   // Get enhanced features
   EnhancedFeatures features;
   CalculateEnhancedFeatures(_Symbol, features);
   
   // Basic momentum features
   score += mlModel.weights[0] * NormalizeValue(signal.strength, 0, 100);
   score += mlModel.weights[1] * NormalizeValue(signal.momentum, 95, 105);
   
   // Pattern-specific weights
   score += mlModel.patternWeights[signal.pattern];
   
   // Market context features
   score += mlModel.weights[10] * features.sessionOverlap;
   score += mlModel.weights[11] * features.marketRegime;
   score += mlModel.weights[12] * features.htfTrend * signal.direction;
   
   // Technical features
   score += mlModel.weights[15] * features.relativeVolume;
   score += mlModel.weights[16] * features.rsiDivergence;
   score += mlModel.weights[17] * features.bollingerPosition;
   
   // Time-based features
   MqlDateTime dt;
   TimeToStruct(signal.time, dt);
   score += mlModel.weights[20] * NormalizeValue(dt.hour, 0, 23);
   score += mlModel.weights[21] * NormalizeValue(dt.day_of_week, 1, 5);
   
   // Apply sigmoid activation
   return 100 / (1 + MathExp(-score));
}

//+------------------------------------------------------------------+
//| Estimate risk/reward ratio                                       |
//+------------------------------------------------------------------+
double EstimateRiskReward(const MomentumSignal &signal, int bar)
{
   double atr[];
   ArraySetAsSeries(atr, true);
   int atr_handle = iATR(_Symbol, _Period, 14);
   CopyBuffer(atr_handle, 0, bar, 5, atr);
   
   double avgATR = 0;
   for(int i = 0; i < 5; i++)
      avgATR += atr[i];
   avgATR /= 5;
   
   // Base risk/reward on pattern type and strength
   double baseRR = 2.0;
   
   switch(signal.pattern)
   {
      case PATTERN_BULLISH_DIVERGENCE:
      case PATTERN_BEARISH_DIVERGENCE:
         baseRR = 3.0; // Higher RR for divergences
         break;
         
      case PATTERN_MOMENTUM_BURST:
         baseRR = 2.5;
         break;
         
      case PATTERN_VOLUME_SURGE:
         baseRR = 2.2;
         break;
   }
   
   // Adjust based on strength
   baseRR *= (0.5 + signal.strength / 100);
   
   return baseRR;
}

//+------------------------------------------------------------------+
//| Sort signals by score                                            |
//+------------------------------------------------------------------+
void SortSignalsByScore(MomentumSignal &signals[])
{
   int n = ArraySize(signals);
   
   for(int i = 0; i < n - 1; i++)
   {
      for(int j = 0; j < n - i - 1; j++)
      {
         if(signals[j].score < signals[j + 1].score)
         {
            MomentumSignal temp = signals[j];
            signals[j] = signals[j + 1];
            signals[j + 1] = temp;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Display momentum signals on chart                                |
//+------------------------------------------------------------------+
void DisplayMomentumSignals()
{
   ObjectsDeleteAll(0, "MOM_");
   
   int maxSignals = MathMin(ArraySize(currentSignals), 5);
   
   for(int i = 0; i < maxSignals; i++)
   {
      string prefix = "MOM_" + IntegerToString(i) + "_";
      MomentumSignal signal = currentSignals[i];
      
      // Draw arrow
      string arrowName = prefix + "Arrow";
      ObjectCreate(0, arrowName, OBJ_ARROW, 0, signal.time, signal.price);
      
      if(signal.direction > 0)
      {
         ObjectSetInteger(0, arrowName, OBJPROP_ARROWCODE, 233); // Up arrow
         ObjectSetInteger(0, arrowName, OBJPROP_COLOR, clrLime);
      }
      else
      {
         ObjectSetInteger(0, arrowName, OBJPROP_ARROWCODE, 234); // Down arrow
         ObjectSetInteger(0, arrowName, OBJPROP_COLOR, clrRed);
      }
      
      ObjectSetInteger(0, arrowName, OBJPROP_WIDTH, 2);
      
      // Add label with pattern info
      string labelName = prefix + "Label";
      string patternText = GetPatternName(signal.pattern);
      string labelText = StringFormat("%s\nScore: %.1f\nRR: %.1f:1", 
                                     patternText, signal.score, signal.riskReward);
      
      double labelPrice = signal.direction > 0 ? 
                         signal.price - 30 * _Point : 
                         signal.price + 30 * _Point;
      
      ObjectCreate(0, labelName, OBJ_TEXT, 0, signal.time, labelPrice);
      ObjectSetString(0, labelName, OBJPROP_TEXT, labelText);
      ObjectSetInteger(0, labelName, OBJPROP_COLOR, clrWhite);
      ObjectSetInteger(0, labelName, OBJPROP_FONTSIZE, 8);
      ObjectSetString(0, labelName, OBJPROP_FONT, "Arial");
   }
}

//+------------------------------------------------------------------+
//| Get pattern name as string                                       |
//+------------------------------------------------------------------+
string GetPatternName(ENUM_MOMENTUM_PATTERN pattern)
{
   switch(pattern)
   {
      case PATTERN_BULLISH_DIVERGENCE: return "Bull Div";
      case PATTERN_BEARISH_DIVERGENCE: return "Bear Div";
      case PATTERN_HIDDEN_BULL_DIV: return "Hidden Bull";
      case PATTERN_HIDDEN_BEAR_DIV: return "Hidden Bear";
      case PATTERN_MOMENTUM_BURST: return "Mom Burst";
      case PATTERN_VOLUME_SURGE: return "Vol Surge";
      case PATTERN_VOLATILITY_BREAKOUT: return "Vol Break";
      default: return "Unknown";
   }
}

//+------------------------------------------------------------------+
//| Evaluate trading opportunities                                   |
//+------------------------------------------------------------------+
void EvaluateTradingOpportunities(const MarketContext &context)
{
   // Check if we can take new trades
   if(PositionsTotal() >= riskParams.maxOpenTrades)
      return;
   
   // Get the best signal
   if(ArraySize(currentSignals) == 0)
      return;
   
   MomentumSignal bestSignal = currentSignals[0];
   
   // Additional context validation
   if(!ShouldTakeTradeBasedOnContext(context, bestSignal.score))
      return;
   
   // Check if signal is recent enough (within last 5 bars)
   int signalBar = iBarShift(_Symbol, _Period, bestSignal.time);
   if(signalBar > 5)
      return;
   
   // Execute trade
   ExecuteMomentumTrade(bestSignal);
}

//+------------------------------------------------------------------+
//| Execute momentum trade                                           |
//+------------------------------------------------------------------+
void ExecuteMomentumTrade(const MomentumSignal &signal)
{
   double entryPrice = signal.direction > 0 ? 
                      SymbolInfoDouble(_Symbol, SYMBOL_ASK) : 
                      SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // Calculate dynamic stop loss
   double stopLoss = CalculateDynamicStopLoss(_Symbol, entryPrice, signal.direction);
   
   // Calculate take profit based on risk/reward
   double stopDistance = MathAbs(entryPrice - stopLoss);
   double takeProfit = signal.direction > 0 ?
                      entryPrice + stopDistance * signal.riskReward :
                      entryPrice - stopDistance * signal.riskReward;
   
   // Calculate position size
   double lotSize = CalculateDynamicPositionSize(_Symbol, stopDistance, riskParams);
   
   if(lotSize == 0) return;
   
   // Validate trade
   if(!ValidateTradeRisk(_Symbol, entryPrice, stopLoss, takeProfit, lotSize, riskParams))
      return;
   
   // Place trade
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lotSize;
   request.type = signal.direction > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   request.price = entryPrice;
   request.sl = stopLoss;
   request.tp = takeProfit;
   request.comment = "ML_Momentum_" + GetPatternName(signal.pattern);
   request.magic = 54321;
   
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         Print("Momentum trade executed: ", GetPatternName(signal.pattern),
               " ", signal.direction > 0 ? "BUY" : "SELL",
               " Score: ", signal.score);
         
         mlModel.signalsAnalyzed++;
         signalCount++;
      }
   }
}

//+------------------------------------------------------------------+
//| Update all trailing stops                                        |
//+------------------------------------------------------------------+
void UpdateAllTrailingStops()
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      string comment = PositionGetString(POSITION_COMMENT);
      
      if(StringFind(comment, "ML_Momentum") >= 0)
      {
         UpdateTrailingStop(ticket, riskParams);
      }
   }
}

//+------------------------------------------------------------------+
//| Analyze historical patterns for ML training                      |
//+------------------------------------------------------------------+
void AnalyzeHistoricalPatterns()
{
   Print("Analyzing historical momentum patterns...");
   
   int patterns_found = 0;
   double total_return = 0;
   
   for(int i = InpPatternHistory; i > 100; i -= 50)
   {
      MomentumSignal signal;
      
      // Simplified pattern detection for historical analysis
      double momentum[];
      ArraySetAsSeries(momentum, true);
      int momentum_handle = iMomentum(_Symbol, _Period, InpMomentumPeriod, PRICE_CLOSE);
      CopyBuffer(momentum_handle, 0, i, 50, momentum);
      
      // Check if pattern led to profitable move
      if(MathAbs(momentum[0] - 100) > 3.0)
      {
         double entry = iClose(_Symbol, _Period, i);
         double exit = iClose(_Symbol, _Period, i - 20);
         double return_pct = (exit - entry) / entry * 100;
         
         if(momentum[0] > 100 && return_pct > 0)
         {
            patterns_found++;
            total_return += return_pct;
            mlModel.successfulSignals++;
         }
         else if(momentum[0] < 100 && return_pct < 0)
         {
            patterns_found++;
            total_return += MathAbs(return_pct);
            mlModel.successfulSignals++;
         }
         
         mlModel.signalsAnalyzed++;
      }
   }
   
   if(patterns_found > 0)
   {
      mlModel.avgReturn = total_return / patterns_found;
      mlModel.performance = (double)mlModel.successfulSignals / mlModel.signalsAnalyzed * 100;
   }
   
   Print("Historical analysis complete. Patterns found: ", patterns_found);
   Print("Success rate: ", mlModel.performance, "%");
}

//+------------------------------------------------------------------+
//| Update momentum model based on recent trades                     |
//+------------------------------------------------------------------+
void UpdateMomentumModel()
{
   // Check recent trade performance
   if(HistorySelect(mlModel.lastUpdate, TimeCurrent()))
   {
      int deals = HistoryDealsTotal();
      double totalReturn = 0;
      int tradeCount = 0;
      
      for(int i = 0; i < deals; i++)
      {
         ulong ticket = HistoryDealGetTicket(i);
         string comment = HistoryDealGetString(ticket, DEAL_COMMENT);
         
         if(StringFind(comment, "ML_Momentum") >= 0)
         {
            double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
            totalReturn += profit;
            tradeCount++;
            
            if(profit > 0)
               mlModel.successfulSignals++;
         }
      }
      
      // Update performance metrics
      if(tradeCount > 0)
      {
         UpdatePerformanceMetrics(totalReturn, AccountInfoDouble(ACCOUNT_BALANCE));
         
         // Adjust pattern weights based on performance
         for(int i = 0; i < 10; i++)
         {
            // Simple reinforcement learning
            if(g_performance.winRate > 0.55)
               mlModel.patternWeights[i] *= 1.05;
            else if(g_performance.winRate < 0.45)
               mlModel.patternWeights[i] *= 0.95;
         }
      }
   }
   
   mlModel.lastUpdate = TimeCurrent();
   SaveMomentumModel();
}

//+------------------------------------------------------------------+
//| Initialize momentum model                                        |
//+------------------------------------------------------------------+
void InitializeMomentumModel()
{
   // Initialize weights randomly
   for(int i = 0; i < 40; i++)
      mlModel.weights[i] = (MathRand() / 32767.0 - 0.5) * 0.1;
   
   for(int i = 0; i < 10; i++)
      mlModel.patternWeights[i] = 0.5;
   
   mlModel.bias = 0;
   mlModel.performance = 0;
   mlModel.signalsAnalyzed = 0;
   mlModel.successfulSignals = 0;
   mlModel.avgReturn = 0;
   mlModel.lastUpdate = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Save/Load momentum model                                         |
//+------------------------------------------------------------------+
bool SaveMomentumModel()
{
   int handle = FileOpen(InpModelFile, FILE_WRITE|FILE_BIN);
   if(handle != INVALID_HANDLE)
   {
      FileWriteStruct(handle, mlModel);
      FileClose(handle);
      return true;
   }
   return false;
}

bool LoadMomentumModel()
{
   if(!FileIsExist(InpModelFile)) return false;
   
   int handle = FileOpen(InpModelFile, FILE_READ|FILE_BIN);
   if(handle != INVALID_HANDLE)
   {
      FileReadStruct(handle, mlModel);
      FileClose(handle);
      return true;
   }
   return false;
}