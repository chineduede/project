//+------------------------------------------------------------------+
//|                                     EnhancedMLRangeBreakout.mq5  |
//|                   Enhanced ML Range Breakout Scanner             |
//|                 With Dynamic Features & Risk Management          |
//+------------------------------------------------------------------+
#property copyright "Enhanced ML Range Breakout Scanner"
#property link      ""
#property version   "3.00"
#property description "Advanced ML scanner with market context and risk management"

#include "RangeAnalysis.mqh"
#include "EnhancedMLFeatures.mqh"
#include "MarketContextFilter.mqh"
#include "AdvancedRiskManagement.mqh"

//--- Input parameters
input group "Machine Learning Settings"
input bool     InpEnableLearning = true;         // Enable machine learning
input int      InpInitialLookback = 5000;        // Initial training period (bars)
input double   InpLearningRate = 0.1;            // Learning rate (0.01-1.0)
input int      InpUpdateFrequency = 100;         // Update model every N bars
input string   InpModelFile = "EnhancedMLModel.bin"; // Model save file

input group "Risk Management"
input double   InpMaxRiskPerTrade = 1.0;         // Max risk per trade (%)
input double   InpMaxDailyRisk = 3.0;            // Max daily risk (%)
input int      InpMaxOpenTrades = 3;             // Max concurrent trades
input bool     InpUseKellyCriterion = true;      // Use Kelly Criterion
input bool     InpUseTrailingStop = true;        // Enable trailing stop
input double   InpTrailingStopATR = 2.0;         // Trailing stop (ATR multiples)

input group "Market Context Filters"
input bool     InpUseMarketContext = true;       // Enable market context filtering
input bool     InpUseMultiTimeframe = true;      // Enable multi-timeframe analysis
input double   InpMinSessionScore = 0.5;         // Min session overlap score

input group "Display Settings"
input bool     InpShowDebugInfo = false;         // Show debug information
input bool     InpShowRiskReport = true;         // Show risk management report
input int      InpMaxRangesToShow = 10;          // Maximum ranges to display

//--- Enhanced ML Model structure
struct EnhancedMLModel
{
   AdaptiveParams params;
   double featureWeights[30];    // Extended feature weights
   double contextWeights[10];    // Market context weights
   double bias;
   double performance;
   int tradesAnalyzed;
   int successfulTrades;
   datetime lastUpdate;
   PerformanceMetrics metrics;   // Integrated performance tracking
};

//--- Global variables
EnhancedMLModel enhancedModel;
RangeData detectedRanges[];
MarketContext currentContext;
RiskParameters riskParams;
int updateCounter = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("Enhanced ML Range Breakout Scanner v3.0 initialized");
   
   // Initialize risk parameters
   riskParams.maxRiskPerTrade = InpMaxRiskPerTrade;
   riskParams.maxDailyRisk = InpMaxDailyRisk;
   riskParams.maxOpenTrades = InpMaxOpenTrades;
   riskParams.winRateTarget = 0.55;
   riskParams.profitFactorTarget = 1.8;
   riskParams.useKellyCriterion = InpUseKellyCriterion;
   riskParams.useTrailingStop = InpUseTrailingStop;
   riskParams.trailingStopATR = InpTrailingStopATR;
   
   // Initialize model
   if(!LoadEnhancedModel())
   {
      InitializeEnhancedModel();
      Print("New enhanced model initialized");
   }
   else
   {
      Print("Enhanced model loaded from file");
      Print("Previous performance - Win Rate: ", enhancedModel.metrics.winRate * 100, "%");
   }
   
   // Initial training
   if(InpEnableLearning)
   {
      TrainEnhancedModel(InpInitialLookback);
   }
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   SaveEnhancedModel();
   ObjectsDeleteAll(0, "ML_");
   
   if(InpShowRiskReport)
   {
      Print(GenerateRiskReport());
   }
   
   Print("Enhanced ML Range Breakout Scanner stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Update market context
   if(InpUseMarketContext)
   {
      GetMarketContext(_Symbol, currentContext);
      
      // Skip if market conditions are unfavorable
      if(!currentContext.isValidSession || currentContext.isLowLiquidity)
         return;
   }
   
   // Update trailing stops for open positions
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionGetString(POSITION_COMMENT) == "ML_Range_Breakout")
      {
         UpdateTrailingStop(ticket, riskParams);
      }
   }
   
   // Detect new bars
   static datetime lastBarTime = 0;
   datetime currentBarTime = iTime(_Symbol, _Period, 0);
   
   if(currentBarTime == lastBarTime) return;
   lastBarTime = currentBarTime;
   
   updateCounter++;
   
   // Scan for ranges and breakouts
   ScanForEnhancedBreakouts();
   
   // Update model periodically
   if(InpEnableLearning && updateCounter >= InpUpdateFrequency)
   {
      UpdateEnhancedModel();
      updateCounter = 0;
   }
}

//+------------------------------------------------------------------+
//| Scan for enhanced breakout opportunities                         |
//+------------------------------------------------------------------+
void ScanForEnhancedBreakouts()
{
   ArrayResize(detectedRanges, 0);
   EnhancedFeatures features;
   
   // Calculate enhanced features once
   if(InpUseMultiTimeframe)
   {
      CalculateEnhancedFeatures(_Symbol, features);
   }
   
   // Scan recent bars for ranges
   int barsToScan = MathMin(500, iBars(_Symbol, _Period) - enhancedModel.params.maxRangePeriod);
   
   for(int i = enhancedModel.params.maxRangePeriod; i < barsToScan; i += 10)
   {
      RangeData range;
      
      // Identify range using ML parameters
      if(IdentifyRangeML(i, enhancedModel.params, range))
      {
         // Calculate range metrics
         RangeMetrics metrics;
         CalculateRangeMetrics(range.startTime, range.endTime, 
                             range.highPrice, range.lowPrice, metrics);
         
         // Check for breakout
         if(DetectEnhancedBreakout(i, range, metrics, features))
         {
            // Apply market context filter
            if(InpUseMarketContext)
            {
               if(!ShouldTakeTradeBasedOnContext(currentContext, range.breakoutScore))
                  continue;
            }
            
            // Add to detected ranges
            int size = ArraySize(detectedRanges);
            ArrayResize(detectedRanges, size + 1);
            detectedRanges[size] = range;
         }
      }
   }
   
   // Sort by enhanced score
   SortRangesByScore(detectedRanges);
   
   // Display top ranges
   ObjectsDeleteAll(0, "ML_");
   int rangesToShow = MathMin(ArraySize(detectedRanges), InpMaxRangesToShow);
   
   for(int i = 0; i < rangesToShow; i++)
   {
      DrawEnhancedRange(i, detectedRanges[i]);
      
      // Check for trading opportunity
      if(i == 0 && detectedRanges[i].breakoutScore >= enhancedModel.params.scoreThreshold)
      {
         CheckTradingOpportunity(detectedRanges[i], features);
      }
   }
   
   if(InpShowDebugInfo)
   {
      PrintEnhancedDebugInfo(rangesToShow);
   }
}

//+------------------------------------------------------------------+
//| Detect enhanced breakout with ML scoring                         |
//+------------------------------------------------------------------+
bool DetectEnhancedBreakout(int startBar, RangeData &range, 
                           const RangeMetrics &metrics, const EnhancedFeatures &features)
{
   // Check basic breakout
   double currentPrice = iClose(_Symbol, _Period, 0);
   double breakoutThreshold = (range.highPrice - range.lowPrice) * enhancedModel.params.breakoutThreshold * 0.1;
   
   bool breakoutUp = currentPrice > range.highPrice + breakoutThreshold;
   bool breakoutDown = currentPrice < range.lowPrice - breakoutThreshold;
   
   if(!breakoutUp && !breakoutDown) return false;
   
   // Calculate ML score
   double mlScore = CalculateMLScore(range, metrics, features);
   
   // Calculate enhanced score with all features
   double enhancedScore = CalculateEnhancedScore(range, features);
   
   // Combine scores
   range.breakoutScore = mlScore * 0.6 + enhancedScore * 0.4;
   range.breakoutDirection = breakoutUp ? 1 : -1;
   range.breakoutTime = TimeCurrent();
   
   // Pattern recognition bonus
   if(metrics.pattern == PATTERN_FLAG || metrics.pattern == PATTERN_PENNANT)
      range.breakoutScore += 10;
   
   return range.breakoutScore >= 50; // Minimum score threshold
}

//+------------------------------------------------------------------+
//| Calculate ML score using neural network                          |
//+------------------------------------------------------------------+
double CalculateMLScore(const RangeData &range, const RangeMetrics &metrics, 
                       const EnhancedFeatures &features)
{
   double score = enhancedModel.bias;
   
   // Range-based features
   score += enhancedModel.featureWeights[0] * NormalizeValue(range.rangeSize, 10, 300);
   score += enhancedModel.featureWeights[1] * NormalizeValue(range.barsInRange, 5, 100);
   score += enhancedModel.featureWeights[2] * NormalizeValue(metrics.volatilityScore, 0, 100);
   score += enhancedModel.featureWeights[3] * NormalizeValue(metrics.touchesHigh, 2, 10);
   score += enhancedModel.featureWeights[4] * NormalizeValue(metrics.touchesLow, 2, 10);
   
   // Enhanced features
   score += enhancedModel.featureWeights[5] * features.sessionOverlap;
   score += enhancedModel.featureWeights[6] * features.marketRegime;
   score += enhancedModel.featureWeights[7] * features.htfTrend * range.breakoutDirection;
   score += enhancedModel.featureWeights[8] * NormalizeValue(features.relativeVolume, 0.5, 2.0);
   score += enhancedModel.featureWeights[9] * features.liquidityScore;
   
   // Market context weights
   if(InpUseMarketContext)
   {
      score += enhancedModel.contextWeights[0] * currentContext.correlationScore;
      score += enhancedModel.contextWeights[1] * currentContext.marketSentiment;
      score += enhancedModel.contextWeights[2] * (currentContext.isMajorLevel ? 1.0 : 0.0);
   }
   
   // Apply sigmoid activation
   return 100 / (1 + MathExp(-score));
}

//+------------------------------------------------------------------+
//| Check and execute trading opportunity                            |
//+------------------------------------------------------------------+
void CheckTradingOpportunity(const RangeData &range, const EnhancedFeatures &features)
{
   // Check if we already have a position
   if(PositionsTotal() >= riskParams.maxOpenTrades) return;
   
   double entryPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(range.breakoutDirection < 0)
      entryPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // Calculate dynamic stop loss
   double stopLoss = CalculateDynamicStopLoss(_Symbol, entryPrice, range.breakoutDirection);
   
   // Calculate take profit (dynamic based on volatility)
   double atr[];
   ArraySetAsSeries(atr, true);
   int atr_handle = iATR(_Symbol, _Period, 14);
   CopyBuffer(atr_handle, 0, 0, 1, atr);
   
   double takeProfit;
   if(range.breakoutDirection > 0)
      takeProfit = entryPrice + atr[0] * 3; // 3:1 RR ratio
   else
      takeProfit = entryPrice - atr[0] * 3;
   
   // Calculate position size
   double stopLossPoints = MathAbs(entryPrice - stopLoss);
   double lotSize = CalculateDynamicPositionSize(_Symbol, stopLossPoints, riskParams);
   
   if(lotSize == 0) return; // Risk limit reached
   
   // Validate trade
   if(!ValidateTradeRisk(_Symbol, entryPrice, stopLoss, takeProfit, lotSize, riskParams))
      return;
   
   // Execute trade
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lotSize;
   request.type = range.breakoutDirection > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   request.price = entryPrice;
   request.sl = stopLoss;
   request.tp = takeProfit;
   request.comment = "ML_Range_Breakout";
   request.magic = 12345;
   
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         Print("Trade executed: ", _Symbol, " ", 
               range.breakoutDirection > 0 ? "BUY" : "SELL",
               " Lots: ", lotSize,
               " Score: ", range.breakoutScore);
         
         // Update model
         enhancedModel.tradesAnalyzed++;
      }
   }
}

//+------------------------------------------------------------------+
//| Initialize enhanced model                                        |
//+------------------------------------------------------------------+
void InitializeEnhancedModel()
{
   // Initialize adaptive parameters
   enhancedModel.params.minRangeSize = 30;
   enhancedModel.params.maxRangeSize = 200;
   enhancedModel.params.minRangePeriod = 10;
   enhancedModel.params.maxRangePeriod = 50;
   enhancedModel.params.breakoutThreshold = 1.5;
   enhancedModel.params.volatilityFilter = 70;
   enhancedModel.params.consolidationRatio = 70;
   enhancedModel.params.scoreThreshold = 65;
   
   // Initialize weights randomly
   for(int i = 0; i < 30; i++)
      enhancedModel.featureWeights[i] = (MathRand() / 32767.0 - 0.5) * 0.1;
   
   for(int i = 0; i < 10; i++)
      enhancedModel.contextWeights[i] = (MathRand() / 32767.0 - 0.5) * 0.1;
   
   enhancedModel.bias = 0;
   enhancedModel.performance = 0;
   enhancedModel.tradesAnalyzed = 0;
   enhancedModel.successfulTrades = 0;
   enhancedModel.lastUpdate = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Train enhanced model with genetic algorithm                      |
//+------------------------------------------------------------------+
void TrainEnhancedModel(int lookbackBars)
{
   Print("Training enhanced model with ", lookbackBars, " bars of data...");
   
   // Similar to original but with enhanced features
   AdaptiveParams population[];
   ArrayResize(population, 20);
   
   // Initialize population
   for(int i = 0; i < 20; i++)
   {
      population[i] = GenerateRandomParams();
   }
   
   // Genetic algorithm with enhanced fitness function
   for(int generation = 0; generation < 10; generation++)
   {
      double fitness[];
      ArrayResize(fitness, 20);
      
      // Evaluate each member with enhanced features
      for(int i = 0; i < 20; i++)
      {
         fitness[i] = EvaluateEnhancedParameters(population[i], lookbackBars);
      }
      
      // Sort by fitness
      SortPopulationByFitness(population, fitness);
      
      // Create new generation
      for(int i = 10; i < 20; i++)
      {
         if(MathRand() % 2 == 0)
         {
            // Crossover
            int parent1 = MathRand() % 5;
            int parent2 = MathRand() % 5;
            population[i] = CrossoverParams(population[parent1], population[parent2]);
         }
         else
         {
            // Mutation
            population[i] = MutateParams(population[MathRand() % 10]);
         }
      }
      
      Print("Generation ", generation + 1, " best fitness: ", fitness[0]);
   }
   
   enhancedModel.params = population[0];
   Print("Enhanced training complete.");
}

//+------------------------------------------------------------------+
//| Evaluate parameters with enhanced features                       |
//+------------------------------------------------------------------+
double EvaluateEnhancedParameters(const AdaptiveParams &params, int lookbackBars)
{
   int successCount = 0;
   int totalTrades = 0;
   double totalProfit = 0;
   
   for(int i = params.maxRangePeriod; i < lookbackBars - 100; i += 20)
   {
      RangeData range;
      if(IdentifyRangeML(i, params, range))
      {
         // Calculate enhanced features for historical bar
         EnhancedFeatures histFeatures;
         // Simplified feature calculation for backtesting
         histFeatures.sessionOverlap = GetSessionOverlapScore();
         histFeatures.marketRegime = 0.5; // Neutral for backtesting
         
         // Simulate trade with enhanced scoring
         RangeMetrics metrics;
         CalculateRangeMetrics(range.startTime, range.endTime,
                             range.highPrice, range.lowPrice, metrics);
         
         double mlScore = CalculateMLScore(range, metrics, histFeatures);
         
         if(mlScore >= params.scoreThreshold)
         {
            double result = SimulateBreakoutTrade(i, range, params);
            if(result > 0)
            {
               successCount++;
               totalProfit += result;
            }
            totalTrades++;
         }
      }
   }
   
   if(totalTrades == 0) return 0;
   
   // Enhanced fitness function
   double winRate = (double)successCount / totalTrades;
   double avgProfit = totalProfit / totalTrades;
   double frequency = (double)totalTrades / (lookbackBars / 100.0);
   double consistency = 1.0 - MathAbs(winRate - 0.6); // Prefer 60% win rate
   
   return winRate * 40 + avgProfit * 30 + frequency * 20 + consistency * 10;
}

//+------------------------------------------------------------------+
//| Update enhanced model based on recent performance                |
//+------------------------------------------------------------------+
void UpdateEnhancedModel()
{
   // Check recent trade performance
   if(HistorySelect(enhancedModel.lastUpdate, TimeCurrent()))
   {
      int deals = HistoryDealsTotal();
      for(int i = 0; i < deals; i++)
      {
         ulong ticket = HistoryDealGetTicket(i);
         if(HistoryDealGetString(ticket, DEAL_COMMENT) == "ML_Range_Breakout")
         {
            double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
            UpdatePerformanceMetrics(profit, AccountInfoDouble(ACCOUNT_BALANCE));
            
            if(profit > 0)
               enhancedModel.successfulTrades++;
         }
      }
   }
   
   // Update model performance
   enhancedModel.performance = g_performance.winRate * 50 + 
                              g_performance.profitFactor * 30 + 
                              (100 - g_performance.maxDrawdown) * 20;
   
   enhancedModel.lastUpdate = TimeCurrent();
   enhancedModel.metrics = g_performance;
   
   // Adjust parameters based on performance
   if(g_performance.totalTrades >= 20)
   {
      if(g_performance.winRate < 0.5)
      {
         // Tighten parameters
         enhancedModel.params.scoreThreshold = MathMin(90, enhancedModel.params.scoreThreshold + 5);
         enhancedModel.params.consolidationRatio = MathMin(90, enhancedModel.params.consolidationRatio + 5);
      }
      else if(g_performance.winRate > 0.65 && g_performance.profitFactor > 2.0)
      {
         // Loosen parameters slightly
         enhancedModel.params.scoreThreshold = MathMax(50, enhancedModel.params.scoreThreshold - 2);
      }
   }
   
   SaveEnhancedModel();
}

//+------------------------------------------------------------------+
//| Draw enhanced range with additional information                  |
//+------------------------------------------------------------------+
void DrawEnhancedRange(int index, const RangeData &range)
{
   string prefix = "ML_Range_" + IntegerToString(index) + "_";
   
   // Draw rectangle
   string rectName = prefix + "Box";
   color boxColor = range.breakoutDirection > 0 ? clrDodgerBlue : clrTomato;
   
   ObjectCreate(0, rectName, OBJ_RECTANGLE, 0, range.startTime, range.highPrice, range.endTime, range.lowPrice);
   ObjectSetInteger(0, rectName, OBJPROP_COLOR, boxColor);
   ObjectSetInteger(0, rectName, OBJPROP_STYLE, STYLE_SOLID);
   ObjectSetInteger(0, rectName, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, rectName, OBJPROP_BACK, true);
   ObjectSetInteger(0, rectName, OBJPROP_FILL, true);
   
   // Add enhanced score label
   string labelName = prefix + "Score";
   string scoreText = StringFormat("Score: %.1f | Vol: %.1f | MTF: %s", 
                                  range.breakoutScore,
                                  range.qualityScore,
                                  range.breakoutDirection > 0 ? "↑" : "↓");
   
   ObjectCreate(0, labelName, OBJ_TEXT, 0, range.startTime, range.highPrice + 20 * _Point);
   ObjectSetString(0, labelName, OBJPROP_TEXT, scoreText);
   ObjectSetInteger(0, labelName, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, labelName, OBJPROP_FONTSIZE, 9);
   ObjectSetString(0, labelName, OBJPROP_FONT, "Arial Bold");
}

//+------------------------------------------------------------------+
//| Print enhanced debug information                                 |
//+------------------------------------------------------------------+
void PrintEnhancedDebugInfo(int rangesFound)
{
   Print("=== Enhanced ML Debug Info ===");
   Print("Ranges found: ", rangesFound);
   Print("Model performance: ", DoubleToString(enhancedModel.performance, 2));
   Print("Win rate: ", DoubleToString(g_performance.winRate * 100, 1), "%");
   Print("Profit factor: ", DoubleToString(g_performance.profitFactor, 2));
   Print("Current drawdown: ", DoubleToString(g_performance.currentDrawdown, 1), "%");
   Print("Market regime: ", GetMarketRegimeString());
   Print("Session score: ", DoubleToString(currentContext.correlationScore, 2));
}

//+------------------------------------------------------------------+
//| Get market regime as string                                      |
//+------------------------------------------------------------------+
string GetMarketRegimeString()
{
   ENUM_MARKET_REGIME regime = GetMarketRegime(_Symbol, _Period);
   switch(regime)
   {
      case REGIME_TRENDING_UP: return "Trending Up";
      case REGIME_TRENDING_DOWN: return "Trending Down";
      case REGIME_RANGING: return "Ranging";
      case REGIME_VOLATILE: return "Volatile";
      case REGIME_QUIET: return "Quiet";
      default: return "Unknown";
   }
}

//+------------------------------------------------------------------+
//| Save/Load enhanced model                                        |
//+------------------------------------------------------------------+
bool SaveEnhancedModel()
{
   int handle = FileOpen(InpModelFile, FILE_WRITE|FILE_BIN);
   if(handle != INVALID_HANDLE)
   {
      FileWriteStruct(handle, enhancedModel);
      FileClose(handle);
      return true;
   }
   return false;
}

bool LoadEnhancedModel()
{
   if(!FileIsExist(InpModelFile)) return false;
   
   int handle = FileOpen(InpModelFile, FILE_READ|FILE_BIN);
   if(handle != INVALID_HANDLE)
   {
      FileReadStruct(handle, enhancedModel);
      FileClose(handle);
      g_performance = enhancedModel.metrics;
      return true;
   }
   return false;
}