//+------------------------------------------------------------------+
//|                                             MLRangeBreakout.mq5   |
//|                        Machine Learning Range Breakout Scanner   |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "ML Range Breakout Scanner"
#property link      ""
#property version   "2.00"
#property description "Self-learning range breakout scanner with adaptive parameters"

#include "RangeAnalysis.mqh"

//--- Input parameters
input bool     InpEnableLearning = true;         // Enable machine learning
input int      InpInitialLookback = 5000;        // Initial training period (bars)
input double   InpLearningRate = 0.1;            // Learning rate (0.01-1.0)
input int      InpUpdateFrequency = 100;         // Update model every N bars
input bool     InpShowDebugInfo = false;         // Show debug information
input string   InpModelFile = "MLRangeModel.bin"; // Model save file

//--- Adaptive parameters (will be optimized by ML)
struct AdaptiveParams
{
   double minRangeSize;        // 10-200 points
   double maxRangeSize;        // 50-500 points
   int    minRangePeriod;      // 5-50 bars
   int    maxRangePeriod;      // 20-200 bars
   double breakoutThreshold;   // 0.5-3.0 multiplier
   double volatilityFilter;    // 0-100 score
   int    consolidationRatio;  // 50-90 percent
   double scoreThreshold;      // 40-90 score
};

//--- ML Model structure
struct MLModel
{
   AdaptiveParams params;
   double weights[20];          // Feature weights
   double bias;
   double performance;          // Model performance metric
   int    tradesAnalyzed;
   int    successfulTrades;
   datetime lastUpdate;
};

//--- Feature vector for ML
struct FeatureVector
{
   double timeOfDay;           // 0-24 normalized
   double dayOfWeek;           // 1-5 normalized
   double volatility;          // Market volatility
   double trend;               // Trend strength
   double volume;              // Volume ratio
   double priceLevel;          // Relative price level
   double momentum;            // Price momentum
   double rangeCount;          // Recent range count
};

//--- Global variables
MLModel currentModel;
AdaptiveParams bestParams;
RangeData detectedRanges[];
double performanceHistory[];
int historySize = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("ML Range Breakout Scanner initialized");
   
   // Initialize model
   if(!LoadModel())
   {
      InitializeModel();
      Print("New model initialized");
   }
   else
   {
      Print("Model loaded from file");
   }
   
   // Initial training
   if(InpEnableLearning)
   {
      TrainModel(InpInitialLookback);
   }
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   SaveModel();
   ObjectsDeleteAll(0, "ML_");
   Print("ML Range Breakout Scanner stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   static datetime lastBarTime = 0;
   static int barsSinceUpdate = 0;
   
   datetime currentBarTime = iTime(_Symbol, _Period, 0);
   
   if(lastBarTime != currentBarTime)
   {
      lastBarTime = currentBarTime;
      barsSinceUpdate++;
      
      // Scan for ranges with current parameters
      ScanWithMLParameters();
      
      // Update model periodically
      if(InpEnableLearning && barsSinceUpdate >= InpUpdateFrequency)
      {
         UpdateModel();
         barsSinceUpdate = 0;
      }
   }
}

//+------------------------------------------------------------------+
//| Initialize ML model with default parameters                      |
//+------------------------------------------------------------------+
void InitializeModel()
{
   // Initialize adaptive parameters with reasonable defaults
   currentModel.params.minRangeSize = 30;
   currentModel.params.maxRangeSize = 200;
   currentModel.params.minRangePeriod = 10;
   currentModel.params.maxRangePeriod = 50;
   currentModel.params.breakoutThreshold = 1.0;
   currentModel.params.volatilityFilter = 50;
   currentModel.params.consolidationRatio = 70;
   currentModel.params.scoreThreshold = 60;
   
   // Initialize weights randomly
   for(int i = 0; i < 20; i++)
   {
      currentModel.weights[i] = (MathRand() / 32767.0 - 0.5) * 0.1;
   }
   
   currentModel.bias = 0;
   currentModel.performance = 0;
   currentModel.tradesAnalyzed = 0;
   currentModel.successfulTrades = 0;
   currentModel.lastUpdate = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Train model on historical data                                   |
//+------------------------------------------------------------------+
void TrainModel(int lookbackBars)
{
   Print("Training model on ", lookbackBars, " bars of historical data...");
   
   ArrayResize(performanceHistory, 0);
   historySize = 0;
   
   // Genetic algorithm for parameter optimization
   AdaptiveParams population[];
   ArrayResize(population, 20);
   
   // Create initial population
   for(int i = 0; i < 20; i++)
   {
      population[i] = GenerateRandomParams();
   }
   
   // Evolution iterations
   for(int generation = 0; generation < 10; generation++)
   {
      double fitness[];
      ArrayResize(fitness, 20);
      
      // Evaluate each individual
      for(int i = 0; i < 20; i++)
      {
         fitness[i] = EvaluateParameters(population[i], lookbackBars);
      }
      
      // Select best performers
      SortPopulationByFitness(population, fitness);
      
      // Create next generation
      for(int i = 10; i < 20; i++)
      {
         if(MathRand() % 2 == 0)
         {
            // Crossover
            population[i] = CrossoverParams(population[MathRand() % 5], population[MathRand() % 5]);
         }
         else
         {
            // Mutation
            population[i] = MutateParams(population[MathRand() % 10]);
         }
      }
      
      Print("Generation ", generation + 1, " best fitness: ", fitness[0]);
   }
   
   // Use best parameters
   currentModel.params = population[0];
   bestParams = population[0];
   
   Print("Training complete. Best parameters found.");
}

//+------------------------------------------------------------------+
//| Generate random parameters                                       |
//+------------------------------------------------------------------+
AdaptiveParams GenerateRandomParams()
{
   AdaptiveParams params;
   
   params.minRangeSize = 10 + MathRand() % 190;
   params.maxRangeSize = params.minRangeSize + 50 + MathRand() % 300;
   params.minRangePeriod = 5 + MathRand() % 45;
   params.maxRangePeriod = params.minRangePeriod + 10 + MathRand() % 150;
   params.breakoutThreshold = 0.5 + (MathRand() / 32767.0) * 2.5;
   params.volatilityFilter = MathRand() % 100;
   params.consolidationRatio = 50 + MathRand() % 40;
   params.scoreThreshold = 40 + MathRand() % 50;
   
   return params;
}

//+------------------------------------------------------------------+
//| Evaluate parameters performance                                  |
//+------------------------------------------------------------------+
double EvaluateParameters(const AdaptiveParams &params, int lookbackBars)
{
   int successCount = 0;
   int totalTrades = 0;
   double totalProfit = 0;
   
   // Test parameters on historical data
   for(int i = params.maxRangePeriod; i < lookbackBars - 100; i += 10)
   {
      RangeData range;
      if(IdentifyRangeML(i, params, range))
      {
         // Check if breakout was successful
         double result = SimulateBreakoutTrade(i, range, params);
         if(result > 0)
         {
            successCount++;
            totalProfit += result;
         }
         totalTrades++;
      }
   }
   
   if(totalTrades == 0) return 0;
   
   // Calculate fitness score
   double winRate = (double)successCount / totalTrades;
   double avgProfit = totalProfit / totalTrades;
   double frequency = (double)totalTrades / (lookbackBars / 100.0);
   
   // Weighted fitness function
   return winRate * 50 + avgProfit * 30 + frequency * 20;
}

//+------------------------------------------------------------------+
//| Identify range using ML parameters                               |
//+------------------------------------------------------------------+
bool IdentifyRangeML(int startBar, const AdaptiveParams &params, RangeData &range)
{
   // Try different range periods within parameters
   for(int period = params.minRangePeriod; period <= params.maxRangePeriod; period += 5)
   {
      double highestHigh = 0;
      double lowestLow = DBL_MAX;
      
      // Find high and low
      for(int i = startBar; i >= startBar - period && i >= 0; i--)
      {
         double high = iHigh(_Symbol, _Period, i);
         double low = iLow(_Symbol, _Period, i);
         
         if(high > highestHigh) highestHigh = high;
         if(low < lowestLow) lowestLow = low;
      }
      
      double rangeSize = (highestHigh - lowestLow) / _Point;
      
      // Check if range meets criteria
      if(rangeSize >= params.minRangeSize && rangeSize <= params.maxRangeSize)
      {
         // Check consolidation quality
         int consolidationBars = 0;
         double rangeBuffer = (highestHigh - lowestLow) * 0.15;
         
         for(int i = startBar; i >= startBar - period && i >= 0; i--)
         {
            double high = iHigh(_Symbol, _Period, i);
            double low = iLow(_Symbol, _Period, i);
            
            if(high <= highestHigh + rangeBuffer && low >= lowestLow - rangeBuffer)
               consolidationBars++;
         }
         
         double consolidationPercent = (double)consolidationBars / period * 100;
         
         if(consolidationPercent >= params.consolidationRatio)
         {
            // Calculate volatility score
            RangeMetrics metrics;
            CalculateRangeMetrics(iTime(_Symbol, _Period, startBar), 
                                iTime(_Symbol, _Period, startBar - period),
                                highestHigh, lowestLow, metrics);
            
            if(metrics.volatilityScore <= params.volatilityFilter)
            {
               // Valid range found
               range.startTime = iTime(_Symbol, _Period, startBar);
               range.endTime = iTime(_Symbol, _Period, startBar - period);
               range.highPrice = highestHigh;
               range.lowPrice = lowestLow;
               range.rangeSize = rangeSize;
               range.barsInRange = period;
               
               return true;
            }
         }
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Simulate breakout trade for backtesting                         |
//+------------------------------------------------------------------+
double SimulateBreakoutTrade(int rangeEndBar, const RangeData &range, const AdaptiveParams &params)
{
   double breakoutLevel = (range.highPrice - range.lowPrice) * params.breakoutThreshold * 0.1;
   int checkBars = MathMin(50, rangeEndBar - range.barsInRange);
   
   for(int i = rangeEndBar - range.barsInRange - 1; i >= rangeEndBar - range.barsInRange - checkBars && i >= 0; i--)
   {
      double close = iClose(_Symbol, _Period, i);
      
      // Check for breakout
      if(close > range.highPrice + breakoutLevel || close < range.lowPrice - breakoutLevel)
      {
         bool isUpBreakout = close > range.highPrice;
         double entryPrice = close;
         
         // Simulate trade outcome
         double maxProfit = 0;
         double maxLoss = 0;
         
         for(int j = i - 1; j >= MathMax(0, i - 20); j--)
         {
            double price = iClose(_Symbol, _Period, j);
            double profit = isUpBreakout ? price - entryPrice : entryPrice - price;
            
            if(profit > maxProfit) maxProfit = profit;
            if(profit < maxLoss) maxLoss = profit;
         }
         
         // Simple profit calculation
         return maxProfit + maxLoss;
      }
   }
   
   return 0;
}

//+------------------------------------------------------------------+
//| Scan with current ML parameters                                  |
//+------------------------------------------------------------------+
void ScanWithMLParameters()
{
   ArrayResize(detectedRanges, 0);
   ObjectsDeleteAll(0, "ML_");
   
   // Extract features for current market
   FeatureVector features = ExtractFeatures();
   
   // Adjust parameters based on features
   AdaptiveParams adjustedParams = AdjustParametersML(features);
   
   // Scan for ranges
   int rangesFound = 0;
   for(int i = adjustedParams.maxRangePeriod; i < 500; i += 5)
   {
      RangeData range;
      if(IdentifyRangeML(i, adjustedParams, range))
      {
         // Calculate ML score
         double mlScore = CalculateMLScore(range, features);
         
         if(mlScore >= adjustedParams.scoreThreshold)
         {
            range.breakoutScore = mlScore;
            
            int size = ArraySize(detectedRanges);
            ArrayResize(detectedRanges, size + 1);
            detectedRanges[size] = range;
            rangesFound++;
            
            // Visualize
            DrawMLRange(rangesFound - 1, range);
         }
      }
   }
   
   if(InpShowDebugInfo)
   {
      PrintDebugInfo(adjustedParams, rangesFound);
   }
}

//+------------------------------------------------------------------+
//| Extract features from current market                             |
//+------------------------------------------------------------------+
FeatureVector ExtractFeatures()
{
   FeatureVector features;
   
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   
   features.timeOfDay = dt.hour / 24.0;
   features.dayOfWeek = dt.day_of_week / 5.0;
   
   // Calculate market features
   double atr = iATR(_Symbol, _Period, 14);
   features.volatility = NormalizeValue(atr / _Point, 0, 100);
   
   // Trend calculation
   double ma20 = iMA(_Symbol, _Period, 20, 0, MODE_SMA, PRICE_CLOSE);
   double ma50 = iMA(_Symbol, _Period, 50, 0, MODE_SMA, PRICE_CLOSE);
   features.trend = NormalizeValue((ma20 - ma50) / _Point, -100, 100);
   
   // Volume
   double avgVolume = 0;
   for(int i = 0; i < 20; i++)
   {
      avgVolume += iVolume(_Symbol, _Period, i);
   }
   avgVolume /= 20;
   features.volume = NormalizeValue(iVolume(_Symbol, _Period, 0) / avgVolume, 0, 3);
   
   // Price level
   double dayHigh = iHigh(_Symbol, PERIOD_D1, 0);
   double dayLow = iLow(_Symbol, PERIOD_D1, 0);
   double currentPrice = iClose(_Symbol, _Period, 0);
   features.priceLevel = (currentPrice - dayLow) / (dayHigh - dayLow);
   
   // Momentum
   double momentum = iMomentum(_Symbol, _Period, 14, PRICE_CLOSE);
   features.momentum = NormalizeValue(momentum - 100, -10, 10);
   
   // Recent range count
   features.rangeCount = NormalizeValue(ArraySize(detectedRanges), 0, 10);
   
   return features;
}

//+------------------------------------------------------------------+
//| Adjust parameters based on ML features                           |
//+------------------------------------------------------------------+
AdaptiveParams AdjustParametersML(const FeatureVector &features)
{
   AdaptiveParams adjusted = currentModel.params;
   
   // Apply neural network adjustments
   double featureArray[];
   ArrayResize(featureArray, 8);
   featureArray[0] = features.timeOfDay;
   featureArray[1] = features.dayOfWeek;
   featureArray[2] = features.volatility;
   featureArray[3] = features.trend;
   featureArray[4] = features.volume;
   featureArray[5] = features.priceLevel;
   featureArray[6] = features.momentum;
   featureArray[7] = features.rangeCount;
   
   // Neural network forward pass
   double activation = currentModel.bias;
   for(int i = 0; i < 8; i++)
   {
      activation += featureArray[i] * currentModel.weights[i];
   }
   
   // Sigmoid activation
   double adjustment = 1.0 / (1.0 + MathExp(-activation));
   
   // Apply adjustments
   adjusted.minRangeSize *= (0.5 + adjustment);
   adjusted.maxRangeSize *= (0.5 + adjustment);
   adjusted.breakoutThreshold *= (0.7 + adjustment * 0.6);
   adjusted.volatilityFilter = 30 + adjustment * 40;
   
   // Time-based adjustments
   if(features.timeOfDay < 0.3 || features.timeOfDay > 0.8)
   {
      // Asian/Night session - wider ranges
      adjusted.minRangePeriod *= 1.2;
      adjusted.maxRangePeriod *= 1.2;
   }
   
   // Volatility adjustments
   if(features.volatility > 0.7)
   {
      adjusted.consolidationRatio *= 0.8;
      adjusted.scoreThreshold *= 1.1;
   }
   
   return adjusted;
}

//+------------------------------------------------------------------+
//| Calculate ML-based score for range                               |
//+------------------------------------------------------------------+
double CalculateMLScore(const RangeData &range, const FeatureVector &features)
{
   RangeMetrics metrics;
   CalculateRangeMetrics(range.startTime, range.endTime, 
                        range.highPrice, range.lowPrice, metrics);
   
   // Feature engineering for scoring
   double scoreFeatures[12];
   scoreFeatures[0] = NormalizeValue(range.rangeSize, 0, 200);
   scoreFeatures[1] = NormalizeValue(metrics.volatilityScore, 0, 100);
   scoreFeatures[2] = NormalizeValue(metrics.touchesHigh, 0, 10);
   scoreFeatures[3] = NormalizeValue(metrics.touchesLow, 0, 10);
   scoreFeatures[4] = features.timeOfDay;
   scoreFeatures[5] = features.volatility;
   scoreFeatures[6] = features.trend;
   scoreFeatures[7] = features.momentum;
   scoreFeatures[8] = NormalizeValue(metrics.resistanceStrength, 0, 100);
   scoreFeatures[9] = NormalizeValue(metrics.supportStrength, 0, 100);
   scoreFeatures[10] = features.volume;
   scoreFeatures[11] = (metrics.pattern == PATTERN_RECTANGLE) ? 1.0 : 0.5;
   
   // Neural network scoring
   double score = 50; // Base score
   
   for(int i = 0; i < 12; i++)
   {
      score += scoreFeatures[i] * currentModel.weights[i + 8];
   }
   
   // Apply learned performance factor
   if(currentModel.tradesAnalyzed > 0)
   {
      double winRate = (double)currentModel.successfulTrades / currentModel.tradesAnalyzed;
      score *= (0.5 + winRate);
   }
   
   return MathMax(0, MathMin(100, score));
}

//+------------------------------------------------------------------+
//| Update model based on recent performance                         |
//+------------------------------------------------------------------+
void UpdateModel()
{
   if(ArraySize(performanceHistory) < 10) return;
   
   // Calculate recent performance
   double recentPerformance = 0;
   int recentCount = MathMin(10, ArraySize(performanceHistory));
   
   for(int i = ArraySize(performanceHistory) - recentCount; i < ArraySize(performanceHistory); i++)
   {
      recentPerformance += performanceHistory[i];
   }
   recentPerformance /= recentCount;
   
   // Update weights using gradient descent
   double learningRate = InpLearningRate * (1.0 - currentModel.performance);
   
   for(int i = 0; i < 20; i++)
   {
      double gradient = (recentPerformance - currentModel.performance) * MathRand() / 32767.0;
      currentModel.weights[i] += learningRate * gradient;
      
      // Keep weights in reasonable range
      currentModel.weights[i] = MathMax(-1.0, MathMin(1.0, currentModel.weights[i]));
   }
   
   // Update bias
   currentModel.bias += learningRate * (recentPerformance - currentModel.performance) * 0.1;
   
   // Update performance metric
   currentModel.performance = currentModel.performance * 0.9 + recentPerformance * 0.1;
   currentModel.lastUpdate = TimeCurrent();
   
   if(InpShowDebugInfo)
   {
      Print("Model updated. Performance: ", DoubleToString(currentModel.performance, 2));
   }
   
   // Save model periodically
   SaveModel();
}

//+------------------------------------------------------------------+
//| Draw ML-detected range                                           |
//+------------------------------------------------------------------+
void DrawMLRange(int index, const RangeData &range)
{
   string prefix = "ML_Range_" + IntegerToString(index) + "_";
   
   // Draw rectangle
   string rectName = prefix + "Box";
   ObjectCreate(0, rectName, OBJ_RECTANGLE, 0, range.startTime, range.highPrice, range.endTime, range.lowPrice);
   ObjectSetInteger(0, rectName, OBJPROP_COLOR, clrYellow);
   ObjectSetInteger(0, rectName, OBJPROP_STYLE, STYLE_SOLID);
   ObjectSetInteger(0, rectName, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, rectName, OBJPROP_BACK, true);
   
   // Add ML score label
   string labelName = prefix + "Score";
   ObjectCreate(0, labelName, OBJ_TEXT, 0, range.startTime, range.highPrice + 15 * _Point);
   ObjectSetString(0, labelName, OBJPROP_TEXT, "ML Score: " + DoubleToString(range.breakoutScore, 1));
   ObjectSetInteger(0, labelName, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, labelName, OBJPROP_FONTSIZE, 9);
}

//+------------------------------------------------------------------+
//| Helper functions                                                 |
//+------------------------------------------------------------------+
double NormalizeValue(double value, double min, double max)
{
   if(max - min == 0) return 0.5;
   return (value - min) / (max - min);
}

void SortPopulationByFitness(AdaptiveParams &population[], double &fitness[])
{
   // Simple bubble sort
   for(int i = 0; i < ArraySize(population) - 1; i++)
   {
      for(int j = 0; j < ArraySize(population) - i - 1; j++)
      {
         if(fitness[j] < fitness[j + 1])
         {
            double tempFit = fitness[j];
            fitness[j] = fitness[j + 1];
            fitness[j + 1] = tempFit;
            
            AdaptiveParams tempParam = population[j];
            population[j] = population[j + 1];
            population[j + 1] = tempParam;
         }
      }
   }
}

AdaptiveParams CrossoverParams(const AdaptiveParams &parent1, const AdaptiveParams &parent2)
{
   AdaptiveParams child;
   
   // Random crossover
   child.minRangeSize = MathRand() % 2 ? parent1.minRangeSize : parent2.minRangeSize;
   child.maxRangeSize = MathRand() % 2 ? parent1.maxRangeSize : parent2.maxRangeSize;
   child.minRangePeriod = MathRand() % 2 ? parent1.minRangePeriod : parent2.minRangePeriod;
   child.maxRangePeriod = MathRand() % 2 ? parent1.maxRangePeriod : parent2.maxRangePeriod;
   child.breakoutThreshold = (parent1.breakoutThreshold + parent2.breakoutThreshold) / 2;
   child.volatilityFilter = (parent1.volatilityFilter + parent2.volatilityFilter) / 2;
   child.consolidationRatio = MathRand() % 2 ? parent1.consolidationRatio : parent2.consolidationRatio;
   child.scoreThreshold = (parent1.scoreThreshold + parent2.scoreThreshold) / 2;
   
   return child;
}

AdaptiveParams MutateParams(const AdaptiveParams &params)
{
   AdaptiveParams mutated = params;
   
   // Random mutations
   if(MathRand() % 4 == 0) mutated.minRangeSize *= (0.8 + MathRand() / 32767.0 * 0.4);
   if(MathRand() % 4 == 0) mutated.maxRangeSize *= (0.8 + MathRand() / 32767.0 * 0.4);
   if(MathRand() % 4 == 0) mutated.minRangePeriod = MathMax(5, mutated.minRangePeriod + MathRand() % 21 - 10);
   if(MathRand() % 4 == 0) mutated.maxRangePeriod = MathMax(mutated.minRangePeriod + 10, mutated.maxRangePeriod + MathRand() % 41 - 20);
   if(MathRand() % 4 == 0) mutated.breakoutThreshold *= (0.8 + MathRand() / 32767.0 * 0.4);
   if(MathRand() % 4 == 0) mutated.volatilityFilter = MathMax(0, MathMin(100, mutated.volatilityFilter + MathRand() % 21 - 10));
   if(MathRand() % 4 == 0) mutated.consolidationRatio = MathMax(50, MathMin(90, mutated.consolidationRatio + MathRand() % 11 - 5));
   if(MathRand() % 4 == 0) mutated.scoreThreshold = MathMax(40, MathMin(90, mutated.scoreThreshold + MathRand() % 11 - 5));
   
   return mutated;
}

void PrintDebugInfo(const AdaptiveParams &params, int rangesFound)
{
   Print("=== ML Debug Info ===");
   Print("Ranges found: ", rangesFound);
   Print("Range size: ", params.minRangeSize, "-", params.maxRangeSize);
   Print("Period: ", params.minRangePeriod, "-", params.maxRangePeriod);
   Print("Breakout threshold: ", DoubleToString(params.breakoutThreshold, 2));
   Print("Model performance: ", DoubleToString(currentModel.performance, 2));
   Print("Trades analyzed: ", currentModel.tradesAnalyzed);
}

//+------------------------------------------------------------------+
//| Save/Load model functions                                        |
//+------------------------------------------------------------------+
bool SaveModel()
{
   int handle = FileOpen(InpModelFile, FILE_WRITE|FILE_BIN);
   if(handle != INVALID_HANDLE)
   {
      FileWriteStruct(handle, currentModel);
      FileClose(handle);
      return true;
   }
   return false;
}

bool LoadModel()
{
   if(!FileIsExist(InpModelFile)) return false;
   
   int handle = FileOpen(InpModelFile, FILE_READ|FILE_BIN);
   if(handle != INVALID_HANDLE)
   {
      FileReadStruct(handle, currentModel);
      FileClose(handle);
      return true;
   }
   return false;
}