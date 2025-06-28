//+------------------------------------------------------------------+
//|                                      MLMeanReversionStrategy.mq5  |
//|                    Machine Learning Mean Reversion Strategy       |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "ML Mean Reversion Strategy"
#property link      ""
#property version   "1.00"
#property description "Advanced mean reversion strategy with ML optimization and statistical analysis"

#include "EnhancedMLFeatures.mqh"
#include "MarketContextFilter.mqh"
#include "AdvancedRiskManagement.mqh"
#include "AdvancedMarketAnalysis.mqh"
#include "UltraAdvancedTrading.mqh"
#include "EliteQuantTrading.mqh"

//--- Input parameters
input group "Strategy Settings"
input int      InpZScorePeriod = 20;             // Z-Score calculation period
input double   InpZScoreEntry = 2.0;             // Z-Score entry threshold
input double   InpZScoreExit = 0.5;              // Z-Score exit threshold
input int      InpBollingerPeriod = 20;          // Bollinger Bands period
input double   InpBollingerDeviation = 2.0;      // Bollinger Bands deviation
input int      InpRSIPeriod = 14;                // RSI period for overbought/oversold
input double   InpRSIOverbought = 70;            // RSI overbought level
input double   InpRSIOversold = 30;              // RSI oversold level

input group "Mean Reversion Filters"
input bool     InpUseVolumeFilter = true;        // Use volume confirmation
input double   InpMinVolumeRatio = 1.2;          // Minimum volume ratio vs average
input bool     InpUseTrendFilter = true;         // Avoid trading against strong trends
input int      InpTrendPeriod = 50;              // Trend detection period
input double   InpMaxTrendStrength = 30;         // Maximum trend angle (degrees)
input bool     InpUseMTFConfirmation = true;     // Use multi-timeframe confirmation
input ENUM_TIMEFRAMES InpHTF1 = PERIOD_H1;       // Higher timeframe 1
input ENUM_TIMEFRAMES InpHTF2 = PERIOD_H4;       // Higher timeframe 2

input group "Machine Learning"
input bool     InpEnableML = true;               // Enable machine learning
input int      InpMLUpdateFreq = 50;             // Update ML model every N trades
input double   InpLearningRate = 0.1;            // ML learning rate
input string   InpModelFile = "MLMeanRev.bin";   // Model save file

input group "Risk Management"
input double   InpRiskPerTrade = 1.0;            // Risk per trade (%)
input double   InpMaxDailyRisk = 3.0;            // Maximum daily risk (%)
input int      InpMaxPositions = 3;              // Maximum open positions
input bool     InpUseATRStop = true;             // Use ATR-based stops
input double   InpATRMultiplier = 1.5;           // ATR multiplier for stops
input bool     InpScalePositions = true;         // Scale into positions
input bool     InpUseKellyCriterion = true;      // Use Kelly Criterion for position sizing
input double   InpKellySafety = 0.25;            // Kelly safety factor (0.25 = 25% of Kelly)

input group "Advanced Features"
input bool     InpUseMarketRegime = true;        // Use market regime detection
input bool     InpUseMicrostructure = true;      // Use market microstructure analysis
input bool     InpUseSentiment = true;           // Use sentiment indicators
input bool     InpUsePairTrading = true;         // Enable pair trading mode
input string   InpPairSymbol = "";               // Pair trading symbol (if enabled)
input bool     InpUseSmartExecution = true;      // Use TWAP/VWAP execution
input double   InpMinLiquidity = 50000;          // Minimum daily volume for trading

//--- Mean reversion patterns
enum ENUM_REVERSION_PATTERN
{
   REV_PATTERN_NONE,
   REV_PATTERN_ZSCORE_EXTREME,
   REV_PATTERN_BOLLINGER_SQUEEZE,
   REV_PATTERN_RSI_DIVERGENCE,
   REV_PATTERN_VOLUME_CLIMAX,
   REV_PATTERN_DOUBLE_TOP_BOTTOM,
   REV_PATTERN_CHANNEL_REJECTION,
   REV_PATTERN_VOLATILITY_MEAN_REV,
   REV_PATTERN_CORRELATION_BREAK
};

//--- Reversion signal structure
struct ReversionSignal
{
   datetime time;
   double price;
   double zScore;
   double deviation;
   ENUM_REVERSION_PATTERN pattern;
   int direction;        // 1 for long (oversold), -1 for short (overbought)
   double score;        // ML-calculated score
   double targetPrice;  // Mean reversion target
   double confidence;   // Confidence level (0-100)
   double expectedReturn;
};

//--- ML Model for mean reversion
struct MeanReversionMLModel
{
   double weights[50];          // Feature weights
   double patternWeights[10];   // Pattern-specific weights
   double bias;
   double performance;
   int signalsAnalyzed;
   int successfulSignals;
   double avgReturn;
   double sharpeRatio;
   datetime lastUpdate;
   
   // Statistical tracking
   double meanReversionRate;    // How quickly price reverts to mean
   double falseSignalRate;      // Rate of signals that don't revert
   double avgHoldingPeriod;     // Average bars until reversion
};

//--- Statistical data structure
struct StatisticalData
{
   double mean;
   double stdDev;
   double skewness;
   double kurtosis;
   double currentZScore;
   double halfLife;        // Mean reversion half-life
   double hurst;          // Hurst exponent (trend vs mean reversion)
};

//--- Global variables
MeanReversionMLModel mlModel;
ReversionSignal currentSignals[];
StatisticalData statistics[];
RiskParameters riskParams;
int signalCount = 0;
double g_accountBalance = 0;

// Advanced analysis variables
MarketMicrostructure g_microstructure;
MarketRegime g_marketRegime;
MarketSentiment g_sentiment;
OrderFlowAnalysis g_orderFlow;
KellyPosition g_kellyPosition;
CorrelationData g_pairCorrelation;

// Ultra-advanced variables
NeuralNetwork g_neuralNet;
HiddenLiquidity g_hiddenLiquidity;
OrderBookDynamics g_orderBookDynamics;
MarketMakerInventory g_marketMaker;
AdaptiveStopLoss g_adaptiveStop;
CrossAssetContagion g_contagion;
HFMicrostructure g_hfMicro;
MetaStrategy g_metaStrategy;

// Elite quant variables
QuantumPortfolio g_quantumPortfolio;
MarketImpactModel g_marketImpact;
SyntheticAsset g_synthetic;
TailRiskHedge g_tailHedge;
BehavioralSignals g_behavioral;
GeneticOptimizer g_genetic;
FractalAnalysis g_fractal;
ExtremeValueModel g_extremeValue;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("ML Mean Reversion Strategy initialized");
   
   // Initialize risk parameters
   riskParams.maxRiskPerTrade = InpRiskPerTrade;
   riskParams.maxDailyRisk = InpMaxDailyRisk;
   riskParams.maxOpenTrades = InpMaxPositions;
   riskParams.useTrailingStop = false;  // Mean reversion typically doesn't use trailing
   
   g_accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   
   // Load or initialize ML model
   if(!LoadMeanReversionModel())
   {
      InitializeMeanReversionModel();
      Print("New mean reversion model initialized");
   }
   else
   {
      Print("Mean reversion model loaded. Performance: ", mlModel.performance);
   }
   
   // Initial statistical analysis if ML enabled
   if(InpEnableML)
   {
      AnalyzeHistoricalReversions();
   }
   
   // Initialize ultra-advanced components
   InitializeUltraAdvanced();
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   SaveMeanReversionModel();
   ObjectsDeleteAll(0, "REV_");
   
   Print("ML Mean Reversion Strategy stopped");
   Print("Final performance: ", mlModel.performance, "%");
   Print("Signals analyzed: ", mlModel.signalsAnalyzed);
   Print("Average reversion rate: ", mlModel.meanReversionRate);
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
   
   // Update advanced analytics
   if(InpUseMicrostructure)
   {
      CalculateMarketMicrostructure(_Symbol, g_microstructure);
      CalculateOrderFlow(_Symbol, g_orderFlow);
      
      // Ultra-advanced microstructure
      DetectHiddenLiquidity(_Symbol, g_hiddenLiquidity);
      AnalyzeOrderBookDynamics(_Symbol, g_orderBookDynamics);
      AnalyzeHFMicrostructure(_Symbol, g_hfMicro);
   }
   
   if(InpUseMarketRegime)
      DetectMarketRegime(_Symbol, g_marketRegime);
   
   if(InpUseSentiment)
      CalculateMarketSentiment(_Symbol, g_sentiment);
   
   // Update pair correlation if pair trading enabled
   if(InpUsePairTrading && InpPairSymbol != "")
      CalculateCorrelation(_Symbol, InpPairSymbol, 100, g_pairCorrelation);
   
   // Ultra-advanced updates
   UpdateMarketMakerModel(_Symbol, PositionsTotal(), g_marketMaker);
   
   // Cross-asset contagion (if multiple symbols)
   string symbols[] = {_Symbol};  // Add more symbols as needed
   if(ArraySize(symbols) > 1)
      AnalyzeCrossAssetContagion(symbols, g_contagion);
   
   // Update meta strategy
   int regimeIndex = g_marketRegime.currentRegime == MarketRegime::REGIME_TRENDING_UP || 
                    g_marketRegime.currentRegime == MarketRegime::REGIME_TRENDING_DOWN ? 0 :
                    (g_marketRegime.currentRegime == MarketRegime::REGIME_RANGING ? 1 : 2);
   UpdateMetaStrategy(g_metaStrategy, regimeIndex);
   
   // Update Kelly position sizing
   if(InpUseKellyCriterion)
      UpdateKellyPosition();
   
   // Elite quant analysis
   UpdateEliteAnalysis();
   
   // Update statistics
   UpdateStatisticalData();
   
   // Check existing positions for exit
   ManageReversionPositions();
   
   // Get market context
   MarketContext context;
   GetMarketContext(_Symbol, context);
   
   // Advanced filtering
   if(!PassAdvancedFilters(context))
      return;
   
   // Scan for mean reversion signals
   ScanReversionSignals();
   
   // Check for trade opportunities
   if(ArraySize(currentSignals) > 0)
   {
      EvaluateReversionOpportunities(context);
   }
   
   // Update ML model periodically
   if(InpEnableML && signalCount >= InpMLUpdateFreq)
   {
      UpdateMeanReversionModel();
      signalCount = 0;
   }
}

//+------------------------------------------------------------------+
//| Update statistical data                                          |
//+------------------------------------------------------------------+
void UpdateStatisticalData()
{
   ArrayResize(statistics, 1);
   
   // Get price data
   double close[];
   ArraySetAsSeries(close, true);
   CopyClose(_Symbol, _Period, 0, InpZScorePeriod + 50, close);
   
   // Calculate rolling statistics
   statistics[0].mean = CalculateMean(close, 0, InpZScorePeriod);
   statistics[0].stdDev = CalculateStdDev(close, 0, InpZScorePeriod, statistics[0].mean);
   statistics[0].currentZScore = (close[0] - statistics[0].mean) / statistics[0].stdDev;
   
   // Calculate additional statistics
   statistics[0].skewness = CalculateSkewness(close, 0, InpZScorePeriod);
   statistics[0].kurtosis = CalculateKurtosis(close, 0, InpZScorePeriod);
   
   // Calculate mean reversion metrics
   statistics[0].halfLife = CalculateHalfLife(close, InpZScorePeriod);
   statistics[0].hurst = CalculateHurstExponent(close, InpZScorePeriod);
}

//+------------------------------------------------------------------+
//| Calculate mean                                                   |
//+------------------------------------------------------------------+
double CalculateMean(const double &data[], int start, int period)
{
   double sum = 0;
   for(int i = start; i < start + period && i < ArraySize(data); i++)
      sum += data[i];
   return sum / period;
}

//+------------------------------------------------------------------+
//| Calculate standard deviation                                     |
//+------------------------------------------------------------------+
double CalculateStdDev(const double &data[], int start, int period, double mean)
{
   double sum = 0;
   for(int i = start; i < start + period && i < ArraySize(data); i++)
      sum += MathPow(data[i] - mean, 2);
   return MathSqrt(sum / period);
}

//+------------------------------------------------------------------+
//| Calculate skewness                                              |
//+------------------------------------------------------------------+
double CalculateSkewness(const double &data[], int start, int period)
{
   double mean = CalculateMean(data, start, period);
   double stdDev = CalculateStdDev(data, start, period, mean);
   
   double sum = 0;
   for(int i = start; i < start + period && i < ArraySize(data); i++)
      sum += MathPow((data[i] - mean) / stdDev, 3);
   
   return sum / period;
}

//+------------------------------------------------------------------+
//| Calculate kurtosis                                               |
//+------------------------------------------------------------------+
double CalculateKurtosis(const double &data[], int start, int period)
{
   double mean = CalculateMean(data, start, period);
   double stdDev = CalculateStdDev(data, start, period, mean);
   
   double sum = 0;
   for(int i = start; i < start + period && i < ArraySize(data); i++)
      sum += MathPow((data[i] - mean) / stdDev, 4);
   
   return sum / period - 3;  // Excess kurtosis
}

//+------------------------------------------------------------------+
//| Calculate half-life of mean reversion                           |
//+------------------------------------------------------------------+
double CalculateHalfLife(const double &data[], int period)
{
   // Using Ornstein-Uhlenbeck process estimation
   double y[], x[];
   ArrayResize(y, period - 1);
   ArrayResize(x, period - 1);
   
   for(int i = 0; i < period - 1; i++)
   {
      y[i] = data[i] - data[i + 1];
      x[i] = data[i + 1];
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
   return -MathLog(2) / beta;
}

//+------------------------------------------------------------------+
//| Calculate Hurst exponent                                         |
//+------------------------------------------------------------------+
double CalculateHurstExponent(const double &data[], int period)
{
   // Simplified R/S analysis
   double returns[];
   ArrayResize(returns, period - 1);
   
   for(int i = 0; i < period - 1; i++)
      returns[i] = MathLog(data[i] / data[i + 1]);
   
   double mean = CalculateMean(returns, 0, period - 1);
   double cumDev = 0;
   double maxDev = 0, minDev = 0;
   
   for(int i = 0; i < period - 1; i++)
   {
      cumDev += returns[i] - mean;
      if(cumDev > maxDev) maxDev = cumDev;
      if(cumDev < minDev) minDev = cumDev;
   }
   
   double range = maxDev - minDev;
   double stdDev = CalculateStdDev(returns, 0, period - 1, mean);
   
   // Hurst = log(R/S) / log(N/2)
   return MathLog(range / stdDev) / MathLog(period / 2.0);
}

//+------------------------------------------------------------------+
//| Scan for mean reversion signals                                  |
//+------------------------------------------------------------------+
void ScanReversionSignals()
{
   ArrayResize(currentSignals, 0);
   
   // Get indicator values
   double close[], high[], low[], volume[];
   double bb_upper[], bb_middle[], bb_lower[];
   double rsi[];
   
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(volume, true);
   ArraySetAsSeries(bb_upper, true);
   ArraySetAsSeries(bb_middle, true);
   ArraySetAsSeries(bb_lower, true);
   ArraySetAsSeries(rsi, true);
   
   // Copy data
   CopyClose(_Symbol, _Period, 0, 100, close);
   CopyHigh(_Symbol, _Period, 0, 100, high);
   CopyLow(_Symbol, _Period, 0, 100, low);
   CopyTickVolume(_Symbol, _Period, 0, 100, volume);
   
   // Get indicators
   int bb_handle = iBands(_Symbol, _Period, InpBollingerPeriod, 0, InpBollingerDeviation, PRICE_CLOSE);
   int rsi_handle = iRSI(_Symbol, _Period, InpRSIPeriod, PRICE_CLOSE);
   
   CopyBuffer(bb_handle, 0, 0, 100, bb_upper);
   CopyBuffer(bb_handle, 1, 0, 100, bb_middle);
   CopyBuffer(bb_handle, 2, 0, 100, bb_lower);
   CopyBuffer(rsi_handle, 0, 0, 100, rsi);
   
   // Check trend filter if enabled
   double trendAngle = 0;
   if(InpUseTrendFilter)
   {
      trendAngle = CalculateTrendAngle(close, InpTrendPeriod);
      if(MathAbs(trendAngle) > InpMaxTrendStrength)
         return;  // Skip if trend is too strong
   }
   
   // Look for reversion patterns
   for(int i = 1; i < 20; i++)
   {
      ReversionSignal signal;
      signal.time = iTime(_Symbol, _Period, i);
      signal.price = close[i];
      signal.zScore = statistics[0].currentZScore;
      
      // Check for various reversion patterns
      ENUM_REVERSION_PATTERN pattern = DetectReversionPattern(i, close, bb_upper, bb_middle, 
                                                             bb_lower, rsi, volume);
      
      if(pattern != REV_PATTERN_NONE)
      {
         signal.pattern = pattern;
         signal.direction = DetermineReversionDirection(i, close, bb_upper, bb_lower, rsi);
         signal.deviation = CalculateDeviation(i, close, bb_middle);
         signal.targetPrice = bb_middle[i];  // Initial target is the mean
         
         // Calculate ML score if enabled
         if(InpEnableML)
         {
            signal.score = CalculateEnhancedMLScore(signal, i);
            signal.confidence = CalculateReversionConfidence(signal, i);
         }
         else
         {
            signal.score = CalculateBasicReversionScore(signal, i);
            signal.confidence = signal.score;
         }
         
         // Only keep high-scoring signals
         if(signal.score >= 60)
         {
            signal.expectedReturn = EstimateReversionReturn(signal, i);
            
            int size = ArraySize(currentSignals);
            ArrayResize(currentSignals, size + 1);
            currentSignals[size] = signal;
         }
      }
   }
   
   // Sort signals by score
   SortReversionSignalsByScore(currentSignals);
   
   // Display top signals
   DisplayReversionSignals();
}

//+------------------------------------------------------------------+
//| Detect reversion patterns                                        |
//+------------------------------------------------------------------+
ENUM_REVERSION_PATTERN DetectReversionPattern(int bar, const double &close[],
                                             const double &bb_upper[], const double &bb_middle[],
                                             const double &bb_lower[], const double &rsi[],
                                             const long &volume[])
{
   // Z-Score extreme
   if(MathAbs(statistics[0].currentZScore) >= InpZScoreEntry)
      return REV_PATTERN_ZSCORE_EXTREME;
   
   // Bollinger Band squeeze and expansion
   double bb_width = bb_upper[bar] - bb_lower[bar];
   double bb_width_prev = bb_upper[bar + 5] - bb_lower[bar + 5];
   
   if(bb_width < bb_width_prev * 0.7)  // Squeeze detected
   {
      if(close[bar] > bb_upper[bar] || close[bar] < bb_lower[bar])
         return REV_PATTERN_BOLLINGER_SQUEEZE;
   }
   
   // RSI divergence
   if(CheckRSIDivergence(bar, close, rsi))
      return REV_PATTERN_RSI_DIVERGENCE;
   
   // Volume climax
   if(CheckVolumeClimax(bar, volume, close))
      return REV_PATTERN_VOLUME_CLIMAX;
   
   // Double top/bottom
   if(CheckDoubleTopBottom(bar, high, low))
      return REV_PATTERN_DOUBLE_TOP_BOTTOM;
   
   // Channel rejection
   if(CheckChannelRejection(bar, high, low, close))
      return REV_PATTERN_CHANNEL_REJECTION;
   
   // Volatility mean reversion
   if(CheckVolatilityMeanReversion(bar))
      return REV_PATTERN_VOLATILITY_MEAN_REV;
   
   return REV_PATTERN_NONE;
}

//+------------------------------------------------------------------+
//| Check RSI divergence for mean reversion                         |
//+------------------------------------------------------------------+
bool CheckRSIDivergence(int bar, const double &close[], const double &rsi[])
{
   // Look for price extremes with RSI divergence
   if(rsi[bar] > InpRSIOverbought)
   {
      // Check if price is making higher high but RSI is not
      for(int i = bar + 5; i < bar + 20 && i < ArraySize(close); i++)
      {
         if(close[i] < close[bar] && rsi[i] > rsi[bar])
            return true;  // Bearish divergence
      }
   }
   else if(rsi[bar] < InpRSIOversold)
   {
      // Check if price is making lower low but RSI is not
      for(int i = bar + 5; i < bar + 20 && i < ArraySize(close); i++)
      {
         if(close[i] > close[bar] && rsi[i] < rsi[bar])
            return true;  // Bullish divergence
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Check volume climax                                              |
//+------------------------------------------------------------------+
bool CheckVolumeClimax(int bar, const long &volume[], const double &close[])
{
   // Calculate average volume
   double avgVolume = 0;
   for(int i = bar + 5; i < bar + 25 && i < ArraySize(volume); i++)
      avgVolume += volume[i];
   avgVolume /= 20;
   
   // Check for volume spike with price extreme
   if(volume[bar] > avgVolume * 2.5)
   {
      // Check if price is at extreme (using Bollinger Bands or Z-score)
      if(MathAbs(statistics[0].currentZScore) > 1.5)
         return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Check double top/bottom                                          |
//+------------------------------------------------------------------+
bool CheckDoubleTopBottom(int bar, const double &high[], const double &low[])
{
   double tolerance = 10 * _Point;  // Price tolerance for double top/bottom
   
   // Look for double top
   for(int i = bar + 10; i < bar + 30 && i < ArraySize(high); i++)
   {
      if(MathAbs(high[bar] - high[i]) < tolerance)
      {
         // Confirm with lower high between
         bool hasLowerHigh = false;
         for(int j = bar + 1; j < i; j++)
         {
            if(high[j] < high[bar] - tolerance * 2)
            {
               hasLowerHigh = true;
               break;
            }
         }
         if(hasLowerHigh) return true;
      }
   }
   
   // Look for double bottom
   for(int i = bar + 10; i < bar + 30 && i < ArraySize(low); i++)
   {
      if(MathAbs(low[bar] - low[i]) < tolerance)
      {
         // Confirm with higher low between
         bool hasHigherLow = false;
         for(int j = bar + 1; j < i; j++)
         {
            if(low[j] > low[bar] + tolerance * 2)
            {
               hasHigherLow = true;
               break;
            }
         }
         if(hasHigherLow) return true;
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Check channel rejection                                          |
//+------------------------------------------------------------------+
bool CheckChannelRejection(int bar, const double &high[], const double &low[], 
                          const double &close[])
{
   // Calculate dynamic channel using recent highs/lows
   double upperChannel = 0, lowerChannel = 0;
   int channelPeriod = 20;
   
   // Find highest high and lowest low
   for(int i = bar; i < bar + channelPeriod && i < ArraySize(high); i++)
   {
      if(i == bar || high[i] > upperChannel)
         upperChannel = high[i];
      if(i == bar || low[i] < lowerChannel)
         lowerChannel = low[i];
   }
   
   // Check for rejection from channel boundaries
   double channelWidth = upperChannel - lowerChannel;
   double upperZone = upperChannel - channelWidth * 0.1;
   double lowerZone = lowerChannel + channelWidth * 0.1;
   
   // Upper channel rejection
   if(high[bar] >= upperZone && close[bar] < high[bar] - (high[bar] - low[bar]) * 0.3)
      return true;
   
   // Lower channel rejection
   if(low[bar] <= lowerZone && close[bar] > low[bar] + (high[bar] - low[bar]) * 0.3)
      return true;
   
   return false;
}

//+------------------------------------------------------------------+
//| Check volatility mean reversion                                 |
//+------------------------------------------------------------------+
bool CheckVolatilityMeanReversion(int bar)
{
   // Get ATR values
   double atr[];
   ArraySetAsSeries(atr, true);
   int atr_handle = iATR(_Symbol, _Period, 14);
   CopyBuffer(atr_handle, 0, bar, 30, atr);
   
   // Calculate ATR statistics
   double meanATR = CalculateMean(atr, 5, 20);
   double currentATR = atr[0];
   
   // Check if volatility is extreme
   if(currentATR > meanATR * 2.0 || currentATR < meanATR * 0.5)
      return true;
   
   return false;
}

//+------------------------------------------------------------------+
//| Determine reversion direction                                    |
//+------------------------------------------------------------------+
int DetermineReversionDirection(int bar, const double &close[],
                               const double &bb_upper[], const double &bb_lower[],
                               const double &rsi[])
{
   // Multiple confirmation approach
   int votes = 0;
   
   // Z-Score indication
   if(statistics[0].currentZScore > InpZScoreEntry)
      votes--;  // Overbought, expect down
   else if(statistics[0].currentZScore < -InpZScoreEntry)
      votes++;  // Oversold, expect up
   
   // Bollinger Bands indication
   if(close[bar] > bb_upper[bar])
      votes--;
   else if(close[bar] < bb_lower[bar])
      votes++;
   
   // RSI indication
   if(rsi[bar] > InpRSIOverbought)
      votes--;
   else if(rsi[bar] < InpRSIOversold)
      votes++;
   
   return votes > 0 ? 1 : (votes < 0 ? -1 : 0);
}

//+------------------------------------------------------------------+
//| Calculate deviation from mean                                    |
//+------------------------------------------------------------------+
double CalculateDeviation(int bar, const double &close[], const double &bb_middle[])
{
   return MathAbs(close[bar] - bb_middle[bar]) / bb_middle[bar] * 100;
}

//+------------------------------------------------------------------+
//| Calculate basic reversion score                                  |
//+------------------------------------------------------------------+
double CalculateBasicReversionScore(const ReversionSignal &signal, int bar)
{
   double score = 0;
   
   // Z-Score component (30%)
   double zScoreComponent = MathMin(MathAbs(signal.zScore) / 3.0, 1.0) * 30;
   score += zScoreComponent;
   
   // Deviation component (30%)
   double deviationComponent = MathMin(signal.deviation / 5.0, 1.0) * 30;
   score += deviationComponent;
   
   // Pattern strength (20%)
   double patternStrength = GetPatternStrength(signal.pattern) * 20;
   score += patternStrength;
   
   // Statistical favorability (20%)
   if(statistics[0].hurst < 0.5)  // Mean reverting market
      score += 20;
   else if(statistics[0].hurst < 0.6)
      score += 10;
   
   return score;
}

//+------------------------------------------------------------------+
//| Calculate ML reversion score                                     |
//+------------------------------------------------------------------+
double CalculateMLReversionScore(const ReversionSignal &signal, int bar)
{
   double score = mlModel.bias;
   
   // Get enhanced features
   EnhancedFeatures features;
   CalculateEnhancedFeatures(_Symbol, features);
   
   // Statistical features
   score += mlModel.weights[0] * NormalizeValue(MathAbs(signal.zScore), 0, 4);
   score += mlModel.weights[1] * NormalizeValue(signal.deviation, 0, 10);
   score += mlModel.weights[2] * statistics[0].skewness;
   score += mlModel.weights[3] * statistics[0].kurtosis;
   score += mlModel.weights[4] * NormalizeValue(statistics[0].halfLife, 0, 50);
   score += mlModel.weights[5] * statistics[0].hurst;
   
   // Pattern-specific weights
   score += mlModel.patternWeights[signal.pattern];
   
   // Market context features
   score += mlModel.weights[10] * features.marketRegime;
   score += mlModel.weights[11] * features.relativeVolume;
   score += mlModel.weights[12] * features.bollingerWidth;
   score += mlModel.weights[13] * features.pricePosition;
   
   // Technical features
   score += mlModel.weights[15] * features.rsiDivergence;
   score += mlModel.weights[16] * features.volumeProfile;
   score += mlModel.weights[17] * features.atrPosition;
   
   // Time-based features
   MqlDateTime dt;
   TimeToStruct(signal.time, dt);
   score += mlModel.weights[20] * NormalizeValue(dt.hour, 0, 23);
   score += mlModel.weights[21] * NormalizeValue(dt.day_of_week, 1, 5);
   
   // Historical performance of similar setups
   score += mlModel.weights[25] * mlModel.meanReversionRate;
   
   // Apply sigmoid activation
   return 100 / (1 + MathExp(-score));
}

//+------------------------------------------------------------------+
//| Calculate reversion confidence                                   |
//+------------------------------------------------------------------+
double CalculateReversionConfidence(const ReversionSignal &signal, int bar)
{
   double confidence = 0;
   double weights = 0;
   
   // Statistical confidence
   if(statistics[0].hurst < 0.5)
   {
      confidence += (0.5 - statistics[0].hurst) * 200;
      weights += 1;
   }
   
   // Half-life confidence (faster reversion = higher confidence)
   if(statistics[0].halfLife > 0 && statistics[0].halfLife < 30)
   {
      confidence += (30 - statistics[0].halfLife) / 30 * 100;
      weights += 1;
   }
   
   // Pattern reliability from ML model
   if(mlModel.signalsAnalyzed > 100)
   {
      confidence += mlModel.performance;
      weights += 1;
   }
   
   // Market condition confidence
   MarketContext context;
   GetMarketContext(_Symbol, context);
   
   if(!context.isHighVolatility)
   {
      confidence += 20;
      weights += 0.2;
   }
   
   return weights > 0 ? confidence / weights : 50;
}

//+------------------------------------------------------------------+
//| Get pattern strength                                             |
//+------------------------------------------------------------------+
double GetPatternStrength(ENUM_REVERSION_PATTERN pattern)
{
   switch(pattern)
   {
      case REV_PATTERN_ZSCORE_EXTREME: return 0.9;
      case REV_PATTERN_BOLLINGER_SQUEEZE: return 0.8;
      case REV_PATTERN_RSI_DIVERGENCE: return 0.85;
      case REV_PATTERN_VOLUME_CLIMAX: return 0.75;
      case REV_PATTERN_DOUBLE_TOP_BOTTOM: return 0.8;
      case REV_PATTERN_CHANNEL_REJECTION: return 0.7;
      case REV_PATTERN_VOLATILITY_MEAN_REV: return 0.65;
      case REV_PATTERN_CORRELATION_BREAK: return 0.7;
      default: return 0.5;
   }
}

//+------------------------------------------------------------------+
//| Estimate reversion return                                        |
//+------------------------------------------------------------------+
double EstimateReversionReturn(const ReversionSignal &signal, int bar)
{
   // Calculate expected move to mean
   double currentPrice = signal.price;
   double targetPrice = signal.targetPrice;
   double expectedMove = MathAbs(targetPrice - currentPrice) / currentPrice * 100;
   
   // Adjust based on historical reversion rate
   if(mlModel.meanReversionRate > 0)
      expectedMove *= mlModel.meanReversionRate;
   
   // Consider partial reversion (rarely goes all the way to mean)
   expectedMove *= 0.7;  // Expect 70% of the move
   
   return expectedMove;
}

//+------------------------------------------------------------------+
//| Sort reversion signals by score                                  |
//+------------------------------------------------------------------+
void SortReversionSignalsByScore(ReversionSignal &signals[])
{
   int n = ArraySize(signals);
   
   for(int i = 0; i < n - 1; i++)
   {
      for(int j = 0; j < n - i - 1; j++)
      {
         if(signals[j].score < signals[j + 1].score)
         {
            ReversionSignal temp = signals[j];
            signals[j] = signals[j + 1];
            signals[j + 1] = temp;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Display reversion signals on chart                               |
//+------------------------------------------------------------------+
void DisplayReversionSignals()
{
   ObjectsDeleteAll(0, "REV_");
   
   int maxSignals = MathMin(ArraySize(currentSignals), 5);
   
   for(int i = 0; i < maxSignals; i++)
   {
      string prefix = "REV_" + IntegerToString(i) + "_";
      ReversionSignal signal = currentSignals[i];
      
      // Draw arrow
      string arrowName = prefix + "Arrow";
      ObjectCreate(0, arrowName, OBJ_ARROW, 0, signal.time, signal.price);
      
      if(signal.direction > 0)
      {
         ObjectSetInteger(0, arrowName, OBJPROP_ARROWCODE, 233); // Up arrow
         ObjectSetInteger(0, arrowName, OBJPROP_COLOR, clrAqua);
      }
      else
      {
         ObjectSetInteger(0, arrowName, OBJPROP_ARROWCODE, 234); // Down arrow
         ObjectSetInteger(0, arrowName, OBJPROP_COLOR, clrOrange);
      }
      
      ObjectSetInteger(0, arrowName, OBJPROP_WIDTH, 2);
      
      // Add label with pattern info
      string labelName = prefix + "Label";
      string patternText = GetReversionPatternName(signal.pattern);
      string labelText = StringFormat("%s\nZ-Score: %.2f\nScore: %.1f\nConf: %.1f%%", 
                                     patternText, signal.zScore, signal.score, signal.confidence);
      
      double labelPrice = signal.direction > 0 ? 
                         signal.price - 40 * _Point : 
                         signal.price + 40 * _Point;
      
      ObjectCreate(0, labelName, OBJ_TEXT, 0, signal.time, labelPrice);
      ObjectSetString(0, labelName, OBJPROP_TEXT, labelText);
      ObjectSetInteger(0, labelName, OBJPROP_COLOR, clrYellow);
      ObjectSetInteger(0, labelName, OBJPROP_FONTSIZE, 8);
      ObjectSetString(0, labelName, OBJPROP_FONT, "Arial");
      
      // Draw target line
      string targetName = prefix + "Target";
      ObjectCreate(0, targetName, OBJ_HLINE, 0, 0, signal.targetPrice);
      ObjectSetInteger(0, targetName, OBJPROP_COLOR, clrGray);
      ObjectSetInteger(0, targetName, OBJPROP_STYLE, STYLE_DOT);
   }
}

//+------------------------------------------------------------------+
//| Get reversion pattern name                                       |
//+------------------------------------------------------------------+
string GetReversionPatternName(ENUM_REVERSION_PATTERN pattern)
{
   switch(pattern)
   {
      case REV_PATTERN_ZSCORE_EXTREME: return "Z-Score Extreme";
      case REV_PATTERN_BOLLINGER_SQUEEZE: return "BB Squeeze";
      case REV_PATTERN_RSI_DIVERGENCE: return "RSI Divergence";
      case REV_PATTERN_VOLUME_CLIMAX: return "Volume Climax";
      case REV_PATTERN_DOUBLE_TOP_BOTTOM: return "Double Top/Bot";
      case REV_PATTERN_CHANNEL_REJECTION: return "Channel Reject";
      case REV_PATTERN_VOLATILITY_MEAN_REV: return "Vol Mean Rev";
      case REV_PATTERN_CORRELATION_BREAK: return "Correlation Break";
      default: return "Unknown";
   }
}

//+------------------------------------------------------------------+
//| Evaluate reversion opportunities                                 |
//+------------------------------------------------------------------+
void EvaluateReversionOpportunities(const MarketContext &context)
{
   // Check if we can take new trades
   if(PositionsTotal() >= riskParams.maxOpenTrades)
      return;
   
   // Get the best signal
   if(ArraySize(currentSignals) == 0)
      return;
   
   ReversionSignal bestSignal = currentSignals[0];
   
   // Additional validation for mean reversion
   if(!ValidateReversionSetup(bestSignal, context))
      return;
   
   // Check if signal is recent enough (within last 3 bars for reversion)
   int signalBar = iBarShift(_Symbol, _Period, bestSignal.time);
   if(signalBar > 3)
      return;
   
   // Execute trade with enhanced features
   ExecuteReversionTradeEnhanced(bestSignal);
}

//+------------------------------------------------------------------+
//| Validate reversion setup                                         |
//+------------------------------------------------------------------+
bool ValidateReversionSetup(const ReversionSignal &signal, const MarketContext &context)
{
   // Don't trade mean reversion in trending markets
   if(InpUseTrendFilter && context.trendStrength > 0.7)
      return false;
   
   // Require minimum deviation
   if(signal.deviation < 1.5)
      return false;
   
   // Check volume if filter enabled
   if(InpUseVolumeFilter)
   {
      double volume[];
      ArraySetAsSeries(volume, true);
      CopyTickVolume(_Symbol, _Period, 0, 20, volume);
      
      double avgVolume = CalculateMean(volume, 5, 15);
      if(volume[0] < avgVolume * InpMinVolumeRatio)
         return false;
   }
   
   // Multi-timeframe confirmation
   if(InpUseMTFConfirmation)
   {
      if(!CheckMTFConfirmation(signal))
         return false;
   }
   
   // Validate based on market conditions
   if(context.isLowLiquidity || context.isHighImpactNews)
      return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Execute reversion trade                                          |
//+------------------------------------------------------------------+
void ExecuteReversionTrade(const ReversionSignal &signal)
{
   double entryPrice = signal.direction > 0 ? 
                      SymbolInfoDouble(_Symbol, SYMBOL_ASK) : 
                      SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // Calculate stop loss (wider for mean reversion)
   double atr = 0;
   double atr_array[];
   ArraySetAsSeries(atr_array, true);
   int atr_handle = iATR(_Symbol, _Period, 14);
   CopyBuffer(atr_handle, 0, 0, 1, atr_array);
   atr = atr_array[0];
   
   double stopDistance = atr * InpATRMultiplier;
   double stopLoss = signal.direction > 0 ?
                    entryPrice - stopDistance :
                    entryPrice + stopDistance;
   
   // Calculate take profit (target is mean)
   double takeProfit = signal.targetPrice;
   
   // Adjust TP if it's too close
   double tpDistance = MathAbs(takeProfit - entryPrice);
   if(tpDistance < stopDistance * 0.5)
   {
      takeProfit = signal.direction > 0 ?
                   entryPrice + stopDistance :
                   entryPrice - stopDistance;
   }
   
   // Calculate position size based on volatility
   double lotSize = CalculateDynamicPositionSize(_Symbol, stopDistance, riskParams);
   
   // Scale position if enabled
   if(InpScalePositions && signal.confidence < 80)
   {
      lotSize *= (signal.confidence / 100);
   }
   
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
   request.comment = "ML_MeanRev_" + GetReversionPatternName(signal.pattern);
   request.magic = 12345;
   
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         Print("Mean reversion trade executed: ", GetReversionPatternName(signal.pattern),
               " ", signal.direction > 0 ? "BUY" : "SELL",
               " Score: ", signal.score, " Confidence: ", signal.confidence);
         
         mlModel.signalsAnalyzed++;
         signalCount++;
      }
   }
}

//+------------------------------------------------------------------+
//| Manage reversion positions                                       |
//+------------------------------------------------------------------+
void ManageReversionPositions()
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      
      string comment = PositionGetString(POSITION_COMMENT);
      
      if(StringFind(comment, "ML_MeanRev") >= 0)
      {
         double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
         double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
         int posType = (int)PositionGetInteger(POSITION_TYPE);
         
         // Check for early exit based on z-score
         double currentZScore = statistics[0].currentZScore;
         
         // Exit if z-score has reversed significantly
         if(posType == POSITION_TYPE_BUY && currentZScore > InpZScoreExit)
         {
            CloseReversionPosition(ticket, "Z-Score reversal");
         }
         else if(posType == POSITION_TYPE_SELL && currentZScore < -InpZScoreExit)
         {
            CloseReversionPosition(ticket, "Z-Score reversal");
         }
         
         // Partial profit taking
         if(InpScalePositions)
         {
            double profit = posType == POSITION_TYPE_BUY ?
                          currentPrice - openPrice : openPrice - currentPrice;
            double profitPercent = profit / openPrice * 100;
            
            if(profitPercent > 1.0 && PositionGetDouble(POSITION_VOLUME) > 0.01)
            {
               // Take partial profit
               double partialVolume = PositionGetDouble(POSITION_VOLUME) * 0.5;
               ClosePartialPosition(ticket, partialVolume);
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Close reversion position                                         |
//+------------------------------------------------------------------+
void CloseReversionPosition(ulong ticket, string reason)
{
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   if(!PositionSelectByTicket(ticket)) return;
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = PositionGetString(POSITION_SYMBOL);
   request.volume = PositionGetDouble(POSITION_VOLUME);
   request.type = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ?
                 ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   request.price = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ?
                  SymbolInfoDouble(request.symbol, SYMBOL_BID) :
                  SymbolInfoDouble(request.symbol, SYMBOL_ASK);
   request.position = ticket;
   request.comment = "Close: " + reason;
   
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         Print("Position closed: ", reason);
      }
   }
}

//+------------------------------------------------------------------+
//| Close partial position                                           |
//+------------------------------------------------------------------+
void ClosePartialPosition(ulong ticket, double volume)
{
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   if(!PositionSelectByTicket(ticket)) return;
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = PositionGetString(POSITION_SYMBOL);
   request.volume = NormalizeDouble(volume, 2);
   request.type = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ?
                 ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   request.price = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ?
                  SymbolInfoDouble(request.symbol, SYMBOL_BID) :
                  SymbolInfoDouble(request.symbol, SYMBOL_ASK);
   request.position = ticket;
   request.comment = "Partial close";
   
   OrderSend(request, result);
}

//+------------------------------------------------------------------+
//| Calculate trend angle                                            |
//+------------------------------------------------------------------+
double CalculateTrendAngle(const double &close[], int period)
{
   // Linear regression to find trend angle
   double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
   
   for(int i = 0; i < period && i < ArraySize(close); i++)
   {
      sumX += i;
      sumY += close[i];
      sumXY += i * close[i];
      sumX2 += i * i;
   }
   
   double slope = (period * sumXY - sumX * sumY) / (period * sumX2 - sumX * sumX);
   
   // Convert to angle in degrees
   return MathArctan(slope) * 180 / M_PI;
}

//+------------------------------------------------------------------+
//| Check multi-timeframe confirmation                               |
//+------------------------------------------------------------------+
bool CheckMTFConfirmation(const ReversionSignal &signal)
{
   // Check HTF1 confirmation
   double htf1_close[];
   double htf1_bb_upper[], htf1_bb_middle[], htf1_bb_lower[];
   double htf1_rsi[];
   
   ArraySetAsSeries(htf1_close, true);
   ArraySetAsSeries(htf1_bb_upper, true);
   ArraySetAsSeries(htf1_bb_middle, true);
   ArraySetAsSeries(htf1_bb_lower, true);
   ArraySetAsSeries(htf1_rsi, true);
   
   // Copy HTF1 data
   CopyClose(_Symbol, InpHTF1, 0, 50, htf1_close);
   
   int htf1_bb = iBands(_Symbol, InpHTF1, InpBollingerPeriod, 0, InpBollingerDeviation, PRICE_CLOSE);
   int htf1_rsi_handle = iRSI(_Symbol, InpHTF1, InpRSIPeriod, PRICE_CLOSE);
   
   CopyBuffer(htf1_bb, 0, 0, 50, htf1_bb_upper);
   CopyBuffer(htf1_bb, 1, 0, 50, htf1_bb_middle);
   CopyBuffer(htf1_bb, 2, 0, 50, htf1_bb_lower);
   CopyBuffer(htf1_rsi_handle, 0, 0, 50, htf1_rsi);
   
   // Calculate HTF1 z-score
   double htf1_mean = CalculateMean(htf1_close, 0, InpZScorePeriod);
   double htf1_stdDev = CalculateStdDev(htf1_close, 0, InpZScorePeriod, htf1_mean);
   double htf1_zScore = (htf1_close[0] - htf1_mean) / htf1_stdDev;
   
   // HTF1 should show similar extreme condition
   bool htf1_confirmed = false;
   
   if(signal.direction > 0)  // Looking for oversold confirmation
   {
      htf1_confirmed = (htf1_zScore < -1.0 || htf1_close[0] < htf1_bb_lower[0] || htf1_rsi[0] < 40);
   }
   else  // Looking for overbought confirmation
   {
      htf1_confirmed = (htf1_zScore > 1.0 || htf1_close[0] > htf1_bb_upper[0] || htf1_rsi[0] > 60);
   }
   
   if(!htf1_confirmed) return false;
   
   // Check HTF2 confirmation if different from HTF1
   if(InpHTF2 != InpHTF1)
   {
      double htf2_close[];
      double htf2_rsi[];
      
      ArraySetAsSeries(htf2_close, true);
      ArraySetAsSeries(htf2_rsi, true);
      
      CopyClose(_Symbol, InpHTF2, 0, 50, htf2_close);
      
      int htf2_rsi_handle = iRSI(_Symbol, InpHTF2, InpRSIPeriod, PRICE_CLOSE);
      CopyBuffer(htf2_rsi_handle, 0, 0, 50, htf2_rsi);
      
      // HTF2 trend check - should not be strongly trending against us
      double htf2_trend = CalculateTrendAngle(htf2_close, 20);
      
      if(signal.direction > 0 && htf2_trend < -20)  // Strong downtrend on HTF2
         return false;
      else if(signal.direction < 0 && htf2_trend > 20)  // Strong uptrend on HTF2
         return false;
      
      // HTF2 RSI should not be at opposite extreme
      if(signal.direction > 0 && htf2_rsi[0] > 70)
         return false;
      else if(signal.direction < 0 && htf2_rsi[0] < 30)
         return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Analyze historical reversions for ML training                    |
//+------------------------------------------------------------------+
void AnalyzeHistoricalReversions()
{
   Print("Analyzing historical mean reversions...");
   
   int reversions_found = 0;
   double total_return = 0;
   double total_time = 0;
   
   double close[];
   ArraySetAsSeries(close, true);
   CopyClose(_Symbol, _Period, 0, 1000, close);
   
   for(int i = 100; i < 900; i++)
   {
      // Calculate historical z-score
      double mean = CalculateMean(close, i, InpZScorePeriod);
      double stdDev = CalculateStdDev(close, i, InpZScorePeriod, mean);
      double zScore = (close[i] - mean) / stdDev;
      
      // Check if extreme z-score reverted
      if(MathAbs(zScore) > InpZScoreEntry)
      {
         // Find reversion point
         for(int j = i - 1; j > i - 50 && j >= 0; j--)
         {
            double futureZScore = (close[j] - mean) / stdDev;
            
            if(MathAbs(futureZScore) < InpZScoreExit)
            {
               reversions_found++;
               double return_pct = MathAbs(close[j] - close[i]) / close[i] * 100;
               total_return += return_pct;
               total_time += (i - j);
               
               if(zScore > 0 && close[j] < close[i])
                  mlModel.successfulSignals++;
               else if(zScore < 0 && close[j] > close[i])
                  mlModel.successfulSignals++;
               
               mlModel.signalsAnalyzed++;
               break;
            }
         }
      }
   }
   
   if(reversions_found > 0)
   {
      mlModel.avgReturn = total_return / reversions_found;
      mlModel.performance = (double)mlModel.successfulSignals / mlModel.signalsAnalyzed * 100;
      mlModel.meanReversionRate = (double)reversions_found / mlModel.signalsAnalyzed;
      mlModel.avgHoldingPeriod = total_time / reversions_found;
   }
   
   Print("Historical analysis complete. Reversions found: ", reversions_found);
   Print("Success rate: ", mlModel.performance, "%");
   Print("Average holding period: ", mlModel.avgHoldingPeriod, " bars");
}

//+------------------------------------------------------------------+
//| Update mean reversion model based on recent trades               |
//+------------------------------------------------------------------+
void UpdateMeanReversionModel()
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
         
         if(StringFind(comment, "ML_MeanRev") >= 0)
         {
            double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
            totalReturn += profit;
            tradeCount++;
            
            if(profit > 0)
            {
               mlModel.successfulSignals++;
               
               // Update pattern weights based on success
               string patternName = comment;
               for(int p = 0; p < 10; p++)
               {
                  if(StringFind(patternName, GetReversionPatternName((ENUM_REVERSION_PATTERN)p)) >= 0)
                  {
                     mlModel.patternWeights[p] *= 1.02;  // Increase weight for successful patterns
                     break;
                  }
               }
            }
            else
            {
               // Decrease weight for unsuccessful patterns
               string patternName = comment;
               for(int p = 0; p < 10; p++)
               {
                  if(StringFind(patternName, GetReversionPatternName((ENUM_REVERSION_PATTERN)p)) >= 0)
                  {
                     mlModel.patternWeights[p] *= 0.98;
                     break;
                  }
               }
            }
         }
      }
      
      // Update performance metrics
      if(tradeCount > 0)
      {
         double winRate = (double)mlModel.successfulSignals / mlModel.signalsAnalyzed;
         mlModel.performance = winRate * 100;
         
         // Calculate Sharpe ratio
         double returns[];
         ArrayResize(returns, tradeCount);
         // Simplified Sharpe calculation
         mlModel.sharpeRatio = totalReturn > 0 ? 
                              (totalReturn / g_accountBalance) / MathSqrt(tradeCount) : 0;
      }
   }
   
   mlModel.lastUpdate = TimeCurrent();
   SaveMeanReversionModel();
}

//+------------------------------------------------------------------+
//| Initialize mean reversion model                                  |
//+------------------------------------------------------------------+
void InitializeMeanReversionModel()
{
   // Initialize weights with small random values
   for(int i = 0; i < 50; i++)
      mlModel.weights[i] = (MathRand() / 32767.0 - 0.5) * 0.1;
   
   // Initialize pattern weights
   for(int i = 0; i < 10; i++)
      mlModel.patternWeights[i] = 0.5;
   
   mlModel.bias = 0;
   mlModel.performance = 0;
   mlModel.signalsAnalyzed = 0;
   mlModel.successfulSignals = 0;
   mlModel.avgReturn = 0;
   mlModel.sharpeRatio = 0;
   mlModel.lastUpdate = TimeCurrent();
   mlModel.meanReversionRate = 0.7;  // Initial assumption
   mlModel.falseSignalRate = 0.3;
   mlModel.avgHoldingPeriod = 10;
}

//+------------------------------------------------------------------+
//| Save/Load mean reversion model                                   |
//+------------------------------------------------------------------+
bool SaveMeanReversionModel()
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

bool LoadMeanReversionModel()
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

//+------------------------------------------------------------------+
//| Normalize value to 0-1 range                                     |
//+------------------------------------------------------------------+
double NormalizeValue(double value, double min, double max)
{
   if(max == min) return 0.5;
   return MathMax(0, MathMin(1, (value - min) / (max - min)));
}

//+------------------------------------------------------------------+
//| Pass advanced filters                                            |
//+------------------------------------------------------------------+
bool PassAdvancedFilters(const MarketContext &context)
{
   // Basic context check
   if(!context.isValidSession || context.isHighImpactNews)
      return false;
   
   // Market regime filter
   if(InpUseMarketRegime)
   {
      // Avoid mean reversion in strong trending regimes
      if(g_marketRegime.currentRegime == MarketRegime::REGIME_TRENDING_UP ||
         g_marketRegime.currentRegime == MarketRegime::REGIME_TRENDING_DOWN)
      {
         if(g_marketRegime.regimeStrength > 0.8)
            return false;
      }
      
      // Prefer ranging or reversal regimes
      if(g_marketRegime.currentRegime != MarketRegime::REGIME_RANGING &&
         g_marketRegime.currentRegime != MarketRegime::REGIME_REVERSAL &&
         g_marketRegime.currentRegime != MarketRegime::REGIME_VOLATILE)
      {
         if(g_marketRegime.transitionProbability < 0.6)
            return false;
      }
   }
   
   // Microstructure filter
   if(InpUseMicrostructure)
   {
      // Avoid toxic markets
      if(g_microstructure.toxicity > 70)
         return false;
      
      // Check liquidity
      if(g_microstructure.liquidityScore < 10)
         return false;
      
      // Check spread
      if(g_microstructure.spreadPercentage > 0.1)  // 0.1% spread threshold
         return false;
      
      // Ultra-advanced filters
      
      // Hidden liquidity checks
      if(g_hiddenLiquidity.liquidityMirage > 0.7)
         return false;  // Too much fake liquidity
      
      if(g_hiddenLiquidity.darkPoolActivity > 0.8)
         return false;  // Too much hidden activity
      
      // Order book manipulation checks
      if(g_orderBookDynamics.spoofingScore > 0.5)
         return false;  // Spoofing detected
      
      if(g_orderBookDynamics.layeringDetected > 0)
         return false;  // Layering detected
      
      if(g_orderBookDynamics.quoteStuffing > 0.3)
         return false;  // Quote stuffing
      
      // HF competition checks
      if(g_hfMicro.competitionIntensity > 0.8)
         return false;  // Too much HFT competition
      
      if(g_hfMicro.latencyAdvantage > 50)
         return false;  // We're too slow
      
      // Market maker model checks
      if(g_marketMaker.adverseSelection > 5)
         return false;  // High adverse selection
      
      if(g_marketMaker.toxicFlow > 0.7)
         return false;  // Toxic flow detected
      
      if(g_marketMaker.informedTraderProb > 0.3)
         return false;  // Too many informed traders
      
      // Cross-asset contagion checks
      if(g_contagion.systemicRisk > 0.7)
         return false;  // High systemic risk
      
      if(g_contagion.cascadeProbability > 0.5)
         return false;  // Risk of contagion cascade
   }
   
   // Sentiment filter
   if(InpUseSentiment)
   {
      // Extreme VIX levels
      if(g_sentiment.vix > 40 || g_sentiment.vix < 10)
         return false;
   }
   
   // Meta strategy weight check
   if(g_metaStrategy.meanReversionWeight < 0.1)
      return false;  // Meta strategy doesn't favor mean reversion
   
   // Elite quant filters
   
   // Behavioral extremes
   if(g_behavioral.euphoria > 80 || g_behavioral.panic > 80)
      return false;  // Extreme sentiment
   
   if(g_behavioral.herding > 70)
      return false;  // Too much herding
   
   // Fractal analysis
   if(g_fractal.hurstExponent > 0.65)
      return false;  // Strong trend persistence
   
   if(g_fractal.multifractalWidth > 0.8)
      return false;  // Too complex/unstable
   
   // Extreme value risk
   if(g_extremeValue.varExtreme > 0.05)
      return false;  // Tail risk too high
   
   if(g_extremeValue.extremalIndex < 0.5)
      return false;  // Clustering of extremes
   
   // Quantum coherence check
   if(g_quantumPortfolio.coherence < 0.3)
      return false;  // Low quantum coherence
   
   return true;
}

//+------------------------------------------------------------------+
//| Update Kelly position sizing                                     |
//+------------------------------------------------------------------+
void UpdateKellyPosition()
{
   // Calculate win rate and avg win/loss from history
   if(HistorySelect(0, TimeCurrent()))
   {
      int wins = 0, losses = 0;
      double totalWin = 0, totalLoss = 0;
      
      for(int i = 0; i < HistoryDealsTotal(); i++)
      {
         ulong ticket = HistoryDealGetTicket(i);
         double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
         
         if(profit > 0)
         {
            wins++;
            totalWin += profit;
         }
         else if(profit < 0)
         {
            losses++;
            totalLoss += MathAbs(profit);
         }
      }
      
      if(wins + losses > 30)  // Need minimum sample size
      {
         double winRate = (double)wins / (wins + losses);
         double avgWin = wins > 0 ? totalWin / wins : 0;
         double avgLoss = losses > 0 ? totalLoss / losses : 1;
         
         CalculateKellyPosition(winRate, avgWin, avgLoss, g_accountBalance, g_kellyPosition);
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate enhanced ML score with advanced features               |
//+------------------------------------------------------------------+
double CalculateEnhancedMLScore(const ReversionSignal &signal, int bar)
{
   // Use neural network if available
   double score = CalculateNeuralNetworkScore(signal);
   
   // Fallback to traditional ML if neural network fails
   if(score == 0)
      score = CalculateMLReversionScore(signal, bar);
   
   // Regime adjustment
   if(InpUseMarketRegime)
   {
      switch(g_marketRegime.currentRegime)
      {
         case MarketRegime::REGIME_RANGING:
            score *= 1.2;  // Boost in ranging markets
            break;
         case MarketRegime::REGIME_TRENDING_UP:
         case MarketRegime::REGIME_TRENDING_DOWN:
            score *= 0.8;  // Reduce in trending markets
            break;
         case MarketRegime::REGIME_REVERSAL:
            score *= 1.1;  // Slight boost at reversals
            break;
      }
   }
   
   // Microstructure adjustment
   if(InpUseMicrostructure)
   {
      // Order flow imbalance confirmation
      if(signal.direction > 0 && g_microstructure.orderFlowImbalance < -0.3)
         score *= 1.15;  // Oversold with selling pressure
      else if(signal.direction < 0 && g_microstructure.orderFlowImbalance > 0.3)
         score *= 1.15;  // Overbought with buying pressure
      
      // Toxicity penalty
      score *= (1 - g_microstructure.toxicity / 200);
   }
   
   // Sentiment adjustment
   if(InpUseSentiment)
   {
      // VIX consideration
      if(g_sentiment.vix > 20 && g_sentiment.vix < 30)
         score *= 1.1;  // Moderate volatility good for mean reversion
      
      // Smart money alignment
      if(signal.direction * g_sentiment.smartMoney > 0)
         score *= 0.9;  // Fade smart money for mean reversion
   }
   
   // Order flow adjustment
   if(InpUseMicrostructure && g_orderFlow.poc > 0)
   {
      double priceVsPOC = (signal.price - g_orderFlow.poc) / g_orderFlow.poc * 100;
      
      // Boost score if far from POC
      if(MathAbs(priceVsPOC) > 1.0)
         score *= (1 + MathMin(MathAbs(priceVsPOC) / 10, 0.2));
   }
   
   return MathMin(score, 100);
}

//+------------------------------------------------------------------+
//| Execute trade with smart execution                               |
//+------------------------------------------------------------------+
void ExecuteReversionTradeEnhanced(const ReversionSignal &signal)
{
   double entryPrice = signal.direction > 0 ? 
                      SymbolInfoDouble(_Symbol, SYMBOL_ASK) : 
                      SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // Use adaptive stop loss system
   double stopLoss;
   if(g_adaptiveStop.averageReward > 0)
   {
      stopLoss = UpdateAdaptiveStopLoss(_Symbol, entryPrice, signal.direction, g_adaptiveStop);
   }
   else
   {
      // Fallback to traditional stop calculation
      double atr = 0;
      double atr_array[];
      ArraySetAsSeries(atr_array, true);
      int atr_handle = iATR(_Symbol, _Period, 14);
      CopyBuffer(atr_handle, 0, 0, 1, atr_array);
      atr = atr_array[0];
      
      // Adjust stop based on toxicity and hidden liquidity
      double stopMultiplier = InpATRMultiplier;
      if(InpUseMicrostructure)
      {
         stopMultiplier *= (1 + g_microstructure.toxicity / 100);
         
         // Widen stop if liquidity mirage detected
         if(g_hiddenLiquidity.liquidityMirage > 0.5)
            stopMultiplier *= 1.2;
      }
      
      double stopDistance = atr * stopMultiplier;
      stopLoss = signal.direction > 0 ?
                entryPrice - stopDistance :
                entryPrice + stopDistance;
   }
   
   // Enhanced take profit using multiple advanced methods
   double takeProfit = signal.targetPrice;
   
   if(InpUseMicrostructure)
   {
      // Primary: Use POC as target
      if(g_orderFlow.poc > 0)
         takeProfit = g_orderFlow.poc;
      
      // Secondary: Check for liquidity voids
      if(ArraySize(g_orderFlow.liquidityVoids) > 0)
      {
         // Find nearest liquidity void as potential target
         for(int i = 0; i < ArraySize(g_orderFlow.liquidityVoids); i++)
         {
            double voidPrice = g_orderFlow.liquidityVoids[i];
            if(signal.direction > 0 && voidPrice > entryPrice && voidPrice < takeProfit)
               takeProfit = voidPrice * 0.98;  // Just before void
            else if(signal.direction < 0 && voidPrice < entryPrice && voidPrice > takeProfit)
               takeProfit = voidPrice * 1.02;  // Just before void
         }
      }
      
      // Tertiary: Market maker fair value
      if(g_marketMaker.fairValue > 0)
      {
         double fairTarget = g_marketMaker.fairValue;
         // Blend with existing target
         takeProfit = takeProfit * 0.7 + fairTarget * 0.3;
      }
      
      // Adjust if too close to entry
      double stopDistance = MathAbs(stopLoss - entryPrice);
      double tpDistance = MathAbs(takeProfit - entryPrice);
      if(tpDistance < stopDistance * 0.5)
      {
         takeProfit = signal.direction > 0 ?
                      entryPrice + stopDistance * 1.5 :
                      entryPrice - stopDistance * 1.5;
      }
   }
   
   // Calculate position size with elite methods
   double lotSize;
   
   if(InpUseKellyCriterion && g_kellyPosition.adjustedKelly > 0)
   {
      double baseRisk = g_accountBalance * g_kellyPosition.adjustedKelly;
      
      // Adjust for behavioral factors
      if(g_behavioral.dispositionEffect > 50)
         baseRisk *= 0.8;  // Reduce when disposition effect high
      
      if(g_behavioral.lossAversion > 30)
         baseRisk *= 0.9;  // Reduce when loss aversion high
      
      // Adjust for fractal properties
      if(g_fractal.antipersistence > 50)
         baseRisk *= 1.1;  // Increase for mean reverting markets
      
      // Adjust for tail risk
      if(g_tailHedge.hedgeRatio > 0)
         baseRisk *= (1 + g_tailHedge.hedgeRatio * 0.5);  // Can risk more with hedge
      
      // Quantum optimization adjustment
      if(g_quantumPortfolio.interferenceBonus > 0)
         baseRisk *= (1 + g_quantumPortfolio.interferenceBonus);
      
      // Market impact adjustment
      if(g_marketImpact.expectedCost > 0)
      {
         double impactPenalty = g_marketImpact.expectedCost / g_accountBalance;
         baseRisk *= (1 - impactPenalty);
      }
      
      lotSize = CalculatePositionSizeFromRisk(_Symbol, baseRisk, stopDistance);
   }
   else
   {
      lotSize = CalculateDynamicPositionSize(_Symbol, stopDistance, riskParams);
   }
   
   // Apply genetic algorithm optimized size multiplier
   if(g_genetic.bestFitness > 2.0)  // Sharpe > 2 from genetic optimization
   {
      lotSize *= 1.2;  // Increase size for well-optimized parameters
   }
   
   // Pair trading adjustment
   if(InpUsePairTrading && InpPairSymbol != "" && g_pairCorrelation.isTradeable)
   {
      // Check if we should trade the pair instead
      if(MathAbs(g_pairCorrelation.zScore) > MathAbs(signal.zScore))
      {
         Print("Pair trading opportunity detected with higher z-score");
         // Would implement pair trade here
      }
   }
   
   // Scale position based on confidence and regime
   if(InpScalePositions)
   {
      lotSize *= (signal.confidence / 100);
      
      if(InpUseMarketRegime)
      {
         if(g_marketRegime.currentRegime == MarketRegime::REGIME_VOLATILE)
            lotSize *= 0.7;  // Reduce size in volatile regimes
      }
   }
   
   if(lotSize == 0) return;
   
   // Validate trade
   if(!ValidateTradeRisk(_Symbol, entryPrice, stopLoss, takeProfit, lotSize, riskParams))
      return;
   
   // Smart execution
   if(InpUseSmartExecution && lotSize > 0.1)
   {
      // Use TWAP for large orders
      ExecuteTWAP(_Symbol, lotSize, signal.direction, 5);  // 5 minute TWAP
   }
   else
   {
      // Regular execution
      MqlTradeRequest request = {};
      MqlTradeResult result = {};
      
      request.action = TRADE_ACTION_DEAL;
      request.symbol = _Symbol;
      request.volume = lotSize;
      request.type = signal.direction > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
      request.price = entryPrice;
      request.sl = stopLoss;
      request.tp = takeProfit;
      request.comment = StringFormat("ML_MeanRev_%s_R%.1f", 
                                    GetReversionPatternName(signal.pattern),
                                    g_marketRegime.currentRegime);
      request.magic = 12345;
      
      if(OrderSend(request, result))
      {
         if(result.retcode == TRADE_RETCODE_DONE)
         {
            Print("Enhanced mean reversion trade executed: ", 
                  GetReversionPatternName(signal.pattern),
                  " ", signal.direction > 0 ? "BUY" : "SELL",
                  " Score: ", signal.score, 
                  " Regime: ", g_marketRegime.currentRegime,
                  " Toxicity: ", g_microstructure.toxicity);
            
            mlModel.signalsAnalyzed++;
            signalCount++;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate position size from risk amount                         |
//+------------------------------------------------------------------+
double CalculatePositionSizeFromRisk(string symbol, double riskAmount, double stopDistance)
{
   double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   
   double stopPips = stopDistance / tickSize;
   double lotSize = riskAmount / (stopPips * tickValue);
   
   // Normalize to symbol specifications
   double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   
   lotSize = MathFloor(lotSize / lotStep) * lotStep;
   lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
   
   return lotSize;
}

//+------------------------------------------------------------------+
//| Initialize ultra-advanced components                             |
//+------------------------------------------------------------------+
void InitializeUltraAdvanced()
{
   Print("Initializing ultra-advanced trading components...");
   
   // Initialize neural network
   int hiddenLayers[] = {128, 64, 32};
   InitializeNeuralNetwork(g_neuralNet, 50, hiddenLayers, 1);
   
   // Initialize adaptive stop loss
   InitializeAdaptiveStopLoss(g_adaptiveStop);
   
   // Initialize meta strategy
   InitializeMetaStrategy(g_metaStrategy, 4);
   
   // Initialize market maker model
   g_marketMaker.inventoryLimit = 10;  // Max 10 lots
   g_marketMaker.targetInventory = 0;
   
   // Initialize elite quant components
   InitializeEliteQuant();
   
   Print("Ultra-advanced components initialized");
}

//+------------------------------------------------------------------+
//| Initialize elite quant components                                |
//+------------------------------------------------------------------+
void InitializeEliteQuant()
{
   Print("Initializing elite quant components...");
   
   // Initialize genetic optimizer for strategy parameters
   RunGeneticOptimization(g_genetic, 20, 100);  // 20 params, 100 population
   
   // Initialize market impact model
   g_marketImpact.riskAversion = 0.0001;
   g_marketImpact.urgencyFactor = 0.5;  // Medium urgency
   
   // Initialize tail risk parameters
   g_tailHedge.hedgeRatio = 0;  // Will be calculated dynamically
   
   Print("Elite quant components initialized");
}

//+------------------------------------------------------------------+
//| Update elite analysis                                            |
//+------------------------------------------------------------------+
void UpdateEliteAnalysis()
{
   // Behavioral analysis
   CalculateBehavioralSignals(_Symbol, g_behavioral);
   
   // Fractal market analysis
   AnalyzeFractalMarket(_Symbol, g_fractal);
   
   // Extreme value analysis
   AnalyzeExtremeValues(_Symbol, g_extremeValue);
   
   // Quantum portfolio optimization (if multiple assets)
   string symbols[] = {_Symbol};
   if(ArraySize(symbols) > 1)
   {
      double returns[][];
      // Get returns data
      ArrayResize(returns, ArraySize(symbols));
      for(int i = 0; i < ArraySize(symbols); i++)
      {
         double ret[];
         GetReturnsData(symbols[i], ret, 100);
         ArrayCopy(returns[i], ret);
      }
      
      OptimizeQuantumPortfolio(symbols, returns, g_quantumPortfolio);
   }
   
   // Market impact calculation for current position
   if(PositionsTotal() > 0)
   {
      double totalSize = 0;
      for(int i = 0; i < PositionsTotal(); i++)
      {
         if(PositionSelectByTicket(PositionGetTicket(i)))
            totalSize += PositionGetDouble(POSITION_VOLUME);
      }
      
      double volatility = CalculateVolatility(_Symbol, 20);
      CalculateOptimalExecution(totalSize, 3600, volatility, g_marketImpact);
   }
   
   // Tail risk hedging
   double portfolioValue = AccountInfoDouble(ACCOUNT_BALANCE);
   CalculateTailHedge(_Symbol, portfolioValue, g_tailHedge);
   
   // Create synthetic asset for hedging
   if(g_tailHedge.hedgeRatio > 0.1)
   {
      string availableAssets[] = {"EURUSD", "GBPUSD", "USDJPY"};
      CreateSyntheticAsset(_Symbol, availableAssets, g_synthetic);
   }
}

//+------------------------------------------------------------------+
//| Enhanced scoring with neural network                             |
//+------------------------------------------------------------------+
double CalculateNeuralNetworkScore(const ReversionSignal &signal)
{
   // Prepare input features
   double inputs[];
   ArrayResize(inputs, 50);
   
   // Statistical features
   inputs[0] = NormalizeValue(signal.zScore, -4, 4);
   inputs[1] = NormalizeValue(signal.deviation, 0, 10);
   inputs[2] = NormalizeValue(statistics[0].skewness, -2, 2);
   inputs[3] = NormalizeValue(statistics[0].kurtosis, -2, 10);
   inputs[4] = NormalizeValue(statistics[0].halfLife, 0, 50);
   inputs[5] = statistics[0].hurst;
   
   // Microstructure features
   inputs[6] = NormalizeValue(g_microstructure.orderFlowImbalance, -1, 1);
   inputs[7] = NormalizeValue(g_microstructure.vpin, 0, 1);
   inputs[8] = NormalizeValue(g_microstructure.toxicity, 0, 100);
   inputs[9] = NormalizeValue(g_microstructure.liquidityScore, 0, 100);
   
   // Hidden liquidity features
   inputs[10] = g_hiddenLiquidity.icebergRatio;
   inputs[11] = g_hiddenLiquidity.darkPoolActivity;
   inputs[12] = NormalizeValue(g_hiddenLiquidity.hiddenBuyPressure, 0, 1000);
   inputs[13] = NormalizeValue(g_hiddenLiquidity.hiddenSellPressure, 0, 1000);
   inputs[14] = g_hiddenLiquidity.liquidityMirage;
   
   // Order book dynamics
   inputs[15] = NormalizeValue(g_orderBookDynamics.persistentImbalance, -1, 1);
   inputs[16] = g_orderBookDynamics.spoofingScore;
   inputs[17] = g_orderBookDynamics.layeringDetected;
   inputs[18] = g_orderBookDynamics.quoteStuffing;
   inputs[19] = g_orderBookDynamics.frontRunning;
   
   // Market regime
   inputs[20] = (double)g_marketRegime.currentRegime / 7.0;
   inputs[21] = g_marketRegime.regimeStrength;
   inputs[22] = g_marketRegime.transitionProbability;
   
   // HF microstructure
   inputs[23] = NormalizeValue(g_hfMicro.tickImbalance, -1, 1);
   inputs[24] = g_hfMicro.tickMomentum;
   inputs[25] = g_hfMicro.tickReversal;
   inputs[26] = NormalizeValue(g_hfMicro.quoteIntensity, 0, 100);
   
   // Cross-asset contagion
   inputs[27] = g_contagion.contagionIndex;
   inputs[28] = g_contagion.systemicRisk;
   inputs[29] = g_contagion.cascadeProbability;
   
   // Market maker model
   inputs[30] = NormalizeValue(g_marketMaker.inventoryRisk, 0, 1);
   inputs[31] = NormalizeValue(g_marketMaker.adverseSelection, 0, 10);
   inputs[32] = NormalizeValue(g_marketMaker.expectedFlow, -100, 100);
   inputs[33] = g_marketMaker.toxicFlow;
   inputs[34] = g_marketMaker.informedTraderProb;
   
   // Time features
   MqlDateTime dt;
   TimeToStruct(signal.time, dt);
   inputs[35] = NormalizeValue(dt.hour, 0, 23);
   inputs[36] = NormalizeValue(dt.day_of_week, 1, 5);
   inputs[37] = NormalizeValue(dt.day, 1, 31);
   
   // Technical features
   inputs[38] = NormalizeValue(signal.direction, -1, 1);
   inputs[39] = NormalizeValue(signal.pattern, 0, 8);
   
   // Fill remaining inputs with market data
   for(int i = 40; i < 50; i++)
      inputs[i] = MathRand() / 32767.0;  // Placeholder
   
   // Get neural network prediction
   double outputs[];
   LSTMForward(g_neuralNet, inputs, outputs);
   
   return outputs[0] * 100;  // Convert to 0-100 score
}