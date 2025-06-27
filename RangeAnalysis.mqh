//+------------------------------------------------------------------+
//|                                               RangeAnalysis.mqh   |
//|                                                                  |
//|                        Advanced Range Analysis Functions         |
//+------------------------------------------------------------------+
#property copyright "Range Analysis Library"
#property link      ""

//+------------------------------------------------------------------+
//| Range pattern types                                              |
//+------------------------------------------------------------------+
enum ENUM_RANGE_PATTERN
{
   PATTERN_RECTANGLE,      // Classic rectangle
   PATTERN_ASCENDING,      // Ascending triangle
   PATTERN_DESCENDING,     // Descending triangle
   PATTERN_SYMMETRIC,      // Symmetric triangle
   PATTERN_WEDGE_RISING,   // Rising wedge
   PATTERN_WEDGE_FALLING,  // Falling wedge
   PATTERN_FLAG,           // Flag pattern
   PATTERN_PENNANT         // Pennant pattern
};

//+------------------------------------------------------------------+
//| Advanced range metrics structure                                 |
//+------------------------------------------------------------------+
struct RangeMetrics
{
   double volatilityScore;      // Volatility within range
   double volumeProfile;        // Volume concentration
   double resistanceStrength;   // Strength of upper boundary
   double supportStrength;      // Strength of lower boundary
   int    touchesHigh;          // Number of times high was tested
   int    touchesLow;           // Number of times low was tested
   double avgBarRange;          // Average bar range within consolidation
   ENUM_RANGE_PATTERN pattern;  // Identified pattern type
};

//+------------------------------------------------------------------+
//| Calculate advanced range metrics                                 |
//+------------------------------------------------------------------+
void CalculateRangeMetrics(const datetime startTime, const datetime endTime, 
                          const double highPrice, const double lowPrice,
                          RangeMetrics &metrics)
{
   int startBar = iBarShift(_Symbol, _Period, startTime);
   int endBar = iBarShift(_Symbol, _Period, endTime);
   
   if(startBar < 0 || endBar < 0) return;
   
   // Initialize metrics
   metrics.volatilityScore = 0;
   metrics.volumeProfile = 0;
   metrics.touchesHigh = 0;
   metrics.touchesLow = 0;
   metrics.avgBarRange = 0;
   
   double totalVolume = 0;
   double totalRange = 0;
   int barCount = 0;
   
   // Analyze each bar in the range
   for(int i = startBar; i >= endBar && i >= 0; i--)
   {
      double high = iHigh(_Symbol, _Period, i);
      double low = iLow(_Symbol, _Period, i);
      double close = iClose(_Symbol, _Period, i);
      double volume = iVolume(_Symbol, _Period, i);
      
      // Count touches of boundaries
      if(MathAbs(high - highPrice) < 5 * _Point)
         metrics.touchesHigh++;
      
      if(MathAbs(low - lowPrice) < 5 * _Point)
         metrics.touchesLow++;
      
      // Calculate volatility
      double barRange = high - low;
      totalRange += barRange;
      
      // Volume analysis
      totalVolume += volume;
      
      barCount++;
   }
   
   // Calculate final metrics
   if(barCount > 0)
   {
      metrics.avgBarRange = totalRange / barCount;
      metrics.volatilityScore = (metrics.avgBarRange / (highPrice - lowPrice)) * 100;
      
      // Resistance and support strength based on touches
      metrics.resistanceStrength = MathMin(metrics.touchesHigh * 20, 100);
      metrics.supportStrength = MathMin(metrics.touchesLow * 20, 100);
   }
   
   // Identify pattern type
   metrics.pattern = IdentifyRangePattern(startBar, endBar, highPrice, lowPrice);
}

//+------------------------------------------------------------------+
//| Identify range pattern type                                      |
//+------------------------------------------------------------------+
ENUM_RANGE_PATTERN IdentifyRangePattern(int startBar, int endBar, 
                                        double rangeHigh, double rangeLow)
{
   // Get trend lines for pattern identification
   double upperTrendSlope = 0, lowerTrendSlope = 0;
   CalculateTrendLines(startBar, endBar, upperTrendSlope, lowerTrendSlope);
   
   double slopeThreshold = 0.0001; // Threshold for considering slope as flat
   
   // Pattern identification logic
   if(MathAbs(upperTrendSlope) < slopeThreshold && MathAbs(lowerTrendSlope) < slopeThreshold)
      return PATTERN_RECTANGLE;
   
   if(upperTrendSlope > slopeThreshold && MathAbs(lowerTrendSlope) < slopeThreshold)
      return PATTERN_ASCENDING;
   
   if(MathAbs(upperTrendSlope) < slopeThreshold && lowerTrendSlope < -slopeThreshold)
      return PATTERN_DESCENDING;
   
   if(upperTrendSlope < -slopeThreshold && lowerTrendSlope > slopeThreshold)
      return PATTERN_SYMMETRIC;
   
   if(upperTrendSlope > slopeThreshold && lowerTrendSlope > slopeThreshold)
      return PATTERN_WEDGE_RISING;
   
   if(upperTrendSlope < -slopeThreshold && lowerTrendSlope < -slopeThreshold)
      return PATTERN_WEDGE_FALLING;
   
   return PATTERN_RECTANGLE; // Default
}

//+------------------------------------------------------------------+
//| Calculate trend lines for pattern identification                 |
//+------------------------------------------------------------------+
void CalculateTrendLines(int startBar, int endBar, 
                        double &upperSlope, double &lowerSlope)
{
   // Arrays for regression
   double x[], yHigh[], yLow[];
   int size = startBar - endBar + 1;
   
   ArrayResize(x, size);
   ArrayResize(yHigh, size);
   ArrayResize(yLow, size);
   
   // Fill arrays
   int index = 0;
   for(int i = startBar; i >= endBar && i >= 0; i--)
   {
      x[index] = index;
      yHigh[index] = iHigh(_Symbol, _Period, i);
      yLow[index] = iLow(_Symbol, _Period, i);
      index++;
   }
   
   // Calculate slopes using linear regression
   upperSlope = CalculateSlope(x, yHigh);
   lowerSlope = CalculateSlope(x, yLow);
}

//+------------------------------------------------------------------+
//| Calculate slope using linear regression                          |
//+------------------------------------------------------------------+
double CalculateSlope(const double &x[], const double &y[])
{
   int n = ArraySize(x);
   if(n < 2) return 0;
   
   double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
   
   for(int i = 0; i < n; i++)
   {
      sumX += x[i];
      sumY += y[i];
      sumXY += x[i] * y[i];
      sumX2 += x[i] * x[i];
   }
   
   double denominator = n * sumX2 - sumX * sumX;
   if(MathAbs(denominator) < 0.0000001) return 0;
   
   return (n * sumXY - sumX * sumY) / denominator;
}

//+------------------------------------------------------------------+
//| Calculate breakout probability based on pattern and metrics      |
//+------------------------------------------------------------------+
double CalculateBreakoutProbability(const RangeMetrics &metrics, bool upDirection)
{
   double probability = 50; // Base probability
   
   // Adjust based on pattern type
   switch(metrics.pattern)
   {
      case PATTERN_ASCENDING:
         probability += upDirection ? 15 : -15;
         break;
      case PATTERN_DESCENDING:
         probability += upDirection ? -15 : 15;
         break;
      case PATTERN_WEDGE_RISING:
         probability += upDirection ? -10 : 10;
         break;
      case PATTERN_WEDGE_FALLING:
         probability += upDirection ? 10 : -10;
         break;
   }
   
   // Adjust based on boundary strength
   if(upDirection)
   {
      probability -= metrics.resistanceStrength * 0.2;
      probability += metrics.supportStrength * 0.1;
   }
   else
   {
      probability += metrics.resistanceStrength * 0.1;
      probability -= metrics.supportStrength * 0.2;
   }
   
   // Adjust based on volatility
   if(metrics.volatilityScore < 30)
      probability += 10; // Low volatility increases breakout probability
   
   return MathMax(0, MathMin(100, probability));
}

//+------------------------------------------------------------------+
//| Calculate optimal breakout entry point                           |
//+------------------------------------------------------------------+
double CalculateOptimalEntry(double rangeHigh, double rangeLow, 
                           bool upDirection, const RangeMetrics &metrics)
{
   double rangeSize = rangeHigh - rangeLow;
   double entryOffset = rangeSize * 0.02; // 2% of range as default
   
   // Adjust based on volatility
   if(metrics.volatilityScore > 50)
      entryOffset *= 1.5; // Increase offset for volatile ranges
   
   if(upDirection)
      return rangeHigh + entryOffset;
   else
      return rangeLow - entryOffset;
}

//+------------------------------------------------------------------+
//| Calculate stop loss based on range characteristics               |
//+------------------------------------------------------------------+
double CalculateStopLoss(double entryPrice, double rangeHigh, double rangeLow,
                        bool upDirection, const RangeMetrics &metrics)
{
   double rangeSize = rangeHigh - rangeLow;
   double stopDistance = rangeSize * 0.5; // 50% of range as default
   
   // Adjust based on pattern
   if(metrics.pattern == PATTERN_RECTANGLE)
      stopDistance = rangeSize * 0.3; // Tighter stop for rectangles
   else if(metrics.pattern == PATTERN_WEDGE_RISING || metrics.pattern == PATTERN_WEDGE_FALLING)
      stopDistance = rangeSize * 0.6; // Wider stop for wedges
   
   if(upDirection)
      return entryPrice - stopDistance;
   else
      return entryPrice + stopDistance;
}

//+------------------------------------------------------------------+
//| Calculate take profit based on historical breakout performance   |
//+------------------------------------------------------------------+
double CalculateTakeProfit(double entryPrice, double rangeSize, 
                          bool upDirection, const RangeMetrics &metrics)
{
   double targetMultiplier = 1.5; // Default 1.5x range size
   
   // Adjust based on pattern strength
   if(metrics.resistanceStrength > 60 || metrics.supportStrength > 60)
      targetMultiplier = 2.0; // Strong boundaries suggest stronger breakout
   
   if(metrics.volatilityScore < 20)
      targetMultiplier = 1.2; // Low volatility suggests smaller moves
   
   double targetDistance = rangeSize * targetMultiplier;
   
   if(upDirection)
      return entryPrice + targetDistance;
   else
      return entryPrice - targetDistance;
}

//+------------------------------------------------------------------+
//| Validate range quality for trading                               |
//+------------------------------------------------------------------+
bool ValidateRangeQuality(const RangeMetrics &metrics, double rangeSize)
{
   // Minimum requirements
   if(metrics.touchesHigh < 2 || metrics.touchesLow < 2)
      return false; // Not enough boundary tests
   
   if(metrics.volatilityScore > 80)
      return false; // Too volatile to be a proper range
   
   if(rangeSize < 30 || rangeSize > 500)
      return false; // Range size out of acceptable bounds
   
   return true;
}

//+------------------------------------------------------------------+
//| Get pattern name as string                                       |
//+------------------------------------------------------------------+
string GetPatternName(ENUM_RANGE_PATTERN pattern)
{
   switch(pattern)
   {
      case PATTERN_RECTANGLE:      return "Rectangle";
      case PATTERN_ASCENDING:      return "Ascending Triangle";
      case PATTERN_DESCENDING:     return "Descending Triangle";
      case PATTERN_SYMMETRIC:      return "Symmetric Triangle";
      case PATTERN_WEDGE_RISING:   return "Rising Wedge";
      case PATTERN_WEDGE_FALLING:  return "Falling Wedge";
      case PATTERN_FLAG:           return "Flag";
      case PATTERN_PENNANT:        return "Pennant";
      default:                     return "Unknown";
   }
}