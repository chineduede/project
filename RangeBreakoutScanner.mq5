//+------------------------------------------------------------------+
//|                                          RangeBreakoutScanner.mq5 |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Range Breakout Scanner"
#property link      ""
#property version   "1.00"
#property description "Scans for optimal range breakout opportunities using historical data"

//--- Input parameters
input int      InpRangePeriod = 20;           // Range calculation period (bars)
input double   InpMinRangeSize = 50;          // Minimum range size in points
input double   InpBreakoutThreshold = 1.2;    // Breakout threshold multiplier
input int      InpLookbackBars = 1000;        // Number of bars to analyze
input bool     InpShowVisuals = true;         // Show visual markers
input int      InpMaxRangesToDisplay = 10;    // Maximum ranges to display

//--- Global variables
struct RangeData
{
   datetime startTime;
   datetime endTime;
   double   highPrice;
   double   lowPrice;
   double   rangeSize;
   double   breakoutScore;
   int      barsInRange;
   bool     breakoutDirection; // true = up, false = down
   double   breakoutPrice;
   datetime breakoutTime;
};

RangeData ranges[];
int totalRangesFound = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("Range Breakout Scanner initialized");
   Print("Analyzing last ", InpLookbackBars, " bars");
   
   // Scan historical data on initialization
   ScanHistoricalRanges();
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Clean up visual objects
   ObjectsDeleteAll(0, "Range_");
   Print("Range Breakout Scanner stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Update current ranges on new bar
   static datetime lastBarTime = 0;
   datetime currentBarTime = iTime(_Symbol, _Period, 0);
   
   if(lastBarTime != currentBarTime)
   {
      lastBarTime = currentBarTime;
      ScanHistoricalRanges();
   }
}

//+------------------------------------------------------------------+
//| Scan historical data for range breakouts                         |
//+------------------------------------------------------------------+
void ScanHistoricalRanges()
{
   ArrayResize(ranges, 0);
   totalRangesFound = 0;
   
   // Clear previous visual objects
   if(InpShowVisuals)
      ObjectsDeleteAll(0, "Range_");
   
   // Analyze historical bars
   for(int i = InpRangePeriod; i < InpLookbackBars - InpRangePeriod; i++)
   {
      // Identify potential range
      RangeData range;
      if(IdentifyRange(i, range))
      {
         // Check for breakout
         if(CheckBreakout(i, range))
         {
            // Calculate breakout score
            CalculateBreakoutScore(range);
            
            // Add to array
            int size = ArraySize(ranges);
            ArrayResize(ranges, size + 1);
            ranges[size] = range;
            totalRangesFound++;
         }
      }
   }
   
   // Sort ranges by score
   SortRangesByScore();
   
   // Display results
   DisplayResults();
}

//+------------------------------------------------------------------+
//| Identify range at given position                                 |
//+------------------------------------------------------------------+
bool IdentifyRange(int startBar, RangeData &range)
{
   double highestHigh = 0;
   double lowestLow = DBL_MAX;
   int rangeStart = startBar;
   int rangeEnd = startBar - InpRangePeriod;
   
   // Find high and low of the range
   for(int i = rangeStart; i >= rangeEnd && i >= 0; i--)
   {
      double high = iHigh(_Symbol, _Period, i);
      double low = iLow(_Symbol, _Period, i);
      
      if(high > highestHigh) highestHigh = high;
      if(low < lowestLow) lowestLow = low;
   }
   
   // Calculate range size
   double rangeSize = (highestHigh - lowestLow) / _Point;
   
   // Check if range meets minimum size requirement
   if(rangeSize < InpMinRangeSize)
      return false;
   
   // Check if price stayed within range
   double rangeBuffer = (highestHigh - lowestLow) * 0.1; // 10% buffer
   int consolidationBars = 0;
   
   for(int i = rangeStart; i >= rangeEnd && i >= 0; i--)
   {
      double high = iHigh(_Symbol, _Period, i);
      double low = iLow(_Symbol, _Period, i);
      
      if(high <= highestHigh + rangeBuffer && low >= lowestLow - rangeBuffer)
         consolidationBars++;
   }
   
   // Require at least 70% of bars to be within range
   if(consolidationBars < InpRangePeriod * 0.7)
      return false;
   
   // Fill range data
   range.startTime = iTime(_Symbol, _Period, rangeStart);
   range.endTime = iTime(_Symbol, _Period, rangeEnd);
   range.highPrice = highestHigh;
   range.lowPrice = lowestLow;
   range.rangeSize = rangeSize;
   range.barsInRange = InpRangePeriod;
   
   return true;
}

//+------------------------------------------------------------------+
//| Check if breakout occurred after range                           |
//+------------------------------------------------------------------+
bool CheckBreakout(int rangeStartBar, RangeData &range)
{
   int checkBars = MathMin(20, rangeStartBar - InpRangePeriod);
   double breakoutLevel = (range.highPrice - range.lowPrice) * InpBreakoutThreshold;
   
   for(int i = rangeStartBar - InpRangePeriod - 1; i >= rangeStartBar - InpRangePeriod - checkBars && i >= 0; i--)
   {
      double close = iClose(_Symbol, _Period, i);
      
      // Check upward breakout
      if(close > range.highPrice + breakoutLevel * 0.1)
      {
         range.breakoutDirection = true;
         range.breakoutPrice = close;
         range.breakoutTime = iTime(_Symbol, _Period, i);
         return true;
      }
      
      // Check downward breakout
      if(close < range.lowPrice - breakoutLevel * 0.1)
      {
         range.breakoutDirection = false;
         range.breakoutPrice = close;
         range.breakoutTime = iTime(_Symbol, _Period, i);
         return true;
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Calculate breakout quality score                                 |
//+------------------------------------------------------------------+
void CalculateBreakoutScore(RangeData &range)
{
   double score = 0;
   
   // Factor 1: Range size (normalized)
   double rangeSizeScore = MathMin(range.rangeSize / 200.0, 1.0) * 30;
   
   // Factor 2: Breakout strength
   double breakoutDistance;
   if(range.breakoutDirection)
      breakoutDistance = (range.breakoutPrice - range.highPrice) / _Point;
   else
      breakoutDistance = (range.lowPrice - range.breakoutPrice) / _Point;
   
   double breakoutScore = MathMin(breakoutDistance / 100.0, 1.0) * 40;
   
   // Factor 3: Time in range (consolidation quality)
   double consolidationScore = (range.barsInRange / 50.0) * 30;
   consolidationScore = MathMin(consolidationScore, 30);
   
   // Calculate total score
   range.breakoutScore = rangeSizeScore + breakoutScore + consolidationScore;
}

//+------------------------------------------------------------------+
//| Sort ranges by breakout score                                    |
//+------------------------------------------------------------------+
void SortRangesByScore()
{
   int size = ArraySize(ranges);
   
   // Simple bubble sort (sufficient for small arrays)
   for(int i = 0; i < size - 1; i++)
   {
      for(int j = 0; j < size - i - 1; j++)
      {
         if(ranges[j].breakoutScore < ranges[j + 1].breakoutScore)
         {
            RangeData temp = ranges[j];
            ranges[j] = ranges[j + 1];
            ranges[j + 1] = temp;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Display analysis results                                         |
//+------------------------------------------------------------------+
void DisplayResults()
{
   Print("=== Range Breakout Analysis Results ===");
   Print("Total ranges found: ", totalRangesFound);
   
   int displayCount = MathMin(ArraySize(ranges), InpMaxRangesToDisplay);
   
   for(int i = 0; i < displayCount; i++)
   {
      Print("Range #", i + 1, ":");
      Print("  Score: ", DoubleToString(ranges[i].breakoutScore, 2));
      Print("  Range: ", DoubleToString(ranges[i].rangeSize, 0), " points");
      Print("  Period: ", ranges[i].startTime, " to ", ranges[i].endTime);
      Print("  High: ", DoubleToString(ranges[i].highPrice, _Digits));
      Print("  Low: ", DoubleToString(ranges[i].lowPrice, _Digits));
      Print("  Breakout: ", ranges[i].breakoutDirection ? "UP" : "DOWN", 
            " at ", DoubleToString(ranges[i].breakoutPrice, _Digits));
      
      // Draw visual representation
      if(InpShowVisuals)
         DrawRange(i, ranges[i]);
   }
}

//+------------------------------------------------------------------+
//| Draw range on chart                                              |
//+------------------------------------------------------------------+
void DrawRange(int index, const RangeData &range)
{
   string prefix = "Range_" + IntegerToString(index) + "_";
   
   // Draw rectangle for range
   string rectName = prefix + "Box";
   ObjectCreate(0, rectName, OBJ_RECTANGLE, 0, range.startTime, range.highPrice, range.endTime, range.lowPrice);
   ObjectSetInteger(0, rectName, OBJPROP_COLOR, clrLightBlue);
   ObjectSetInteger(0, rectName, OBJPROP_STYLE, STYLE_SOLID);
   ObjectSetInteger(0, rectName, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, rectName, OBJPROP_BACK, true);
   ObjectSetInteger(0, rectName, OBJPROP_SELECTABLE, true);
   ObjectSetInteger(0, rectName, OBJPROP_SELECTED, false);
   
   // Draw breakout arrow
   string arrowName = prefix + "Arrow";
   ObjectCreate(0, arrowName, OBJ_ARROW, 0, range.breakoutTime, range.breakoutPrice);
   ObjectSetInteger(0, arrowName, OBJPROP_ARROWCODE, range.breakoutDirection ? 233 : 234);
   ObjectSetInteger(0, arrowName, OBJPROP_COLOR, range.breakoutDirection ? clrGreen : clrRed);
   ObjectSetInteger(0, arrowName, OBJPROP_WIDTH, 2);
   
   // Add score label
   string labelName = prefix + "Score";
   ObjectCreate(0, labelName, OBJ_TEXT, 0, range.startTime, range.highPrice + 10 * _Point);
   ObjectSetString(0, labelName, OBJPROP_TEXT, "Score: " + DoubleToString(range.breakoutScore, 1));
   ObjectSetInteger(0, labelName, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, labelName, OBJPROP_FONTSIZE, 8);
}

//+------------------------------------------------------------------+
//| Get best range for trading                                       |
//+------------------------------------------------------------------+
bool GetBestRange(RangeData &bestRange)
{
   if(ArraySize(ranges) > 0)
   {
      bestRange = ranges[0];
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Export analysis results to CSV                                   |
//+------------------------------------------------------------------+
void ExportResults(string filename)
{
   int handle = FileOpen(filename, FILE_WRITE|FILE_CSV);
   if(handle != INVALID_HANDLE)
   {
      FileWrite(handle, "Score", "RangeSize", "StartTime", "EndTime", "High", "Low", "BreakoutDirection", "BreakoutPrice");
      
      for(int i = 0; i < ArraySize(ranges); i++)
      {
         FileWrite(handle,
                   DoubleToString(ranges[i].breakoutScore, 2),
                   DoubleToString(ranges[i].rangeSize, 0),
                   TimeToString(ranges[i].startTime),
                   TimeToString(ranges[i].endTime),
                   DoubleToString(ranges[i].highPrice, _Digits),
                   DoubleToString(ranges[i].lowPrice, _Digits),
                   ranges[i].breakoutDirection ? "UP" : "DOWN",
                   DoubleToString(ranges[i].breakoutPrice, _Digits));
      }
      
      FileClose(handle);
      Print("Results exported to ", filename);
   }
}