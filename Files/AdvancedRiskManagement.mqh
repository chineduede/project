//+------------------------------------------------------------------+
//|                                      AdvancedRiskManagement.mqh   |
//|                         Dynamic Risk and Money Management        |
//+------------------------------------------------------------------+
#property copyright "Advanced Risk Management"
#property link      ""

//+------------------------------------------------------------------+
//| Risk management parameters                                       |
//+------------------------------------------------------------------+
struct RiskParameters
{
   double maxRiskPerTrade;      // Maximum risk per trade (%)
   double maxDailyRisk;         // Maximum daily risk (%)
   double maxOpenTrades;        // Maximum concurrent trades
   double winRateTarget;        // Target win rate
   double profitFactorTarget;   // Target profit factor
   bool useKellyCriterion;      // Use Kelly Criterion for sizing
   bool useTrailingStop;        // Enable trailing stop
   double trailingStopATR;      // Trailing stop in ATR multiples
};

//+------------------------------------------------------------------+
//| Trade performance tracking                                       |
//+------------------------------------------------------------------+
struct PerformanceMetrics
{
   int totalTrades;
   int winningTrades;
   int losingTrades;
   double totalProfit;
   double totalLoss;
   double largestWin;
   double largestLoss;
   double avgWin;
   double avgLoss;
   double winRate;
   double profitFactor;
   double sharpeRatio;
   double maxDrawdown;
   double currentDrawdown;
   datetime lastTradeTime;
};

//+------------------------------------------------------------------+
//| Global performance tracker                                       |
//+------------------------------------------------------------------+
PerformanceMetrics g_performance;

//+------------------------------------------------------------------+
//| Calculate Kelly Criterion position size                         |
//+------------------------------------------------------------------+
double CalculateKellySize(double win_probability, double avg_win, double avg_loss, double max_risk)
{
   if(avg_loss == 0) return max_risk;
   
   double b = avg_win / avg_loss; // Odds
   double p = win_probability;     // Probability of winning
   double q = 1 - p;              // Probability of losing
   
   // Kelly formula: f = (bp - q) / b
   double kelly_percent = (b * p - q) / b;
   
   // Apply Kelly fraction (typically 0.25 to be conservative)
   kelly_percent *= 0.25;
   
   // Ensure within bounds
   if(kelly_percent < 0) kelly_percent = 0;
   if(kelly_percent > max_risk) kelly_percent = max_risk;
   
   return kelly_percent;
}

//+------------------------------------------------------------------+
//| Update performance metrics                                       |
//+------------------------------------------------------------------+
void UpdatePerformanceMetrics(double trade_result, double account_balance)
{
   g_performance.totalTrades++;
   
   if(trade_result > 0)
   {
      g_performance.winningTrades++;
      g_performance.totalProfit += trade_result;
      
      if(trade_result > g_performance.largestWin)
         g_performance.largestWin = trade_result;
   }
   else
   {
      g_performance.losingTrades++;
      g_performance.totalLoss += MathAbs(trade_result);
      
      if(MathAbs(trade_result) > g_performance.largestLoss)
         g_performance.largestLoss = MathAbs(trade_result);
   }
   
   // Update averages
   if(g_performance.winningTrades > 0)
      g_performance.avgWin = g_performance.totalProfit / g_performance.winningTrades;
   
   if(g_performance.losingTrades > 0)
      g_performance.avgLoss = g_performance.totalLoss / g_performance.losingTrades;
   
   // Update win rate
   g_performance.winRate = (double)g_performance.winningTrades / g_performance.totalTrades;
   
   // Update profit factor
   if(g_performance.totalLoss > 0)
      g_performance.profitFactor = g_performance.totalProfit / g_performance.totalLoss;
   else
      g_performance.profitFactor = g_performance.totalProfit > 0 ? 999 : 0;
   
   // Update drawdown
   static double peak_balance = account_balance;
   if(account_balance > peak_balance)
      peak_balance = account_balance;
   
   g_performance.currentDrawdown = (peak_balance - account_balance) / peak_balance * 100;
   
   if(g_performance.currentDrawdown > g_performance.maxDrawdown)
      g_performance.maxDrawdown = g_performance.currentDrawdown;
   
   g_performance.lastTradeTime = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Calculate dynamic position size                                  |
//+------------------------------------------------------------------+
double CalculateDynamicPositionSize(string symbol, double stop_loss_points, 
                                   const RiskParameters &risk_params)
{
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double account_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   
   // Use the lower of balance and equity
   double risk_base = MathMin(account_balance, account_equity);
   
   // Base risk percentage
   double risk_percent = risk_params.maxRiskPerTrade;
   
   // Adjust based on performance
   if(g_performance.totalTrades >= 20) // Need sufficient sample size
   {
      // Reduce risk if performance is poor
      if(g_performance.winRate < 0.4 || g_performance.profitFactor < 1.2)
         risk_percent *= 0.5;
      
      // Increase risk if performance is excellent
      else if(g_performance.winRate > 0.6 && g_performance.profitFactor > 2.0)
         risk_percent *= 1.2;
      
      // Use Kelly Criterion if enabled
      if(risk_params.useKellyCriterion && g_performance.avgLoss > 0)
      {
         double kelly_size = CalculateKellySize(g_performance.winRate, 
                                               g_performance.avgWin, 
                                               g_performance.avgLoss,
                                               risk_params.maxRiskPerTrade);
         risk_percent = kelly_size;
      }
   }
   
   // Check daily risk limit
   double daily_loss = GetDailyPnL();
   double daily_risk_used = MathAbs(daily_loss) / risk_base * 100;
   
   if(daily_risk_used >= risk_params.maxDailyRisk)
      return 0; // No more trades today
   
   // Adjust for remaining daily risk
   double remaining_daily_risk = risk_params.maxDailyRisk - daily_risk_used;
   risk_percent = MathMin(risk_percent, remaining_daily_risk);
   
   // Calculate position size
   double risk_amount = risk_base * risk_percent / 100;
   double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   
   if(stop_loss_points <= 0 || tick_value <= 0 || tick_size <= 0)
      return 0;
   
   double lots = risk_amount / (stop_loss_points / tick_size * tick_value);
   
   // Normalize to symbol specifications
   double min_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   
   lots = MathFloor(lots / lot_step) * lot_step;
   lots = MathMax(min_lot, MathMin(lots, max_lot));
   
   return lots;
}

//+------------------------------------------------------------------+
//| Get daily P&L                                                    |
//+------------------------------------------------------------------+
double GetDailyPnL()
{
   datetime today_start = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
   double daily_pnl = 0;
   
   // Check history
   if(HistorySelect(today_start, TimeCurrent()))
   {
      int deals = HistoryDealsTotal();
      for(int i = 0; i < deals; i++)
      {
         ulong ticket = HistoryDealGetTicket(i);
         if(ticket > 0)
         {
            double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
            double commission = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
            double swap = HistoryDealGetDouble(ticket, DEAL_SWAP);
            
            daily_pnl += profit + commission + swap;
         }
      }
   }
   
   // Add open positions
   int positions = PositionsTotal();
   for(int i = 0; i < positions; i++)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetInteger(POSITION_TIME) >= today_start)
         {
            daily_pnl += PositionGetDouble(POSITION_PROFIT);
         }
      }
   }
   
   return daily_pnl;
}

//+------------------------------------------------------------------+
//| Trailing stop implementation                                     |
//+------------------------------------------------------------------+
void UpdateTrailingStop(ulong position_ticket, const RiskParameters &risk_params)
{
   if(!risk_params.useTrailingStop) return;
   if(!PositionSelectByTicket(position_ticket)) return;
   
   string symbol = PositionGetString(POSITION_SYMBOL);
   double current_price = SymbolInfoDouble(symbol, SYMBOL_BID);
   double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
   double current_sl = PositionGetDouble(POSITION_SL);
   double current_tp = PositionGetDouble(POSITION_TP);
   ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   
   // Get ATR for trailing distance
   double atr[];
   ArraySetAsSeries(atr, true);
   int atr_handle = iATR(symbol, _Period, 14);
   CopyBuffer(atr_handle, 0, 0, 1, atr);
   
   double trail_distance = atr[0] * risk_params.trailingStopATR;
   
   bool modify_needed = false;
   double new_sl = current_sl;
   
   if(pos_type == POSITION_TYPE_BUY)
   {
      double trail_level = current_price - trail_distance;
      
      // Only trail if in profit and new SL is better
      if(current_price > open_price && trail_level > current_sl)
      {
         new_sl = trail_level;
         modify_needed = true;
      }
   }
   else // POSITION_TYPE_SELL
   {
      double trail_level = current_price + trail_distance;
      
      // Only trail if in profit and new SL is better
      if(current_price < open_price && (current_sl == 0 || trail_level < current_sl))
      {
         new_sl = trail_level;
         modify_needed = true;
      }
   }
   
   // Modify position if needed
   if(modify_needed)
   {
      MqlTradeRequest request = {};
      MqlTradeResult result = {};
      
      request.action = TRADE_ACTION_SLTP;
      request.position = position_ticket;
      request.sl = new_sl;
      request.tp = current_tp;
      request.symbol = symbol;
      
      OrderSend(request, result);
   }
}

//+------------------------------------------------------------------+
//| Risk-based trade validation                                      |
//+------------------------------------------------------------------+
bool ValidateTradeRisk(string symbol, double entry_price, double stop_loss, 
                      double take_profit, double lot_size, const RiskParameters &risk_params)
{
   // Check max open trades
   if(PositionsTotal() >= risk_params.maxOpenTrades)
      return false;
   
   // Check risk/reward ratio
   double risk = MathAbs(entry_price - stop_loss);
   double reward = MathAbs(take_profit - entry_price);
   
   if(reward < risk * 1.5) // Minimum 1.5:1 RR ratio
      return false;
   
   // Check correlation with existing positions
   int correlated_positions = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         string pos_symbol = PositionGetString(POSITION_SYMBOL);
         
         // Simple correlation check based on currency
         if(StringFind(symbol, StringSubstr(pos_symbol, 0, 3)) >= 0 ||
            StringFind(symbol, StringSubstr(pos_symbol, 3, 3)) >= 0)
         {
            correlated_positions++;
         }
      }
   }
   
   // Limit correlated positions
   if(correlated_positions >= 2)
      return false;
   
   // Validate lot size
   double min_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   
   if(lot_size < min_lot || lot_size > max_lot)
      return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Generate risk report                                             |
//+------------------------------------------------------------------+
string GenerateRiskReport()
{
   string report = "\n=== Risk Management Report ===\n";
   
   report += StringFormat("Total Trades: %d\n", g_performance.totalTrades);
   report += StringFormat("Win Rate: %.1f%%\n", g_performance.winRate * 100);
   report += StringFormat("Profit Factor: %.2f\n", g_performance.profitFactor);
   report += StringFormat("Average Win: %.2f\n", g_performance.avgWin);
   report += StringFormat("Average Loss: %.2f\n", g_performance.avgLoss);
   report += StringFormat("Largest Win: %.2f\n", g_performance.largestWin);
   report += StringFormat("Largest Loss: %.2f\n", g_performance.largestLoss);
   report += StringFormat("Max Drawdown: %.1f%%\n", g_performance.maxDrawdown);
   report += StringFormat("Current Drawdown: %.1f%%\n", g_performance.currentDrawdown);
   
   // Add recommendations
   report += "\nRecommendations:\n";
   
   if(g_performance.winRate < 0.4)
      report += "- Low win rate detected. Consider tightening entry criteria.\n";
   
   if(g_performance.profitFactor < 1.5)
      report += "- Profit factor below target. Review risk/reward ratios.\n";
   
   if(g_performance.maxDrawdown > 20)
      report += "- High drawdown detected. Consider reducing position sizes.\n";
   
   if(g_performance.avgLoss > g_performance.avgWin * 1.5)
      report += "- Average loss too high. Tighten stop loss management.\n";
   
   return report;
}