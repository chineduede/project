# Quick Start Trading Guide

## 5-Minute Setup

### 1. Installation
```
1. Copy all files to: C:\[MT5 Folder]\MQL5\Experts\
2. Open MetaEditor (F4 in MT5)
3. Open MLMeanReversionStrategy.mq5
4. Compile (F7) - Should show "0 errors"
5. Restart MT5
```

### 2. Attach to Chart
```
1. Open EURUSD M15 chart
2. Drag "MLMeanReversionStrategy" from Navigator
3. Enable "Allow algorithmic trading" ✓
4. Use these starter settings:
   - Risk per trade: 1%
   - Max positions: 2
   - Enable ML: Yes
   - Use microstructure: Yes
```

### 3. First Week Protocol
- **Monday**: Run on demo account, observe signals
- **Tuesday-Wednesday**: Note win/loss patterns
- **Thursday**: Adjust Z-score thresholds if needed
- **Friday**: Review week, prepare for live

## Critical Rules for Success

### The 3 NEVERS
1. **NEVER** trade during NFP (first Friday, 8:30 AM EST)
2. **NEVER** risk more than 2% per trade
3. **NEVER** have more than 6% total exposure

### The 3 ALWAYS
1. **ALWAYS** check economic calendar before trading
2. **ALWAYS** wait for all confirmations
3. **ALWAYS** respect the stop loss

## Signal Quality Guide

### 🟢 BEST Signals (Take these!)
- Z-score > 2.5 or < -2.5
- Toxicity < 50
- Confidence > 80%
- No news in next 4 hours
- Market regime = RANGING

### 🟡 GOOD Signals (Consider these)
- Z-score > 2.0 or < -2.0
- Toxicity < 70
- Confidence > 65%
- Minor news only
- Low volatility

### 🔴 AVOID Signals (Skip these!)
- Any signal 30min before/after major news
- Toxicity > 70
- Friday afternoon (weird flows)
- Month-end (rebalancing)
- Confidence < 60%

## Time Zone Guide

### Best Trading Hours (EST)
- **London Open**: 3:00 AM - 5:00 AM ⭐⭐⭐⭐⭐
- **NY Overlap**: 8:00 AM - 11:00 AM ⭐⭐⭐⭐
- **London Close**: 11:00 AM - 12:00 PM ⭐⭐⭐
- **NY Afternoon**: 1:00 PM - 3:00 PM ⭐⭐

### Avoid These Times
- **Asian Lunch**: 11:00 PM - 1:00 AM (thin)
- **NY Lunch**: 12:00 PM - 1:00 PM (choppy)
- **Friday PM**: After 3:00 PM (positioning)
- **Sunday Open**: First 2 hours (gaps)

## Money Management

### Account Size Guidelines
- **$1,000-5,000**: 0.01 lot per $1,000, 1 position max
- **$5,000-10,000**: 0.5% risk per trade, 2 positions max
- **$10,000-25,000**: 1% risk per trade, 3 positions max
- **$25,000+**: Full system, all features enabled

### Progressive Scaling
Week 1-2: Trade 25% of calculated size
Week 3-4: Trade 50% of calculated size
Week 5-6: Trade 75% of calculated size
Week 7+: Trade 100% if profitable

## Emergency Procedures

### System Says "No Trades"
**Reasons**:
- Market trending strongly (check D1 chart)
- High toxicity environment
- Recent news impact
- System protecting capital

**Action**: WAIT. No trades is a position.

### Losing Streak (3+ losses)
1. Reduce size by 50%
2. Check if market regime changed
3. Verify no news was missed
4. Wait for A+ setups only
5. Consider break if 5+ losses

### Technical Issues
- "Array out of range": Needs more historical data
- "Not enough bars": Wait or load more history
- "Zero divide": Check ATR indicator working
- Frozen chart: Restart MT5, check connection

## Daily Routine (10 minutes)

### Morning (Before London)
- [ ] Check economic calendar
- [ ] Note support/resistance levels
- [ ] Review overnight moves
- [ ] Set daily risk limit

### Midday (After London)
- [ ] Review morning trades
- [ ] Adjust parameters if needed
- [ ] Check correlation changes

### Evening (After NY)
- [ ] Log trades in journal
- [ ] Calculate daily P&L
- [ ] Plan tomorrow's approach

## Performance Expectations

### Realistic Monthly Targets
- **Month 1**: Break even (learning)
- **Month 2**: 2-5% return
- **Month 3**: 5-8% return
- **Month 4+**: 8-15% return

### Warning Signs
- Win rate < 50% for 2 weeks
- Daily losses > 3% occurring
- Same mistakes repeating
- Emotional trading starting

## Top 10 Success Tips

1. **Start Small**: Your ego wants big trades, your account wants survival
2. **Trust the System**: It sees patterns you can't
3. **News is King**: No signal beats high-impact news
4. **Patience Pays**: 3 good trades beat 10 mediocre ones
5. **Document Everything**: Screenshot trades, note thoughts
6. **Correlation Kills**: Don't trade EURUSD + GBPUSD together
7. **Toxicity = Death**: High toxicity means pros are hunting
8. **Respect Drawdowns**: They're normal, panic isn't
9. **Update Monthly**: Markets evolve, so should settings
10. **Capital First**: You can't trade if you're broke

## Cheat Sheet

### Quick Signal Check
```
Z-Score: Beyond ±2? ✓
Toxicity: Under 70? ✓
News: None in 2hr? ✓
Regime: Ranging? ✓
Confidence: Over 65%? ✓
→ TAKE THE TRADE
```

### Position Size Formula
```
Risk = Account × 1%
Lots = Risk ÷ (Stop Loss Pips × Pip Value)
Final = Lots × Confidence %
```

### Stop Loss Rules
- Minimum: 1.5 × ATR
- Toxicity > 50: Add 20%
- News day: Add 30%
- Never move against you

## Remember

> "The market rewards discipline, not intelligence."

This system is like a Formula 1 car - incredibly powerful in the right hands, but requires respect and skill. Start slowly, build confidence, then accelerate.

Your first goal isn't to make money - it's to not lose money while learning. Profits come to those who survive long enough to compound them.

Good luck, and may the probabilities be with you! 🎯