# 🚀 HYBRID APPROACH - WEEKLY ROADMAP

**Start Date:** Feb 3, 2026  
**Goal:** Live trading ready in 3-4 weeks with profitable strategy

---

## 📅 WEEK 1 (Feb 3-9): Backtest Optimization Phase

### Priority: Find a PROFITABLE strategy

**Current Status:**
- EMA(32/45) → -0.11% (LOSING) ❌
- Need to find parameters with:
  - Win rate > 40%
  - Profit factor > 1.2
  - Sharpe ratio > 0.5

### Tasks:

#### Day 1-2 (Today): Parameter Sweep
- [ ] Run `run_optimization_fast.py` with wider EMA ranges
  - Fast: 5-50 (step 2)
  - Slow: 20-200 (step 10)
  - ~437 combinations to test
- [ ] Test multiple EMA strategies:
  - Very fast: EMA(5-15) / EMA(20-40)
  - Medium: EMA(15-30) / EMA(40-80)
  - Slow: EMA(30-50) / EMA(100-200)
- [ ] Save **top 10 results**

#### Day 3-4: Evaluate Best Candidates
- [ ] Take top 3 EMA pairs from optimization
- [ ] **Walk-forward validation** on each:
  - Split data: Train (60 days) → Test (30 days) → OOS (30 days)
  - Must perform similar in OOS as training
  - Check for overfitting
- [ ] Select **BEST performer** (highest Sharpe in OOS)

#### Day 5-7: Risk Analysis & Fine-tuning
- [ ] For best EMA:
  - [ ] Adjust stop-loss: 3% → 1.5% (reduce drawdown)
  - [ ] Adjust take-profit: 5% → 10% (bigger winners)
  - [ ] Run sensitivity analysis
  - [ ] Check drawdown periods
- [ ] Generate detailed backtest report

**Week 1 Goal:** Have **1 profitable strategy** ready

---

## 📅 WEEK 2 (Feb 10-16): Paper Trading Setup

### Objective: Match backtest results in real market

#### Day 1-2: Environment Setup
- [ ] Implement live data feed (CCXT streaming)
  - Real-time 1h candles
  - Proper timestamp handling
  - Latency tolerance

#### Day 3-5: Paper Trading Engine
- [ ] Create `paper_trader.py` that:
  - Pulls live CCXT data
  - Runs strategy in real-time (not backtest)
  - Simulates order execution
  - Tracks fills, commissions, slippage
  - Sends **Telegram alerts** on trades
- [ ] Test for 2-3 days
  - Monitor logs
  - Verify candle timing
  - Check order mechanics

#### Day 6-7: Validation
- [ ] Compare:
  - Historical backtest metrics
  - vs. Paper trading metrics (first week)
  - Should be **similar** (±2-3% variance)
- [ ] If matches → proceed to live
- [ ] If different → debug & adjust

**Week 2 Goal:** Paper trading running, metrics validated

---

## 📅 WEEK 3 (Feb 17-23): Small Live Trading

### Objective: Real money, small size, risk mitigation

#### Day 1-2: Account Setup
- [ ] Configure Binance margin account connection
- [ ] Set trading capital: **$100-500** (test amount)
- [ ] Strict position size: **0.001-0.01 BTC max**
- [ ] Emergency stop-loss at account level

#### Day 3-7: Live Trading Monitoring
- [ ] Deploy to production
- [ ] Monitor 24/7 (set Telegram alerts)
- [ ] Track all trades in database
- [ ] Daily P&L review
- [ ] Document any issues

**Week 3 Goal:** 10-20 real trades, validated system

---

## 📅 WEEK 4+ (Feb 24+): Scale & Optimize

- [ ] Increase position size gradually
- [ ] Add more strategies (multi-asset)
- [ ] Implement advanced risk management
- [ ] Set up performance dashboard

---

## 🔥 CRITICAL SUCCESS FACTORS

| Factor | Target | Current | Status |
|--------|--------|---------|--------|
| Win Rate | >40% | 7.7% | ❌ NEEDS FIX |
| Profit Factor | >1.2 | 0.39 | ❌ NEEDS FIX |
| Sharpe Ratio | >0.5 | N/A | ⚠️ TBD |
| Max Drawdown | <15% | N/A | ⚠️ TBD |
| Backtest vs Paper Match | <5% diff | N/A | ⚠️ TBD |

---

## 💻 CODE CHANGES NEEDED

### 1. Fast Optimization Script (DONE)
```python
run_optimization_fast.py  # 437 combinations in 2-3 min
```

### 2. Live Data Feed (TODO)
```python
src/data/stream.py  # Real-time CCXT candles
```

### 3. Paper Trading Engine (TODO)
```python
scripts/paper_trader.py  # Simulate live trading
```

### 4. Live Trader (TODO)
```python
scripts/live_trader.py  # Real order execution
```

### 5. Monitoring Dashboard (TODO)
```python
src/monitoring/dashboard.py  # Grafana/web dashboard
```

---

## 📊 SUCCESS METRICS

**Weekly Goals:**

| Week | Metric | Target |
|------|--------|--------|
| 1 | Find profitable params | Sharpe > 0.5 in OOS |
| 2 | Paper trading match | Backtest ≈ Paper ±2-3% |
| 3 | Live P&L | Positive in first 20 trades |
| 4+ | Consistent returns | Monthly Sharpe > 1.0 |

---

## ⚠️ RISK MITIGATION

- Start with **small capital** ($100)
- **Hard stop-loss** at -2% account loss
- **Position limits** to 0.5% account per trade
- **Daily review** of all trades
- **Immediate rollback** if something breaks

---

## 🎯 SUCCESS DEFINITION

✅ **Live Trading Success** = First month with:
- At least 15 trades
- Win rate > 35%
- Positive P&L
- Zero system errors
- All Telegram alerts working

Then: Scale to $1,000+ capital

---

**Last Updated:** Feb 3, 2026  
**Next Review:** After Week 1 optimization complete
