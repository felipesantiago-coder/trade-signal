'''CTEV 15m Refine — FASE 2: refinar top do v7 com mais SL/TP/be/trailing
Usa o cache de dados existente.
'''
import sys, os, time, math, json, pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJ)
from backtest import _apply_costs

FEE = 0.016; SPREAD = 2.0; SLIPPAGE = 2.0
COST_RT = (FEE*2 + SPREAD/100 + SLIPPAGE/100)/100
CACHE = '/tmp/ctev_15m_v7_data.pkl'

print(f'Custo RT: {COST_RT*100:.4f}%', flush=True)
with open(CACHE, 'rb') as f: df = pickle.load(f)['df_clean']
n = len(df); bh = (df['close'].values[-1]-df['close'].values[0])/df['close'].values[0]*100
print(f'{n}c ({n//96}d) B&H={bh:+.2f}%', flush=True)

c = df['close'].values.astype(np.float64)
hi = df['high'].values.astype(np.float64)
lo = df['low'].values.astype(np.float64)
regime = df['regime'].values
rsi_a = df['rsi'].values.astype(np.float64)
rd_a = df['rsi_delta'].values.astype(np.float64)
atr = df['atr'].values.astype(np.float64)
adx_a = df['adx'].values.astype(np.float64)
e50 = df['ema50'].values.astype(np.float64)
e200 = df['ema200'].values.astype(np.float64)
mh = df['macd_hist'].values.astype(np.float64)
ap_a = df['atr_percentile'].values.astype(np.float64)
e20 = df['ema20'].values.astype(np.float64)
bbl = df['bb_lower'].values.astype(np.float64)
bbu = df['bb_upper'].values.astype(np.float64)
bbsq = df['bb_squeeze_pct'].values.astype(np.float64)
mid = n // 2

@dataclass
class R:
    score: float = 0.0; strategy: str = ''
    total_trades: int = 0; long_trades: int = 0; short_trades: int = 0
    wins: int = 0; win_rate: float = 0.0
    profit_factor: float = 0.0; total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0; sharpe_ratio: float = 0.0
    avg_win_pct: float = 0.0; avg_loss_pct: float = 0.0
    avg_bars_held: float = 0.0; buy_hold_pct: float = 0.0
    train_pnl: float = 0.0; test_pnl: float = 0.0
    train_wr: float = 0.0; test_wr: float = 0.0
    train_trades: int = 0; test_trades: int = 0
    p: dict = None
    def to_dict(self): d = asdict(self); d['params'] = self.p; return d


def sim(entries, dirs, sls, tps, eps, mb, be_r=0.0, trailing=False):
    trades = []
    for k in range(len(entries)):
        idx=entries[k]; is_l=dirs[k]; sl=sls[k]; tp=tps[k]; ep=eps[k]
        xp=None; bars=0; mj=min(idx+mb,n); be_on=False; tsl=sl
        for j in range(idx+1, mj):
            if is_l:
                if hi[j]>=tp: xp=tp; bars=j-idx; break
                if lo[j]<=tsl: xp=tsl; bars=j-idx; break
            else:
                if lo[j]<=tp: xp=tp; bars=j-idx; break
                if hi[j]>=tsl: xp=tsl; bars=j-idx; break
            if not be_on and be_r > 0:
                ur = ((hi[j]-ep) if is_l else (ep-lo[j])) / max(atr[idx], 1e-9)
                if ur >= be_r: be_on=True; tsl=ep
            if trailing and be_on:
                if is_l and lo[j] > tsl: tsl = lo[j]
                elif not is_l and hi[j] < tsl: tsl = hi[j]
        if xp is None: bars=mj-1-idx; xp=c[mj-1]
        _, adj, _ = _apply_costs(ep, xp, is_l, FEE, SPREAD, SLIPPAGE)
        pnl = (adj-ep)/ep*100 if is_l else (ep-adj)/ep*100
        trades.append((is_l, pnl, bars))
    return trades


def calc_m(trades):
    if not trades: return None
    total=len(trades); wins=sum(1 for t in trades if t[1]>0)
    longs=sum(1 for t in trades if t[0])
    pnls=[t[1] for t in trades]
    wp=[p for p in pnls if p>0]; lp=[p for p in pnls if p<=0]
    gw=sum(wp); gl=abs(sum(lp))
    pf=gw/gl if gl>0 else 999.0; tpnl=sum(pnls)
    eq=[100.0]
    for p in pnls: eq.append(eq[-1]*(1+p/100))
    peak=np.maximum.accumulate(eq)
    dd=(peak-eq)/peak*100; mdd=float(np.max(dd))
    sh=float(np.mean(pnls)/(np.std(pnls)+1e-9)*math.sqrt(365*96/max(total,1)))
    return {'n':total,'w':wins,'wr':wins/total*100,'pf':pf,'pnl':tpnl,'dd':mdd,'sh':sh,
            'aw':np.mean(wp) if wp else 0,'al':np.mean(lp) if lp else 0,
            'ab':np.mean([t[2] for t in trades]),'l':longs,'s':total-longs}


def calc_score(r):
    if r.total_trades<100 or r.max_drawdown_pct>30: return 0.0
    if r.total_pnl_pct<=0 or r.profit_factor<0.9: return 0.0
    if r.train_pnl<=0 and r.test_pnl<=0: return 0.0
    wr=r.win_rate/100.0; pf=r.profit_factor
    dd_pen=max(0, 1.0-r.max_drawdown_pct/100.0)
    freq=min(math.log(max(r.total_trades,1))/math.log(500), 1.5)
    pnl_b=2.0 if r.total_pnl_pct>15 else (1.5 if r.total_pnl_pct>5 else 1.0)
    sig=min(math.sqrt(r.total_trades)/12.0, 2.0)
    wf=1.3 if (r.train_pnl>0 and r.test_pnl>0) else (0.8 if (r.train_pnl>0 or r.test_pnl>0) else 0.3)
    if r.long_trades>5 and r.short_trades>5:
        ratio=min(r.long_trades,r.short_trades)/max(r.long_trades,r.short_trades)
        bal=0.85+0.15*ratio
    else: bal=0.7
    return round(wr*pf*freq*dd_pen*pnl_b*sig*wf*bal, 4)


def gen_momentum(start, end, p):
    E=[]; D=[]; S=[]; T=[]; EP=[]
    sl_m=p['sl_m']; tp_m=p['tp_m']; rl=p['rsi_l']; rs=p['rsi_s']
    adx_min=p.get('adx_min',20); cd=p['cooldown']
    i=start; cooldown=0
    while i<end:
        if i<cooldown: i+=1; continue
        rg=regime[i]
        if rg=='volatile' or rg=='ranging' or i<1: i+=1; continue
        ci=c[i]; r=rsi_a[i]; a=ap_a[i]
        if a<0.1 or a>0.9: i+=1; continue
        is_l=(rg=='trending_up')
        if rg=='trending_down': is_l=False
        elif rg=='transition':
            if mh[i]>0 and mh[i-1]<=0 and c[i]>e50[i]>e200[i] and rl[0]<=r<=rl[1]: is_l=True
            elif mh[i]<0 and mh[i-1]>=0 and c[i]<e50[i]<e200[i] and rs[0]<=r<=rs[1]: is_l=False
            else: i+=1; continue
        else: i+=1; continue
        if adx_a[i]<adx_min: i+=1; continue
        rng=rl if is_l else rs
        if not (rng[0]<=r<=rng[1]): i+=1; continue
        bc=(mh[i]>0 and mh[i-1]<=0) if is_l else (mh[i]<0 and mh[i-1]>=0)
        bm=(mh[i]>0 and rd_a[i]>0) if is_l else (mh[i]<0 and rd_a[i]<0)
        if not (bc or bm): i+=1; continue
        av=atr[i]
        if is_l:
            sl=ci-sl_m*av; tp=ci+tp_m*av
            if sl<=0: i+=1; continue
        else:
            sl=ci+sl_m*av; tp=ci-tp_m*av
        E.append(i); D.append(is_l); S.append(sl); T.append(tp); EP.append(ci)
        cooldown=i+cd; i+=1
    return E, D, S, T, EP


# Refine grid based on v7 winner: momentum, rsi_l=(30,60), rsi_s=(45,75), adx_min=20
print('Building refine grid...', flush=True)
grid = []

# Widen RSI around winner
for rl in [(28,55),(30,60)]:
    for rs in [(42,72),(45,75)]:
        for sl,tp in [(0.75,1.5),(1.0,2.0),(1.0,2.5),(1.5,2.5)]:
            for mb in [24,48]:
                for cd in [4,8]:
                    for be in [0.0, 0.5, 1.0]:
                        for tr in [False, True]:
                            grid.append({'strat':'momentum','rsi_l':rl,'rsi_s':rs,'sl_m':sl,'tp_m':tp,
                                'max_bars':mb,'cooldown':cd,'adx_min':20,'be_after_r':be,'trailing':tr})

# Also try lower ADX with transition
for rl in [(28,55),(30,60)]:
    for rs in [(42,72),(45,75)]:
        for sl,tp in [(0.75,1.5),(1.0,2.0),(1.0,2.5)]:
            for mb in [24,48]:
                for cd in [4,8]:
                    for be in [0.0, 0.5]:
                        for tr in [False]:
                            for adx in [0, 15]:
                                grid.append({'strat':'momentum','rsi_l':rl,'rsi_s':rs,'sl_m':sl,'tp_m':tp,
                                    'max_bars':mb,'cooldown':cd,'adx_min':adx,'be_after_r':be,'trailing':tr})

print(f'{len(grid):,} combos', flush=True)

print(f'Simulating...', flush=True)
results = []; viable = 0; best_score = 0; best_str = ''
t0 = time.time()
for idx, p in enumerate(grid):
    mb=p['max_bars']; be=p.get('be_after_r',0.0); tr=p.get('trailing',False)
    ea,da,sa,ta,epa = gen_momentum(0, n, p)
    if len(ea)<100: continue
    et,dt,st2,tt2,ept = gen_momentum(0, mid, p)
    ete,dte,ste,tte,epte = gen_momentum(mid, n, p)
    ma=calc_m(sim(ea,da,sa,ta,epa,mb,be,tr))
    mt=calc_m(sim(et,dt,st2,tt2,ept,mb,be,tr))
    mte=calc_m(sim(ete,dte,ste,tte,epte,mb,be,tr))
    if not ma: continue
    r=R(p=p,strategy='momentum',total_trades=ma['n'],long_trades=ma['l'],short_trades=ma['s'],
        wins=ma['w'],win_rate=round(ma['wr'],2),profit_factor=round(ma['pf'],4),
        total_pnl_pct=round(ma['pnl'],4),max_drawdown_pct=round(ma['dd'],4),
        sharpe_ratio=round(ma['sh'],4),avg_win_pct=round(ma['aw'],4),
        avg_loss_pct=round(ma['al'],4),avg_bars_held=round(ma['ab'],1),buy_hold_pct=bh,
        train_pnl=round(mt['pnl'],4) if mt else 0,test_pnl=round(mte['pnl'],4) if mte else 0,
        train_wr=round(mt['wr'],2) if mt else 0,test_wr=round(mte['wr'],2) if mte else 0,
        train_trades=mt['n'] if mt else 0,test_trades=mte['n'] if mte else 0)
    r.score=calc_score(r)
    results.append(r)
    if r.score>0: viable+=1
    if r.score>best_score:
        best_score=r.score
        best_str=f'T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% tr={r.train_pnl:+.2f}% te={r.test_pnl:+.2f}% SL={r.p["sl_m"]} TP={r.p["tp_m"]} be={r.p.get("be_after_r",0)} tr={r.p.get("trailing",False)}'
    if (idx+1)%2000==0:
        el=time.time()-t0; spd=(idx+1)/max(el,0.01)
        eta=(len(grid)-idx-1)/max(spd,0.01)
        print(f'  [{idx+1:,}/{len(grid):,}] {spd:.0f}/s v={viable} ETA={eta:.0f}s | {best_str}', flush=True)

print(f'\nDone in {time.time()-t0:.1f}s | viable={viable}', flush=True)
results.sort(key=lambda x: x.score, reverse=True)

print(f'\n{"="*160}')
print(f'  TOP 25 REFINED ({viable} viable de {len(grid):,})')
print(f'{"="*160}')
print(f'{"#":>3} {"Score":>7} {"Trades":>6} {"L/S":>6} {"T/d":>5} {"WR%":>6} {"PF":>6} {"PnL%":>8} {"DD%":>6} {"Sharpe":>6} {"Train":>8} {"Test":>8} {"SL":>4} {"TP":>4} {"MB":>3} {"CD":>3} {"BE":>4} {"TR":>3} {"ADX":>3}')
for i, r in enumerate(results[:25]):
    if r.score<=0: break
    ls=f'{r.long_trades}/{r.short_trades}'; td=r.total_trades/(n/96)
    print(f'{i+1:>3} {r.score:>7.3f} {r.total_trades:>6} {ls:>6} {td:>5.2f} {r.win_rate:>6.1f} {r.profit_factor:>6.2f} '
          f'{r.total_pnl_pct:>8.2f} {r.max_drawdown_pct:>6.2f} {r.sharpe_ratio:>6.2f} '
          f'{r.train_pnl:>+8.2f} {r.test_pnl:>+8.2f} '
          f'{r.p["sl_m"]:>4.2f} {r.p["tp_m"]:>4.2f} {r.p["max_bars"]:>3} {r.p["cooldown"]:>3} '
          f'{r.p.get("be_after_r",0):>4.1f} {"T" if r.p.get("trailing") else "F":>3} {r.p.get("adx_min",20):>3}', flush=True)

winner = results[0] if results and results[0].score>0 else None
if winner:
    print(f'\n{"#"*160}')
    print(f'#  VENCEDOR 15m REFINED')
    print(f'{"#"*160}')
    for k,v in [('Score',f'{winner.score:.4f}'),
        ('Trades',f'{winner.total_trades} ({winner.total_trades/(n/96):.2f}/d) L={winner.long_trades} S={winner.short_trades}'),
        ('Win Rate',f'{winner.win_rate:.1f}%'),('PF',f'{winner.profit_factor:.2f}'),
        ('PnL',f'{winner.total_pnl_pct:+.2f}% (B&H={winner.buy_hold_pct:+.2f}%)'),
        ('Max DD',f'{winner.max_drawdown_pct:.2f}%'),('Sharpe',f'{winner.sharpe_ratio:.2f}'),
        ('Train',f'{winner.train_trades}T PnL={winner.train_pnl:+.2f}% WR={winner.train_wr:.1f}%'),
        ('Test',f'{winner.test_trades}T PnL={winner.test_pnl:+.2f}% WR={winner.test_wr:.1f}%'),
        ('Avg W/L',f'{winner.avg_win_pct:.3f}% / {winner.avg_loss_pct:.3f}%'),
        ('SL/TP',f'{winner.p["sl_m"]}x / {winner.p["tp_m"]}x ({winner.p["tp_m"]/winner.p["sl_m"]:.1f}:1)'),
        ('MaxBars',winner.p['max_bars']),('Cooldown',f'{winner.p["cooldown"]}c ({winner.p["cooldown"]*15}min)'),
        ('BE',winner.p.get('be_after_r',0)),('Trailing',winner.p.get('trailing',False)),
        ('ADX_min',winner.p.get('adx_min',20))]:
        print(f'#  {k:>12}: {v}')
    print(f'#  RSI_L: {winner.p["rsi_l"]} | RSI_S: {winner.p["rsi_s"]}')
    print(f'#  Params: {json.dumps({k:v for k,v in winner.p.items()}, default=str)}')
    print(f'{"#"*160}\n')
    out = os.path.join(PROJ, 'download'); os.makedirs(out, exist_ok=True)
    pd.DataFrame([r.to_dict() for r in results]).to_csv(os.path.join(out,'grid_15m_v7_results.csv'), index=False)
    cp={}
    for k,v in winner.p.items(): cp[k]=list(v) if isinstance(v,(tuple,list)) else v
    with open(os.path.join(out,'winner_15m.json'),'w') as f:
        json.dump({'version':'15m-v7-refined-wf','timeframe':'15m','symbol':'BTC/USDT',
            'score':winner.score,'strategy':winner.strategy,'params':cp,
            'metrics':{'total_trades':winner.total_trades,
                'trades_per_day':round(winner.total_trades/(n/96),3),
                'win_rate':winner.win_rate,'profit_factor':winner.profit_factor,
                'total_pnl_pct':winner.total_pnl_pct,'max_drawdown_pct':winner.max_drawdown_pct,
                'sharpe_ratio':winner.sharpe_ratio,'buy_hold_pct':winner.buy_hold_pct,
                'train_pnl':winner.train_pnl,'test_pnl':winner.test_pnl,
                'train_wr':winner.train_wr,'test_wr':winner.test_wr,
                'train_trades':winner.train_trades,'test_trades':winner.test_trades,
                'avg_win_pct':winner.avg_win_pct,'avg_loss_pct':winner.avg_loss_pct},
            'costs':{'fee_pct':FEE,'spread_bps':SPREAD,'slippage_bps':SLIPPAGE,'rt_pct':COST_RT*100},
            'grid':{'total':len(grid),'viable':viable,'time_sec':round(time.time()-t0,1)}},f,indent=2)
    print('Saved: winner_15m.json + grid_15m_v7_results.csv')
else:
    print(f'\n!!! No viable !!!')
    bp=sorted(results, key=lambda x: x.total_pnl_pct, reverse=True)
    for r in bp[:10]:
        print(f'  T={r.total_trades} WR={r.win_rate:.1f}% PF={r.profit_factor:.2f} PnL={r.total_pnl_pct:+.2f}% tr={r.train_pnl:+.2f}% te={r.test_pnl:+.2f}%')