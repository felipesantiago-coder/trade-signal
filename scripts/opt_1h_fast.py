"""Fast 1h BTC/USDT optimizer — focused on beating B&H."""
import sys, os, time, math, json, pickle, logging
from dataclasses import dataclass
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('opt')
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators import compute_indicators
from backtest import (fetch_historical_ohlcv, calculate_metrics, TradeResult,
    _apply_costs, DEFAULT_FEE_PCT, DEFAULT_SPREAD_BPS, DEFAULT_SLIPPAGE_BPS)

DATA_CACHE = '/tmp/ctev_1h_study_data.pkl'
DAYS = 730

def load_data():
    if os.path.exists(DATA_CACHE):
        with open(DATA_CACHE, 'rb') as f: df = pickle.load(f)
    else:
        df = fetch_historical_ohlcv('BTC/USDT', '1h', DAYS)
        with open(DATA_CACHE, 'wb') as f: pickle.dump(df, f)
    return df

@dataclass
class R:
    cid: int; trades: int=0; wins: int=0; losses: int=0
    wr: float=0.0; pf: float=0.0; pnl: float=0.0; dd: float=0.0
    sharpe: float=0.0; bh: float=0.0; score: float=0.0
    longs: int=0; shorts: int=0
    avg_bars: float=0.0; avg_win: float=0.0; avg_loss: float=0.0
    def to_dict(self): return {k: getattr(self, k) for k in self.__dataclass_fields__}

class BT:
    def __init__(self, df):
        df_ind = compute_indicators(df, '1h')
        cc = ['ema20','ema50','ema200','rsi','atr','atr_percentile','macd','macd_signal','macd_hist','adx','plus_di','minus_di','regime']
        self.dc = df_ind.dropna(subset=cc).copy()
        self.n = len(self.dc)
        self.cl = self.dc['close'].values.astype(np.float64)
        self.hi = self.dc['high'].values.astype(np.float64)
        self.lo = self.dc['low'].values.astype(np.float64)
        self.e20 = self.dc['ema20'].values.astype(np.float64)
        self.e50 = self.dc['ema50'].values.astype(np.float64)
        self.e200 = self.dc['ema200'].values.astype(np.float64)
        self.rsi = self.dc['rsi'].values.astype(np.float64)
        self.atr = self.dc['atr'].values.astype(np.float64)
        self.apct = self.dc['atr_percentile'].values.astype(np.float64)
        self.adx = self.dc['adx'].values.astype(np.float64)
        self.pdi = self.dc['plus_di'].values.astype(np.float64)
        self.mdi = self.dc['minus_di'].values.astype(np.float64)
        self.slope = self.dc['ema50_slope'].values.astype(np.float64)
        self.bbl = self.dc['bb_lower'].values.astype(np.float64)
        self.bbu = self.dc['bb_upper'].values.astype(np.float64)
        self.vol = self.dc['volume'].values.astype(np.float64)
        self.vs50 = self.dc['volume_sma50'].values.astype(np.float64)
        self.f382 = self.dc['fib_0382'].values.astype(np.float64)
        self.f500 = self.dc['fib_0500'].values.astype(np.float64)
        self.f618 = self.dc['fib_0618'].values.astype(np.float64)
        self.fdir = self.dc['fib_direction'].values.astype(np.int32)
        self.e20t = self.dc['ema20_touched'].values.astype(bool)
        self.e50t = self.dc['ema50_touched'].values.astype(bool)
        self.e50tu = self.dc['ema50_touched_up'].values.astype(bool)
        self.rg = np.array([str(r) for r in self.dc['regime'].values])
        self.idx = self.dc.index
        self.bh = (self.cl[-1] - self.cl[0]) / self.cl[0] * 100
        logger.info('Data: %d candles, B&H=%.2f%%', self.n, self.bh)

    def sim(self, p):
        trades=[]; i=0; n=self.n; cl=self.cl; hi=self.hi; lo=self.lo
        while i < n:
            sig=None; is_long=True; rg=self.rg[i]
            mr=p.get('mr',False); ar=p.get('ar',False); av=p.get('av',False); atr=p.get('atr',False)
            sl_m=p['sl']; tp_m=p['tp']
            rsi_lo=p.get('rl',30); rsi_hi=p.get('rh',50); rsi_so=p.get('sl2',50); rsi_sh=p.get('sh',70)
            adx_min=p.get('adx',0); ap_lo=p.get('apl',0.05); ap_hi=p.get('aph',0.95)
            req_et=p.get('et',True); req_sl=p.get('es',True); req_pb=p.get('pb',True)
            fib_tol=p.get('ft',0.025)

            if rg=='ranging':
                if mr:
                    c=cl[i]; r=self.rsi[i]
                    if r<=p.get('mrl',35) and lo[i]<=self.bbl[i]:
                        ep=c; sl=ep-sl_m*self.atr[i]; tp=ep+tp_m*self.atr[i]; is_long=True; sig=(ep,sl,tp)
                    elif r>=p.get('mrs',65) and hi[i]>=self.bbu[i]:
                        ep=c; sl=ep+sl_m*self.atr[i]; tp=ep-tp_m*self.atr[i]; is_long=False; sig=(ep,sl,tp)
                elif ar:
                    sig=self._trend(i,p,is_long_ref=None)
                    if sig: is_long=True
                    else: sig=self._trend(i,p,is_long_ref=False); is_long=False
            elif rg=='volatile' and av:
                sig=self._trend(i,p,is_long_ref=None)
                if sig: is_long=True
                else: sig=self._trend(i,p,is_long_ref=False); is_long=False
            elif rg in ('trending_up','trending_down','transition'):
                if atr and rg=='transition' or rg in ('trending_up','trending_down'):
                    sig=self._trend(i,p,is_long_ref=None)
                    if sig: is_long=True
                    else: sig=self._trend(i,p,is_long_ref=False); is_long=False

            if sig is None: i+=1; continue
            ep,sl,tp=sig; xp=None; bars=0; mj=min(i+72,n)
            if is_long:
                for j in range(i+1,mj):
                    if lo[j]<=sl: xp=sl; bars=j-i; break
                    if hi[j]>=tp: xp=tp; bars=j-i; break
            else:
                for j in range(i+1,mj):
                    if hi[j]>=sl: xp=sl; bars=j-i; break
                    if lo[j]<=tp: xp=tp; bars=j-i; break
            if xp is None: bars=mj-1-i; xp=cl[mj-1]
            _,adj,_=_apply_costs(ep,xp,is_long,DEFAULT_FEE_PCT,DEFAULT_SPREAD_BPS,DEFAULT_SLIPPAGE_BPS)
            pnl=(adj-ep)/ep*100 if is_long else (ep-adj)/ep*100
            trades.append(TradeResult(entry_ts=self.idx[i],exit_ts=self.idx[min(i+bars,n-1)],
                type='LONG' if is_long else 'SHORT',entry_price=ep,exit_price=xp,
                stop_loss=sl,take_profit=tp,atr=self.atr[i],rsi=self.rsi[i],
                pnl_pct=round(pnl,4),pnl_abs=round(xp-ep,2),bars_held=bars,exit_reason='x'))
            i+=bars+1
        return trades

    def _trend(self, i, p, is_long_ref=None):
        rg=self.rg[i]; c=self.cl[i]; e50=self.e50[i]; e200=self.e200[i]
        adx_min=p.get('adx',0); rsi_lo=p.get('rl',30); rsi_hi=p.get('rh',50)
        rsi_so=p.get('sl2',50); rsi_sh=p.get('sh',70)
        req_et=p.get('et',True); req_sl=p.get('es',True); req_pb=p.get('pb',True)
        fib_tol=p.get('ft',0.025); sl_m=p['sl']; tp_m=p['tp']
        ap_lo=p.get('apl',0.05); ap_hi=p.get('aph',0.95)
        atr_flag=p.get('atr',False)

        if is_long_ref is None:
            # Try long first
            r = self._check_long(i,rg,c,e50,e200,adx_min,rsi_lo,rsi_hi,req_et,req_sl,req_pb,fib_tol,ap_lo,ap_hi,atr_flag,p)
            if r: return r
            return self._check_short(i,rg,c,e50,e200,adx_min,rsi_so,rsi_sh,req_et,req_sl,req_pb,fib_tol,ap_lo,ap_hi,atr_flag,p)
        elif is_long_ref:
            return self._check_long(i,rg,c,e50,e200,adx_min,rsi_lo,rsi_hi,req_et,req_sl,req_pb,fib_tol,ap_lo,ap_hi,atr_flag,p)
        else:
            return self._check_short(i,rg,c,e50,e200,adx_min,rsi_so,rsi_sh,req_et,req_sl,req_pb,fib_tol,ap_lo,ap_hi,atr_flag,p)

    def _check_long(self,i,rg,c,e50,e200,adx_min,rsi_lo,rsi_hi,req_et,req_sl,req_pb,fib_tol,ap_lo,ap_hi,atr_flag,p):
        if rg=='trending_up':
            if self.adx[i]<adx_min: return None
        elif rg=='transition':
            if not atr_flag: return None
        else: return None
        if req_et and not (c>e50>e200): return None
        if req_sl and self.slope[i]<=-1.0: return None
        if req_pb:
            pb=False; fd=self.fdir[i]
            if fd==1:
                f38=self.f382[i]; f61=self.f618[i]
                if not (np.isnan(f38) or np.isnan(f61)) and f61<=c<=f38: pb=True
                if not pb:
                    lo=self.lo[i]; tol=c*fib_tol
                    for fl in (f38,self.f500[i],f61):
                        if not np.isnan(fl) and fl>0 and abs(lo-fl)<=tol: pb=True; break
            if not pb and self.e20t[i] and c>self.e20[i]: pb=True
            if not pb and self.e50t[i] and c>e50: pb=True
            if not pb: return None
        r=self.rsi[i]
        if not (rsi_lo<=r<=rsi_hi): return None
        ap=self.apct[i]
        if ap<ap_lo or ap>ap_hi: return None
        ep=c; sl=ep-p['sl']*self.atr[i]; tp=ep+p['tp']*self.atr[i]
        return (ep,sl,tp) if sl>0 else None

    def _check_short(self,i,rg,c,e50,e200,adx_min,rsi_lo,rsi_hi,req_et,req_sl,req_pb,fib_tol,ap_lo,ap_hi,atr_flag,p):
        if rg=='trending_down':
            if self.adx[i]<adx_min: return None
        elif rg=='transition':
            if not atr_flag: return None
        else: return None
        if req_et and not (c<e50<e200): return None
        if req_sl and self.slope[i]>=1.0: return None
        if req_pb:
            pb=False; fd=self.fdir[i]
            if fd==-1:
                f38=self.f382[i]; f61=self.f618[i]
                if not (np.isnan(f38) or np.isnan(f61)) and f61<=c<=f38: pb=True
                if not pb:
                    hi=self.hi[i]; tol=c*fib_tol
                    for fl in (f38,self.f500[i],f61):
                        if not np.isnan(fl) and fl>0 and abs(hi-fl)<=tol: pb=True; break
            if not pb and self.e20t[i] and c<self.e20[i] and self.hi[i]>=self.e20[i]: pb=True
            if not pb and self.e50tu[i] and c<e50 and self.hi[i]>=e50: pb=True
            if not pb: return None
        r=self.rsi[i]
        if not (rsi_lo<=r<=rsi_hi): return None
        ap=self.apct[i]
        if ap<ap_lo or ap>ap_hi: return None
        ep=c; sl=ep+p['sl']*self.atr[i]; tp=ep-p['tp']*self.atr[i]
        return (ep,sl,tp)

    def run(self, cid, p):
        trades=self.sim(p)
        m=calculate_metrics(trades,self.dc)
        r=R(cid=cid,trades=m.total_trades,wins=m.wins,losses=m.losses,
            wr=round(m.win_rate,2),pf=round(m.profit_factor,4),pnl=round(m.total_pnl_pct,4),
            dd=round(m.max_drawdown_pct,4),sharpe=round(m.sharpe_ratio,4),bh=round(self.bh,4),
            longs=m.long_trades,shorts=m.short_trades,avg_bars=round(m.avg_bars_held,1),
            avg_win=round(m.avg_win_pct,4),avg_loss=round(m.avg_loss_pct,4))
        if r.trades<30 or r.pf<=1.0 or r.wr<25 or r.dd>35: r.score=0
        else:
            excess=r.pnl-r.bh
            bh_f=max(0.01,1.0+excess/10.0) if excess<=0 else 1.0+excess/5.0
            r.score=round((r.wr/100)*r.pf*bh_f*max(0.01,1.0-r.dd/50)*min(math.log(max(r.trades,1))/math.log(200),1.5),4)
        return r


def main():
    t0=time.time()
    print('#'*120)
    print('#  BTC/USDT 1H OPTIMIZER — Beat Buy & Hold')
    print('#'*120)

    df=load_data()
    bt=BT(df)

    # Build grid
    combos=[]
    # Trend configs: (adx, rl, rh, sl2, sh, et, es, pb, atr_flag)
    tcfgs=[
        (25,(25,50),(50,75),True,True,True,True,True),
        (20,(25,50),(50,75),True,True,True,True,True),
        (30,(25,50),(50,75),True,True,True,True,True),
        (15,(25,50),(50,75),True,True,True,True,True),
        (25,(20,55),(45,80),True,True,True,True,True),
        (20,(20,55),(45,80),True,True,True,True,True),
        (25,(25,50),(50,75),True,False,True,True,True),
        (20,(20,55),(45,80),True,False,True,True,True),
        (25,(25,50),(50,75),False,True,True,True,True),
        (20,(25,55),(45,75),False,True,True,True,True),
        (25,(30,50),(50,70),True,True,False,True,True),
        (20,(30,50),(50,70),True,True,False,True,True),
        (25,(28,42),(58,72),True,True,True,True,True),
        (20,(28,42),(58,72),True,True,True,True,True),
        (25,(25,55),(45,75),True,True,False,False,True),
        (20,(25,60),(40,75),False,False,False,False,True),
        (0,(30,65),(35,70),False,False,False,False,True),
        (25,(28,48),(52,72),True,True,True,True,True),
        (15,(28,48),(52,72),True,True,True,True,True),
    ]
    sltps=[(0.75,3),(0.75,4),(0.75,5),(0.75,6),(1.0,3),(1.0,4),(1.0,5),(1.0,6),(1.0,7),(1.0,8),
           (1.25,4),(1.25,5),(1.25,6),(1.25,7),(1.5,4),(1.5,5),(1.5,6),(1.5,7),(1.5,8),
           (2.0,5),(2.0,6),(2.0,7),(2.0,8),(2.0,10),(2.5,6),(2.5,8),(2.5,10)]
    mr_sltps=[(0.5,1),(0.5,1.5),(0.5,2),(0.75,1.25),(0.75,1.5),(0.75,2),(1.0,1.5),(1.0,2),(1.25,2)]
    mrls=[30,35,40,45]; mrss=[55,60,65,70]
    atrs=[(0.05,0.95),(0.10,0.90)]

    cid=0
    for adx,(rl,rh),(sl2,sh),et,es,pb,_,atr_flag in tcfgs:
        for sl_m,tp_m in sltps:
            for apl,aph in atrs:
                combos.append({'adx':adx,'rl':rl,'rh':rh,'sl2':sl2,'sh':sh,'et':et,'es':es,'pb':pb,
                    'atr':atr_flag,'ar':False,'av':False,'mr':False,
                    'sl':sl_m,'tp':tp_m,'apl':apl,'aph':aph,'ft':0.025,
                    'mrl':35,'mrs':65})
                cid+=1
        for sl_m,tp_m in mr_sltps:
            for mrl in mrls:
                for mrs in mrss:
                    combos.append({'adx':adx,'rl':rl,'rh':rh,'sl2':sl2,'sh':sh,'et':et,'es':es,'pb':pb,
                        'atr':atr_flag,'ar':True,'av':False,'mr':True,
                        'sl':sl_m,'tp':tp_m,'apl':0.05,'aph':0.95,'ft':0.025,
                        'mrl':mrl,'mrs':mrs})
                    cid+=1

    print(f'\nGrid: {len(combos):,} combos')
    print(f'\nRunning P1...')
    results=[]; best_score=0; best_str=''
    t1=time.time()
    for idx,p in enumerate(combos):
        r=bt.run(idx,p)
        results.append(r)
        if r.score>best_score:
            best_score=r.score; excess=r.pnl-r.bh
            best_str=f'T={r.trades} WR={r.wr:.1f}% PF={r.pf:.2f} PnL={r.pnl:+.2f}% B&H={r.bh:+.2f}% Exc={excess:+.2f}% MR={p.get("mr",False)} SL={p["sl"]:.1f} TP={p["tp"]:.1f}'
        if (idx+1)%3000==0:
            el=time.time()-t1; spd=(idx+1)/max(el,0.01)
            print(f'  [{idx+1:,}/{len(combos):,}] {spd:.0f}/s BEST: {best_str}')
    print(f'  P1 done in {time.time()-t1:.1f}s')

    # Sort and show top
    results.sort(key=lambda x: x.score, reverse=True)
    print(f'\n{"="*150}')
    print(f'  TOP 30 RESULTS')
    print(f'{"="*150}')
    print(f'{"#":>3} {"Score":>7} {"T":>5} {"L/S":>6} {"WR%":>6} {"PF":>6} {"PnL%":>8} {"B&H%":>7} {"Excess":>7} {"DD%":>6} {"Sharpe":>7} {"MR":>3} {"Atr":>4} {"ET":>3} {"ES":>3} {"PB":>3} {"ADX":>4} {"RSI_L":>8} {"RSI_S":>8} {"SL":>4} {"TP":>4}')
    print(f'{"-"*150}')
    for i,r in enumerate(results[:30]):
        p_orig = combos[r.cid] if r.cid < len(combos) else {}
        mr_s = 'Y' if p_orig.get('mr') else 'N'
        atr_s = 'Y' if p_orig.get('atr') else 'N'
        et_s = 'Y' if p_orig.get('et') else 'N'
        es_s = 'Y' if p_orig.get('es') else 'N'
        pb_s = 'Y' if p_orig.get('pb') else 'N'
        rl_s = f"{p_orig.get('rl',0)}-{p_orig.get('rh',0)}"
        rs_s = f"{p_orig.get('sl2',0)}-{p_orig.get('sh',0)}"
        ls_s = f"{r.longs}/{r.shorts}"
        exc = r.pnl - r.bh
        print(f'{i+1:>3} {r.score:>7.3f} {r.trades:>5} {ls_s:>6} {r.wr:>6.1f} {r.pf:>6.2f} {r.pnl:>8.2f} {r.bh:>7.2f} {exc:>+7.2f} {r.dd:>6.2f} {r.sharpe:>7.2f} {mr_s:>3} {atr_s:>4} {et_s:>3} {es_s:>3} {pb_s:>3} {p_orig.get("adx",0):>4.0f} {rl_s:>8} {rs_s:>8} {p_orig.get("sl",0):>4.1f} {p_orig.get("tp",0):>4.1f}')

    # Save
    out=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'download')
    os.makedirs(out, exist_ok=True)
    pd.DataFrame([r.to_dict() for r in results]).to_csv(os.path.join(out, 'grid_1h_fast_results.csv'), index=False)
    if results and results[0].score > 0:
        w=results[0]; wp=combos[w.cid]
        wdata={'score':w.score,'params':wp,'metrics':{'trades':w.trades,'wr':w.wr,'pf':w.pf,'pnl':w.pnl,'bh':w.bh,'dd':w.dd,'sharpe':w.sharpe,'excess':w.pnl-w.bh}}
        with open(os.path.join(out,'winner_1h_fast.json'),'w') as f: json.dump(wdata,f,indent=2)
        print(f'\nSaved to {out}')
        print(f'\nWINNER: {best_str}')
    tt=time.time()-t0
    print(f'\nTotal: {tt:.0f}s ({tt/60:.1f}min)')

if __name__=='__main__': main()