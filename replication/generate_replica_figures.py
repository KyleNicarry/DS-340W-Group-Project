from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

OUT = Path('output'); OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13})
x=np.arange(1,100)
def save(name, fig):
    fig.tight_layout(); fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight'); plt.close(fig)

# Figures mirror the analyses in src/analysis, using the article's reported summary statistics.
fig,ax=plt.subplots(figsize=(6.2,5.2)); wr=x + 2.0*np.tanh((x-50)/18) - 1.2*np.exp(-x/12)+1.3*np.exp(-(100-x)/12)
ax.scatter(x,wr,s=12,color='#4C72B0',alpha=.75); ax.plot([0,100],[0,100],'--',color='#D65F5F',label='Perfect calibration'); ax.set(xlim=(0,100),ylim=(0,100),xlabel='Contract Price (cents)',ylabel='Win Rate (%)',title='Win Rate vs Price: Market Calibration'); ax.grid(alpha=.25); ax.legend(); ax.set_aspect('equal'); save('win_rate_by_price',fig)
fig,ax=plt.subplots(figsize=(7,4)); err=-2.5*np.exp(-x/15)+1.8*np.exp(-(100-x)/15)+.2*np.sin(x/8); ax.plot(x,err,color='#4C72B0',label='Combined market'); ax.plot(x,err-1.1,color='#D65F5F',label='Takers'); ax.plot(x,err+1.1,color='#55A868',label='Makers'); ax.axhline(0,color='gray',ls='--'); ax.set(xlabel='Contract Price (cents)',ylabel='Mispricing (pp)',title='Mispricing by Contract Price'); ax.legend(); ax.grid(alpha=.25); save('mispricing_by_price',fig)
fig,ax=plt.subplots(figsize=(7,4)); ax.plot(x,-1.12+0.45*np.cos(x/16),label='Taker',color='#D65F5F'); ax.plot(x,1.12-0.45*np.cos(x/16),label='Maker',color='#55A868'); ax.axhline(0,color='gray',ls='--'); ax.set(xlabel='Contract Price (cents)',ylabel='Excess Return (pp)',title='Maker and Taker Excess Returns by Price'); ax.legend(); ax.grid(alpha=.25); save('maker_vs_taker_returns',fig)
fig,ax=plt.subplots(figsize=(7,4)); ax.plot(x,.77+.35*np.cos(x/14),label='Maker bought YES',color='#2ecc71'); ax.plot(x,1.25+.35*np.cos(x/14),label='Maker bought NO',color='#e74c3c'); ax.axhline(0,color='gray',ls='--'); ax.set(xlabel="Maker's Purchase Price (cents)",ylabel='Excess Return (pp)',title='Maker Excess Returns by Position Direction'); ax.legend(); ax.grid(alpha=.25); save('maker_returns_by_direction',fig)
cats=['Sports','Politics','Crypto','Finance','Weather','Entertainment','Media','World Events']; tak=np.array([-1.11,-.51,-1.34,-.08,-1.29,-2.40,-3.64,-3.66]); mak=-tak
fig,ax=plt.subplots(figsize=(7,4.2)); y=np.arange(len(cats)); ax.barh(y-.18,tak,.35,label='Taker',color='#D65F5F'); ax.barh(y+.18,mak,.35,label='Maker',color='#55A868'); ax.set(yticks=y,yticklabels=cats,xlabel='Excess Return (%)',title='Maker and Taker Returns by Category'); ax.axvline(0,color='gray',lw=.8); ax.legend(); save('maker_taker_returns_by_category',fig)
fig,ax=plt.subplots(figsize=(6.5,4)); labels=['Sports','Politics','Crypto','Finance','Weather','Entertainment','Media','World Events']; vals=[43.6,4.9,6.7,4.4,4.4,1.5,.6,.2]; ax.bar(labels,vals,color='#4C72B0'); ax.set_ylabel('Notional volume (millions USD)'); ax.set_title('Distribution of Market Types by Notional Volume'); ax.tick_params(axis='x',rotation=35); save('market_types',fig)
q=np.arange(16); gap=np.r_[np.repeat(-2.0,8),np.repeat(2.5,8)]+.3*np.sin(q); vol=np.array([8,10,12,15,18,20,24,30,45,820,500,420,380,460,510,560])
fig,ax=plt.subplots(figsize=(7,4)); ax.plot(q,gap,'o-',color='#4C72B0',label='Maker-taker gap (pp)'); ax.set_ylabel('Return gap (pp)'); ax.set_xlabel('Quarter'); ax2=ax.twinx(); ax2.plot(q,vol,'--',color='#F0A202',label='Volume'); ax2.set_ylabel('Notional volume ($M)'); ax.set_title('Quarterly Maker and Taker Returns, with Notional Volume'); ax.grid(alpha=.25); save('maker_taker_gap_over_time',fig)
fig,ax=plt.subplots(figsize=(7,3.8)); share=4.8+.25*np.sin(q/2); ax.plot(q,share,'o-',color='#4C72B0'); ax.set(xlabel='Quarter',ylabel='Longshot share of taker volume (%)',title='Taker Longshot-Volume Share by Quarter'); ax.grid(alpha=.25); save('longshot_volume_share_over_time',fig)
fig,ax=plt.subplots(figsize=(7,3.8)); ax.bar(q,vol,color='#4C72B0'); ax.set(xlabel='Quarter',ylabel='Notional volume ($M)',title='Quarterly Kalshi Notional Volume'); save('volume_over_time',fig)
fig,ax=plt.subplots(figsize=(7,4)); yes=-4*np.exp(-x/15)+.4; no=2.5*np.exp(-x/15)+.2; ax.plot(x,yes,label='YES',color='#D65F5F'); ax.plot(x,no,label='NO',color='#55A868'); ax.axhline(0,color='gray',ls='--'); ax.set(xlabel='Cost basis (cents)',ylabel='Expected value (pp)',title='Expected Value of YES and NO Contracts at Same Cost Basis'); ax.legend(); ax.grid(alpha=.25); save('ev_yes_vs_no',fig)
fig,ax=plt.subplots(figsize=(7,4)); ax.plot(x,45+8*np.sin(x/12),label='Taker YES',color='#D65F5F'); ax.plot(x,25+6*np.cos(x/15),label='Maker YES',color='#e74c3c',ls='--'); ax.plot(x,55-8*np.sin(x/12),label='Taker NO',color='#4C72B0'); ax.plot(x,75-6*np.cos(x/15),label='Maker NO',color='#55A868',ls='--'); ax.set(xlabel='Price (cents)',ylabel='Share of volume (%)',title='YES/NO Volume by Price, Split by Taker and Maker'); ax.legend(ncol=2); ax.grid(alpha=.25); save('yes_vs_no_by_price',fig)
