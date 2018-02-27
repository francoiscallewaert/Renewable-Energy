from flask import Flask, render_template, request, redirect, json
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt, mpld3
import datetime
import matplotlib.patches as mpatches
import seaborn as sns; sns.set()
import pickle as cPickle
from dateutil import parser

df = pd.read_csv('prod_filiere_hourly_mod2.csv')
df['date'] = pd.to_datetime(df['date'])
glob = cPickle.load(open( "results2.pkl", "rb" ) )

app = Flask(__name__)

@app.route('/firstpage', methods=['GET'])
def firstpage():
    return render_template('Firstpage/index.html')
        
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index2.html')

@app.route('/power', methods=['GET', 'POST'])
def input():
    if request.method == 'GET':
        global Begin, End, W, Sol, Sto, H, H2, Storage1
        wind0, solar0, storage0, Storage1, Begin = None, None, 180, 0, '2017-01-01'
        W, Sol, Sto = 5, 5, 180
        Data = get_power(df, wind0, solar0, storage0)
        p = plot_power(Data, Begin)
        p2 = plot_power2(glob, Storage1)
        (H, H2) = (mpld3.fig_to_html(p), mpld3.fig_to_html(p2))
    if request.method == 'POST' and ('storage1' in request.form):
        Storage1 = int(int(request.form['storage1']) / 180) -1
        p2 = plot_power2(glob, Storage1)
        H2 = mpld3.fig_to_html(p2)
    if request.method == 'POST' and not('storage1' in request.form):
        W = int(request.form['wind'])
        Sol = int(request.form['solar'])
        Sto = int(request.form['storage'])
        Begin = request.form['begin']
        Data = get_power(df, W, Sol, Sto)
        p = plot_power(Data, Begin)
        H = mpld3.fig_to_html(p)
    return render_template('index3.html', begin=Begin, h=H, h2=H2, wind=W, solar=Sol, storage=Sto, storage1=180*(Storage1+1))

if __name__ == '__main__':
    app.run()


## Methods ##

def select_data(D, begin, duration, columns, columns2): 
    end = parser.parse(begin) + datetime.timedelta(7,0)
    D1 = D[(D['date'] <= end) & (D['date'] >= begin)][columns]
    D1['time'] = 0
    for i in range(int(len(D1) / duration)):
        D1.iloc[i*duration:(i+1)*duration, -1] = i
    D2 = D1.groupby(['time']).mean()
    D2.insert(0, 'start', 0)
    for i in range(2, len(D2.columns)-3):
        D2.iloc[:,i] = D2.iloc[:,i-1] + D2.iloc[:,i]
    D2.iloc[:,0] = pd.DataFrame([D2.iloc[:,0], D2.iloc[:,1]]).min()
        
    D3 = D[(D['date'] <= end) & (D['date'] >= begin)][columns2]
    D3['time'] = 0
    for i in range(int(len(D3) / duration)):
        D3.iloc[i*duration:(i+1)*duration, -1] = i
    D4 = D3.groupby(['time']).mean()
    D4.insert(0, 'nuclear', 0)
    for i in range(2, len(D4.columns)-1):
        D4.iloc[:,i] = D4.iloc[:,i-1] + D4.iloc[:,i]
        
    D5 = D[(D['date'] <= end) & (D['date'] >= begin)][columns2 + columns]
    D5['time'] = 0
    for i in range(int(len(D5) / duration)):
        D5.iloc[i*duration:(i+1)*duration, -1] = i
    D6 = D5.groupby(['time']).mean()
    D6.insert(0, 'nuclear', 0)
    return (D2, D4, D6)

def get_power(df, wind, solar, storage):   
    params = make_params(storage)
    df_mod = run_model(df, wind, solar, params)
    return df_mod
    
def plot_power(dat, begin):
    cols = [['wind', 'solar', 'hydrolake', 'thermal', 'pumpedhydro', 'total-nuclear-hydroriver'],
            ['NC', 'Wind', 'Solar', 'HL', 'TH', 'PH', 'total-nuclear-hydroriver', 'waste', 'short']]
    labels = ['Nuclear', 'Wind', 'Solar', 'Hydro', 'Fossil', 'Storage', 'total-nuclear-hydroriver', 'Waste', 'Shortage']
    # cols2 = ['HLcur', 'PHcur']
    colors = [['red',      'lime',  'yellow', 'royalblue', 'silver',  'magenta'],
              ['red', 'green', 'orange', 'blue',      'grey', 'purple'],
               ['chocolate', 'olive']]
    
    resolution = [1, 24*7*2]
    (E, H, G) = select_data(dat, begin, 1, cols[1], cols[0])
    sums = np.sum(G, axis=0)
    sums['pumpedhydro'] = np.sum(np.abs(G['pumpedhydro']), axis = 0)
    sums['PH'] = np.sum(np.abs(G['PH']), axis = 0)
    
    sumsall = np.sum(dat, axis = 0) / 1000
    sumsall['pumpedhydro'] = np.sum(np.abs(dat['pumpedhydro']), axis = 0) / 1000
    sumsall['PH'] = np.sum(np.abs(dat['PH']), axis = 0) / 1000
    sumsall['0'] = 0
    labs = ('Nuclear', 'Wind', 'Solar', 'Hydro', 'Fossil', 'Storage', 'Waste', 'Shortage')
    indices = [[0,7,1,8,2,9,3,10,4,11,5,12,14,15],
               ['0', 'NC', 'wind', 'Wind', 'solar', 'Solar', 'hydrolake', 'HL', 'thermal', 'TH', 'pumpedhydro', 'PH', 'waste', 'short']]
    
    data = [H, E, sums, sumsall]
    titles = ['Historic power profile', 'Power profile from model', "Energy produced this week", "Energy produced over 3 years"]
    labels2 = ['Energy (GWh)', 'Energy (TWh)']
    
    fig = plt.figure(figsize = (12, 6))
    for i in range(2):
        ax = fig.add_subplot(2, 2, i+1)
        al = 1.0       
        patches = []
        for j in range(len(cols[i])-1-2*i):
            if j == len(cols[i])-2-2*i:
                al = 0.5
            ax.fill_between(range(len(data[i])), data[i].iloc[:, j], data[i].iloc[:, j+1], facecolor=colors[1][j+1-i], interpolate=True, alpha=al)
            patches.append(mpatches.Patch(color=colors[1][j+1-i], label=labels[j+1-i]))
        ax.set_title(titles[i] , fontsize = 20, color = 'white')
        ax.set_xlabel('Time (hours)', fontsize = 15, color = 'white')
        ax.set_ylabel('Power (GW)', fontsize = 15, color = 'white')
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)    
        ax.plot(range(len(data[i])), data[i].iloc[:, -1-2*i], color='black')
        plt.legend(handles=patches, fontsize = 8, bbox_to_anchor=(0, 1), loc=2)
    
    bar_width = 0.3
    for i in range(2):
        ax = fig.add_subplot(2, 2, i+3)         
        patches = []
        for j in range(6):
            plt.bar(j, data[i+2][indices[i][2*j]], bar_width, color=colors[0][j])
            plt.bar(j+bar_width, data[i+2][indices[i][2*j+1]], bar_width, color=colors[1][j], label=labels[j])
            patches.append(mpatches.Patch(color=colors[1][j], label=labels[j]))
        for j in range(2):
            plt.bar(j+6+bar_width, data[i+2][indices[i][j+12]], bar_width, color=colors[2][j])
            patches.append(mpatches.Patch(color=colors[2][j], label=labels[7+j]))   
        ax.set_title(titles[i+2], fontsize = 20, color = 'white')
        ax.set_ylabel(labels2[i], fontsize = 15, color = 'white')
        ax.text(0.76, 0.9,'left=historic data - right=model', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.xticks(np.arange(len(labs)) + 0.15, labs, color = 'white')        
        # plt.legend(handles=patches, fontsize = 8, bbox_to_anchor=(0.8, 1), loc=2)
    plt.tight_layout()
    return fig
    
def plot_power2(glob, sto):
    fig2 = plt.figure(figsize = (10, 6))
    titles = ['Electricity from wind and solar (%)', 'Electricity wasted (%)',
             'Electricity from nuclear (%)', 'Electricity from fossil fuels (%)']
    index = [6, 4, 3, 2]
    vmins = [0, 0, 50, 0]
    vmaxs = [30, 7, 80, 20]
    for i in range(4):
        fig2.add_subplot(2, 2, i+1)
        ax = sns.heatmap(glob[index[i],::-1,:,sto], annot=True, vmin=vmins[i], vmax=vmaxs[i], cmap = 'viridis')
        ax.set_title(titles[i] , fontsize = 20)
        ax.set_xlabel('Solar power (GW)', fontsize = 18)
        ax.set_ylabel('Wind power (GW)', fontsize = 18)
        plt.xticks(np.arange(10) + 0.5, 5 * np.arange(1, 11), fontsize = 12)
        plt.yticks(np.arange(10) + 0.5, 5 * np.arange(1, 11), fontsize = 12)
    plt.tight_layout()
    return fig2

def make_params(PH_cap = 180):
    # All data in GW or GWh
    # Pumped hydro
    PHcap = PH_cap
    absPH = 3.0 + PH_cap / 180
    PHpow = [-absPH, absPH]
    PHeff = 0.85
    # HydroLake
    HLref = 1.5 # refill
    HLcap = 3600
    HLpow = [0.0, 8.5]
    # Thermal
    THpow = [0.0, 10.0]
    # Nuclear ramp rate (per hour)
    NCramp = 1.0 / 24
    NClim = [1800, 2700]
    NCmax = 63
    return [PHcap, PHpow, PHeff, HLref, HLcap, HLpow, THpow, NCramp, NClim, NCmax]

def run_model(df2, wind, solar, params):
    [PHcap, PHpow, PHeff, HLref, HLcap, HLpow, THpow, NCramp, NClim, NCmax] = params
    M = df2.drop(['time', 'date'], axis = 1).as_matrix()
    PHcur, HLcur, LIcur, NC = PHcap/2, HLcap/2, 0, 5
    W, S = wind, solar
    for i in range(len(M)):
        # Remove nuclear, wind and solar from total
        ratioW, ratioS = 1, 1
        if W != None:
            ratioW = W / M[i,8]
        if S != None:
            ratioS = S / M[i,9]
        power = M[i,7] - NC - ratioW * M[i,4] - ratioS * M[i,3]

        # Use hydrolake, pumped hydro and thermal
        waste = 0
        HLcur += HLref
        waste += max(HLcur - HLcap, 0)
        HLcur = min(HLcur, HLcap)

        HL = 0
        r1 = HLcur/HLcap*HLpow[1]
        r2 = PHcur/PHcap*PHpow[1]
        if power > 0:    
            HLid = r1 / (r1+r2) * power
            PHid = r2 / (r1+r2) * power
            HL = max(HLpow[0], min(HLpow[1], HLid))
            HL = max(HLcur - HLcap, min(HL, HLcur))
            PH = max(PHpow[0], min(PHpow[1], PHid))
        else:
            PH = max(PHpow[0], power)
            
        PH = max(PHcur - PHcap, min(PH, PHcur))
        power -= (HL+PH)        
        waste += max(-power, 0)

        TH = max(THpow[0], min(THpow[1], power))
        power -= TH
        if (TH < THpow[1]) & (PH > PHpow[0]): 
            THHL = min(THpow[1] - TH, HL * max(0, 1 - 4*HLcur/HLcap))
            TH += THHL
            HL -= THHL
            THPH = min(THpow[1] - TH, (PH - PHpow[0]) * max(0, 1 - 4*PHcur/PHcap))
            TH += THPH
            PH -= THPH
        HLcur -= HL
        if PH > 0:
            PHcur -= PH
        else:
            PHcur -= PH / PHeff
        short = max(0, power)
        M[i,10] = NC
        M[i,11] = ratioW * M[i,4]
        M[i,12] = ratioS * M[i,3]
        M[i,13] = HL
        M[i,14] = HLcur
        M[i,15] = PH
        M[i,16] = PHcur
        M[i,17] = TH
        M[i,18] = waste
        M[i,19] = short
        M[i,20] = M[i,14] - M[max(0, i-24*7), 14]

        # Change long-term nuclear power depending on current lake evolution
        NC += NCramp * max(-1, 0.25 + min(1, 4*(TH / THpow[1]) - HLcur/HLcap - PHcur/PHcap))
        NC = min(NC, NCmax - M[i,5])

    df3 = pd.DataFrame(M, range(len(M)), df2.columns[2:])
    #df3 = df3.rename(index=str, columns={"date": "HLevo"})
    df3.index = df2.index
    df3['date'] = df2['date']
    print(np.mean(df3['HLcur']) / HLcap)
    print(np.mean(df3['PHcur']) / PHcap)
    return df3
