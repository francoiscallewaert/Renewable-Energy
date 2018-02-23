from flask import Flask, render_template, request, redirect, json
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt, mpld3
import datetime
import matplotlib.patches as mpatches
import seaborn as sns
import _pickle as cPickle

df = pd.read_csv('prod_filiere_hourly_mod2.csv')
df['date'] = pd.to_datetime(df['date'])
#cols = ['total', 'nuclear', 'thermal', 'hydro', 'solar', 'wind']
#colors = {'total':'black', 'total-nuclear':'black', 'total-nuclear-hydroriver':'black', 'nuclear':'red', 
#          'hydro':'blue', 'solar':'yellow', 'wind':'cyan'}

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
        wind0, solar0, storage0, begin, end = None, None, 180, '2017-01-01', '2017-01-07'
        w, sol, sto = 5, 5, 180
        data0 = get_power(df, wind0, solar0, storage0)
        (p0, p1, p2, p3) = plot_power(data0, begin, end)
        (h0, h1, h2, h3) = (mpld3.fig_to_html(p0), mpld3.fig_to_html(p1), mpld3.fig_to_html(p2), mpld3.fig_to_html(p3))
    if request.method == 'POST':
        w = int(request.form['wind'])
        sol = int(request.form['solar'])
        sto = int(request.form['storage'])
        begin = request.form['begin']
        end = request.form['end']
        data = get_power(df, w, sol, sto)
        (p0, p1, p2, p3) = plot_power(data, begin, end)
        (h0, h1, h2, h3) = (mpld3.fig_to_html(p0), mpld3.fig_to_html(p1), mpld3.fig_to_html(p2), mpld3.fig_to_html(p3))
    return render_template('index2.html', begin=begin, end=end, h0=h0, h1=h1, h2=h2, h3=h3, wind=w, solar=sol, storage=sto)

if __name__ == '__main__':
    app.run(host = '0.0.0.0')


## Methods ##

def select_data(D, begin, end, duration, columns, columns2):    
    D1 = D[(D['date'] <= end) & (D['date'] >= begin)][columns]
    D1['time'] = 0
    for i in range(int(len(D1) / duration)):
        D1.iloc[i*duration:(i+1)*duration, -1] = i
    D2 = D1.groupby(['time']).mean()
    D3 = D1.groupby(['time']).mean()
    D3.insert(0, 'start', 0)
    for i in range(2, len(D3.columns)-3):
        D3.iloc[:,i] = D3.iloc[:,i-1] + D3.iloc[:,i]
        
    D4 = D[(D['date'] <= end) & (D['date'] >= begin)][columns2]
    D4['time'] = 0
    for i in range(int(len(D4) / duration)):
        D4.iloc[i*duration:(i+1)*duration, -1] = i
    D5 = D4.groupby(['time']).mean()
    D5.insert(0, 'nuclear', 0)
    D6 = D5.copy()
    for i in range(2, len(D6.columns)-1):
        D6.iloc[:,i] = D6.iloc[:,i-1] + D6.iloc[:,i]
    return (D2, D3, D5, D6)

def get_power(df, wind, solar, storage):   
    params = make_params(storage)
    df_mod = run_model(df, wind, solar, params)
    return df_mod
    
def plot_power(dat, begin, end):
    cols = [['wind', 'solar', 'hydrolake', 'thermal', 'pumpedhydro', 'total-nuclear-hydroriver'],
            ['NC', 'Wind', 'Solar', 'HL', 'TH', 'PH', 'total-nuclear-hydroriver', 'waste', 'short']]
    labels = ['Nuclear', 'Wind', 'Solar', 'Hydro', 'Fossil', 'Storage', 'total-nuclear-hydroriver', 'Waste', 'Shortage']
    # cols2 = ['HLcur', 'PHcur']
    colors = [['red',      'lime',  'yellow', 'royalblue', 'silver',  'magenta'],
              ['red', 'green', 'orange', 'blue',      'grey', 'purple'],
               ['chocolate', 'olive']]
    resolution = [1, 24*7*2]
    (F, E, G, H) = select_data(dat, begin, end, 1, cols[1], cols[0])
    
    al = 1.0    
    fig0, ax0 = plt.subplots(figsize = (6, 3))       
    patches = []
    for j in range(len(cols[0])-1):
        if j == len(cols[0])-2:
            al = 0.5
        ax0.fill_between(range(len(H)), H.iloc[:, j], H.iloc[:, j+1], facecolor=colors[1][j+1], interpolate=True, alpha=al)
        patches.append(mpatches.Patch(color=colors[1][j+1], label=labels[j+1]))
    ax0.set_title('Historic power profile' , fontsize = 20, color = 'white')
    ax0.set_xlabel('Time (hours)', fontsize = 15, color = 'white')
    ax0.set_ylabel('Power (GW)', fontsize = 15, color = 'white')
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)    
    ax0.plot(range(len(H)), H.iloc[:, -1], color='black')
    plt.legend(handles=patches, fontsize = 8, bbox_to_anchor=(0, 1), loc=2)
    
    al = 1.0    
    fig1, ax1 = plt.subplots(figsize = (6, 3))
    patches = []
    for j in range(len(cols[1])-3):
        if j == len(cols[1])-4:
            al = 0.5
        ax1.fill_between(range(len(E)), E.iloc[:, j], E.iloc[:, j+1], facecolor=colors[1][j], interpolate=True, alpha=al)
        patches.append(mpatches.Patch(color=colors[1][j], label=labels[j]))
    ax1.set_title('Power profile from model' , fontsize = 20, color = 'white')
    ax1.set_xlabel('Time (hours)', fontsize = 15, color = 'white')
    ax1.set_ylabel('Power (GW)', fontsize = 15, color = 'white')
    plt.xticks(fontsize = 12, color = 'white')
    plt.yticks(fontsize = 12, color = 'white')       
    ax1.plot(range(len(E)), E.iloc[:, -3], color='black')
    plt.legend(handles=patches, fontsize = 8, bbox_to_anchor=(0, 1), loc=2)
    
    sums0 = np.sum(G, axis=0)
    sums0['pumpedhydro'] = np.sum(np.abs(G['pumpedhydro']), axis = 0)
    sums = np.sum(F, axis = 0)
    sums['PH'] = np.sum(np.abs(F['PH']), axis = 0)
    sumsall = np.sum(dat, axis = 0) / 1000
    sumsall['pumpedhydro'] = np.sum(np.abs(dat['pumpedhydro']), axis = 0) / 1000
    sumsall['PH'] = np.sum(np.abs(dat['PH']), axis = 0) / 1000
    sumsall['0'] = 0
    labs = ('Nuclear', 'Wind', 'Solar', 'Hydro', 'Thermal', 'Storage', 'Waste', 'Shortage')
    
    fig2, ax2 = plt.subplots(figsize = (6, 3))
    bar_width = 0.35
    patches = []
    for j in range(len(cols[1])-3):
        plt.bar(j, sums0[j], bar_width, color=colors[0][j])
        plt.bar(j+bar_width, sums[j], bar_width, color=colors[1][j], label=labels[j])
        patches.append(mpatches.Patch(color=colors[1][j], label=labels[j]))
    for j in range(2):
        plt.bar(j+6+bar_width, sums[j+7], bar_width, color=colors[2][j])
        patches.append(mpatches.Patch(color=colors[2][j], label=labels[7+j]))   
    ax2.set_title("Energy produced this week", fontsize = 20, color = 'white')
    ax2.set_ylabel('Energy (GWh)', fontsize = 15, color = 'white')
    ax2.text(0.78, 0.9,'left=historic - right=model', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    plt.xticks(np.arange(len(labs)) + 0.15, labs, color = 'white')        
    # plt.legend(handles=patches, fontsize = 8, bbox_to_anchor=(0.8, 1), loc=2)
    
    fig3, ax3 = plt.subplots(figsize = (6, 3))
    bar_width = 0.3
    patches = []
    indices = ['0', 'NC', 'wind', 'Wind', 'solar', 'Solar', 'hydrolake', 'HL', 'thermal', 'TH', 'pumpedhydro', 'PH', 'waste', 'short']
    for j in range(6):
        plt.bar(j, sumsall[indices[2*j]], bar_width, color=colors[0][j])
        plt.bar(j+bar_width, sumsall[indices[2*j+1]], bar_width, color=colors[1][j], label=labels[j])
        patches.append(mpatches.Patch(color=colors[1][j], label=labels[j]))
    for j in range(2):
        plt.bar(j+6+bar_width, sumsall[indices[j+12]], bar_width, color=colors[2][j])
        patches.append(mpatches.Patch(color=colors[2][j], label=labels[7+j]))       
    ax3.set_title("Energy produced during 3 years", fontsize = 20, color = 'white')
    ax3.set_ylabel('Energy (TWh)', fontsize = 15, color = 'white')
    ax3.text(0.78, 0.9,'left=historic - right=model', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
    plt.xticks(np.arange(len(labs)) + 0.15, labs, color = 'white')        
    # plt.legend(handles=patches, fontsize = 8, bbox_to_anchor=(0.8, 1), loc=2)
    
    return (fig0, fig1, fig2, fig3)

def make_params(PH_cap = 180):
    # All data in GW or GWh
    # Pumped hydro
    PHcap = PH_cap
    multPH = 0.75 + PH_cap / 180 / 4
    PHpow = [-4*multPH, 4*multPH]
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

        PH = max(PHpow[0], min(PHpow[1], power))
        PH = max(PHcur - PHcap, min(PH, PHcur))
        power -= PH

        HL = 0
        if power > 0:
            HL = max(HLpow[0], min(HLpow[1], power))
            HL = max(HLcur - HLcap, min(HL, HLcur))
        power -= HL
        waste += max(-power, 0)

        TH = max(THpow[0], min(THpow[1], power))
        power -= TH
        if (TH < THpow[1]) & (PH > PHpow[0]): 
            THHL = min(THpow[1] - TH, HL * max(0, min(1, 4*(0.25*HLcap-HLcur)/HLcap)))
            TH += THHL
            HL -= THHL
            THPH = min(THpow[1] - TH, (PH - PHpow[0]) * max(0, min(1, 2*(0.5*PHcap-PHcur)/PHcap)))
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
        NC -= NCramp * max(-1, min(1, (M[i,20] / 100 + (HLcur - 0.75*HLcap) / HLcap)))
        NC = min(NC, NCmax - M[i,5])

    df3 = pd.DataFrame(M, range(len(M)), df2.columns[2:])
    #df3 = df3.rename(index=str, columns={"date": "HLevo"})
    df3.index = df2.index
    df3['date'] = df2['date']
    return df3
