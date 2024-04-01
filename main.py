import pickle
import time  # to measure time to analyse
import math  # for mathematical functions such as square root
import awkward as ak

import uproot3 # for reading .root files
import pandas as pd  # to store data as dataframe
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt  # for plotting
from matplotlib.ticker import AutoMinorLocator  # for minor ticks

import infofile  # local file containing info on cross-sections, sums of weights, dataset IDs

lumi = 10  # fb-1 # data_A+B+C+D

fraction = 0.03  # reduce this is you want the code to run quicker
MeV = 0.001
GeV = 1.0
rerun = False

tuple_path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/"

def calc_mllll(lep_pt, lep_eta, lep_phi, lep_E):
    # construct awkward 4-vector array
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "E": lep_E})
    # calculate invariant mass of first 4 leptons
    # [:, i] selects the i-th lepton in each event
    # .M calculates the invariant mass
    return (p4[:, 0] + p4[:, 1] + p4[:, 2] + p4[:, 3]).M * MeV

samples = {

    'ZZ': {
        'list': ['llll']
    },

    r'$H \rightarrow ZZ \rightarrow \ell\ell\ell\ell$': {  # H -> ZZ -> llll
        'list': ['ggH125_ZZ4lep']  # gluon-gluon fusion
    }

}


def get_data_from_files():

    data = {}  # define empty dictionary to hold dataframes
    for s in samples:  # loop over samples
        print('Processing '+s+' samples')  # print which sample
        frames = []  # define empty list to hold data
        for val in samples[s]['list']:  # loop over each file
            if s == 'data':
                prefix = "Data/"  # Data prefix
            else:  # MC prefix
                prefix = "MC/mc_"+str(infofile.infos[val]["DSID"])+"."
            fileString = tuple_path+prefix+val+".4lep.root"  # file name to open
            # call the function read_file defined below
            temp = read_file(fileString, val)
            # append dataframe returned from read_file to list of dataframes
            frames.append(temp)
        # dictionary entry is concatenated dataframes
        data[s] = pd.concat(frames)

    return data  # return dictionary of dataframes


def get_xsec_weight(sample):
    info = infofile.infos[sample]  # open infofile
    # *1000 to go from fb-1 to pb-1
    xsec_weight = (lumi*1000*info["xsec"])/(info["sumw"]*info["red_eff"])
    return xsec_weight  # return cross-section weight


def calc_weight(xsec_weight, mcWeight, scaleFactor_PILEUP,
                scaleFactor_ELE, scaleFactor_MUON,
                scaleFactor_LepTRIGGER):
    return xsec_weight*mcWeight*scaleFactor_PILEUP*scaleFactor_ELE*scaleFactor_MUON*scaleFactor_LepTRIGGER

def calc_weight2(xsec_weight, events):
    return (
        xsec_weight
        * events.mcWeight
        * events.scaleFactor_PILEUP
        * events.scaleFactor_ELE
        * events.scaleFactor_MUON 
        * events.scaleFactor_LepTRIGGER
    )


def calc_lep_pt_i(lep_pt, i):
    return lep_pt[i]/1000  # /1000 to go from MeV to GeV

# cut on lepton charge
# paper: "selecting two pairs of isolated leptons, each of which is comprised of two leptons with the same flavour and opposite charge"


def cut_lep_charge(lep_charge):
    # throw away when sum of lepton charges is not equal to 0
    # first lepton is [0], 2nd lepton is [1] etc
    return lep_charge[0] + lep_charge[1] + lep_charge[2] + lep_charge[3] != 0

# cut on lepton type
# paper: "selecting two pairs of isolated leptons, each of which is comprised of two leptons with the same flavour and opposite charge"


def cut_lep_type(lep_type):
    # for an electron lep_type is 11
    # for a muon lep_type is 13
    # throw away when none of eeee, mumumumu, eemumu
    sum_lep_type = lep_type[0] + lep_type[1] + lep_type[2] + lep_type[3]
    return (sum_lep_type != 44) and (sum_lep_type != 48) and (sum_lep_type != 52)


def read_file(path, sample):
    start = time.time()  # start the clock
    print("\tProcessing: "+sample)  # print which sample is being processed
    # define empty pandas DataFrame to hold all data for this sample
    data_all = pd.DataFrame()
    tree = uproot3.open(path)["mini"]  # open the tree called mini
    numevents = uproot3.numentries(path, "mini")  # number of events
    if 'data' not in sample:
        xsec_weight = get_xsec_weight(sample)  # get cross-section weight
    for data in tree.iterate(['lep_charge', 'lep_type', 'lep_pt',
                              # uncomment these variables if you want to calculate masses
                              # ,'lep_eta','lep_phi','lep_E',
                              # add more variables here if you make cuts on them
                              'mcWeight', 'scaleFactor_PILEUP',
                              'scaleFactor_ELE', 'scaleFactor_MUON',
                              'scaleFactor_LepTRIGGER'
                              ],  # variables to calculate Monte Carlo weight
                             outputtype=pd.DataFrame,  # choose output type as pandas DataFrame
                             entrystop=numevents*fraction):  # process up to numevents*fraction

        nIn = len(data.index)  # number of events in this batch

        if 'data' not in sample:  # only do this for Monte Carlo simulation files
            # multiply all Monte Carlo weights and scale factors together to give total weight
            data['totalWeight'] = np.vectorize(calc_weight)(xsec_weight,
                                                            data.mcWeight,
                                                            data.scaleFactor_PILEUP,
                                                            data.scaleFactor_ELE,
                                                            data.scaleFactor_MUON,
                                                            data.scaleFactor_LepTRIGGER)

        # cut on lepton charge using the function cut_lep_charge defined above
        fail = data[np.vectorize(cut_lep_charge)(data.lep_charge)].index
        data.drop(fail, inplace=True)

        # cut on lepton type using the function cut_lep_type defined above
        fail = data[np.vectorize(cut_lep_type)(data.lep_type)].index
        data.drop(fail, inplace=True)

        # return the individual lepton transverse momenta in GeV
        data['lep_pt_1'] = np.vectorize(calc_lep_pt_i)(data.lep_pt, 1)
        data['lep_pt_2'] = np.vectorize(calc_lep_pt_i)(data.lep_pt, 2)

        # dataframe contents can be printed at any stage like this
        # print(data)

        # dataframe column can be printed at any stage like this
        # print(data['lep_pt'])

        # multiple dataframe columns can be printed at any stage like this
        # print(data[['lep_pt','lep_eta']])

        nOut = len(data.index)  # number of events passing cuts in this batch
        # append dataframe from this batch to the dataframe for the whole sample
        data_all = data_all.append(data)
        elapsed = time.time() - start  # time taken to process
        print("\t\t nIn: "+str(nIn)+",\t nOut: \t"+str(nOut)+"\t in " +
              str(round(elapsed, 1))+"s")  # events before and after

    return data_all  # return dataframe containing events passing all cuts


if rerun:
    start = time.time()  # time at start of whole processing
    data = get_data_from_files()  # process all files
    elapsed = time.time() - start  # time after whole processing
    with open("data.pkl", "wb") as data_file:
        pickle.dump(data, data_file)
    # print total time taken to process every file
    print("Time taken: "+str(round(elapsed, 1))+"s")

else:
    with open("data.pkl", "rb") as data_file:
        data = pickle.load(data_file)


lep_pt_2 = {  # dictionary containing plotting parameters for the lep_pt_2 histogram
    # change plotting parameters
    'bin_width': 1,  # width of each histogram bin
    'num_bins': 13,  # number of histogram bins
    'xrange_min': 7,  # minimum on x-axis
    'xlabel': r'$lep\_pt$[2] [GeV]',  # x-axis label
}

lep_pt_1 = {  # dictionary containing plotting parameters for the lep_pt_1 histogram
    # change plotting parameters
    'bin_width': 1,  # width of each histogram bin
    'num_bins': 28,  # number of histogram bins
    'xrange_min': 7,  # minimum on x-axis
    'xlabel': r'$lep\_pt$[1] [GeV]',  # x-axis label
}

SoverB_hist_dict = {'lep_pt_2': lep_pt_2, 'lep_pt_1': lep_pt_1}
# add a histogram here if you want it plotted


def plot_SoverB(data):

    # which sample is the signal
    signal = r'$H \rightarrow ZZ \rightarrow \ell\ell\ell\ell$'

    # *******************
    # general definitions (shouldn't need to change)

    # access the dictionary of histograms defined in the cell above
    for x_variable, hist in SoverB_hist_dict.items():

        # get the bin width defined in the cell above
        h_bin_width = hist['bin_width']
        # get the number of bins defined in the cell above
        h_num_bins = hist['num_bins']
        # get the x-range minimum defined in the cell above
        h_xrange_min = hist['xrange_min']
        # get the x-axis label defined in the cell above
        h_xlabel = hist['xlabel']

        bin_edges = [h_xrange_min + x *
                     h_bin_width for x in range(h_num_bins+1)]  # bin limits
        bin_centres = [h_xrange_min+h_bin_width/2 + x *
                       h_bin_width for x in range(h_num_bins)]  # bin centres

        signal_x = data[signal][x_variable]  # histogram the signal

        mc_x = []  # define list to hold the Monte Carlo histogram entries

        for s in samples:  # loop over samples
            if s not in ['data', signal]:  # if not data nor signal
                # append to the list of Monte Carlo histogram entries
                mc_x = [*mc_x, *data[s][x_variable]]

        # *************
        # Signal and background distributions
        # *************
        distributions_axes = plt.gca()  # get current axes

        mc_heights = distributions_axes.hist(mc_x, bins=bin_edges, color='red',
                                             label='Total background',
                                             histtype='step',  # lineplot that's unfilled
                                             density=True)  # normalize to form probability density
        signal_heights = distributions_axes.hist(signal_x, bins=bin_edges, color='blue',
                                                 label=signal,
                                                 histtype='step',  # lineplot that's unfilled
                                                 density=True,  # normalize to form probability density
                                                 linestyle='--')  # dashed line

        # x-limits of the distributions axes
        distributions_axes.set_xlim(left=bin_edges[0], right=bin_edges[-1])
        # y-axis label for distributions axes
        distributions_axes.set_ylabel('Arbitrary units')
        distributions_axes.set_ylim(
            top=max(signal_heights[0])*1.3)  # set y-axis limits
        plt.title('Signal and background '+x_variable +
                  ' distributions')  # add title
        distributions_axes.legend()  # draw the legend
        distributions_axes.set_xlabel(h_xlabel)  # x-axis label

        # Add text 'ATLAS Open Data' on plot
        plt.text(0.05,  # x
                 0.93,  # y
                 'ATLAS Open Data',  # text
                 # coordinate system used is that of distributions_axes
                 transform=distributions_axes.transAxes,
                 fontsize=13)
        # Add text 'for education' on plot
        plt.text(0.05,  # x
                 0.88,  # y
                 'for education',  # text
                 # coordinate system used is that of distributions_axes
                 transform=distributions_axes.transAxes,
                 style='italic',
                 fontsize=8)

        plt.show()  # show the Signal and background distributions

        # *************
        # Signal to background ratio
        # *************
        plt.figure()  # start new figure
        SoverB = []  # list to hold S/B values
        for cut_value in bin_edges:  # loop over bins
            signal_weights_passing_cut = sum(
                data[signal][data[signal][x_variable] > cut_value].totalWeight)
            # start counter for background weights passing cut
            background_weights_passing_cut = 0
            for s in samples:  # loop over samples
                if s not in ['data', signal]:  # if not data nor signal
                    background_weights_passing_cut += sum(
                        data[s][data[s][x_variable] > cut_value].totalWeight)
            if background_weights_passing_cut != 0:  # some background passes cut
                SoverB_value = signal_weights_passing_cut/background_weights_passing_cut
                SoverB_percent = 100*SoverB_value  # multiply by 100 for percentage
                SoverB.append(SoverB_percent)  # append to list of S/B values

        SoverB_axes = plt.gca()  # get current axes
        # plot the data points
        SoverB_axes.plot(bin_edges[:len(SoverB)], SoverB)
        # set the x-limit of the main axes
        SoverB_axes.set_xlim(left=bin_edges[0], right=bin_edges[-1])
        SoverB_axes.set_ylabel('S/B (%)')  # write y-axis label for main axes
        plt.title('Signal to background ratio for different ' +
                  x_variable+' cut values', family='sans-serif')
        SoverB_axes.set_xlabel(h_xlabel)  # x-axis label

        plt.show()  # show S/B plot

    return


plot_SoverB(data)

# define empty dictionary to hold dataframes that will be used to train the BDT
data_for_BDT = {}
BDT_inputs = ['lep_pt_1', 'lep_pt_2']  # list of features for BDT
for key in data:  # loop over the different keys in the dictionary of dataframes
    data_for_BDT[key] = data[key][BDT_inputs].copy()

# for sklearn data is usually organised
# into one 2D array of shape (n_samples x n_features)
# containing all the data and one array of categories
# of length n_samples

all_MC = []  # define empty list that will contain all features for the MC
for key in data:  # loop over the different keys in the dictionary of dataframes
    if key != 'data':  # only MC should pass this
        # append the MC dataframe to the list containing all MC features
        all_MC.append(data_for_BDT[key])
# concatenate the list of MC dataframes into a single 2D array of features, called X
X = np.concatenate(all_MC)

all_y = []  # define empty list that will contain labels whether an event in signal or background
for key in data:  # loop over the different keys in the dictionary of dataframes
    # only background MC should pass this
    if key != r'$H \rightarrow ZZ \rightarrow \ell\ell\ell\ell$' and key != 'data':
        # background events are labelled with 0
        all_y.append(np.zeros(data_for_BDT[key].shape[0]))
# signal events are labelled with 1
all_y.append(np.ones(
    data_for_BDT[r'$H \rightarrow ZZ \rightarrow \ell\ell\ell\ell$'].shape[0]))
# concatenate the list of lables into a single 1D array of labels, called y
y = np.concatenate(all_y)


# make train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=492)


dt = DecisionTreeClassifier(max_depth=2)  # maximum depth of the tree
bdt = AdaBoostClassifier(dt,
                         algorithm='SAMME',  # SAMME discrete boosting algorithm
                         n_estimators=12,  # max number of estimators at which boosting is terminated
                         learning_rate=0.5)  # shrinks the contribution of each classifier by learning_rate

start = time.time()  # time at start of BDT fit
bdt.fit(X_train, y_train)  # fit BDT to training set
elapsed = time.time() - start  # time after fitting BDT
# print total time taken to fit BDT
print("Time taken to fit BDT: "+str(round(elapsed, 1))+"s")
print(bdt)

y_predicted = bdt.predict(X_test)  # get predicted y for test set
print(classification_report(y_test, y_predicted,
                            target_names=["background", "signal"]))
print("Area under ROC curve for test data: %.4f" % (roc_auc_score(y_test,
                                                                  bdt.decision_function(X_test))))

def plot_data(data):

    xmin = 80 * GeV
    xmax = 250 * GeV
    step_size = 5 * GeV

    bin_edges = np.arange(start=xmin, # The interval includes this value
                     stop=xmax+step_size, # The interval doesn't include this value
                     step=step_size ) # Spacing between values
    bin_centres = np.arange(start=xmin+step_size/2, # The interval includes this value
                            stop=xmax+step_size/2, # The interval doesn't include this value
                            step=step_size ) # Spacing between values

    data_x,_ = np.histogram(ak.to_numpy(data['data']['mllll']), 
                            bins=bin_edges ) # histogram the data
    data_x_errors = np.sqrt( data_x ) # statistical error on the data

    signal_x = ak.to_numpy(data[r'Signal ($m_H$ = 125 GeV)']['mllll']) # histogram the signal
    signal_weights = ak.to_numpy(data[r'Signal ($m_H$ = 125 GeV)'].totalWeight) # get the weights of the signal events
    signal_color = samples[r'Signal ($m_H$ = 125 GeV)']['color'] # get the colour for the signal bar

    mc_x = [] # define list to hold the Monte Carlo histogram entries
    mc_weights = [] # define list to hold the Monte Carlo weights
    mc_colors = [] # define list to hold the colors of the Monte Carlo bars
    mc_labels = [] # define list to hold the legend labels of the Monte Carlo bars

    for s in samples: # loop over samples
        if s not in ['data', r'Signal ($m_H$ = 125 GeV)']: # if not data nor signal
            mc_x.append( ak.to_numpy(data[s]['mllll']) ) # append to the list of Monte Carlo histogram entries
            mc_weights.append( ak.to_numpy(data[s].totalWeight) ) # append to the list of Monte Carlo weights
            mc_colors.append( samples[s]['color'] ) # append to the list of Monte Carlo bar colors
            mc_labels.append( s ) # append to the list of Monte Carlo legend labels
    


    # *************
    # Main plot 
    # *************
    main_axes = plt.gca() # get current axes
    
    # plot the data points
    main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                       fmt='ko', # 'k' means black and 'o' is for circles 
                       label='Data') 
    
    # plot the Monte Carlo bars
    mc_heights = main_axes.hist(mc_x, bins=bin_edges, 
                                weights=mc_weights, stacked=True, 
                                color=mc_colors, label=mc_labels )
    
    mc_x_tot = mc_heights[0][-1] # stacked background MC y-axis value
    
    # calculate MC statistical uncertainty: sqrt(sum w^2)
    mc_x_err = np.sqrt(np.histogram(np.hstack(mc_x), bins=bin_edges, weights=np.hstack(mc_weights)**2)[0])
    
    # plot the signal bar
    main_axes.hist(signal_x, bins=bin_edges, bottom=mc_x_tot, 
                   weights=signal_weights, color=signal_color,
                   label=r'Signal ($m_H$ = 125 GeV)')
    
    # plot the statistical uncertainty
    main_axes.bar(bin_centres, # x
                  2*mc_x_err, # heights
                  alpha=0.5, # half transparency
                  bottom=mc_x_tot-mc_x_err, color='none', 
                  hatch="////", width=step_size, label='Stat. Unc.' )

    # set the x-limit of the main axes
    main_axes.set_xlim( left=xmin, right=xmax ) 
    
    # separation of x axis minor ticks
    main_axes.xaxis.set_minor_locator( AutoMinorLocator() ) 
    
    # set the axis tick parameters for the main axes
    main_axes.tick_params(which='both', # ticks on both x and y axes
                          direction='in', # Put ticks inside and outside the axes
                          top=True, # draw ticks on the top axis
                          right=True ) # draw ticks on right axis
    
    # x-axis label
    main_axes.set_xlabel(r'4-lepton invariant mass $\mathrm{m_{4l}}$ [GeV]',
                        fontsize=13, x=1, horizontalalignment='right' )
    
    # write y-axis label for main axes
    main_axes.set_ylabel('Events / '+str(step_size)+' GeV',
                         y=1, horizontalalignment='right') 
    
    # set y-axis limits for main axes
    main_axes.set_ylim( bottom=0, top=np.amax(data_x)*1.6 )
    
    # add minor ticks on y-axis for main axes
    main_axes.yaxis.set_minor_locator( AutoMinorLocator() ) 

    # Add text 'ATLAS Open Data' on plot
    plt.text(0.05, # x
             0.93, # y
             'ATLAS Open Data', # text
             transform=main_axes.transAxes, # coordinate system used is that of main_axes
             fontsize=13 ) 
    
    # Add text 'for education' on plot
    plt.text(0.05, # x
             0.88, # y
             'for education', # text
             transform=main_axes.transAxes, # coordinate system used is that of main_axes
             style='italic',
             fontsize=8 ) 
    
    # Add energy and luminosity
    lumi_used = str(lumi*fraction) # luminosity to write on the plot
    plt.text(0.05, # x
             0.82, # y
             '$\sqrt{s}$=13 TeV,$\int$L dt = '+lumi_used+' fb$^{-1}$', # text
             transform=main_axes.transAxes ) # coordinate system used is that of main_axes
    
    # Add a label for the analysis carried out
    plt.text(0.05, # x
             0.76, # y
             r'$H \rightarrow ZZ^* \rightarrow 4\ell$', # text 
             transform=main_axes.transAxes ) # coordinate system used is that of main_axes

    # draw the legend
    main_axes.legend( frameon=False ) # no box around the legend
    
    return
def bdt_cut(data, cut_value):
    data_for_BDT = {} # define empty dictionary to hold dataframes that will be used to train the BDT
    BDT_inputs = ['lep_pt_1','lep_pt_2'] # list of features for BDT
    for key in data: # loop over the different keys in the dictionary of dataframes
        data_for_BDT[key] = data[key][BDT_inputs].copy()
    all_MC = [] # define empty list that will contain all features for the MC
    for key in data: # loop over the different keys in the dictionary of dataframes
        if key!='data': # only MC should pass this
            all_MC.append(data_for_BDT[key]) # append the MC dataframe to the list containing all MC features
    X = np.concatenate(all_MC) # concatenate the list of MC dataframes into a single 2D array of features, called X
    y_predicted = bdt.decision_function(X)

    return y_predicted < cut_value

def read_file2(path,sample):
    start = time.time() # start the clock
    print("\tProcessing: "+sample) # print which sample is being processed
    data_all = [] # define empty list to hold all data for this sample
    
    # open the tree called mini using a context manager (will automatically close files/resources)
    with uproot.open(path + ":mini") as tree:
        numevents = tree.num_entries # number of events
        if 'data' not in sample: xsec_weight = get_xsec_weight(sample) # get cross-section weight
        for data in tree.iterate(['lep_pt','lep_eta','lep_phi',
                                  'lep_E','lep_charge','lep_type', 
                                  # add more variables here if you make cuts on them 
                                  'mcWeight','scaleFactor_PILEUP',
                                  'scaleFactor_ELE','scaleFactor_MUON',
                                  'scaleFactor_LepTRIGGER'], # variables to calculate Monte Carlo weight
                                 library="ak", # choose output type as awkward array
                                 entry_stop=numevents*fraction): # process up to numevents*fraction

            nIn = len(data) # number of events in this batch

            if 'data' not in sample: # only do this for Monte Carlo simulation files
                # multiply all Monte Carlo weights and scale factors together to give total weight
                data['totalWeight'] = calc_weight2(xsec_weight, data)

            # cut on lepton charge using the function cut_lep_charge defined above
            data = data[~bdt_cut(data, 0.0)]

            # calculation of 4-lepton invariant mass using the function calc_mllll defined above
            data['mllll'] = calc_mllll(data.lep_pt, data.lep_eta, data.lep_phi, data.lep_E)

            # array contents can be printed at any stage like this
            #print(data)

            # array column can be printed at any stage like this
            #print(data['lep_pt'])

            # multiple array columns can be printed at any stage like this
            #print(data[['lep_pt','lep_eta']])

            nOut = len(data) # number of events passing cuts in this batch
            data_all.append(data) # append array from this batch
            elapsed = time.time() - start # time taken to process
            print("\t\t nIn: "+str(nIn)+",\t nOut: \t"+str(nOut)+"\t in "+str(round(elapsed,1))+"s") # events before and after
    
    return ak.concatenate(data_all) # return array containing events passing all cuts

def get_data_from_files2():

    data = {} # define empty dictionary to hold awkward arrays
    for s in samples: # loop over samples
        print('Processing '+s+' samples') # print which sample
        frames = [] # define empty list to hold data
        for val in samples[s]['list']: # loop over each file
            if s == 'data': prefix = "Data/" # Data prefix
            else: # MC prefix
                prefix = "MC/mc_"+str(infofile.infos[val]["DSID"])+"."
            fileString = tuple_path+prefix+val+".4lep.root" # file name to open
            temp = read_file2(fileString,val) # call the function read_file defined below
            frames.append(temp) # append array returned from read_file to list of awkward arrays
        data[s] = ak.concatenate(frames) # dictionary entry is concatenated awkward arrays
    
    return data # return dictionary of awkward arrays


start = time.time() # time at start of whole processing
data = get_data_from_files() # process all files
elapsed = time.time() - start # time after whole processing
print("Time taken: "+str(round(elapsed,1))+"s") # print total time taken to process every file
print(data)
#plot_data(data)
