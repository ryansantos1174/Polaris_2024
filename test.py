###################################################################
# To get to the virtual environment, from your home directory run #
# source polaris/bin/activate                                     #
###################################################################




import numpy as np
import uproot
import infofile

def get_data_from_files(samples, database_path, variable):

    data = {} # define empty dictionary to hold awkward arrays
    for s in samples: # loop over samples
        print('Processing '+s+' samples') # print which sample
        frames = [] # define empty list to hold data
        for val in samples[s]['list']: # loop over each file
            if s == 'data': prefix = "Data/" # Data prefix
            else: # MC prefix
                prefix = "MC/mc_"+str(infofile.infos[val]["DSID"])+"."
            fileString = database_path+prefix+val+".4lep.root" # file name to open
            with uproot.open(fileString + ":mini") as tree:
                frames.append(tree[variable].array(library="np"))
        data[s] = np.concatenate(frames)
    return data # return dictionary of awkward arrays

samples = {

    'data': {
        'list' : ['data_A','data_B','data_C','data_D'],
    },

    r'Background $Z,t\bar{t}$' : { # Z + ttbar
        'list' : ['Zee','Zmumu','ttbar_lep'],
        'color' : "#6b59d3" # purple
    },

    r'Background $ZZ^*$' : { # ZZ
        'list' : ['llll'],
        'color' : "#ff0000" # red
    },

    r'Signal ($m_H$ = 125 GeV)' : { # H -> ZZ -> llll
        'list' : ['ggH125_ZZ4lep','VBFH125_ZZ4lep','WH125_ZZ4lep','ZH125_ZZ4lep'],
        'color' : "#00cdff" # light blue
    },

}
database ="https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/"
data = get_data_from_files(samples, database, "lep_phi")

for key, values in data.items():

    # Use this to combine data
    values = np.concatenate(values)

