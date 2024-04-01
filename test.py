# Code needed for BDT optimization 
y_predicted = bdt.decision_function(X)
print(y_predicted)

# Load data into dataframes in a really janky way. 
cumulative_events = 0
for key in data:
    data[key]['BDT_output'] = y_predicted[cumulative_events:cumulative_events+len(data[key])]
    cumulative_events += len(data[key])
    print(data[key]['BDT_output'])

# Defining parameters to plot BDT output
BDT_output = {
    'bin_width':0.1,
    'num_bins':14,
    'xrange_min':-1,
    'xlabel':'BDT output',
    }
SoverB_hist_dict = {'BDT_output':BDT_output}
plot_SoverB(data)
