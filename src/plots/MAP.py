import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

deciles = list(range(1, 11))

# chromium_baseline = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
chromium_farsec_values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
chromium_de_values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
chromium_smote_values = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
chromium_smotuned_values = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
chromium_swift_values = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

# fig,axs=plt.subplots(nrows,ncols,figsize=(width,height))
fig, axs = plt.subplots(3, 2, figsize=(15, 28))

# axs[0, 0].plot(deciles, chromium_baseline, '->', label='Baseline')
axs[0, 0].plot(deciles, chromium_farsec_values, '-d', label='FARSEC')
axs[0, 0].plot(deciles, chromium_de_values, '-o', label='DE+Learner')
axs[0, 0].plot(deciles, chromium_smote_values, '-x', label='SMOTE')
axs[0, 0].plot(deciles, chromium_smotuned_values, '-s', label='SMOTUNED')
axs[0, 0].plot(deciles, chromium_swift_values, '-v', label='SWIFT')
axs[0, 0].set(xlabel='Deciles',
              ylabel='MAP',
              title='Chromium')

axs[0, 0].legend()

# wicket_baseline = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
wicket_farsec_values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
wicket_de_values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
wicket_smote_values = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
wicket_smotuned_values = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
wicket_swift_values = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

# axs[0, 1].plot(deciles, wicket_baseline, '->', label='Baseline')
axs[0, 1].plot(deciles, wicket_farsec_values, '-d', label='FARSEC')
axs[0, 1].plot(deciles, wicket_de_values, '-o', label='DE+Learner')
axs[0, 1].plot(deciles, wicket_smote_values, '-x', label='SMOTE')
axs[0, 1].plot(deciles, wicket_smotuned_values, '-s', label='SMOTUNED')
axs[0, 1].plot(deciles, wicket_swift_values, '-v', label='SWIFT')
axs[0, 1].set(xlabel='Deciles',
              ylabel='MAP',
              title='Wicket')

axs[0, 1].legend()

# amberi_baseline = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
amberi_farsec_values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
amberi_de_values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
amberi_smote_values = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
amberi_smotuned_values = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
amberi_swift_values = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

# axs[1, 0].plot(deciles, amberi_baseline, '->', label='Baseline')
axs[1, 0].plot(deciles, amberi_farsec_values, '-d', label='FARSEC')
axs[1, 0].plot(deciles, amberi_de_values, '-o', label='DE+Learner')
axs[1, 0].plot(deciles, amberi_smote_values, '-x', label='SMOTE')
axs[1, 0].plot(deciles, amberi_smotuned_values, '-s', label='SMOTUNED')
axs[1, 0].plot(deciles, amberi_swift_values, '-v', label='SWIFT')
axs[1, 0].set(xlabel='Deciles',
              ylabel='MAP',
              title='Amberi')

axs[1, 0].legend()

# camel_baseline = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
camel_farsec_values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
camel_de_values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
camel_smote_values = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
camel_smotuned_values = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
camel_swift_values = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

# axs[1, 1].plot(deciles, camel_baseline, '->', label='Baseline')
axs[1, 1].plot(deciles, camel_farsec_values, '-d', label='FARSEC')
axs[1, 1].plot(deciles, camel_de_values, '-o', label='DE+Learner')
axs[1, 1].plot(deciles, camel_smote_values, '-x', label='SMOTE')
axs[1, 1].plot(deciles, camel_smotuned_values, '-s', label='SMOTUNED')
axs[1, 1].plot(deciles, camel_swift_values, '-v', label='SWIFT')
axs[1, 1].set(xlabel='Deciles',
              ylabel='MAP',
              title='Camel')

axs[1, 1].legend()

# derby_baseline = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
derby_farsec_values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
derby_de_values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
derby_smote_values = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
derby_smotuned_values = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
derby_swift_values = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

# axs[2, 0].plot(deciles, derby_baseline, '->', label='Baseline')
axs[2, 0].plot(deciles, derby_farsec_values, '-d', label='FARSEC')
axs[2, 0].plot(deciles, derby_de_values, '-o', label='DE+Learner')
axs[2, 0].plot(deciles, derby_smote_values, '-x', label='SMOTE')
axs[2, 0].plot(deciles, derby_smotuned_values, '-s', label='SMOTUNED')
axs[2, 0].plot(deciles, derby_swift_values, '-v', label='SWIFT')
axs[2, 0].set(xlabel='Deciles',
              ylabel='MAP',
              title='Derby')

axs[2, 0].legend()

fig.subplots_adjust(hspace=0.5)
fig.delaxes(axs[2, 1])

plt.show()
