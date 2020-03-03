import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

deciles = list(range(1, 11))

# chromium_baseline = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
chromium_farsec_values = [7.4, 4.8, 3.7, 2.9, 2.5, 2.2, 2.0, 1.9, 1.8, 1.6]
chromium_de_values = [8.3, 5.37, 4.24, 3.64, 3.34, 2.91, 2.53, 2.22, 1.97, 1.77]
chromium_smote_values = [8.52, 5.51, 4.35, 3.73, 3.42, 2.99, 2.59, 2.27, 2.03, 1.81]
chromium_smotuned_values = [8.92, 5.77, 4.56, 3.91, 3.58, 3.13, 2.71, 2.38, 2.12, 1.9]
chromium_swift_values = [9.63, 6.23, 4.92, 4.22, 3.87, 3.38, 2.93, 2.57, 2.29, 2.05]

# fig,axs=plt.subplots(nrows,ncols,figsize=(width,height))
fig, axs = plt.subplots(3, 2, figsize=(15, 28))

# axs[0, 0].plot(deciles, chromium_baseline, '->', label='Baseline')
axs[0, 0].plot(deciles, chromium_farsec_values, '-d', label='clnifarsecsq')
axs[0, 0].plot(deciles, chromium_de_values, '-o', label='DE+Learner')
axs[0, 0].plot(deciles, chromium_smote_values, '-x', label='SMOTE')
axs[0, 0].plot(deciles, chromium_smotuned_values, '-s', label='SMOTUNED')
axs[0, 0].plot(deciles, chromium_swift_values, '-v', label='SWIFT')
axs[0, 0].set(xlabel='Deciles',
              ylabel='MAP',
              title='Chromium')

axs[0, 0].legend()

# wicket_baseline = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
wicket_farsec_values = [1.0, 2.5, 3.0, 2.7, 2.6, 2.4, 2.3, 2.2, 2.1, 2.03]
wicket_de_values = [8.02, 5.34, 3.88, 3.36, 3.19, 3.02, 2.93, 2.84, 2.76, 2.72]
wicket_smote_values = [8.53, 5.69, 4.13, 3.58, 3.39, 3.21, 3.12, 3.03, 2.94, 2.89]
wicket_smotuned_values = [8.86, 5.9, 4.29, 3.71, 3.52, 3.33, 3.24, 3.14, 3.05, 3.0]
wicket_swift_values = [9.3, 6.2, 4.5, 3.9, 3.7, 3.5, 3.4, 3.3, 3.2, 3.15]

# axs[0, 1].plot(deciles, wicket_baseline, '->', label='Baseline')
axs[0, 1].plot(deciles, wicket_farsec_values, '-d', label='clnifarsec')
axs[0, 1].plot(deciles, wicket_de_values, '-o', label='DE+Learner')
axs[0, 1].plot(deciles, wicket_smote_values, '-x', label='SMOTE')
axs[0, 1].plot(deciles, wicket_smotuned_values, '-s', label='SMOTUNED')
axs[0, 1].plot(deciles, wicket_swift_values, '-v', label='SWIFT')
axs[0, 1].set(xlabel='Deciles',
              ylabel='MAP',
              title='Wicket')

axs[0, 1].legend()

# amberi_baseline = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
amberi_farsec_values = [18.1, 13, 10.1, 8.35, 7.4, 6.5, 6.0, 5.6, 5.3, 5.1]
amberi_de_values = [18.12, 13.01, 10.11, 8.36, 7.41, 6.5, 6.01, 5.6, 5.31, 5.11]
amberi_smote_values = [18.45, 13.25, 10.29, 8.51, 7.54, 6.62, 6.12, 5.71, 5.4, 5.2]
amberi_smotuned_values = [18.45, 13.25, 10.29, 8.51, 7.54, 6.62, 6.12, 5.71, 5.4, 5.2]
amberi_swift_values = [19.72, 14.16, 11.0, 9.1, 8.06, 7.08, 6.54, 6.1, 5.77, 5.56]

# axs[1, 0].plot(deciles, amberi_baseline, '->', label='Baseline')
axs[1, 0].plot(deciles, amberi_farsec_values, '-d', label='farsectwo')
axs[1, 0].plot(deciles, amberi_de_values, '-o', label='DE+Learner')
axs[1, 0].plot(deciles, amberi_smote_values, '-x', label='SMOTE')
axs[1, 0].plot(deciles, amberi_smotuned_values, '-s', label='SMOTUNED')
axs[1, 0].plot(deciles, amberi_swift_values, '-v', label='SWIFT')
axs[1, 0].set(xlabel='Deciles',
              ylabel='MAP',
              title='Amberi')

axs[1, 0].legend()

# camel_baseline = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
camel_farsec_values = [16.9, 10.7, 8.4, 7.3, 6.0, 5.6, 5.2, 5.1, 5.05, 5.0]
camel_de_values = [20.83, 13.97, 11.57, 9.92, 8.93, 8.1, 7.11, 6.28, 5.7, 5.37]
camel_smote_values = [21.72, 14.57, 12.07, 10.34, 9.31, 8.45, 7.41, 6.55, 5.95, 5.6]
camel_smotuned_values = [22.5, 15.09, 12.5, 10.71, 9.64, 8.75, 7.68, 6.79, 6.16, 5.8]
camel_swift_values = [25.2, 16.9, 14, 12.0, 10.8, 9.8, 8.6, 7.6, 6.9, 6.5]

# axs[1, 1].plot(deciles, camel_baseline, '->', label='Baseline')
axs[1, 1].plot(deciles, camel_farsec_values, '-d', label='clnifarsecsq')
axs[1, 1].plot(deciles, camel_de_values, '-o', label='DE+Learner')
axs[1, 1].plot(deciles, camel_smote_values, '-x', label='SMOTE')
axs[1, 1].plot(deciles, camel_smotuned_values, '-s', label='SMOTUNED')
axs[1, 1].plot(deciles, camel_swift_values, '-v', label='SWIFT')
axs[1, 1].set(xlabel='Deciles',
              ylabel='MAP',
              title='Camel')

axs[1, 1].legend()

# derby_baseline = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
derby_farsec_values = [53, 37, 29.5, 25, 23, 20, 18, 17, 16, 15]
derby_de_values = [51.69, 40.68, 33.9, 29.66, 26.27, 24.58, 22.03, 20.34, 19.07, 17.8]
derby_smote_values = [54.46, 42.86, 35.71, 31.25, 27.68, 25.89, 23.21, 21.43, 20.09, 18.75]
derby_smotuned_values = [57.01, 44.86, 37.38, 32.71, 28.97, 27.1, 24.3, 22.43, 21.03, 19.63]
derby_swift_values = [61, 48, 40, 35, 31, 29, 26, 24, 22.5, 21]

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
