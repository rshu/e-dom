import matplotlib.pyplot as plt

# plt.style.use('seaborn-whitegrid')

font = {'family': 'normal',
        'size': 20}
plt.rc('font', **font)

deciles = list(range(1, 11))

chromium_baseline = [0.15, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.39, 0.41]
chromium_farsec_values = [7.41, 4.95, 3.72, 2.91, 2.43, 2.10, 1.92, 1.81, 1.73, 1.68]
chromium_de_values = [8.07, 5.93, 4.69, 3.87, 3.18, 2.74, 2.34, 2.11, 1.92, 1.84]
chromium_smote_values = [9.32, 6.85, 5.41, 4.47, 3.79, 3.16, 2.7, 2.39, 2.1, 1.93]
chromium_smotuned_values = [9.58, 7.04, 5.56, 4.6, 3.9, 3.25, 2.78, 2.46, 2.16, 1.98]
chromium_swift_values = [10.25, 7.53, 5.95, 4.92, 4.17, 3.48, 2.95, 2.63, 2.31, 2.12]

# fig,axs=plt.subplots(nrows,ncols,figsize=(width,height))
fig, axs = plt.subplots(3, 2, figsize=(15, 28))

axs[0, 0].plot(deciles, chromium_baseline, '->', label='Baseline', linewidth=3.0)
axs[0, 0].plot(deciles, chromium_farsec_values, '-d', label='clnifarsecsq', linewidth=3.0)
axs[0, 0].plot(deciles, chromium_de_values, '-o', label='DE+Learner', linewidth=3.0)
axs[0, 0].plot(deciles, chromium_smote_values, '-x', label='SMOTE', linewidth=3.0)
axs[0, 0].plot(deciles, chromium_smotuned_values, '-s', label='SMOTUNED', linewidth=3.0)
axs[0, 0].plot(deciles, chromium_swift_values, '-v', label='SWIFT', linewidth=3.0)
axs[0, 0].set(xlabel='Deciles',
              ylabel='MAP',
              title='Chromium')

axs[0, 0].legend()

wicket_baseline = [0.0, 0.21, 0.21, 0.2, 0.2, 0.2, 0.2, 0.2, 0.21, 0.25]
wicket_farsec_values = [1.00, 2.48, 2.85, 2.76, 2.63, 2.54, 2.45, 2.36, 2.29, 2.22]
wicket_de_values = [5.55, 4.3, 3.4, 2.98, 2.72, 2.58, 2.43, 2.33, 2.26, 2.23]
wicket_smote_values = [6.65, 4.81, 3.8, 3.33, 3.05, 2.88, 2.72, 2.6, 2.53, 2.46]
wicket_smotuned_values = [6.84, 4.94, 3.91, 3.42, 3.13, 2.96, 2.79, 2.67, 2.6, 2.53]
wicket_swift_values = [7.11, 5.14, 4.06, 3.55, 3.25, 3.08, 2.9, 2.78, 2.7, 2.63]

axs[0, 1].plot(deciles, wicket_baseline, '->', label='Baseline', linewidth=3.0)
axs[0, 1].plot(deciles, wicket_farsec_values, '-d', label='clnifarsec', linewidth=3.0)
axs[0, 1].plot(deciles, wicket_de_values, '-o', label='DE+Learner', linewidth=3.0)
axs[0, 1].plot(deciles, wicket_smote_values, '-x', label='SMOTE', linewidth=3.0)
axs[0, 1].plot(deciles, wicket_smotuned_values, '-s', label='SMOTUNED', linewidth=3.0)
axs[0, 1].plot(deciles, wicket_swift_values, '-v', label='SWIFT', linewidth=3.0)
axs[0, 1].set(xlabel='Deciles',
              ylabel='MAP',
              title='Wicket')

axs[0, 1].legend()

amberi_baseline = [5.81, 5.75, 5.22, 4.95, 4.81, 4.73, 4.65, 4.58, 4.49, 4.46]
amberi_farsec_values = [18.10, 13.52, 10.17, 8.35, 7.26, 6.52, 6.03, 5.64, 5.35, 5.18]
amberi_de_values = [18.11, 13.66, 10.62, 8.7, 7.4, 6.5, 6.01, 5.6, 5.3, 5.1]
amberi_smote_values = [18.12, 13.68, 10.63, 8.71, 7.41, 6.51, 6.02, 5.61, 5.3, 5.11]
amberi_smotuned_values = [18.16, 13.7, 10.65, 8.73, 7.42, 6.52, 6.03, 5.62, 5.31, 5.12]
amberi_swift_values = [19.52, 14.73, 11.45, 9.38, 7.98, 7.01, 6.48, 6.04, 5.71, 5.5]

axs[1, 0].plot(deciles, amberi_baseline, '->', label='Baseline', linewidth=3.0)
axs[1, 0].plot(deciles, amberi_farsec_values, '-d', label='farsectwo', linewidth=3.0)
axs[1, 0].plot(deciles, amberi_de_values, '-o', label='DE+Learner', linewidth=3.0)
axs[1, 0].plot(deciles, amberi_smote_values, '-x', label='SMOTE', linewidth=3.0)
axs[1, 0].plot(deciles, amberi_smotuned_values, '-s', label='SMOTUNED', linewidth=3.0)
axs[1, 0].plot(deciles, amberi_swift_values, '-v', label='SWIFT', linewidth=3.0)
axs[1, 0].set(xlabel='Deciles',
              ylabel='MAP',
              title='Amberi')

axs[1, 0].legend()

camel_baseline = [1.73, 2.65, 2.68, 2.685, 2.690, 2.695, 2.70, 2.705, 2.71, 2.72]
camel_farsec_values = [16.90, 10.91, 8.43, 7.25, 6.21, 5.60, 5.22, 5.07, 5.02, 5.00]
camel_de_values = [19.16, 14.56, 11.96, 10.36, 9.1, 8.08, 7.12, 6.33, 5.87, 5.65]
camel_smote_values = [19.67, 14.95, 12.27, 10.64, 9.35, 8.29, 7.31, 6.5, 6.03, 5.8]
camel_smotuned_values = [19.67, 14.95, 12.27, 10.64, 9.35, 8.29, 7.31, 6.5, 6.03, 5.8]
camel_swift_values = [22.23, 16.89, 13.87, 12.02, 10.56, 9.37, 8.26, 7.34, 6.81, 6.55]

axs[1, 1].plot(deciles, camel_baseline, '->', label='Baseline', linewidth=3.0)
axs[1, 1].plot(deciles, camel_farsec_values, '-d', label='clnifarsecsq', linewidth=3.0)
axs[1, 1].plot(deciles, camel_de_values, '-o', label='DE+Learner', linewidth=3.0)
axs[1, 1].plot(deciles, camel_smote_values, '-x', label='SMOTE', linewidth=3.0)
axs[1, 1].plot(deciles, camel_smotuned_values, '-s', label='SMOTUNED', linewidth=3.0)
axs[1, 1].plot(deciles, camel_swift_values, '-v', label='SWIFT', linewidth=3.0)
axs[1, 1].set(xlabel='Deciles',
              ylabel='MAP',
              title='Camel')

axs[1, 1].legend()

derby_baseline = [10.00, 9.00, 9.00, 9.00, 8.75, 8.50, 8.50, 8.50, 8.50, 8.50]
derby_farsec_values = [55.00, 38.00, 29.50, 25.00, 22.25, 20.00, 17.85, 16.75, 16.00, 15.75]
derby_de_values = [55.96, 44.04, 36.7, 32.11, 28.44, 25.69, 23.39, 22.02, 21.33, 20.64]
derby_smote_values = [56.48, 44.44, 37.04, 32.41, 28.7, 25.93, 23.61, 22.22, 21.53, 20.83]
derby_smotuned_values = [59.22, 46.6, 38.83, 33.98, 30.1, 27.18, 24.76, 23.3, 22.57, 21.84]
derby_swift_values = [61, 48, 40, 35, 31, 28, 25.50, 24, 23.25, 22.50]

axs[2, 0].plot(deciles, derby_baseline, '->', label='Baseline', linewidth=3.0)
axs[2, 0].plot(deciles, derby_farsec_values, '-d', label='clni', linewidth=3.0)
axs[2, 0].plot(deciles, derby_de_values, '-o', label='DE+Learner', linewidth=3.0)
axs[2, 0].plot(deciles, derby_smote_values, '-x', label='SMOTE', linewidth=3.0)
axs[2, 0].plot(deciles, derby_smotuned_values, '-s', label='SMOTUNED', linewidth=3.0)
axs[2, 0].plot(deciles, derby_swift_values, '-v', label='SWIFT', linewidth=3.0)
axs[2, 0].set(xlabel='Deciles',
              ylabel='MAP',
              title='Derby')

axs[2, 0].legend()

fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig.delaxes(axs[2, 1])

plt.savefig("MAP.png", bbox_inches='tight')
# plt.show()
