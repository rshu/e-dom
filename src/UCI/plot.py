import os
import plotly
plotly.tools.set_credentials_file(username='amritbhanu', api_key='cuaXxPfbSxptk2irXf7P')
import plotly.plotly as py
import plotly.graph_objs as go
import pickle


chirp={'optdigits': [0.009, 0.012, 0.013, 0.05], 'satellite': [0.009, 0.009, 0.009, 0.012], 'covtype': [0.0, 0.0, 0.007, 0.707], 'pendigits': [0.0, 0.001, 0.001, 0.001], 'cancer': [0.044, 0.048, 0.048, 0.05], 'diabetic': [0.319, 0.323, 0.329, 0.333], 'waveform': [0.087, 0.088, 0.089, 0.091], 'adult': [0.279, 0.284, 0.286, 0.289], 'shuttle': [0.001, 0.001, 0.001, 0.001]}

cwd = os.getcwd()
data_path = os.path.join(cwd,"..","..","dump", "UCI")

with open(os.path.join(data_path, 'dodge.pickle'), 'rb') as handle:
    dodge = pickle.load(handle)

with open(os.path.join(data_path, 'fft.pickle'), 'rb') as handle:
    details = pickle.load(handle)


n1, n2, n3, n4, n5, n6, n7, n8, n9 = 'optdigits', 'waveform', 'pendigits', 'diabetic', 'adult', 'cancer', 'shuttle', 'covtype', 'satellite'

classifiers = ["DT", "RF", "SVM", "FFT-Dist2Heaven","Chirp", "Dodge_0.2_30"]
colors = ["#AED6F1", "#5DADE2", "#2874A6", "#1B4F72", "#000000", "#FF5722"]#, "#E53935"]

data = []
l = len(details[n1][classifiers[0]])
## fft
x = [n1] * l + [n2] * l + [n3] * l + [n4] * l+ [n5] * l+ [n6] * l+ [n7] * l+ [n8] * l+ [n9] * l

## dodge
x1 = [n1] * 21 + [n2] * 21 + [n3] * 21 + [n4] * 21+ [n5] * 21+ [n6] * 21+ [n7] * 21+ [n8] * 21+ [n9] * 21

## chirp
x2 = [n1] * 4 + [n2] * 4 + [n3] * 4 + [n4] * 4+ [n5] * 4+ [n6] * 4+ [n7] * 4+ [n8] * 4+ [n9] * 4

for i, clf in enumerate(classifiers):
    if clf != "Dodge_0.2_30" and clf!="Chirp":
        tmp_bar = go.Box(
            y=sorted(details[n1][clf]) +
            sorted(details[n2][clf]) +
            sorted(details[n3][clf]) +
            sorted(details[n4][clf]) +
            sorted(details[n5][clf]) +
            sorted(details[n6][clf]) +
            sorted(details[n7][clf]) +
            sorted(details[n8][clf]) +
            sorted(details[n9][clf]),
            x=x,
            name=clf,
            marker=dict(
                color=colors[i]
            )
        )
    elif clf=="Chirp":
        tmp_bar = go.Box(
            y=sorted(chirp[n1]) +
              sorted(chirp[n2]) +
              sorted(chirp[n3]) +
              sorted(chirp[n4]) +
              sorted(chirp[n5]) +
              sorted(chirp[n6]) +
              sorted(chirp[n7]) +
              sorted(chirp[n8]) +
              sorted(chirp[n9]),
            x=x2,
            name=clf,
            marker=dict(
                color=colors[i]
            )
        )
    elif clf=="Dodge_0.2_30":
        tmp_bar = go.Box(
            y=sorted(dodge[n1]) +
              sorted(dodge[n2]) +
              sorted(dodge[n3]) +
              sorted(dodge[n4]) +
              sorted(dodge[n5]) +
              sorted(dodge[n6]) +
              sorted(dodge[n7]) +
              sorted(dodge[n8]) +
              sorted(dodge[n9]),
            x=x1,
            name=clf,
            marker=dict(
                color=colors[i]
            )
        )
    data.append(tmp_bar)

layout = go.Layout(
    autosize=True,
    title="UCI - 25 Times",
    font=dict(size=18),
    yaxis=dict(
        title='Distance to Heaven',
        zeroline=False,
        titlefont=dict(size=20),
        tickfont=dict(size=24),
        automargin=True,
    ),
    xaxis=dict(
        title='UCI Datasets',
        zeroline=False,
        titlefont=dict(size=24),
        tickfont=dict(size=20),
        tickangle=-45,
        automargin=True,
    ),
    boxmode='group',
    legend=dict(font=dict(size=20)
    )
)
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename="UCI - 25 Times")
