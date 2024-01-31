import streamlit as st

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from torch.utils.data import Dataset

import numpy as np

import HydroErr

# load model and data
embedding = torch.load(
    "data/final_lstm_embedding_test.pt", map_location=torch.device("cpu")
)
decoder = torch.load(
    "data/final_lstm_decoder_test.pt", map_location=torch.device("cpu")
)

embedding.eval()
decoder.eval()

data_train = np.genfromtxt("./data/app_train.csv", delimiter=",")
x = torch.from_numpy(data_train[:, 0:3]).unsqueeze(0).to(dtype=torch.float32)
x = x[0:1500]

# model parameter input
st.sidebar.markdown(
    "## Select the eight parameter values to generate a model instance that can be used for hydrological prediction."
)
number1 = st.sidebar.slider(
    "Select parameter 1",
    min_value=-11.0,
    max_value=11.0,
    value=-3.087250730755500161e00,
)
number2 = st.sidebar.slider(
    "Select parameter 2",
    min_value=-11.0,
    max_value=11.0,
    value=-2.015848556240998235e00,
)
number3 = st.sidebar.slider(
    "Select parameter 3",
    min_value=-11.0,
    max_value=11.0,
    value=-8.476807188488493239e00,
)
number4 = st.sidebar.slider(
    "Select parameter 4", min_value=-11.0, max_value=11.0, value=7.177632989891177928e00
)
number5 = st.sidebar.slider(
    "Select parameter 5",
    min_value=-11.0,
    max_value=11.0,
    value=-3.922460983060631623e-01,
)
number6 = st.sidebar.slider(
    "Select parameter 6",
    min_value=-11.0,
    max_value=11.0,
    value=1.808005759465904916e-01,
)
number7 = st.sidebar.slider(
    "Select parameter 7",
    min_value=-11.0,
    max_value=11.0,
    value=-9.858767797305613811e00,
)
number8 = st.sidebar.slider(
    "Select parameter 8", min_value=-11.0, max_value=11.0, value=3.532966359419816627e00
)


solution = np.array(
    [number1, number2, number3, number4, number5, number6, number7, number8]
)

solution = torch.from_numpy(solution).unsqueeze(0).to(dtype=torch.float32)
solution = solution.expand(x.shape[0], -1)

# Prediction
pred = decoder.decode(solution, x).view(-1).detach().cpu().numpy()
d = {
    "Simulated [mm/day]": pred.tolist(),
    "Observation [mm/day]": data_train[365:, 3].tolist(),
}

chart_data = pd.DataFrame(data=d)

# Plotting
st.title("Comparison of simulated and observed hydrographs.")
st.markdown("*Select paramater values to create model instances from the sidebar.*")

st.markdown("Fish River near Fort Kent, Maine, US (USGS gauge ID: 01013500)")
st.line_chart(chart_data[0:1000], color=["#0457ac", "#a7e237"])

# Prediction accuracy
kge = HydroErr.kge_2009(
    simulated_array=pred[0:1000], observed_array=data_train[365:1365:, 3]
)
kge = round(kge, 3)

nse = HydroErr.nse(
    simulated_array=pred[0:1000], observed_array=data_train[365:1365:, 3]
)
nse = round(nse, 3)

f"Performance of the generated model instance: **KGE={kge}**, **NSE={nse}**."

# Reference:

st.markdown(
    "The method for developing is discribed in the paper: [Learning to Generate Lumped Hydrological Models](https://arxiv.org/abs/2309.09904)."
)
st.caption(
    "The data are derived from the [CAMELS dataset](https://ral.ucar.edu/solutions/products/camels), and further proceeded by [Knoben et al. (2020)](http://dx.doi.org/10.1029/2019WR025975)."
)

st.markdown(
    '<a href="mailto:yyang90@connect.hku.hk">Contact the authors. </a>',
    unsafe_allow_html=True,
)
