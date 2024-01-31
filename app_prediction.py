import streamlit as st

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from torch.utils.data import Dataset

import numpy as np

import HydroErr

# load model
embedding = torch.load(
    "data/final_lstm_embedding_test.pt", map_location=torch.device("cpu")
)
decoder = torch.load(
    "data/final_lstm_decoder_test.pt", map_location=torch.device("cpu")
)

embedding.eval()
decoder.eval()


# Input time series 
st.sidebar.markdown(
    "## Upload climate forcing and discharge time series data [Optional]."
)

uploaded_file = st.sidebar.file_uploader(
    "Select a comma-separated CSV file with no headers. The four columns are P, T, PET, and Q."
)

if uploaded_file is not None:
    input_data = np.genfromtxt(uploaded_file, delimiter=",")
    x = torch.from_numpy(input_data[:, 0:3]).unsqueeze(0).to(dtype=torch.float32)
else:
    input_data = np.genfromtxt("./data/app_train.csv", delimiter=",")
    x = torch.from_numpy(input_data[:, 0:3]).unsqueeze(0).to(dtype=torch.float32)


# input parameter values
st.sidebar.markdown(
    "## Select the eight parameter values below to generate a model instance that can be used for hydrological prediction."
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


# input warm-up period
st.sidebar.markdown("## Select a warm-up period.")

warm_up = st.sidebar.number_input(
    "How many days of input are used for warm up?",
    min_value=0,
    max_value=x.shape[1],
    value=min(365, x.shape[1]),
)


# input display period
st.sidebar.markdown("## Select the hydrograph length to display.")

dispaly_days = st.sidebar.slider(
    "How many days of hydrograph to display?",
    min_value=1,
    max_value=x.shape[1]-warm_up,
    value=min(1500, x.shape[1]-warm_up),
)


# prediction
solution = np.array(
    [number1, number2, number3, number4, number5, number6, number7, number8]
)

solution = torch.from_numpy(solution).unsqueeze(0).to(dtype=torch.float32)
solution = solution.expand(x.shape[0], -1)

pred = decoder.decode(solution, x, base_length=warm_up).view(-1).detach().cpu().numpy()

# process prediction result
d = {
    "Simulated [mm/day]": pred.tolist(),
    "Observation [mm/day]": input_data[warm_up:, 3].tolist(),
}

chart_data = pd.DataFrame(data=d)

# Plotting
st.subheader("Comparison of simulated and observed hydrographs.")
st.markdown("*Select paramater values from the sidebar to generate model instances.*")
st.markdown("*[Optional] Upload climate forcing and discharge time series data. If no data are uploaded, data of Fish River near Fort Kent, Maine, US (USGS gauge ID: 01013500) will be used.*")


st.line_chart(chart_data[0:dispaly_days], color=["#0457ac", "#a7e237"])

# Prediction accuracy

kge = HydroErr.kge_2009(
    simulated_array=pred, observed_array=input_data[warm_up:, 3]
)
kge = round(kge, 3)

nse = HydroErr.nse(simulated_array=pred, observed_array=input_data[warm_up:, 3])
nse = round(nse, 3)

f"Performance of the generated model instance on all data: :red[**KGE={kge}**], :red[**NSE={nse}**]."


# Prediction accuracy of displayed period
kge2 = HydroErr.kge_2009(
    simulated_array=pred[0:dispaly_days], observed_array=input_data[warm_up:(dispaly_days+warm_up), 3]
)
kge2 = round(kge2, 3)

nse2 = HydroErr.nse(simulated_array=pred[0:dispaly_days], observed_array=input_data[warm_up:(dispaly_days+warm_up), 3])
nse2 = round(nse2, 3)

f"Performance of the generated model instance in the displayed period: :red[**KGE={kge2}**], :red[**NSE={nse2}**]."

# Display data
st.divider()

st.write("The predicted and observed discharge data without the warm-up period:")
st.write(chart_data)

# References:
st.divider()

st.markdown(
    "The method for developing is discribed in the paper: [Learning to Generate Lumped Hydrological Models](https://arxiv.org/abs/2309.09904)."
)
st.caption(
    "The Fish River data was derived from the [CAMELS dataset](https://ral.ucar.edu/solutions/products/camels), which was further proceeded by [Knoben et al. (2020)](http://dx.doi.org/10.1029/2019WR025975)."
)

st.markdown(
    '<a href="mailto:yyang90@connect.hku.hk">Contact the authors. </a>',
    unsafe_allow_html=True,
)