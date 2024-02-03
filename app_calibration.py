import streamlit as st

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from torch.utils.data import Dataset

import numpy as np

import HydroErr

import pygad

# load catchment list
catchments = pd.read_csv("./data/Caravan-CAMELS/catchments.csv", dtype=str)

# write title texts
st.header("Generative hydrological modeling")

st.subheader(
    "Hydrological model calibration: search the latent space to identify the optimal numerical vectors for generating hydrological model instances."
)

st.markdown("*Use the sidebar to specify data sources and genetic algorithm (GA) parameter values.*")


st.markdown(
    'Click the :blue["Run optimization"] button to see calibration results, simulated hydrographs, predictions, etc.'
)

# load model
decoder = torch.load(
    "data/final_lstm_decoder_test.pt", map_location=torch.device("cpu")
)

decoder.eval()


# Input calibration time series

st.sidebar.markdown(
    "## Select a catchment from the CAMELS dataset or upload a climate forcing and discharge time series."
)

selected_catchment = st.sidebar.selectbox(
    "Which CAMELS catchment do you want to model?",
    catchments["gauge_name"].tolist(),
    index=0,
    placeholder="Choose a catchment...",
)

st.sidebar.markdown(
    "## Or, upload *calibration* and *test* climate forcing and discharge time series data."
)

uploaded_file_calibration = st.sidebar.file_uploader(
    "Select a *calibration* period comma-separated CSV file with no headers. The four columns are P, T, PET, and Q."
)

uploaded_file_test = st.sidebar.file_uploader(
    "Select a *test* period comma-separated CSV file with no headers. The four columns are P, T, PET, and Q."
)


if uploaded_file_calibration is not None:
    input_data = np.genfromtxt(uploaded_file_calibration, delimiter=",")
    x_cal = torch.from_numpy(input_data[:, 0:3]).unsqueeze(0).to(dtype=torch.float32)
    y_cal = torch.from_numpy(input_data[:, 3]).unsqueeze(0).to(dtype=torch.float32)
else:
    file_name = catchments[catchments["gauge_name"] == selected_catchment][
        "data_train"
    ].to_string(index=False)
    input_data = np.genfromtxt(file_name, delimiter=",")
    x_cal = torch.from_numpy(input_data[:, 0:3]).unsqueeze(0).to(dtype=torch.float32)
    y_cal = torch.from_numpy(input_data[:, 3]).unsqueeze(0).to(dtype=torch.float32)


if uploaded_file_test is not None:
    input_data = np.genfromtxt(uploaded_file_test, delimiter=",")
    x_test = torch.from_numpy(input_data[:, 0:3]).unsqueeze(0).to(dtype=torch.float32)
    y_test = torch.from_numpy(input_data[:, 3]).unsqueeze(0).to(dtype=torch.float32)
else:
    file_name = catchments[catchments["gauge_name"] == selected_catchment][
        "data_test"
    ].to_string(index=False)
    input_data = np.genfromtxt(file_name, delimiter=",")
    x_test = torch.from_numpy(input_data[:, 0:3]).unsqueeze(0).to(dtype=torch.float32)
    y_test = torch.from_numpy(input_data[:, 3]).unsqueeze(0).to(dtype=torch.float32)


# input warm-up period
st.sidebar.markdown("## Select a warm-up period.")

warm_up = st.sidebar.number_input(
    "How many days of input are used for warm up?",
    min_value=0,
    max_value=x_cal.shape[1],
    value=min(365, x_cal.shape[1]),
)


# input number of generaion
st.sidebar.markdown("## Select the number of generations of GA.")

num_generations = st.sidebar.number_input(
    "Input the number of generations.",
    min_value=2,
    max_value=500,
    value=30,
)

# input sol_per_pop
st.sidebar.markdown("## Select the population size of GA.")

sol_per_pop = st.sidebar.number_input(
    "Input the population size.",
    min_value=20,
    max_value=500,
    value=40,
)


# input sol_per_pop
st.sidebar.markdown("## Select the number of solutions to be selected as parents.")

num_parents_mating = st.sidebar.number_input(
    "Input the number of solutions to be selected as parents.",
    min_value=2,
    max_value=sol_per_pop,
    value=10,
)


class Objective_builder:
    def __init__(self, x, y):
        self.x = x.contiguous()
        self.y = y.contiguous()

    def eval(self, ga_instance, solution, solution_idx):
        # numpy to torch tensor
        solution = torch.from_numpy(solution).unsqueeze(0).to(dtype=torch.float32)
        solution = solution.expand(self.x.shape[0], -1)

        # BASE_LENGTH is from global
        pred = (
            decoder.decode(solution, self.x, base_length=warm_up)
            .view(-1)
            .detach()
            .cpu()
            .numpy()
        )

        ob = self.y.view(-1).detach().cpu().numpy()[warm_up:]

        return HydroErr.kge_2009(simulated_array=pred, observed_array=ob)

    def pred(self, solution):
        # numpy to torch tensor
        solution = torch.from_numpy(solution).unsqueeze(0).to(dtype=torch.float32)
        solution = solution.expand(self.x.shape[0], -1)

        # BASE_LENGTH is from global
        pred = (
            decoder.decode(solution, self.x, base_length=warm_up)
            .view(-1)
            .detach()
            .cpu()
            .numpy()
        )

        ob = self.y.view(-1).detach().cpu().numpy()[warm_up:]

        d = {
            "Simulated [mm/day]": pred.tolist(),
            "Observation [mm/day]": ob.tolist(),
        }

        chart_data = pd.DataFrame(data=d)

        return chart_data

# Hyperparameters of GA
num_genes = 8

init_range_low = -11
init_range_high = 11

parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"
mutation_probability = 0.25

fn_cal = Objective_builder(x_cal, y_cal)
fn_test = Objective_builder(x_test, y_test)

# run optimization
if "clicked" not in st.session_state:
    st.session_state.clicked = False

progress_text = (
    "Optimization in progress. Please wait. It may take from 1 to several minutes."
)

if st.session_state.clicked:
    my_bar = st.progress(0, text=progress_text)


def on_generation(instance):
    generations_completed = instance.generations_completed

    my_bar.progress(generations_completed / num_generations, text=progress_text)


ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fn_cal.eval,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    init_range_low=init_range_low,
    init_range_high=init_range_high,
    parent_selection_type=parent_selection_type,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_probability=mutation_probability,
    on_generation=on_generation,
)

def click_button():
    st.session_state.clicked = True
    

st.button(":blue[Run optimization]", on_click=click_button)

if st.session_state.clicked:
    
    ga_instance.run()
    
    # Gof
    kge_cal = round(ga_instance.best_solution()[1], 3)
    kge_test = round(fn_test.eval(0, ga_instance.best_solution()[0], 0), 3)
    
    if uploaded_file_calibration is not None:
        st.markdown("Calibration results of user supplied catchment data")
    else:
        st.markdown(
            f"Calibration results of the :red[{selected_catchment}, USA]."
        )

    f"Performance of the optimal (i.e., calibrated) model instance: :red[**Calibration KGE={kge_cal}**], :red[**Test KGE={kge_test}**]."

    st.divider()

    # Optimal model parameters
    optimal_para = pd.DataFrame(ga_instance.best_solution()[0])
    optimal_para["Parameter no."] = range(1, len(optimal_para) + 1)

    optimal_para.columns = ["Optimal value", "Parameter number"]
    st.dataframe(optimal_para[["Parameter number","Optimal value"]], hide_index=True)
    
    st.divider()

    # Show test result:
    st.write(
        "### The *test period* simulated and observed discharge data without the warm-up period:"
    )

    chart_test = fn_test.pred(ga_instance.best_solution()[0])

    st.line_chart(chart_test, color=["#0457ac", "#a7e237"])
    f":red[**Test KGE={kge_test}**]."

    chart_test

    st.divider()

    # Show calibration result:
    st.write(
        "### The *calibration period* simulated and observed discharge data without the warm-up period:"
    )

    chart_cal = fn_cal.pred(ga_instance.best_solution()[0])

    st.line_chart(chart_cal, color=["#0457ac", "#a7e237"])

    f":red[**Calibration KGE={kge_cal}**]."

    chart_cal
    
    st.session_state.clicked = False


# References:
st.divider()

st.markdown(
    "The method for developing generative hydrological model is discribed in the paper: [Learning to Generate Lumped Hydrological Models](https://arxiv.org/abs/2309.09904)."
)
st.caption(
    "The CAMELS catchment data was derived from the [Caravan dataset](https://doi.org/10.1038/s41597-023-01975-w). License of the dataset can be found in the GitHub page of this web appplication."
)

st.markdown(
    '<a href="mailto:yyang90@connect.hku.hk">Contact the authors. </a>',
    unsafe_allow_html=True,
)
