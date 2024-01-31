import streamlit as st

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from torch.utils.data import Dataset

import numpy as np

import HydroErr

import pygad

# load model
decoder = torch.load(
    "data/final_lstm_decoder_test.pt", map_location=torch.device("cpu")
)

decoder.eval()


# Input calibration time series 
st.sidebar.markdown(
    "## Upload a *calibration* climate forcing and discharge time series data [Optional]."
)

uploaded_file_calibration = st.sidebar.file_uploader(
    "Select a calibration period comma-separated CSV file with no headers. The four columns are P, T, PET, and Q."
)


# Input test time series 
st.sidebar.markdown(
    "## Upload a *test* climate forcing and discharge time series data [Optional]."
)

uploaded_file_test = st.sidebar.file_uploader(
    "Select a test period comma-separated CSV file with no headers. The four columns are P, T, PET, and Q."
)


if uploaded_file_calibration is not None:
    input_data = np.genfromtxt(uploaded_file_calibration, delimiter=",")
    x_cal = torch.from_numpy(input_data[:, 0:3]).unsqueeze(0).to(dtype=torch.float32)
    y_cal = torch.from_numpy(input_data[:, 3]).unsqueeze(0).to(dtype=torch.float32)
else:
    input_data = np.genfromtxt("./data/app_train.csv", delimiter=",")
    x_cal = torch.from_numpy(input_data[:, 0:3]).unsqueeze(0).to(dtype=torch.float32)
    y_cal = torch.from_numpy(input_data[:, 3]).unsqueeze(0).to(dtype=torch.float32)


if uploaded_file_test is not None:
    input_data = np.genfromtxt(uploaded_file_test, delimiter=",")
    x_test = torch.from_numpy(input_data[:, 0:3]).unsqueeze(0).to(dtype=torch.float32)
    y_test = torch.from_numpy(input_data[:, 3]).unsqueeze(0).to(dtype=torch.float32)
else:
    input_data = np.genfromtxt("./data/app_test.csv", delimiter=",")
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


class Objective_builder:
    def __init__(self, x, y):
        self.x = x.contiguous()
        self.y = y.contiguous()
    
    def eval(self, ga_instance, solution, solution_idx):
        
        # numpy to torch tensor
        solution = torch.from_numpy(solution).unsqueeze(0).to(dtype=torch.float32)
        solution = solution.expand(self.x.shape[0], -1)
        
        # BASE_LENGTH is from global
        pred = decoder.decode(solution, self.x, base_length=warm_up).view(-1).detach().cpu().numpy()

        ob = self.y.view(-1).detach().cpu().numpy()[warm_up:]
                
        return HydroErr.kge_2009(simulated_array=pred, observed_array=ob)

# Hyperparameters of GA
num_generations = 30
num_parents_mating = 10

sol_per_pop = 40
num_genes = 8

init_range_low = -11
init_range_high = 11

parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"
mutation_probability = 0.25


fn_cal = Objective_builder(x_cal,y_cal)
fn_test = Objective_builder(x_test,y_test)


# Identifying optimal number of generations

progress_text = "Optimization in progress. Please wait."


my_bar = st.progress(0, text=progress_text)

def on_generation(instance):
    
    generations_completed = instance.generations_completed
    
    my_bar.progress(generations_completed/num_generations, text = progress_text)
    
ga_instance = pygad.GA(num_generations=num_generations,
                    num_parents_mating=num_parents_mating,
                    fitness_func=fn_cal.eval,
                    sol_per_pop=sol_per_pop,
                    num_genes=num_genes,
                    init_range_low=init_range_low,
                    init_range_high=init_range_high,
                    parent_selection_type=parent_selection_type,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    mutation_probability = mutation_probability,
                    on_generation=on_generation)

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

st.button('Run optimization', on_click=click_button)

if st.session_state.clicked:
    ga_instance.run()
    
    kge_cal = round(ga_instance.best_solution()[1], 3)
    kge_test = round(fn_test.eval(0, ga_instance.best_solution()[0], 0), 3)

    f"Performance of the calibrated model instance: :red[**KGE={kge_cal}**], :red[**Test KGE={kge_test}**]."
    
    st.session_state.clicked = False

