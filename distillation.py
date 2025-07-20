import pandas as pd
import openpyxl
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from scipy.optimize import differential_evolution

# Load data
input_df = pd.read_excel("output.xlsx", engine="openpyxl")
#chunk size for the list iterations
chunk_size = 5
#random seed
np.random.seed(2025)
#threshold
threshold = 0.1

# Clean data
input_df["Result"] = input_df["Result"].apply(lambda x: max(x, 0))
input_df_copy = input_df.copy()
columns_initial = input_df.columns.difference(["Wavelength", "Result"])
for column in columns_initial:
    input_df[column] = input_df[column].apply(lambda x: 0 if x < 0 else x)

#Data Standardisation
# input_df[' Water vapor_0'] *= 10 
# input_df["carbon dioxide"] /= 500
cols = input_df.columns.difference(["Wavelength", "Result", " Water vapor_0", "carbon dioxide"])
for column in cols:
    input_df[column] /= 20
input_df = input_df[input_df["Result"] <= 0.35]
input_df_copy = input_df.copy()
size = len(input_df["Result"])

#Analysis area of water
water_df = input_df[(input_df['Wavelength'] >= 3200) & (input_df['Wavelength'] <= 3400)].reset_index(drop=True)

test_df = pd.read_excel("water_one_percent.xlsx", engine="openpyxl")
X_water = test_df.iloc[:,1] #water column
y_water = water_df["Result"] #sample column
water_size = len(water_df["Result"])
def cost_water(coef, X, y):
    return np.mean((y - X * coef)**2)        

#water_bound_col = [water_df.iloc[n]["Result"] / water_df.iloc[n][" Water vapor_0"] if water_df.iloc[n][" Water vapor_0"]  != 0 else 1_000_000 for n in range(water_size)]

#water_bound = sorted(set(water_bound_col))[0] #[0,x1,....]

result = minimize(
            cost_water,
            [0.001],
            args=(X_water, y_water),
            method='L-BFGS-B'
            #bounds=[(0.0,water_bound)]
        )

coef_water = result.x


input_df["Result"] = input_df["Result"] - input_df[" Water vapor_0"] * coef_water
input_df.to_excel("after_water.xlsx", engine="openpyxl")
input_df["Result"] = input_df["Result"].apply(lambda x: max(x, 0))
print(f"Water coefficient: {coef_water}")

# #Analysis area of CO2
co2_df = input_df[(input_df['Wavelength'] >= 1900) & (input_df['Wavelength'] <= 2298)].reset_index(drop=True)
co2_size = len(co2_df['Result'])
X_co2 = co2_df["carbon dioxide"]
y_co2 = co2_df["Result"]

def cost_co2(coef, X, y):
    return np.mean((y - X * coef)**2)        

#co2_bound_col = [co2_df.iloc[n]["Result"] / co2_df.iloc[n]["carbon dioxide"] if co2_df.iloc[n]["carbon dioxide"] != 0 else 1_000_000 for n in range(co2_size)]
#co2_bound = sorted(set(co2_bound_col))[1]

result = minimize(
            cost_co2,
            [0.001],
            args=(X_co2, y_co2),
            method='L-BFGS-B'
            #bounds=[(0.0,co2_bound)]
        )

coef_co2 = result.x

input_df["Result"] = input_df["Result"] - input_df["carbon dioxide"] * coef_co2
input_df.to_excel("after_co2.xlsx", engine="openpyxl")
input_df["Result"] = input_df["Result"].apply(lambda x: max(x, 0))
print(f"CO2 coefficient: {coef_co2 * 500}")

# inorganic gases
# co, n2o,
# read from excel or a dictionary:
# all the wavelengths
# also read the ref conc
# noise
# temp and pressure 
inor_df = input_df[(input_df['Wavelength'] >= 1900) & (input_df['Wavelength'] <= 2250)].reset_index(drop=True)

def cost_inor(coef, X_1, X_2, y):
    # coef_co = coef[0]
    # coef_n2o = coef[1]
    mse = np.mean((y - X_1 * coef[0] - X_2 * coef[1])**2)
    print(f"Coef: {coef}, MSE: {mse}")
    return mse

X_1 = inor_df["carbon monoxide"]
X_2 = inor_df["Nitrous oxide"]
y_inor = inor_df["Result"]

best_result = None
best_cost = float('inf')

# Try multiple random starting points
for i in range(10000):
    initial_guess = np.random.uniform(0.08, 0.35, 2)
    print(initial_guess)
    result = minimize(
        cost_inor,
        initial_guess,
        bounds=[(0.0, 0.35), (0.0, 0.35)],
        args=(X_1, X_2, y_inor),
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )
    
    if result.fun < best_cost:
        best_cost = result.fun
        best_result = result
        print(f"New best: {result.x}, cost: {result.fun}")

print(f"Final best: {best_result.x}, cost: {best_result.fun}")
print(np.mean((y_inor - X_1 * 0.20 - X_2*0.019)**2))
# #################### OTHER COMPONENTS #####################################
##next range
##### Now the area on the right: 800 to 1300
##### Components that have the strongest in this range
##### ethanol, methanol, isopropyl acetate, 
# columns_to_edit = input_df_copy.columns.difference(["Wavelength", "Result", " Water vapor_0", "carbon dioxide"])
# ratio_dict = {}
# for column in columns_to_edit:
#     ratio_col = [
#         input_df_copy.iloc[n]["Result"] / input_df_copy.iloc[n][column] if input_df_copy.iloc[n][column] != 0 else 1_000_000
#         for n in range(size)
#     ]
#     ratio_assorted = sorted(set(ratio_col))
#     ratio_dict[column] = [ratio_assorted[1]]

# ratio_dict = dict(sorted(ratio_dict.items(), key=lambda item: item[1], reverse=True))

# ratio_df = pd.DataFrame.from_dict(ratio_dict, orient="index")
# ratio_df.columns = [f"Ratio {i+1}" for i in range(ratio_df.shape[1])]
# ratio_df.index.name = "Wavelength"
# file_path = "ratio.xlsx"
# ratio_df.to_excel(file_path, index=True)
# filtered_dict = {key: value for key, value in ratio_dict.items() if not key.lower().endswith('ane')}

# keys = list(filtered_dict.keys())
# # keys = list(ratio_dict.keys())

# def iterative_selection(components, df, filtered_dict):
#     residuals = df["Result"].values
    
#     for i in range(0, len(components), chunk_size):
#         print(f"Processing batch: {components[i:i+chunk_size]}")
#         current_batch = components[i:i+chunk_size]
#         analysis_dict = {key: filtered_dict[key] for key in current_batch if key in filtered_dict}
#         coefficient_bounds = [(0, values[0]) for values in analysis_dict.values()]
#         initial_guess = np.array([0 for bound in coefficient_bounds])
#         X_batch = df[current_batch].values

#         def cost_function(coefs, X, y):
#             return np.mean((y - X @ coefs) ** 2)

#         # Perform bounded optimization
#         result = minimize(
#             cost_function,
#             initial_guess,
#             args=(X_batch, residuals),
#             method='L-BFGS-B',
#             bounds=coefficient_bounds
#         )

#         #Detection limit check to modify the coefficients of the linear combination
#         optimized_coefs = result.x  
#         print(optimized_coefs)




# iterative_selection(keys, input_df, filtered_dict)
# # # iterative_selection(keys, input_df, ratio_dict)