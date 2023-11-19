import time
import pandas as pd

start_time = time.time()
cc = pd.read_csv("cleaned_data.csv", low_memory=False)
end_time = time.time()
time2read = end_time - start_time
print(f'It takes {time2read} seconds to read the cleaned file')


start_time = time.time()
og = pd.read_csv("lc_loan_training.csv", low_memory=False)
end_time = time.time()
time2read = end_time - start_time
print(f'It takes {time2read} seconds to read the training file')

