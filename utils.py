

def prompt_user_for_inputs():
    """
    Prompt the user to specify file paths, parameter, year, and indices.
    """
    wind_file = input("Enter the path to the Wind data file (e.g., data/Wind_2017_data.csv): ").strip()
    dscovr_file = input("Enter the path to the DSCOVR data file (e.g., data/DSCOVR_2017_data.csv): ").strip()
    parameter = input("Enter the parameter of interest (e.g., Speed, Density, Temperature): ").strip()
    year = input("Enter the year of interest (e.g., 2017, 2018, 2019): ").strip()
    train_start = int(input("Enter the starting index for training: "))
    train_end = int(input("Enter the ending index for training: "))
    eval_start = int(input("Enter the starting index for first-month evaluation: "))
    eval_end = int(input("Enter the ending index for first-month evaluation: "))

    future_month_indices = []
    while True:
        start_idx = input("Enter the starting index for evaluation of another month (or 'done' to finish): ").strip()
        if start_idx.lower() == "done":
            break
        end_idx = int(input("Enter the ending index for this evaluation month: ").strip())
        future_month_indices.append((int(start_idx), end_idx))

    return wind_file, dscovr_file, parameter, year, train_start, train_end, eval_start, eval_end, future_month_indices

