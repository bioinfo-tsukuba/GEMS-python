import numpy as np
import polars as pl
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from typing import Tuple, List


def logistic(t: np.ndarray, r: float, n0: float, K: float = 1.0) -> np.ndarray:
    return K / (1 + (K/n0 - 1) * np.exp(-r * t))

def calc_optimal_time(target_density: float, r: float, n0: float, K: float = 1.0) -> float:
    if (target_density <= 0) | (1 <= target_density):
        raise ValueError('target_density should be between 0 and 1')
    nt = target_density
    passage_time_delta = np.log(((K - n0) * nt) / ((K - nt) * n0)) / r
    return passage_time_delta

def calculate_optimal_time_from_df(df: pl.DataFrame, target_density: float,
                                K: float = 1.0, time_column: str = "time", density_column: str = "density", operation_column: str = "operation", passage_group_column: str = "passage_group") -> float:
    # time, density, operation があることを確認
    assert time_column in df.columns, f"Column {time_column} as time column is not found."
    assert density_column in df.columns, f"Column {density_column} as density column is not found."
    assert operation_column in df.columns, f"Column {operation_column} as operation column is not found."

    # If the number of rows(density != null) is less than 3, return inf
    if len(df.filter(pl.col(density_column).is_not_null())) < 3:
        return float("inf")

    k, r, n0_fitted, df = fit_param_with_weight(df)
    n0 = n0_fitted[-1]

    # 最終Passage groupを取得
    last_passage_group = df[passage_group_column].max()
    df = df.filter(pl.col(passage_group_column) == last_passage_group)

    # 基準時間を取得
    base_time = df[time_column].min()

    # 密度が目標値になるまでの時間を計算
    rest_time = calc_optimal_time(target_density, r, n0, K)

    return base_time + rest_time

def group_by_passage(df: pl.DataFrame, time_column: str = "time", operation_column: str = "operation") -> pl.DataFrame:
    """
    Groups the DataFrame by passage.

    Args:
        df (pl.DataFrame): The DataFrame containing the data.
        time_column (str, optional): The name of the column representing time. Defaults to "time".
        operation_column (str, optional): The name of the column representing operation. Defaults to "operation".

    Returns:
        pl.DataFrame: The modified DataFrame containing an additional column "passage_group" that groups the data by passage.

    Raises:
        AssertionError: If the specified time or operation column is not found in the DataFrame.

    Notes:
        - The DataFrame is expected to be sorted by the time column.
        - The top row of the DataFrame is assumed to be a "Passage" operation.

    Example:
        >>> df = pl.DataFrame({
        ...     "time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ...     "operation": ["Passage, "GetImage", "GetImage", "GetImage", "Passage", "GetImage", "GetImage", "GetImage", "GetImage", "Passage"]
        ...
        ... })
        >>> df = group_by_passage(df)
    """
    # time, operation があることを確認
    assert time_column in df.columns, f"Column {time_column} as time column is not found."
    assert operation_column in df.columns, f"Column {operation_column} as operation column is not found."

    # time でソート
    df = df.sort(time_column)

    # Passage ごとに時間を取得し、グループ分け
    passage_times = df.filter(pl.col(operation_column) == "Passage")[time_column].to_list()

    # passage_group列を作成
    df = df.with_columns([
        pl.Series("passage_group", np.searchsorted(passage_times[1:], df[time_column], side="right"))
    ])

    return df


def fit_param(df: pl.DataFrame, time_column: str = "time", density_column: str = "density", operation_column: str = "operation") -> Tuple[float, float, List[float], pl.DataFrame]:
    """
    Fits parameters for a given DataFrame.

    Args:
        df (pl.DataFrame): The DataFrame containing the data.
        time_column (str, optional): The name of the column representing time. Defaults to "time".
        density_column (str, optional): The name of the column representing density. Defaults to "density".
        operation_column (str, optional): The name of the column representing operation. Defaults to "operation".

    Returns:
        Tuple[float, float, List[float], pl.DataFrame]: A tuple containing the fitted parameters (k, r, n0) and the modified DataFrame.
        - Modified DataFrame contains an additional column "passage_group" that groups the data by passage.

    Raises:
        AssertionError: If the specified time, density, or operation column is not found in the DataFrame.
        AssertionError: If the number of fitted n0 values is not equal to the number of passage groups.

    Notes:
        - The DataFrame is expected to be sorted by the time column.
        - The top row of the DataFrame is assumed to be a "Passage" operation, and the density value is assumed to be null.

    Example:
        >>> df = pl.DataFrame({
        ...     "time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ...     "density": [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ...     "operation": ["Passage, "GetImage", "GetImage", "GetImage", "Passage", "GetImage", "GetImage", "GetImage", "GetImage", "Passage"]
    """


    # time, density, operation があることを確認
    assert time_column in df.columns, f"Column {time_column} as time column is not found."
    assert density_column in df.columns, f"Column {density_column} as density column is not found."
    assert operation_column in df.columns, f"Column {operation_column} as operation column is not found."

    if "passage_group" not in df.columns:
        # passage_group列を作成
        df = group_by_passage(df, time_column, operation_column)

    # 各 passage_group ごとに最小密度 (n0) を推定
    n0_estimates = df.filter(pl.col(density_column).is_not_null()) \
        .group_by("passage_group") \
        .agg(pl.min(density_column).alias("n0")) \
        .to_dict(as_series=False)["n0"]

    # 全体のフィッティング (k, r の推定)
    def combined_model(t: np.ndarray, k: float, r: float, *n0_values: float) -> np.ndarray:
        result = np.empty_like(t)
        t = np.array(t)
        passage_groups = df["passage_group"].to_numpy()
        for i, n0 in enumerate(n0_values):
            mask = passage_groups == i
            times = t[mask]
            times.sort()
            base_time = times[0]
            base_time_index = np.where(t == base_time)[0][0]
            times = times - base_time
            result[mask] = logistic(times, r, n0)
            # base_time (null) の部分は 0 にする
            result[base_time_index] = 0
        


        return result

    # フィッティングを実行
    initial_guess = [1.0, 0.00001] + n0_estimates  # k, r, n0s の初期値
    popt, _ = curve_fit(
        combined_model,
        df[time_column].to_numpy(),
        df[density_column].fill_null(0).to_numpy(),
        p0=initial_guess,
    )

    k, r = popt[0], popt[1]
    n0_fitted = popt[2:]


    group_num = df["passage_group"].max()
    # Group num == len(n0_fitted) であることを確認
    assert len(n0_fitted) == group_num + 1, f"Number of n0_fitted ({len(n0_fitted)}) is not equal to the number of passage groups ({group_num + 1})."

    return k, r, n0_fitted, df



def fit_param_with_weight(df: pl.DataFrame, time_column: str = "time", density_column: str = "density", operation_column: str = "operation") -> Tuple[float, float, List[float], pl.DataFrame]:
    # time, density, operation があることを確認
    assert time_column in df.columns, f"Column {time_column} as time column is not found."
    assert density_column in df.columns, f"Column {density_column} as density column is not found."
    assert operation_column in df.columns, f"Column {operation_column} as operation column is not found."
    df_grouped = df

    if "passage_group" not in df_grouped.columns:
        # passage_group列を作成
        df_grouped = group_by_passage(df, time_column, operation_column)
    

    df = df_grouped.filter((pl.col(operation_column) == "GetImage") | (pl.col(operation_column) == "Passage"))

    # 各 passage_group ごとに最小密度 (n0) を推定
    n0_estimates = df.filter(pl.col(density_column).is_not_null()) \
        .group_by("passage_group") \
        .agg(pl.min(density_column).alias("n0")) \
        .to_dict(as_series=False)["n0"]
    
    weights = np.zeros(len(df))
    group_num = df["passage_group"].max()
    groups = df["passage_group"].to_numpy()
    weights = pow(1/2, groups)

    # 全体のフィッティング (k, r の推定)
    def combined_model(t: np.ndarray, k: float, r: float, *n0_values: float) -> np.ndarray:
        result = np.empty_like(t)
        t = np.array(t)
        passage_groups = df["passage_group"].to_numpy()
        for i, n0 in enumerate(n0_values):
            mask = passage_groups == i
            times = t[mask]
            times.sort()
            base_time = times[0]
            base_time_index = np.where(t == base_time)[0][0]
            times = times - base_time
            result[mask] = logistic(times, r, n0)
            # base_time (null) の部分は 0 にする
            result[base_time_index] = 0
        return result

    # フィッティングを実行
    initial_guess = [1.0, 0.00001] + n0_estimates  # k, r, n0s の初期値
    popt, _ = curve_fit(
        combined_model,
        df[time_column].to_numpy(),
        df[density_column].fill_null(0).to_numpy(),
        p0=initial_guess,
        sigma=weights,
        absolute_sigma=False  # sigma を重みとして解釈
    )

    k, r = popt[0], popt[1]
    n0_fitted = popt[2:]

    group_num = df["passage_group"].max()
    # Group num == len(n0_fitted) であることを確認
    assert len(n0_fitted) == group_num + 1, f"Number of n0_fitted ({len(n0_fitted)}) is not equal to the number of passage groups ({group_num + 1})."

    return k, r, n0_fitted, df_grouped


def plot_fit(df: pl.DataFrame, k: float, r: float, n0_fitted: list, time_column: str = "time", density_column: str = "density", operation_column: str = "operation"):
    # passage_group列が存在するか確認
    if "passage_group" not in df.columns:
        raise ValueError("passage_group column is missing in the DataFrame.")
    
    # フィッティングした結果を可視化
    plt.figure(figsize=(10, 6))
    
    # 元のデータをプロット
    plt.scatter(df[time_column], df[density_column], label='Original Data', color='blue')
    
    # フィッティングした曲線を描画
    passage_groups = df["passage_group"].to_numpy()
    time_values = df[time_column].to_numpy()
    for i, n0 in enumerate(n0_fitted):
        mask = passage_groups == i
        t_fit = time_values[mask]
        t_fit.sort()
        base_time = t_fit[0]
        t_fit = t_fit - base_time
        fitted_curve = logistic(t_fit, r, n0)

        t_fit = t_fit + base_time
        plt.plot(t_fit, fitted_curve, label=f'Fitted Curve Group {i+1}', linestyle='--')
    
    plt.xlabel("Time")
    plt.ylabel("Density")
    plt.title("Logistic Growth Fit")
    plt.legend()
    # 0~1の範囲に収める
    plt.ylim(0, 1)
    plt.show()

def generate_input_example(num_passages: int, k: float, r: float, passage_criteria: float = 0.5, time_point_interval: int = 60*24) -> pl.DataFrame:
    time_points = []
    densities = []
    operations = []
    
    current_time = 0
    for passage in range(num_passages):
        # Passage直後の密度を設定
        n0 = 0.3 + np.random.uniform(-0.05, 0.05)
        
        # Passageの時点を追加
        time_points.append(current_time)
        densities.append(None)  # Passage時点のdensityは計測しないと仮定
        operations.append("Passage")
        
        # Passage後の成長をシミュレート
        t = 0
        while True:
            t += time_point_interval
            density = logistic(t, r, n0)
            noised_density = density * np.random.uniform(0.95, 1.05)
            current_time += time_point_interval
            time_points.append(current_time)
            densities.append(noised_density)
            operations.append("GetImage")
            if density > passage_criteria:
                break
        r = r*0.9  # rを少し変更
        
        
        current_time += 1  # 次のPassageまでの時間を空ける
    
    df = pl.DataFrame({
        "time": time_points,
        "density": densities,
        "operation": operations
    })

    print(f"final r: {r/0.9}")
    
    return df



if __name__ == "__main__":
    print("Hello, World!")
    

    # 例として3回のPassageを持つデータを生成
    # 例: Passageが3回行われ、全体の時間が20、k=1.0, r=0.1の場合
    r_original = 0.0007
    passage_criteria: float = 0.7

    for num_passages in range(1, 10):
        example_df = generate_input_example(num_passages=num_passages, k=1.0, r=r_original, passage_criteria=passage_criteria)

        optimal_time = calculate_optimal_time_from_df(example_df, target_density=passage_criteria)
        print(f"Optimal Time: {optimal_time}")

        # パラメータの推定
        k, r, n0_fitted, df = fit_param_with_weight(example_df)
        print(f"k: {k}, r: {r}, n0: {n0_fitted}")
        # for i in df.rows():
        #     print(i)
        print(f"Difference in r: {abs(r - r_original)}, Passage: {num_passages}")

        # フィッティング結果の可視化
        plot_fit(df, k, r, n0_fitted)

    print(example_df)
        
