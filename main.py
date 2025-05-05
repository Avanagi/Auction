import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

NUM_SIMULATIONS = 5010                    # число прогонов
BID_RANGE = (1, 50)                       # диапазон значений ставок и внутренней оценки v_i
NUM_BIDDERS_RANGE = (15, 50)              # диапазон числа участников в одном аукционе
GOODS_RANGE = (1, 5)                      # диапазон числа однотипных лотов
STRATEGIES = [                            # список коэффициентов стратегии b_i = coef * v_i
    1.05, 1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5
]

def first_price_auction(bids, num_goods):
    """
    bids: массив ставок всех участников
    num_goods: число лотов
    возвращает кортеж (x, p):
      x — бинарный массив выигрыша лота (1 — выиграл, 0 — нет)
      p — массив платежей (ставка победителя или 0)
    """
    sorted_indices = np.argsort(bids)[::-1]
    bids_new = np.sort(bids[sorted_indices])[::-1]
    bg = bids_new[:num_goods+1]
    winners = sorted_indices[:num_goods]
    x = np.zeros_like(bids)
    p = np.zeros_like(bids, dtype=float)
    count = 1

    for i in winners:
        x[i] = 1
        p[i] = bids[i]
    for i in winners:
        if x[i] == 1:
            p[i] = bg[count]
            count += 1
    return x, p

def simulate_strategy(v_i, strategy_coef, other_bids, num_goods, i_index):
    """
    v_i: внутренняя оценка фиксированного участника i
    strategy_coef: коэффициент стратегии
    other_bids: массив ставок всех остальных участников (размер N-1)
    num_goods: число лотов
    i_index: позиция участника i в итоговом массиве bids
    возвращает полезность u_i = (v_i - p_i) * x_i
    """
    b_i = strategy_coef * v_i
    bids = np.insert(other_bids, i_index, b_i)
    x, p = first_price_auction(bids, num_goods)
    u = (v_i - p[i_index]) * x[i_index]
    return u, bids, x, p

if __name__ == "__main__":
    results = {coef: [] for coef in STRATEGIES}
    example_printed = False

    for sim in range(NUM_SIMULATIONS):
        num_bidders = random.randint(*NUM_BIDDERS_RANGE)
        num_goods   = random.randint(*GOODS_RANGE)
        i = random.randint(0, num_bidders - 1)
        v_i = random.randint(*BID_RANGE)
        other_bids = np.random.randint(*BID_RANGE,
                                       size=num_bidders - 1)

        for coef in STRATEGIES:
            util, bids, x, p = simulate_strategy(
                v_i, coef, other_bids, num_goods, i
            )
            results[coef].append(util)

            if not example_printed:
                print("=== Пример симуляции ===")
                print(f"Число участников: {num_bidders}, число лотов: {num_goods}")
                print(f"Внутренняя оценка v_i: {v_i}, позиция i: {i}")
                print("Ставки всех участников:", np.round(bids, 2))
                print("Вектор выигрыша x:", x.astype(int))
                print("Вектор платежей p:", p)
                print("="*30, "\n")
                example_printed = True

    avg_utils = {k: np.mean(v) for k, v in results.items()}

    df = pd.DataFrame.from_dict(avg_utils,
                                orient='index',
                                columns=['Средняя полезность'])
    print("=== Результаты моделирования ===")
    print(df.sort_values(by='Средняя полезность', ascending=False), "\n")

    plt.figure(figsize=(10, 6))
    plt.bar(avg_utils.keys(), avg_utils.values(), width=0.04)
    plt.xlabel("Коэффициент стратегии")
    plt.ylabel("Средняя полезность")
    plt.title("Сравнение стратегий участника аукциона")
    plt.grid(True)
    plt.show()
