## {{{ imports

import itertools
import importlib
import os
import numpy as np
import plotter as p
import pandas as pd

## }}}

"""
'individual' es el nombre del individuo: los nidos de hornero A, B, 19, 23, 34, HAC1,
HAC2 y HEC1.
Sus archivos de pitch están en una carpeta en {base_path}/{individual}. Por ejemplo, un
archivo de pitch de HAC1 puede llamarse /home/user/datos_tesis/HAC1/pitch1.txt
"""
## {{{ las funcs


# {{{ generales
def make_sequence_start_at_zero(
    sequence,
):  # recibe una secuencia desde un file y devuelve la secuencia llevada a 0
    if not sequence:
        return []  # Return an empty list if the sequence is empty
    differences = [
        sequence[i] - sequence[i - 1] for i in range(1, len(sequence))
    ]  # Calculate the differences between consecutive elements
    reconstructed_sequence = [
        0
    ]  # Initialize the reconstructed sequence starting with 0
    for diff in differences:
        reconstructed_sequence.append(
            reconstructed_sequence[-1] + diff
        )  # Reconstruct the sequence using the differences
    return reconstructed_sequence


def file_parser(
    individual,
):  # le doy el individual y llega a cada txt en cada canción, y le reemplaza la columna
    individual_path = f"{base_path}/{individual}"
    for song_dir in os.listdir(individual_path):
        song_path = os.path.join(individual_path, song_dir)

        # Process each txt file in the song directory
        for txt_file in os.listdir(song_path):
            if txt_file.endswith(".txt"):
                txt_file_path = os.path.join(song_path, txt_file)
                reemplazar_col_tiempos(txt_file_path)


def reemplazar_col_tiempos(
    file_path,
):
    data = np.genfromtxt(file_path, delimiter=",")

    # Extract frequency values and remove NaN values
    time_values = make_sequence_start_at_zero(list(data[:, 0]))
    data[:, 0] = time_values
    np.savetxt(file_path, data, delimiter=",")


def file_parser_quiendamas(
    individual,
):
    lista_ult_valor_tiempo = []
    lista_max_pitches = []
    lista_medias = []
    lista_final_medias = []
    individual_path = f"{base_path}/{individual}"
    for song_dir in os.listdir(individual_path):
        song_path = os.path.join(individual_path, song_dir)

        # Process each txt file in the song directory
        for txt_file in os.listdir(song_path):
            if txt_file.endswith(".txt"):
                txt_file_path = os.path.join(song_path, txt_file)
                data = np.genfromtxt(txt_file_path, delimiter=",")
                ult_tiempo = data[:, 0][-1]
                lista_ult_valor_tiempo.append(ult_tiempo)
                lista_max_pitches.append(max(data[:, 1]))
        print(f"{individual}:{max(lista_max_pitches)}")


def centrar(
    individual,
):
    lista_ult_valor_tiempo = []
    individual_path = f"{base_path}/{individual}"
    for song_dir in os.listdir(individual_path):
        song_path = os.path.join(individual_path, song_dir)

        # Process each txt file in the song directory
        for txt_file in os.listdir(song_path):
            if txt_file.endswith(".txt"):
                txt_file_path = os.path.join(song_path, txt_file)
                data = np.genfromtxt(txt_file_path, delimiter=",")
                ult_tiempo = data[:, 0][-1]
                dif = (0.1295 - ult_tiempo) / 2
                data[:, 0] = data[:, 0] + dif
                np.savetxt(txt_file_path, data, delimiter=",")


def suavizar(time_series):
    window_size = 5
    ts_series = pd.Series(time_series)
    smoothed_series = ts_series.rolling(window=window_size).mean()

    return smoothed_series


def suavizar_dir(direc):
    file_list = [file for file in os.listdir(direc) if file.endswith(".txt")]
    for filename in file_list:
        full_filepath = os.path.join(direc, filename)
        data = np.genfromtxt(full_filepath, delimiter=",")
        data_suave = suavizar(data[:, 1])[4:]
        data[:, 1][2:-2] = data_suave
        np.savetxt(full_filepath, data, delimiter=",")


def suavizar_una_sola(filepath):
    data = np.genfromtxt(filepath, delimiter=",")
    data_suave = suavizar(data[:, 1])[4:]
    data[:, 1][2:-2] = data_suave
    np.savetxt(filepath, data, delimiter=",")


def generar_pngs(path, individuals):
    # individuals= ["A", "B", "19", "23", "34", "HEC1", "HAC1", "HAC2", "HCH"]:
    for individual in individuals:
        individual_path = os.path.join(path, individual)
        for root, dirs, files in os.walk(individual_path):
            for filename in files:
                if filename.endswith(".txt"):
                    full_filename = os.path.join(root, filename)
                    p.plot_pitch(full_filename, save=True)


def separar_silabas(file):
    data = np.genfromtxt(file, delimiter=",")
    start_index = 0
    filename = 0
    index_last_nan_silaba = 0
    while index_last_nan_silaba + 2 < len(data):
        if np.isnan(data[start_index, 1]):
            start_index += 1
            continue

        index_last_nan_silaba = (
            np.where(np.isnan(data[start_index:, 1]))[0][0] + start_index
        )
        print(index_last_nan_silaba)
        silaba = data[start_index:index_last_nan_silaba]
        filename += 1
        np.savetxt(
            f'{file.split("H")[0]}{filename}.txt',
            silaba,
            delimiter=",",
        )
        start_index = index_last_nan_silaba + 1
        print(index_last_nan_silaba)


def reemplazar_nans_comas(direc):
    file_list = [file for file in os.listdir(direc) if file.endswith(".txt")]
    for filename in file_list:
        full_filepath = os.path.join(direc, filename)
        with open(full_filepath, "r") as f:
            lines = f.readlines()

        # Remove the first row
        lines = lines[1:]

        # Replace three spaces with commas and '--undefined--' with 'nan'
        for i in range(len(lines)):
            lines[i] = lines[i].replace("   ", ",").replace("--undefined--", "nan")

        # Convert the modified lines to a numpy array
        matrix = np.array([line.strip().split(",") for line in lines])

        np.savetxt(full_filepath, matrix, delimiter=",", fmt="%s")


# }}}
# {{{ aumentación de datos combinando trazos


def next_nan_index(array):
    index_last_punto_trazo1 = np.where(np.isnan(array[:, 1]))[0][0] - 1
    trazo1 = array[: index_last_punto_trazo1 + 1]
    index_last_nan_trazo1 = (
        np.where(~np.isnan(array[index_last_punto_trazo1 + 1 :, 1]))[0][0]
        + index_last_punto_trazo1
    )
    return trazo1, index_last_nan_trazo1


def separar_trazos(
    direc_silabas, save_dir, cant_trazos=2
):  # si hay 3 trazos agarro los 2 primeros por un lado, si hay 4, agarro 2 y 2.
    file_list = [file for file in os.listdir(direc) if file.endswith(".txt")]
    for filename in file_list:
        full_filepath = os.path.join(direc, filename)
        data = np.genfromtxt(full_filepath, delimiter=",")
        if np.isnan(data[:, 1]).any():
            if cant_trazos == 2:
                trazo1, index_last_nan_trazo1 = next_nan_index(data)
                last_nan_trazo1 = data[index_last_nan_trazo1]
                trazo1 = np.vstack([trazo1, last_nan_trazo1])
                trazo2 = data[index_last_nan_trazo1 + 1 :]
                np.savetxt(
                    f'{save_dir}/trazo2/{filename.split(".")[0]}-trazo2.txt',
                    trazo2,
                    delimiter=",",
                )
                np.savetxt(
                    f'{save_dir}/trazo1/{filename.split(".")[0]}-trazo1.txt',
                    trazo1,
                    delimiter=",",
                )
            else:
                index_last_nan_1er_trazo = next_nan_index(data)[
                    1
                ]  # uso la func solo para encontrar el nan
                print(full_filepath)
                index_last_nan_segunda_parte = (
                    next_nan_index(data[index_last_nan_1er_trazo + 1 :])[1]
                    + index_last_nan_1er_trazo
                )
                primera_parte = data[: index_last_nan_segunda_parte + 1]
                last_nan_primera_parte = data[index_last_nan_segunda_parte]
                primera_parte = np.vstack([primera_parte, last_nan_primera_parte])
                segunda_parte = data[index_last_nan_segunda_parte + 1 :]
                np.savetxt(
                    f'{save_dir}/parte1/{filename.split(".")[0]}-parte1.txt',
                    primera_parte,
                )
                np.savetxt(
                    f'{save_dir}/parte2/{filename.split(".")[0]}-parte2.txt',
                    segunda_parte,
                )


def dif(trazo_1, trazo_2):
    print(trazo_2)
    # Extract frequency values and remove NaN values
    last_time1 = trazo_1[-1][
        0
    ]  # el momento del último nan; de acá en más debería empezar el 2do trazo
    first_time2 = trazo_2[0][0]
    dif = first_time2 - last_time1  # para restarle esta diferencia a todo first_time2
    trazo_2[:, 0] = trazo_2[:, 0] - dif

    return trazo_2


def merge(par_de_trazos):
    file1, file2 = par_de_trazos[0], par_de_trazos[1]
    print(file1)

    trazo1 = np.genfromtxt(
        file1, delimiter=" ", missing_values="nan", filling_values=np.nan
    )
    trazo2 = np.genfromtxt(
        file2, delimiter=" ", missing_values="nan", filling_values=np.nan
    )

    trazo2 = dif(trazo1, trazo2)

    merged = np.vstack((trazo1, trazo2))

    return merged


def combinaciones(lista_trazos1, lista_trazos2):
    combinations = []
    for item1 in lista_trazos1:
        # Iterate over each element in list2
        for item2 in lista_trazos2:
            # Append the combination [item1, it
            combinations.append(
                [item1, item2, f'{item1.split(".")[0]}-{item2.split(".")[0]}']
            )
    return combinations


def merge_and_save(trazos1, trazos2, save_dir):
    for index, items in enumerate(combinaciones(trazos1, trazos2)):
        print(items)
        merged_array = merge([items[0], items[1]])
        ult_tiempo = merged_array[:, 0][-1]
        dif = (0.16 - ult_tiempo) / 2
        merged_array[:, 0] = merged_array[:, 0] + dif
        rootname = items[2]

        np.savetxt(f"{save_dir}/{index}.txt", merged_array, delimiter=",", fmt="%.8f")


def random_erasing(individual):
    individual_path = f"{base_path}/{individual}"
    for song_dir in os.listdir(individual_path):
        song_path = os.path.join(individual_path, song_dir)

        # Process each txt file in the song directory
        for txt_file in os.listdir(song_path):
            print(txt_file)
            if txt_file.endswith(".txt"):
                txt_file_path = os.path.join(song_path, txt_file)
                data = np.genfromtxt(txt_file_path, delimiter=",")
                window_delete_length = np.random.uniform(0.038, 0.05)
                window_delete_start = np.random.uniform(data[0, 0], data[-1, 0] - 0.05)
                # Find the indices of the time window
                start_index = np.argmax(data[:, 0] >= window_delete_start)
                end_index = np.argmax(
                    data[:, 0] >= window_delete_start + window_delete_length
                )
                # Set the frequency values in the time window to np.nan
                data[start_index:end_index, 1] = np.nan
                np.savetxt(txt_file_path, data, delimiter=",")


def descentrar(
    individual,
):
    individual_path = f"{base_path}/{individual}"
    for song_dir in os.listdir(individual_path):
        song_path = os.path.join(individual_path, song_dir)

        # Process each txt file in the song directory
        for txt_file in os.listdir(song_path):
            if txt_file.endswith(".txt"):
                txt_file_path = os.path.join(song_path, txt_file)
                data = np.genfromtxt(txt_file_path, delimiter=",")
                jitter = np.random.uniform(0.04, 0.06)
                make_negative = np.random.choice([True, False])
                if make_negative:
                    jitter *= -1
                data[:, 0] += jitter
                np.savetxt(txt_file_path, data, delimiter=",")


# }}}

## }}}
