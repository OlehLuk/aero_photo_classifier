import math
from typing import List, Tuple, Union
from PIL import Image, ImageDraw
import datetime
import numpy as np
from matplotlib import pyplot as plt


class AeroPhotoClassifier:
    RESULT_NAME_TEMPLATE = "result_{}_{}"
    RESULT_DATE_FORMAT = "%d-%M-%y_%H%m%S"

    @staticmethod
    def launch(image_name: str, classes_images_names: List[str], colors: List, area_size: int,
               delta: int = None) -> None:
        # Отримуємо значення кольорів пікселів для зображень кожного класу
        classes_values = AeroPhotoClassifier.get_classes_values(classes_images_names)

        # Знаходимо оптимальне значення дельти для СКД
        plot_delta_data = None
        if delta is None:
            delta, plot_delta_data = AeroPhotoClassifier.get_optimal_delta(classes_values)
            AeroPhotoClassifier.RESULT_NAME_TEMPLATE += "_delta_auto"
        else:
            AeroPhotoClassifier.RESULT_NAME_TEMPLATE += f"_delta_{delta}"

        # Переводимо значення у бінарний вигляд та знаходимо еталонні вектори кожного класу
        class_binary_matrices = []
        class_vectors = []
        limit_vector = AeroPhotoClassifier.get_limit_vector(classes_values[0])
        for i in range(len(classes_values)):
            class_binary_matrix = AeroPhotoClassifier.get_binary_matrix(classes_values[i], limit_vector, delta)
            class_binary_matrices.append(class_binary_matrix)
            class_vectors.append(AeroPhotoClassifier.get_vector_from_binary_matrix(class_binary_matrix))

        # Знаходимо радіуси контейнера кожного класу
        radii, data_for_plots = AeroPhotoClassifier.get_radii(class_vectors, class_binary_matrices)
        print("Optimal radii:", radii)

        print("Building plot for the relation between Kulbak criterion value and radius; "
              "Kulbak criterion value and delta value.")
        AeroPhotoClassifier.build_plots(data_for_plots, colors, classes_images_names, image_name=image_name,
                                        plot_delta_data=plot_delta_data)

        print("Starting exam")
        # Класифікуємо зображення (екзамен)
        AeroPhotoClassifier.classify_image(image_name, colors, area_size, delta, limit_vector, class_vectors, radii)

    @staticmethod
    def get_classes_values(classes_images_names: List[str]) -> List[np.ndarray]:
        classes_values = []
        for class_image_name in classes_images_names:
            classes_values.append(AeroPhotoClassifier.img_to_array(class_image_name))
        return classes_values

    @staticmethod
    def get_radii(class_vectors: List[List[int]], class_binary_matrices: List[List[List[int]]]) -> Tuple[
        List[int], List[List[Union[list, List[float]]]]]:
        # Знаходимо сусідні класи
        pairs = AeroPhotoClassifier.make_pairs(class_vectors)
        # Знаходимо значення критерію для можливих значень радіусів
        criterion_values = []
        criterion_values.extend(
            AeroPhotoClassifier.get_criterion_values_for_classes_and_radii(class_vectors, class_binary_matrices, pairs))
        # Знаходимо оптимальні радіуси
        radii = []
        data_for_plots = []
        print("Calculation of radii for classes")
        for i in range(len(criterion_values)):
            print("Class number: " + str(i))

            res = criterion_values[i]
            x = list(range(len(res)))
            y = [x[0] for x in res]
            data_for_plots.append([x, y])

            print("Is working area | radius | criterion value")
            index = -1
            value = -1
            # Проходимо по всім можливив значенням радіуса
            for j in range(len(res)):
                print(f"{res[j][1]} {j} {res[j][0]}")
                # Якщо значення критерію у робочій області для даного радіуса більше за поточне оптимальне, то запам'ятовуємо його та значення радіуса
                if res[j][1] and res[j][0] >= value:
                    value = res[j][0]
                    index = j
            radii.append(index)

        return radii, data_for_plots

    @staticmethod
    def get_binary_matrix(values: List[List[int]], limit_vector: List[float], delta: int) -> List[List[int]]:
        """
        Метод для перетворення значень кольорів зображення у бінарний вигляд відносно СКД
        :param values: значення кольорів зображення, яке необхідно перетворити у бінарну матрицю
        :param limitVector: вектор, який задає СКД
        :param delta: значення дельти для СКД
        :return: бінарну матрицю зображення
        """
        binaryMatrix = [[0] * len(values[0]) for _ in range(len(values))]
        for i in range(len(values)):
            for j in range(len(values[0])):
                if values[i][j] >= limit_vector[j] - delta and values[i][j] <= limit_vector[j] + delta:
                    binaryMatrix[i][j] = 1
                else:
                    binaryMatrix[i][j] = 0

        return binaryMatrix

    @staticmethod
    def get_limit_vector(values: List[List[int]]) -> List[float]:
        """
        Метод для отримання вектора, який задає СКД
        :param values: значення кольорів базового класу
        :return: вектор, який задає СКД
        """
        limit_vector = []
        for i in range(len(values[0])):
            sum_ = 0
            for row in values:
                sum_ += row[i]
            limit_vector.append(sum_ / len(values))

        return limit_vector

    @staticmethod
    def classify_image(image_name: str, colors: List[str], area_size: int, delta: int, limit_vector: List[float],
                       class_vectors: List[List[int]],
                       radii: List[int]) -> None:

        image = Image.open(image_name)
        draw = ImageDraw.Draw(image)
        result_name = AeroPhotoClassifier.RESULT_NAME_TEMPLATE.format(datetime.datetime.now().strftime(
            AeroPhotoClassifier.RESULT_DATE_FORMAT), image_name[:-4].replace('/', '_'))
        width, height = image.size
        print("image size is ", width, height)

        for i in range(0, width, area_size):
            for j in range(0, height, area_size):
                crop = image.crop((i, j, i + area_size, j + area_size))
                crop_values = AeroPhotoClassifier.img_to_array_helper(crop)
                crop_binary_matrix = AeroPhotoClassifier.get_binary_matrix(crop_values, limit_vector, delta)
                class_number = -1
                class_value = 0
                # Проводимо екзамен області відносно кожного класу
                for k in range(len(class_vectors)):
                    res = AeroPhotoClassifier.exam(class_vectors[k], radii[k], crop_binary_matrix)
                    # /* Якщо значення після екзамену більше за поточне значення, то відносимо область до цього класу */
                    if res > class_value:
                        class_number = k
                        class_value = res

                if class_number != -1:
                    color = colors[class_number]
                    draw.text((i + area_size / 2, j + area_size / 2), str(class_number), fill=color)
                    draw.rectangle([(i, j), (i + area_size - 1, j + area_size - 1)], outline=color, width=2)

        image.show()
        image.save(f"output/{result_name}.png")

    @staticmethod
    def get_optimal_delta(classes_values: List[List[List[int]]]) -> int:
        """
        Метод для отримання оптимального значення дельти для СКД
        :param classes_values: значення кольорів пікселів для зображень кожного класу
        :return: оптимальне значення дельти для СКД
        """
        optimal_delta = 0
        optimal_delta_criterion_value = 0

        # Шукаємо оптимальне значення у інтервалі [1, 120]
        print("Calculation of the optimal delta")
        print("Delta | criterion value | criterion value in working area")
        plot_values = []
        for delta in range(1, 121):
            # Розраховуємо вектор, який задає СКД, бінарні матриці та еталонні вектори кожного класу
            class_binary_matrices = []
            class_vectors = [[] for _ in range(len(classes_values))]
            limit_vector = AeroPhotoClassifier.get_limit_vector(classes_values[0])
            for i, class_value in enumerate(classes_values):
                class_binary_matrix = AeroPhotoClassifier.get_binary_matrix(class_value, limit_vector, delta)
                class_binary_matrices.append(class_binary_matrix)
                class_vectors[i] = AeroPhotoClassifier.get_vector_from_binary_matrix(class_binary_matrix)

            # Шукаємо сусідів класів
            pairs = AeroPhotoClassifier.make_pairs(class_vectors)
            criterion_values = []
            # Для кожного класу знаходимо значення критеріїв
            criterion_values.extend(
                AeroPhotoClassifier.get_criterion_values_for_classes_and_radii(class_vectors, class_binary_matrices,
                                                                               pairs))

            # Обчислюємо середнє значення критерію та середнє значення критерію у робочій області
            sum_ = []
            sum_working_area = []
            for criterion_value in criterion_values:
                sum_.append(max([pair[0] for pair in criterion_value]))
                sum_working_area.append(max([pair[0] if pair[1] else -10 for pair in criterion_value]))

            avg_sum_ = np.mean(sum_)
            avg_sum_working_area = np.mean(sum_working_area)

            current_value = avg_sum_working_area
            plot_values.append([delta, avg_sum_, avg_sum_working_area])
            if current_value > optimal_delta_criterion_value:
                optimal_delta = delta
                optimal_delta_criterion_value = current_value

            print(f"{delta} {avg_sum_} {avg_sum_working_area if avg_sum_working_area > 0 else -1}")

        print("Optimal delta: ", optimal_delta)

        return optimal_delta, plot_values

    @staticmethod
    def get_criterion_values_for_classes_and_radii(class_vectors: List[List[int]],
                                                   class_binary_matrices: List[List[List[int]]],
                                                   pairs: List[List[int]]) -> List[List[Tuple[float, bool]]]:
        """
        Метод для обчислення значення критерію для класів та радіусів їх контейнера
        :param class_vectors: еталонні вектори кожного класу
        :param class_binary_matrices: бінарні матриці класів
        :param pairs: сусідні класи
        :return: значення критерію для класів та радіусів їх контейнера
        """

        criterionValues = []
        for _ in class_vectors:
            criterionValues.append([])

        for classNumber in range(len(class_vectors)):
            # does 0 radius make sense?
            for radius in range(61):
                d1 = [1 if i <= radius else 0 for i in
                      AeroPhotoClassifier.get_distances_between_vector_and_binary_matrix(
                          class_vectors[classNumber],
                          class_binary_matrices[classNumber])]
                d1 = np.mean(d1)

                alpha = 1 - d1

                beta = [1 if i <= radius else 0 for i in
                        AeroPhotoClassifier.get_distances_between_vector_and_binary_matrix(
                            class_vectors[classNumber],
                            class_binary_matrices[pairs[classNumber][0]])]

                beta = np.mean(beta)

                criterion_value = AeroPhotoClassifier.calculate_criterion(alpha, beta)
                is_working_area = (d1 >= 0.5 and beta < 0.5)

                criterionValues[classNumber].append((criterion_value, is_working_area))

        return criterionValues

    @staticmethod
    def make_pairs(class_vectors):
        """
        Метод для пошуку сусідів кожного класу
        :param class_vectors: еталонні вектори кожного класу
        :return: сусідів кожного класу
        """
        pairs = [[0, 0] for _ in range(len(class_vectors))]
        value_to_set = len(class_vectors[0]) + 1
        for pair in pairs:
            pair[0] = value_to_set
            pair[1] = value_to_set

        for i in range(len(class_vectors)):
            for j in range(len(class_vectors)):
                if i != j:
                    distance = AeroPhotoClassifier.get_distance_between_vectors(class_vectors[i],
                                                                                class_vectors[j])
                    if distance < pairs[i][1]:
                        pairs[i][0] = j
                        pairs[i][1] = distance
        return pairs

    @staticmethod
    def calculate_criterion(alpha, beta):
        """Method for calculating the criterion"""
        return AeroPhotoClassifier.calculate_kullback(alpha, beta) / AeroPhotoClassifier.calculate_kullback(0, 0)

    @staticmethod
    def calculate_kullback(alpha, beta):
        """Method for calculating Kullback's criterion"""
        return (math.log((2 - (alpha + beta) + 0.1) / (alpha + beta + 0.1)) / math.log(2)) * (1 - (alpha + beta))

    @staticmethod
    def get_distances_between_vector_and_binary_matrix(vector, binary_matrix):
        """
        Метод для пошуку відстаней між вектором та рядками бінарної матриці
        :param vector: вектор
        :param binary_matrix: бінарна матриця
        :return: відстані між вектором та рядками бінарної матриці
        """
        distances = []
        for binaryMatrixVector in binary_matrix:
            distances.append(AeroPhotoClassifier.get_distance_between_vectors(vector, binaryMatrixVector))
        return distances

    @staticmethod
    def get_distance_between_vectors(first_vector, second_vector):
        """
        Метод для пошуку відстаней між двома векторами
        :param first_vector: перший вектор
        :param second_vector: другий вектор
        :return: відстань між двома векторами
        """
        distance = 0
        for i in range(len(first_vector)):
            if first_vector[i] != second_vector[i]:
                distance += 1
        return distance

    @staticmethod
    def img_to_array(image_path):
        """
        Метод для отримання значеннь кольорів пікселів зображення
        :param image_path: шлях до зображення
        :return: значення кольорів пікселів зображення
        """
        image = Image.open(image_path)
        return AeroPhotoClassifier.img_to_array_helper(image)

    @staticmethod
    def img_to_array_helper(image):
        """
        Метод для отримання значеннь кольорів пікселів зображення
        :param image: об'єкт зображення
        :return: значення кольорів пікселів зображення
        """
        image_width, image_height = image.size
        values = [[0] * (image_width * 3) for _ in range(image_height)]
        for i in range(image_height):
            for j in range(image_width):
                color = image.getpixel((j, i))
                values[i][j] = color[0]
                values[i][j + image_width] = color[1]
                values[i][j + image_width * 2] = color[2]
        return values

    @staticmethod
    def exam(class_vector, radius, binary_matrix):
        """
        Метод для проведення екзамену на належність зображення до певного класу

        :param class_vector: еталонний вектор класу на належність до якого відбувається екзамен
        :param radius: радіус контейнера класу на належність до якого відбувається екзамен
        :param binary_matrix: бінарна матриця зображення, для якого відбувається пошук класу
        :return: результат екзамену
        """
        if radius <= 0:
            return 0

        sum = 0
        for aBinaryMatrix in binary_matrix:
            sum += 1 - AeroPhotoClassifier.get_distance_between_vectors(class_vector, aBinaryMatrix) / radius

        return sum / len(binary_matrix)

    @staticmethod
    def get_vector_from_binary_matrix(binary_matrix: List[List[int]]) -> List[int]:
        """
        /**
         * Метод для отримання еталонного вектора із бінарної матриці класу
         *
         * @param binaryMatrix бінарна матриця класу, для якого необхідно знайти еталонний вектор
         * @return еталонний вектор класу
         */
        :param binaryMatrix:
        :return:
        """
        vector = []
        for i in range(len(binary_matrix[0])):
            sum = 0
            for row in binary_matrix:
                sum += row[i]
            vector.append(round(sum / len(binary_matrix)))

        return vector

    @staticmethod
    def build_plots(data_for_plots, colors, classes_images_names, image_name=None, plot_delta_data=None):
        fig = plt.figure()
        for i, data in enumerate(data_for_plots):
            x, y = data
            label = classes_images_names[i].split("/")[-1][:-4]
            print(label, y)
            rgb_01 = tuple([x / 255. for x in colors[i]])
            plt.plot(x, y, color=rgb_01, label=label, figure=fig)

        plt.xlabel("Radius")
        plt.ylabel("Kulbak criterion")
        plt.legend()
        filename = AeroPhotoClassifier.RESULT_NAME_TEMPLATE.format(
            datetime.datetime.now().strftime(AeroPhotoClassifier.RESULT_DATE_FORMAT),
            image_name[:-4].replace('/', '_') if image_name is not None else ""
        )
        plt.savefig(f"output/{filename}_kulbak_radius_plot.png")

        if plot_delta_data is not None:
            fig = plt.figure()
            x = list([x[0] for x in plot_delta_data])
            y_1 = list([x[1] for x in plot_delta_data])
            y_2 = list([x[2] for x in plot_delta_data])
            plt.plot(x, y_1, color='red', label="Average criterion", figure=fig)
            plt.plot(x, y_2, color='blue', label="Average criterion in work area", figure=fig)

            plt.xlabel("Delta")
            plt.ylabel("Kulbak criterion")
            plt.legend()
            filename = AeroPhotoClassifier.RESULT_NAME_TEMPLATE.format(
                datetime.datetime.now().strftime(AeroPhotoClassifier.RESULT_DATE_FORMAT),
                image_name[:-4].replace('/', '_') if image_name is not None else ""
            )
            plt.savefig(f"output/{filename}_kulbak_delta_plot.png")
