# simple-cuda-raytracing

Совместное использование технологии MPI, технологии CUDA и технологии OpenMP для создания фотореалистической визуализации.

# Ключи запуска

--cpu Для расчетов используется центральный процессор

--gpu Для расчетов используется видеокарта

--default В stdout выводится конфигурация входных данных по умолчанию

# Пример запуска (Windows)

mpiexec -n 1 a.exe --default \(Вывести конфигурацию входных данных по умолчанию\)
mpiexec -n 4 a.exe --gpu (Запустить расчет. Рекомендуется использовать входные данные, полученные на предыдущем шаге)

Для корректной работы должна присутствовать директория images. После выполнения программы будут сгенерированы файлы с расширением .data.
Для получения изображений необходимо использовать скрипт conv.py

python conv.py all (Преобразует все файлы с расширением .data в изображения в папке images)
