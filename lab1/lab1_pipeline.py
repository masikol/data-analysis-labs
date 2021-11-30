#!/usr/bin/env python3

import os
import re
import sys
import glob
import shutil
import argparse
import subprocess as sp


def get_classif_marks(output_lines):
    # Функция для того, чтобы парсить stdout YOL'ы
    # Она ищет метки классов в stdout'е и возвращает их
    classif_pattern = r'(.+): [0-9]+%'

    classif_lines = tuple(
        filter(
            lambda l: not re.search(classif_pattern, l) is None,
            output_lines
        )
    )

    for l in classif_lines:
        print(l)
    

    classif_marks = tuple(
        map(
            lambda l: re.search(classif_pattern, l).group(1),
            classif_lines
        )
    )

    return classif_marks
# end def get_classif_marks


# Пути к darknet, конфигурационным файлам
DARKNET_FPATH = '/home/deynonih/Misc_soft/darknet/darknet'
CFG_FPATH = '/home/deynonih/Misc_soft/darknet/cfg/yolov3.cfg'
WEIGHT_FPATH = '/home/deynonih/Misc_soft/darknet/yolov3.weights'

# Парсим аргументы командной строки
parser = argparse.ArgumentParser()

# Директория, где лежат входные картинки
parser.add_argument(
    '-i',
    '--indir',
    help='input directory',
    required=True
)

# Директория, куда надо сложить картинки после классификации
parser.add_argument(
    '-o',
    '--outdir',
    help='input directory',
    required=True
)

args = parser.parse_args()

indir_path = os.path.abspath(args.indir)
outdir_path = os.path.abspath(args.outdir)

# Если не нашлась входная директория -- останавливаем
if not os.path.isdir(indir_path):
    print(f'Error: input directory `{indir_path}` does not exist!')
    sys.exit(1)


# Если директория для выходных данных существует -- позволить
#   пользователю решить, что с ней делать: удалять/не удалять перед работой
if os.path.isdir(outdir_path):
    reply = None
    while reply is None:
        reply = input('Outdir directory exists. Remove it [Y,n]> ')
        if reply == '' or reply.upper() == 'Y':
            print(f'Removing directory `{outdir_path}`')
            try:
                shutil.rmtree(outdir_path)
            except OSError as err:
                print(str(err))
                sys.exit(1)
            
        elif reply.upper() == 'N':
            print(f'Keeping directory `{outdir_path}` untouched')
        else:
            print(f'Invalid reply: `{reply}`')



print(indir_path)
print(outdir_path)

# Получаем пути ко всем входным картинкам
input_fpaths = glob.glob(
    os.path.join(indir_path, '*.jpg')
)

# Заготовка команды для запуска darknet
darknet_command_draft = ' '.join([
    DARKNET_FPATH,
    'detect',
    CFG_FPATH,
    WEIGHT_FPATH,
])

# Словарь для меток классов
class_dict = dict()

# Множества с метками классов, по которым будут детертироваться
#   собаки и люди
DOG_MARKS = {'dog',}
PERSON_MARKS = {'person',}

# Перейти в директорию, где лежит исполняемый файл darknet
# Он (darknet) ищет какие-то свои файлы по относительным путям (относительно места, где лежит darknet).
# Поэтому лучше туда перейти временно, и не париться
os.chdir(
    os.path.dirname(DARKNET_FPATH)
)


# Начинаем классификацию

print('Detecting objects...')

for i, infpath in enumerate(input_fpaths):
    basename = os.path.basename(infpath)
    print(f'Doing image #{i+1}/{len(input_fpaths)}: {basename}')

    # Добавляем путь до входной картинки к заготовке коменды
    command = darknet_command_draft + f' "{infpath}"'

    # Запускаем darknet detect, перехватываем stdout, stderr
    pipe = sp.Popen(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout_stderr = pipe.communicate()
    if pipe.returncode != 0:
        print('Error while running darknet detect!')
        print(stdout_stderr.decode('utf-8'))
        sys.exit(1)
    else:
        out_lines = stdout_stderr[0].decode('utf-8').splitlines()
    

    # Парсим stdout, достаём оттуда метки класса
    classif_marks = set(
        map(
            lambda x: x.lower(),
            get_classif_marks(out_lines)
        )
    )

    # Заносим в class_dict метку класса текущей картинки
    if classif_marks == DOG_MARKS | PERSON_MARKS:
        # И собака, и человек нашлись на картинке
        class_dict[basename] = 'person_and_dog'
    elif DOG_MARKS <= classif_marks:
        # Нашлась собака
        class_dict[basename] = 'dog'
    elif PERSON_MARKS <= classif_marks:
        # Нашлёлся человек
        class_dict[basename] = 'person'
    else:
        # Не нашлось ни собак, ни людей
        class_dict[basename] = 'trash'

    # Развлечём пользователя: покажем ему результат классификации
    print(f'{basename} -- {class_dict[basename]}')



# Копируем файлы в результирующую директорию
print('\nCopying files...')

# Создаём поддиректории для разных классов картинок
for dirname in class_dict.values():
    if not os.path.isdir(os.path.join(outdir_path, dirname)):
        try:
            os.makedirs(os.path.join(outdir_path, dirname))
        except OSError as err:
            print(f'Error: cannot create directory `{os.path.join(outdir_path, dirname)}`')
            print(str(err))
            sys.exit(1)


# Рассортировать картинки
for i, infpath in enumerate(input_fpaths):

    basename = os.path.basename(infpath)

    # Путь до файлу-назначения
    dest = os.path.join(outdir_path, class_dict[basename], basename)

    # Копируем тут
    print(f'{infpath} -> {dest}')
    try:
        shutil.copy(
            infpath,
            dest
        )
    except OSError as err:
        print(f'Error: cannot copy file `{infpath}` -> `{dest}`')
        print(str(err))
        sys.exit(1)


print('Completed!')
print(outdir_path)
